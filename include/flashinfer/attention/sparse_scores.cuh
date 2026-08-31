/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_
#define FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../mma.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace sparse_scores {

// One warp scores 16 columns against up to 16 query heads with one m16n16k16
// tile, walking the feature axis 16 at a time.
constexpr uint32_t kTileM = 16;
constexpr uint32_t kTileN = 16;
constexpr uint32_t kTileK = 16;
// Warps per block, and therefore columns per block. Each warp owns its own
// column tile and its own slice of the key staging buffer; more warps hide the
// gather behind another warp's mma and share the one staged query.
constexpr uint32_t kWarps = 4;
constexpr uint32_t kBlockN = kWarps * kTileM;
constexpr uint32_t kThreads = kWarps * 32;
// Query heads a block handles. More than this needs a second n-tile, which the
// caller currently never asks for.
constexpr uint32_t kMaxHeads = kTileN;
// Elements of padding on each staged row. ldmatrix reads 16 rows at once, and a
// head dimension that is a multiple of the 32 four-byte banks puts all of them
// in the same banks. One 16-byte unit of padding rotates the banks each row
// lands in while keeping every row aligned for the 16-byte loads that fill it.
// The bank geometry this rests on has been the same since sm_70.
constexpr uint32_t kSmemPad = 16 / sizeof(uint32_t) * 2;
// Feature slice staged at a time. A head dimension at or below this is staged
// whole, which lets one tile's keys land while the previous tile is multiplied.
// Above it, two whole-head buffers would leave an SM room for a single block,
// so the pipeline runs over slices of the head instead and the tiles go one at
// a time.
constexpr uint32_t kSliceK = 128;

}  // namespace sparse_scores

/*!
 * \brief Score every visible KV entry of a paged cache against a multi-head query.
 *
 * The score a sparse-attention selector ranks by:
 *
 *   score(row, col) = sum_h max(0, dot(K[col], Q[row, h])) / divisor
 *
 * There is no softmax and no value aggregation -- this produces the logits a
 * top-k runs on, not an attention output.
 *
 * Entries past what the query can see, and entries on a page the block table does
 * not map, come out as -inf so a top-k never selects them. The count of visible
 * entries per row is written out as well, since the selector needs it to bound
 * its own k.
 *
 * The ReLU applies to a head's completed dot product, so the feature axis has to
 * be fully accumulated before the heads are summed.
 *
 * \tparam HEAD_DIM per-head feature width, a multiple of 16
 * \tparam TILES_PER_BLOCK column tiles one block walks, to amortize staging the
 *   query across more columns when there are many rows to score
 */
template <uint32_t HEAD_DIM, uint32_t TILES_PER_BLOCK, typename DType, typename IdType>
__global__ void __launch_bounds__(sparse_scores::kThreads) SparsePagedScoresKernel(
    const DType* __restrict__ q, const DType* __restrict__ k_cache,
    const IdType* __restrict__ page_table, const IdType* __restrict__ token_to_req,
    const IdType* __restrict__ query_positions, const IdType* __restrict__ sequence_lengths,
    IdType* __restrict__ visible_out, float* __restrict__ logits, uint32_t stride_q_row,
    uint32_t stride_q_head, uint32_t stride_cache_page, uint32_t stride_cache_entry,
    uint32_t stride_table_req, uint32_t stride_logits_row, uint32_t rows, uint32_t num_columns,
    uint32_t num_pages, uint32_t num_requests, uint32_t table_width, uint32_t num_heads,
    uint_fastdiv page_size, uint_fastdiv compress_ratio, float divisor) {
  using namespace sparse_scores;
  constexpr uint32_t kSmemRow = HEAD_DIM + kSmemPad;
  constexpr uint32_t kSliceRow = kSliceK + kSmemPad;

  extern __shared__ uint8_t smem_raw[];
  // Query first: it is staged once and read by every column tile.
  DType* q_smem = reinterpret_cast<DType*>(smem_raw);
  DType* k_smem = q_smem + kTileN * kSmemRow;
  int32_t* pages_smem = reinterpret_cast<int32_t*>(k_smem + 2 * kBlockN * kSliceRow);
  uint32_t* entries_smem = reinterpret_cast<uint32_t*>(pages_smem + 2 * kBlockN);

  const uint32_t row = blockIdx.y;
  if (row >= rows) return;

  const int32_t request = static_cast<int32_t>(token_to_req[row]);
  const bool request_valid = request >= 0 && request < static_cast<int32_t>(num_requests);
  const int32_t safe_request = min(max(request, 0), static_cast<int32_t>(num_requests) - 1);
  const int32_t query_position = static_cast<int32_t>(query_positions[row]);
  const int32_t sequence_length =
      request_valid ? static_cast<int32_t>(sequence_lengths[safe_request]) : 0;

  // A query sees only the compressed entries whose tokens are all behind it.
  // Clamp before the unsigned divide: a row with no request carries a length of
  // zero, and a negative position would otherwise divide as a huge number.
  uint32_t q_blocks, k_blocks, ignored;
  compress_ratio.divmod(static_cast<uint32_t>(max(query_position + 1, 0)), q_blocks, ignored);
  compress_ratio.divmod(static_cast<uint32_t>(max(sequence_length, 0)), k_blocks, ignored);
  const uint32_t visible = min(min(q_blocks, k_blocks), num_columns);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    visible_out[row] = static_cast<IdType>(min(q_blocks, k_blocks));
  }

  const uint32_t first_column = blockIdx.x * (kBlockN * TILES_PER_BLOCK);
  if (first_column >= visible) return;

  // Stage the query once: [head][feature], which is the byte layout the
  // row-col mma wants for its column-major B operand.
  const uint32_t heads = min(num_heads, kMaxHeads);
  for (uint32_t i = threadIdx.x; i < kTileN * HEAD_DIM; i += kThreads) {
    const uint32_t h = i / HEAD_DIM;
    const uint32_t d = i - h * HEAD_DIM;
    q_smem[h * kSmemRow + d] = h < heads ? q[row * stride_q_row + h * stride_q_head + d] : DType(0);
  }
  __syncthreads();

  const int32_t* tile_pages = nullptr;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  // ldmatrix quadrant addresses, per the m8n8.x4 fragment layout: A is read as
  // 16 rows of 16 features, B as 16 heads of the same, but their quadrants are
  // ordered differently.
  const uint32_t a_row = lane % 16;
  const uint32_t a_col = (lane / 16) * 8;
  const uint32_t b_row = (lane % 8) + 8 * (lane / 16);
  const uint32_t b_col = ((lane % 16) / 8) * 8;

  constexpr uint32_t kPerVec = 16 / sizeof(DType);
  constexpr uint32_t kVecsPerColumn = HEAD_DIM / kPerVec;

  // Resolve one tile's columns into pages, and issue its key loads. Both are
  // split out so the next tile can be in flight while this one is multiplied.
  auto resolve = [&](uint32_t block_column, uint32_t slot) {
    int32_t page = -1;
    uint32_t entry = 0;
    if (threadIdx.x < kBlockN) {
      const uint32_t column = block_column + threadIdx.x;
      if (column < visible && request_valid) {
        uint32_t logical_page;
        page_size.divmod(column, logical_page, entry);
        if (logical_page < table_width) {
          const int32_t mapped =
              static_cast<int32_t>(page_table[safe_request * stride_table_req + logical_page]);
          if (mapped >= 0 && mapped < static_cast<int32_t>(num_pages)) page = mapped;
        }
      }
      pages_smem[slot * kBlockN + threadIdx.x] = page;
      entries_smem[slot * kBlockN + threadIdx.x] = entry;
    }
  };

  auto stage = [&](uint32_t slot, uint32_t k0) {
    DType* dst_base = k_smem + slot * kBlockN * kSliceRow;
    const uint32_t slice = min(kSliceK, HEAD_DIM - k0);
    for (uint32_t i = threadIdx.x; i < kBlockN * (slice / kPerVec); i += kThreads) {
      const uint32_t vecs = slice / kPerVec;
      const uint32_t c = i / vecs;
      const uint32_t v = i - c * vecs;
      const int32_t page = pages_smem[(HEAD_DIM <= kSliceK ? slot : 0u) * kBlockN + c];
      // A column with no page contributes nothing; zero-filling keeps the mma
      // clean and the -inf below keeps it unselectable.
      const DType* src = page >= 0 ? k_cache + static_cast<int64_t>(page) * stride_cache_page +
                                         entries_smem[(HEAD_DIM <= kSliceK ? slot : 0u) * kBlockN +
                                                      c] *
                                             stride_cache_entry +
                                         k0 + v * kPerVec
                                   : nullptr;
      cp_async::pred_load<128, cp_async::PrefetchMode::kNoPrefetch,
                          cp_async::SharedMemFillMode::kFillZero>(
          dst_base + c * kSliceRow + v * kPerVec, src, page >= 0);
    }
    cp_async::commit_group();
  };

  for (uint32_t t = 0; t < TILES_PER_BLOCK; ++t) {
    const uint32_t block_column = first_column + t * kBlockN;
    if (block_column >= visible) break;

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) acc[i] = 0.f;

    if constexpr (HEAD_DIM <= kSliceK) {
      // One slot holds a whole tile, so the next tile's keys land during this
      // tile's mma.
      const uint32_t slot = t & 1;
      const bool has_next =
          t + 1 < TILES_PER_BLOCK && first_column + (t + 1) * kBlockN < visible;
      if (t == 0) {
        resolve(block_column, 0);
        __syncthreads();
        stage(0, 0);
      }
      if (has_next) resolve(first_column + (t + 1) * kBlockN, slot ^ 1);
      cp_async::wait_group<0>();
      __syncthreads();
      if (has_next) stage(slot ^ 1, 0);

      const DType* tile_keys = k_smem + slot * kBlockN * kSliceRow;
      for (uint32_t k0 = 0; k0 < HEAD_DIM; k0 += kTileK) {
        uint32_t a_frag[4], b_frag[4];
        mma::ldmatrix_m8n8x4(a_frag, tile_keys + (warp * kTileM + a_row) * kSliceRow + k0 + a_col);
        mma::ldmatrix_m8n8x4(b_frag, q_smem + b_row * kSmemRow + k0 + b_col);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<DType>(acc, a_frag, b_frag);
      }
      tile_pages = pages_smem + slot * kBlockN;
    } else {
      // A slot holds one slice of the head, and the pipeline runs inside the
      // tile.
      __syncthreads();
      resolve(block_column, 0);
      __syncthreads();
      stage(0, 0);
      for (uint32_t k0 = 0, slice_idx = 0; k0 < HEAD_DIM; k0 += kSliceK, ++slice_idx) {
        const uint32_t slot = slice_idx & 1;
        const bool has_next = k0 + kSliceK < HEAD_DIM;
        cp_async::wait_group<0>();
        __syncthreads();
        if (has_next) stage(slot ^ 1, k0 + kSliceK);

        const DType* slice_keys = k_smem + slot * kBlockN * kSliceRow;
        const uint32_t slice = min(kSliceK, HEAD_DIM - k0);
        for (uint32_t kk = 0; kk < slice; kk += kTileK) {
          uint32_t a_frag[4], b_frag[4];
          mma::ldmatrix_m8n8x4(a_frag,
                               slice_keys + (warp * kTileM + a_row) * kSliceRow + kk + a_col);
          mma::ldmatrix_m8n8x4(b_frag, q_smem + b_row * kSmemRow + k0 + kk + b_col);
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DType>(acc, a_frag, b_frag);
        }
        // The slice after next reuses this slot.
        __syncthreads();
      }
      tile_pages = pages_smem;
    }

    // C fragment: with g = lane >> 2 and u = lane & 3, this thread holds
    // (g, 2u) (g, 2u+1) (g+8, 2u) (g+8, 2u+1) (g, 8+2u) (g, 9+2u)
    // (g+8, 8+2u) (g+8, 9+2u) -- two columns of the tile and four heads each.
    const uint32_t g = lane >> 2;
    const uint32_t u = lane & 3;
    float s0 = fmaxf(acc[0], 0.f) + fmaxf(acc[1], 0.f) + fmaxf(acc[4], 0.f) + fmaxf(acc[5], 0.f);
    float s1 = fmaxf(acc[2], 0.f) + fmaxf(acc[3], 0.f) + fmaxf(acc[6], 0.f) + fmaxf(acc[7], 0.f);
    // The four lanes sharing a row hold the remaining heads.
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);

    // A column on an unmapped page staged as zeros, which would score 0; it has
    // to be unselectable instead.
    if (u == 0) {
      const uint32_t c0 = block_column + warp * kTileM + g;
      const uint32_t c1 = block_column + warp * kTileM + g + 8;
      if (c0 < visible) {
        logits[row * stride_logits_row + c0] =
            tile_pages[warp * kTileM + g] >= 0 ? s0 / divisor : -INFINITY;
      }
      if (c1 < visible) {
        logits[row * stride_logits_row + c1] =
            tile_pages[warp * kTileM + g + 8] >= 0 ? s1 / divisor : -INFINITY;
      }
    }
    // The tile after next reuses this slot.
    __syncthreads();
  }
}

template <uint32_t HEAD_DIM, typename DType, typename IdType>
cudaError_t SparsePagedScores(const DType* q, const DType* k_cache, const IdType* page_table,
                              const IdType* token_to_req, const IdType* query_positions,
                              const IdType* sequence_lengths, IdType* visible_out, float* logits,
                              uint32_t stride_q_row, uint32_t stride_q_head,
                              uint32_t stride_cache_page, uint32_t stride_cache_entry,
                              uint32_t stride_table_req, uint32_t stride_logits_row, uint32_t rows,
                              uint32_t num_columns, uint32_t num_pages, uint32_t num_requests,
                              uint32_t table_width, uint32_t num_heads, uint32_t page_size,
                              uint32_t compress_ratio, float divisor, cudaStream_t stream) {
  using namespace sparse_scores;
  if (rows == 0 || num_columns == 0) return cudaSuccess;
  if (num_heads > kMaxHeads) return cudaErrorInvalidValue;
  if (HEAD_DIM % kTileK != 0) return cudaErrorInvalidValue;

  constexpr uint32_t kSmemRow = HEAD_DIM + kSmemPad;
  constexpr uint32_t kSliceRow = kSliceK + kSmemPad;
  const size_t smem_size = (kTileN * kSmemRow + 2 * kBlockN * kSliceRow) * sizeof(DType) +
                           2 * kBlockN * (sizeof(int32_t) + sizeof(uint32_t));

  // Few rows leave the device idle unless every column tile is its own block;
  // many rows are better off amortizing the query staging across tiles. The
  // crossover is where the narrow choice stops filling the device, which
  // depends on how many blocks this GPU can hold.
  int dev_id = 0, num_sms = 0, max_smem_per_block_optin = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
      &max_smem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
  if (smem_size > static_cast<size_t>(max_smem_per_block_optin)) {
    std::ostringstream err_msg;
    err_msg << "Required shared memory (" << smem_size << " bytes) for head_dim=" << HEAD_DIM
            << " exceeds this GPU's per-block limit (" << max_smem_per_block_optin
            << " bytes); this configuration is not supported on this architecture.";
    FLASHINFER_ERROR(err_msg.str());
  }

  // One column tile per block gives the most blocks, which is what a handful of
  // rows needs; past a full wave of them the extra blocks only re-stage the
  // query, so a block walks several tiles instead.
  auto launch = [&](auto tiles_tag) -> cudaError_t {
    constexpr uint32_t TILES = decltype(tiles_tag)::value;
    auto kernel = SparsePagedScoresKernel<HEAD_DIM, TILES, DType, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    const dim3 grid(ceil_div(num_columns, kBlockN * TILES), rows);
    kernel<<<grid, kThreads, smem_size, stream>>>(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_out,
        logits, stride_q_row, stride_q_head, stride_cache_page, stride_cache_entry,
        stride_table_req, stride_logits_row, rows, num_columns, num_pages, num_requests,
        table_width, num_heads, uint_fastdiv(page_size), uint_fastdiv(compress_ratio), divisor);
    return cudaGetLastError();
  };

  int blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, SparsePagedScoresKernel<HEAD_DIM, 1, DType, IdType>, kThreads, smem_size));
  const uint32_t narrow_blocks = rows * ceil_div(num_columns, kBlockN);
  if (blocks_per_sm == 0 ||
      narrow_blocks <= static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(blocks_per_sm)) {
    return launch(std::integral_constant<uint32_t, 1>{});
  }
  return launch(std::integral_constant<uint32_t, 8>{});
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_SPARSE_SCORES_CUH_
