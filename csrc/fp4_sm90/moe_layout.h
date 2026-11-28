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
// Host-only scratch layout of the SM90 FP4 MoE (no CUDA): the one place of the byte counts that
// fp4_moe_sm90.cu (run), prep.cuh (the prep workspace) and the CUTLASS binding's workspace size
// query for SM90 NVFP4 share.
#pragma once

#include <cstddef>
#include <cstdint>

namespace flashinfer {
namespace fp4_sm90 {
namespace moe_layout {

constexpr size_t kAlign = 256;
constexpr int kPrepChunk = 4096;  // routing entries per prep sort CTA
// Bytes a query adds so that a 128 B aligned buffer holds the scratch from a kAlign boundary.
constexpr size_t kQueryLead = 128;
constexpr int kMinN = 8, kMaxN = 256;  // n: powers of two in [kMinN, kMaxN]

inline size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

inline int max_tiles_of(int rows, int n, int num_local_experts) {
  const int bound = (rows + n - 1) / n + num_local_experts;
  return bound < rows ? bound : rows;
}

// B tile bytes of one GEMM: max_tiles n-tiles x ceil(K / 128) K tiles x n rows x 256 B.
inline size_t b_tiles_bytes(int max_tiles, int k, int n) {
  return size_t(max_tiles) * size_t((k + 127) / 128) * size_t(n) * 256;
}

// Bytes of the prep workspace of a routed prep: int32 per (sort chunk, local expert).
inline size_t prep_workspace_bytes(int tokens, int top_k, int num_local_experts) {
  int64_t const total = int64_t(tokens) * top_k;
  int64_t const chunks = (total + kPrepChunk - 1) / kPrepChunk;
  return size_t(chunks * num_local_experts) * sizeof(int32_t);
}

// The tile list of GEMM1 and of GEMM1's B tiles (n = n1) is the first set of arrays; the second set
// (expert_tile_base2 .. num_tiles2) is GEMM2's own list when n2 != n1 and takes no bytes otherwise.
struct Layout {
  size_t expert_offset, expert_tile_base, permuted_src, tile_expert, tile_n0, num_tiles;
  size_t tok1, tok2, x_tiles, act, x2_tiles, prep;
  size_t expert_tile_base2, tile_expert2, tile_n02, num_tiles2, total;
};

// n1 / n2: the n of GEMM1 / GEMM2. x_tiles are n1 tiles, act and x2_tiles (GEMM2's B tiles) n2
// tiles.
inline Layout make_layout(int64_t tokens_, int64_t top_k_, int64_t num_local_experts_,
                          int64_t hidden_, int64_t inter_, int64_t n1_, int64_t n2_,
                          bool per_token) {
  const int tokens = static_cast<int>(tokens_), top_k = static_cast<int>(top_k_);
  const int num_local_experts = static_cast<int>(num_local_experts_);
  const int k = static_cast<int>(hidden_), inter = static_cast<int>(inter_);
  const int n = static_cast<int>(n1_), n2 = static_cast<int>(n2_);
  const bool cross = n != n2;
  const int max_tiles = max_tiles_of(tokens * top_k, n, num_local_experts);
  const int max_tiles2 = max_tiles_of(tokens * top_k, n2, num_local_experts);
  const size_t rows = size_t(tokens) * top_k;
  const size_t experts = size_t(num_local_experts) + 1;
  Layout l{};
  size_t at = 0;
  auto take = [&at](size_t bytes) {
    const size_t off = at;
    at += align_up(bytes);
    return off;
  };
  l.expert_offset = take(experts * 4);
  l.expert_tile_base = take(experts * 4);
  l.permuted_src = take(rows * 4);
  l.tile_expert = take(size_t(max_tiles) * 4);
  l.tile_n0 = take(size_t(max_tiles) * 4);
  l.num_tiles = take(4);
#if defined(FP4_SM90_W4A16)
  // W4A16 (BF16 activations, no token scales): no tok1 region; tok2 holds the routing weight
  // of each GEMM2 row under kW4a16FusedFinalize
  l.tok1 = at;
  l.tok2 = take(per_token ? rows * 4 : 0);
#else
  l.tok1 = take(per_token ? rows * 4 : 0);
  l.tok2 = take(per_token ? rows * 4 : 0);
#endif
  l.x_tiles = take(b_tiles_bytes(max_tiles, k, n));
  // per token: BF16 GEMM1 rows for row_quant; per tensor: GEMM2 B tiles written by GEMM1
  l.act = take(per_token ? rows * size_t(inter) * 2 : b_tiles_bytes(max_tiles2, inter, n2));
  l.x2_tiles = take(per_token ? b_tiles_bytes(max_tiles2, inter, n2) : 0);
  l.prep = take(prep_workspace_bytes(tokens, top_k, num_local_experts));
  l.expert_tile_base2 = take(cross ? experts * 4 : 0);
  l.tile_expert2 = take(cross ? size_t(max_tiles2) * 4 : 0);
  l.tile_n02 = take(cross ? size_t(max_tiles2) * 4 : 0);
  l.num_tiles2 = take(cross ? 4 : 0);
  l.total = at;
  return l;
}

// A caller's buffer of one call: the layout, then src_permuted (int32 [rows]) and y
// ([rows, hidden] of out_bytes), each kAlign-aligned, when the call does not return them.
struct Scratch {
  size_t src_off, y_off, end;
};

inline Scratch scratch_layout(size_t layout_bytes, size_t rows, size_t hidden, size_t out_bytes,
                              bool carve_src, bool carve_y) {
  Scratch s{};
  size_t at = align_up(layout_bytes);
  s.src_off = at;
  if (carve_src) at += align_up(rows * 4);
  s.y_off = at;
  if (carve_y) at += align_up(rows * hidden * out_bytes);
  s.end = at;
  return s;
}

// The bytes that hold every execution variant a query's inputs do not tell apart: each pair
// (n1, n2), per-token or per-tensor, src_permuted and y both carved, plus kQueryLead.
inline size_t query_bytes(int64_t max_tokens, int64_t top_k, int64_t num_local_experts,
                          int64_t hidden, int64_t inter, int64_t out_bytes) {
  size_t most = 0;
  for (int n1 = kMinN; n1 <= kMaxN; n1 *= 2) {
    for (int n2 = kMinN; n2 <= kMaxN; n2 *= 2) {
      for (int per_token = 0; per_token < 2; ++per_token) {
        const Layout l = make_layout(max_tokens, top_k, num_local_experts, hidden, inter, n1, n2,
                                     per_token != 0);
        const Scratch s = scratch_layout(l.total, size_t(max_tokens) * size_t(top_k),
                                         size_t(hidden), size_t(out_bytes), true, true);
        if (s.end > most) most = s.end;
      }
    }
  }
  return kQueryLead + most;
}

}  // namespace moe_layout
}  // namespace fp4_sm90
}  // namespace flashinfer
