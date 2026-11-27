// Hand transcription of the PTX ISA 9.5 wgmma figures (ext/ptxisa/img/*.png).
//
// The values below were read off the images, not computed from spec.h, so
// test_spec_cpu.cu can use them as an independent reference. Each table keeps
// the figure number and image file name. Conventions shared by all tables:
//   - "row" / "col" are the matrix row and column labels printed in the figure.
//   - A thread cell "Tn:{a_x, a_x+1, ...}" lists elements low to high and covers
//     consecutive columns (logical columns in the sparse figures, where a cell
//     covers cell_cols logical columns and holds cell_elems compressed values).
//   - Shared-memory figures print one number per element. Inside a 16-byte
//     chunk the numbers always increase by one along the contiguous dimension,
//     so only the first number of each chunk is stored, with the coordinate of
//     that first element.
#pragma once

#include <stdint.h>

#include "wgmma_emu/types.h"

namespace wgemu {
namespace figtab {

// clang-format off

// ---------------------------------------------------------------------------
// Register fragment figures 151-158, 178, 180, 182
// ---------------------------------------------------------------------------

// Rows drawn in every register fragment figure (identical in all of them):
// row label, thread of the left-most cell (the next cells are t_first+1..+3),
// and which element group the row shows (0: the group shown in row 0,
// 1: the group shown in row 8).
struct FragRow {
  int row;
  int t_first;
  int group;
};
static const FragRow kFragRows[] = {
    {0, 0, 0},   {1, 4, 0},   {7, 28, 0},  {8, 0, 1},   {15, 28, 1}, {16, 32, 0},
    {31, 60, 1}, {32, 64, 0}, {47, 92, 1}, {48, 96, 0}, {63, 124, 1},
};
static const int kNumFragRows = sizeof(kFragRows) / sizeof(kFragRows[0]);

// Multiplicand A fragment figures.
struct AFig {
  int fig;
  const char* file;
  EType types[4];
  int ntypes;
  bool sparse;
  int cell_cols;    // matrix columns covered by one thread cell
  int right_col0;   // first column of the right half of the figure
  int cell_elems;   // elements listed in one cell
  int first[2][2];  // [row group][left half, right half]: first element in the cell
};
static const AFig kAFigs[] = {
    // 151: k16, cells {a0,a1} / {a2,a3}, columns 8.. {a4,a5} / {a6,a7}
    {151, "wgmma-64N16-A.png", {EType::F16, EType::BF16}, 2, false, 2, 8, 2, {{0, 4}, {2, 6}}},
    // 153: k8, cells a0 / a1, columns 4.. a2 / a3
    {153, "wgmma-64N8-A.png", {EType::TF32}, 1, false, 1, 4, 1, {{0, 2}, {1, 3}}},
    // 155: k32, cells {a0..a3} / {a4..a7}, columns 16.. {a8..a11} / {a12..a15}
    {155, "wgmma-64N32-A.png", {EType::E4M3, EType::E5M2, EType::S8, EType::U8}, 4, false, 4, 16, 4,
     {{0, 8}, {4, 12}}},
    // 157: k256, cells a0..a31 / a32..a63, columns 128.. a64..a95 / a96..a127
    // (the header of the second cell group is printed as "32 .. 64 | 64 .. 95",
    // an evident typo for 32..63 | 64..95)
    {157, "wgmma-64N256-A.png", {EType::B1}, 1, false, 32, 128, 32, {{0, 64}, {32, 96}}},
    // 178: sparse k32, 4-column chunks hold {a0,a1} / {a2,a3}, columns 16.. {a4,a5} / {a6,a7}
    {178, "sparse-wgmma-64N32-f16-bf16-A.png", {EType::F16, EType::BF16}, 2, true, 4, 16, 2,
     {{0, 4}, {2, 6}}},
    // 180: sparse k16, 2-column chunks hold a0 / a1, columns 8.. a2 / a3
    {180, "sparse-wgmma-64N16-tf32-A.png", {EType::TF32}, 1, true, 2, 8, 1, {{0, 2}, {1, 3}}},
    // 182: sparse k64, 8-column cells hold {a0..a3} / {a4..a7}, columns 32.. {a8..a11} / {a12..a15}
    {182, "sparse-wgmma-64N64-e4m3-e5m2-s8-u8-A.png",
     {EType::E4M3, EType::E5M2, EType::S8, EType::U8}, 4, true, 8, 32, 4, {{0, 8}, {4, 12}}},
};
static const int kNumAFigs = sizeof(kAFigs) / sizeof(kAFigs[0]);

// Accumulator D fragment figures. A block is a group of columns; its first
// column is either absolute or an offset from N (col0_from_n). The first
// element shown in rows of group 0 (top) and group 1 (bot) is either absolute
// or an offset from N/2 (from_half: X = N/2-4, Y = N/2-3, Z = N/2-2, W = N/2-1).
struct DBlock {
  int col0;
  bool col0_from_n;
  int top;
  int bot;
  bool from_half;
};
struct DFig {
  int fig;
  const char* file;
  EType types[4];  // input types whose D the figure describes
  int ntypes;
  int cell_cols;
  int cell_elems;
  int nblocks;
  DBlock blocks[4];
};
// 152 (k16), 154 (k8), 156 (k32) are the same drawing:
//   columns 0-7: {d0,d1} / {d2,d3}; 8-15: {d4,d5} / {d6,d7}; N-8..N-1: {dX,dY} / {dZ,dW}.
static const DFig kDFigs[] = {
    {152, "wgmma-64N16-D.png", {EType::F16, EType::BF16}, 2, 2, 2, 3,
     {{0, false, 0, 2, false}, {8, false, 4, 6, false}, {-8, true, -4, -2, true}}},
    {154, "wgmma-64N8-D.png", {EType::TF32}, 1, 2, 2, 3,
     {{0, false, 0, 2, false}, {8, false, 4, 6, false}, {-8, true, -4, -2, true}}},
    {156, "wgmma-64N32-D.png", {EType::E4M3, EType::E5M2, EType::S8, EType::U8}, 4, 2, 2, 3,
     {{0, false, 0, 2, false}, {8, false, 4, 6, false}, {-8, true, -4, -2, true}}},
};
static const int kNumDFigs = sizeof(kDFigs) / sizeof(kDFigs[0]);

// 158 (k256, b1 input, s32 D) is drawn differently: one column per cell,
//   columns 0-3: d0 / d1; 4-7: d2 / d3; N-8..N-5: dX / dY; N-4..N-1: dZ / dW.
static const DFig kFig158 = {
    158, "wgmma-64N256-D.png", {EType::B1}, 1, 1, 1, 4,
    {{0, false, 0, 1, false}, {4, false, 2, 3, false}, {-8, true, -4, -3, true}, {-4, true, -2, -1, true}}};

// One cell of figure 158 deviates from the figure's own pattern: row 15,
// column 7 reads "T31:d1" where every other row of group 1 in columns 4-7
// reads d3.
struct DCellNote {
  int fig;
  int row;
  int col;
  int t;
  int literal_elem;
  int pattern_elem;
};
static const DCellNote kFig158Typo = {158, 15, 7, 31, 1, 3};

// ---------------------------------------------------------------------------
// Sparse metadata figures 179, 181, 183, 184
// ---------------------------------------------------------------------------

// Cell "T_k: [hi..lo]": the metadata bits [hi..lo] of lane k (warp-relative,
// rows 0-15 of a warp). In 179 and 181 the lane is printed as T_{2i+k} with
// i = sparsity selector; `lane` stores k and lane_plus_2sel is set.
struct MetaCell {
  int lane;
  int hi;
  int lo;
};
struct MetaFig {
  int fig;
  const char* file;
  EType types[4];
  int ntypes;
  bool lane_plus_2sel;
  int chunk0;  // chunk index of the first column group in the figure
  int width;   // logical columns per chunk (column header groups)
  int nrows;
  int rows[6];
  MetaCell cells[6][8];
};
static const MetaFig kMetaFigs[] = {
    {179, "sparse-mma-metadata-16832-f16bf16.png", {EType::F16, EType::BF16}, 2, true, 0, 4, 6,
     {0, 1, 7, 8, 9, 15},
     {{{0, 3, 0}, {0, 7, 4}, {0, 11, 8}, {0, 15, 12}, {1, 3, 0}, {1, 7, 4}, {1, 11, 8}, {1, 15, 12}},
      {{4, 3, 0}, {4, 7, 4}, {4, 11, 8}, {4, 15, 12}, {5, 3, 0}, {5, 7, 4}, {5, 11, 8}, {5, 15, 12}},
      {{28, 3, 0}, {28, 7, 4}, {28, 11, 8}, {28, 15, 12}, {29, 3, 0}, {29, 7, 4}, {29, 11, 8}, {29, 15, 12}},
      {{0, 19, 16}, {0, 23, 20}, {0, 27, 24}, {0, 31, 28}, {1, 19, 16}, {1, 23, 20}, {1, 27, 24}, {1, 31, 28}},
      {{4, 19, 16}, {4, 23, 20}, {4, 27, 24}, {4, 31, 28}, {5, 19, 16}, {5, 23, 20}, {5, 27, 24}, {5, 31, 28}},
      {{28, 19, 16}, {28, 23, 20}, {28, 27, 24}, {28, 31, 28}, {29, 19, 16}, {29, 23, 20}, {29, 27, 24}, {29, 31, 28}}}},
    {181, "sparse-mma-metadata-16816-tf32.png", {EType::TF32}, 1, true, 0, 2, 6,
     {0, 1, 7, 8, 9, 15},
     {{{0, 3, 0}, {0, 7, 4}, {0, 11, 8}, {0, 15, 12}, {1, 3, 0}, {1, 7, 4}, {1, 11, 8}, {1, 15, 12}},
      {{4, 3, 0}, {4, 7, 4}, {4, 11, 8}, {4, 15, 12}, {5, 3, 0}, {5, 7, 4}, {5, 11, 8}, {5, 15, 12}},
      {{28, 3, 0}, {28, 7, 4}, {28, 11, 8}, {28, 15, 12}, {29, 3, 0}, {29, 7, 4}, {29, 11, 8}, {29, 15, 12}},
      {{0, 19, 16}, {0, 23, 20}, {0, 27, 24}, {0, 31, 28}, {1, 19, 16}, {1, 23, 20}, {1, 27, 24}, {1, 31, 28}},
      {{4, 19, 16}, {4, 23, 20}, {4, 27, 24}, {4, 31, 28}, {5, 19, 16}, {5, 23, 20}, {5, 27, 24}, {5, 31, 28}},
      {{28, 19, 16}, {28, 23, 20}, {28, 27, 24}, {28, 31, 28}, {29, 19, 16}, {29, 23, 20}, {29, 27, 24}, {29, 31, 28}}}},
    // 183: columns 0-31 (chunks 0-7).
    {183, "sparse-mma-metadata-16864-u8s8-first32col.png",
     {EType::E4M3, EType::E5M2, EType::S8, EType::U8}, 4, false, 0, 4, 6,
     {0, 1, 7, 8, 9, 15},
     {{{0, 3, 0}, {0, 7, 4}, {0, 11, 8}, {0, 15, 12}, {0, 19, 16}, {0, 23, 20}, {0, 27, 24}, {0, 31, 28}},
      {{4, 3, 0}, {4, 7, 4}, {4, 11, 8}, {4, 15, 12}, {4, 19, 16}, {4, 23, 20}, {4, 27, 24}, {4, 31, 28}},
      {{28, 3, 0}, {28, 7, 4}, {28, 11, 8}, {28, 15, 12}, {28, 19, 16}, {28, 23, 20}, {28, 27, 24}, {28, 31, 28}},
      {{1, 3, 0}, {1, 7, 4}, {1, 11, 8}, {1, 15, 12}, {1, 19, 16}, {1, 23, 20}, {1, 27, 24}, {1, 31, 28}},
      {{5, 3, 0}, {5, 7, 4}, {5, 11, 8}, {5, 15, 12}, {5, 19, 16}, {5, 23, 20}, {5, 27, 24}, {5, 31, 28}},
      {{29, 3, 0}, {29, 7, 4}, {29, 11, 8}, {29, 15, 12}, {29, 19, 16}, {29, 23, 20}, {29, 27, 24}, {29, 31, 28}}}},
    // 184: columns 32-63 (chunks 8-15); the column header is printed high to low ("35..32").
    {184, "sparse-mma-metadata-16864-u8s8-last32col.png",
     {EType::E4M3, EType::E5M2, EType::S8, EType::U8}, 4, false, 8, 4, 6,
     {0, 1, 7, 8, 9, 15},
     {{{2, 3, 0}, {2, 7, 4}, {2, 11, 8}, {2, 15, 12}, {2, 19, 16}, {2, 23, 20}, {2, 27, 24}, {2, 31, 28}},
      {{6, 3, 0}, {6, 7, 4}, {6, 11, 8}, {6, 15, 12}, {6, 19, 16}, {6, 23, 20}, {6, 27, 24}, {6, 31, 28}},
      {{30, 3, 0}, {30, 7, 4}, {30, 11, 8}, {30, 15, 12}, {30, 19, 16}, {30, 23, 20}, {30, 27, 24}, {30, 31, 28}},
      {{3, 3, 0}, {3, 7, 4}, {3, 11, 8}, {3, 15, 12}, {3, 19, 16}, {3, 23, 20}, {3, 27, 24}, {3, 31, 28}},
      {{7, 3, 0}, {7, 7, 4}, {7, 11, 8}, {7, 15, 12}, {7, 19, 16}, {7, 23, 20}, {7, 27, 24}, {7, 31, 28}},
      {{31, 3, 0}, {31, 7, 4}, {31, 11, 8}, {31, 15, 12}, {31, 19, 16}, {31, 23, 20}, {31, 27, 24}, {31, 31, 28}}}},
};
static const int kNumMetaFigs = sizeof(kMetaFigs) / sizeof(kMetaFigs[0]);

// Metadata examples 174-177. Value codes: 0 = zero element, 1 = x, 2 = y.
// The chunk and the operand are printed most significant element first.
// In 175 the index row splits the 64-bit two-wide chunk into four 16-bit
// halves labelled 11 10 01 00, and the metadata 11 10 points at the halves of
// element 1.
struct MetaExample {
  int fig;
  const char* file;
  EType types[4];
  int ntypes;
  int width;
  int chunk_hi_to_lo[4];  // first `width` entries used
  int nnz;
  int operand_hi_to_lo[2];  // first `nnz` entries used
  uint32_t field;
};
static const MetaExample kMetaExamples[] = {
    {174, "f16-metadata-example.png", {EType::F16, EType::BF16}, 2, 4, {2, 0, 0, 1}, 2, {2, 1}, 0xCu},
    {175, "tf32-metadata-example.png", {EType::TF32}, 1, 2, {1, 0, 0, 0}, 1, {1, 0}, 0xEu},
    {176, "u8s8-metadata-example.png", {EType::E4M3, EType::E5M2}, 2, 4, {2, 0, 0, 1}, 2, {2, 1}, 0xCu},
    {177, "u8s8-metadata-example.png", {EType::S8, EType::U8}, 2, 4, {2, 0, 0, 1}, 2, {2, 1}, 0xCu},
};
static const int kNumMetaExamples = sizeof(kMetaExamples) / sizeof(kMetaExamples[0]);

// ---------------------------------------------------------------------------
// Shared memory layout figures 159-173
// ---------------------------------------------------------------------------

// Each stored value is the first number printed in a 16-byte chunk. For the
// swizzle atom figures 159-166 the numbers count 16-byte chunks (stored here
// with 8-bit elements, so a chunk is 16 elements and value_unit_bytes = 16);
// for 167-173 they count elements. K-major figures: rows are mn, columns are k.
// MN-major figures: rows are mn, columns are k as well (the figures draw the
// contiguous MN dimension vertically).
// lbo / sbo: bytes stated in the example text (0 when the figure shows one
// atom and the offset never applies).
struct SmemFig {
  int fig;
  const char* file;
  bool mn_major;
  int swizzle_bytes;
  int elem_bits;
  int value_unit_bytes;
  uint32_t lbo;
  uint32_t sbo;
  int nrows;
  int ncols;
  int row_mn[16];
  int col_k[16];
  int value[16][16];
};
static const SmemFig kSmemFigs[] = {
    {159, "async-warpgroup-smem-layout-128B-mn.png", true, 128, 8, 16, 0, 0, 8, 8,
     {0, 16, 32, 48, 64, 80, 96, 112}, {0, 1, 2, 3, 4, 5, 6, 7},
     {{0, 9, 18, 27, 36, 45, 54, 63},
      {1, 8, 19, 26, 37, 44, 55, 62},
      {2, 11, 16, 25, 38, 47, 52, 61},
      {3, 10, 17, 24, 39, 46, 53, 60},
      {4, 13, 22, 31, 32, 41, 50, 59},
      {5, 12, 23, 30, 33, 40, 51, 58},
      {6, 15, 20, 29, 34, 43, 48, 57},
      {7, 14, 21, 28, 35, 42, 49, 56}}},
    {160, "async-warpgroup-smem-layout-128B-k.png", false, 128, 8, 16, 0, 0, 8, 8,
     {0, 1, 2, 3, 4, 5, 6, 7}, {0, 16, 32, 48, 64, 80, 96, 112},
     {{0, 1, 2, 3, 4, 5, 6, 7},
      {9, 8, 11, 10, 13, 12, 15, 14},
      {18, 19, 16, 17, 22, 23, 20, 21},
      {27, 26, 25, 24, 31, 30, 29, 28},
      {36, 37, 38, 39, 32, 33, 34, 35},
      {45, 44, 47, 46, 41, 40, 43, 42},
      {54, 55, 52, 53, 50, 51, 48, 49},
      {63, 62, 61, 60, 59, 58, 57, 56}}},
    {161, "async-warpgroup-smem-layout-64B-mn.png", true, 64, 8, 16, 0, 0, 4, 8,
     {0, 16, 32, 48}, {0, 1, 2, 3, 4, 5, 6, 7},
     {{0, 4, 9, 13, 18, 22, 27, 31},
      {1, 5, 8, 12, 19, 23, 26, 30},
      {2, 6, 11, 15, 16, 20, 25, 29},
      {3, 7, 10, 14, 17, 21, 24, 28}}},
    {162, "async-warpgroup-smem-layout-64B-k.png", false, 64, 8, 16, 0, 0, 8, 4,
     {0, 1, 2, 3, 4, 5, 6, 7}, {0, 16, 32, 48},
     {{0, 1, 2, 3},
      {4, 5, 6, 7},
      {9, 8, 11, 10},
      {13, 12, 15, 14},
      {18, 19, 16, 17},
      {22, 23, 20, 21},
      {27, 26, 25, 24},
      {31, 30, 29, 28}}},
    {163, "async-warpgroup-smem-layout-32B-mn.png", true, 32, 8, 16, 0, 0, 2, 8,
     {0, 16}, {0, 1, 2, 3, 4, 5, 6, 7},
     {{0, 2, 4, 6, 9, 11, 13, 15},
      {1, 3, 5, 7, 8, 10, 12, 14}}},
    {164, "async-warpgroup-smem-layout-32B-k.png", false, 32, 8, 16, 0, 0, 8, 2,
     {0, 1, 2, 3, 4, 5, 6, 7}, {0, 16},
     {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {9, 8}, {11, 10}, {13, 12}, {15, 14}}},
    {165, "async-warpgroup-smem-layout-mn-interleaved.png", true, 0, 8, 16, 0, 0, 1, 8,
     {0}, {0, 1, 2, 3, 4, 5, 6, 7},
     {{0, 1, 2, 3, 4, 5, 6, 7}}},
    {166, "async-warpgroup-smem-layout-k-interleaved.png", false, 0, 8, 16, 0, 0, 8, 1,
     {0, 1, 2, 3, 4, 5, 6, 7}, {0},
     {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}}},
    {167, "async-warpgroup-smem-layout-128B-k-tf32.png", false, 128, 32, 4, 0, 0, 8, 8,
     {0, 1, 2, 3, 4, 5, 6, 7}, {0, 4, 8, 12, 16, 20, 24, 28},
     {{0, 4, 8, 12, 16, 20, 24, 28},
      {36, 32, 44, 40, 52, 48, 60, 56},
      {72, 76, 64, 68, 88, 92, 80, 84},
      {108, 104, 100, 96, 124, 120, 116, 112},
      {144, 148, 152, 156, 128, 132, 136, 140},
      {180, 176, 188, 184, 164, 160, 172, 168},
      {216, 220, 208, 212, 200, 204, 192, 196},
      {252, 248, 244, 240, 236, 232, 228, 224}}},
    {168, "async-warpgroup-smem-layout-128B-mn-tf32.png", true, 128, 32, 4, 0, 0, 8, 8,
     {0, 4, 8, 12, 16, 20, 24, 28}, {0, 1, 2, 3, 4, 5, 6, 7},
     {{0, 36, 72, 108, 144, 180, 216, 252},
      {4, 32, 76, 104, 148, 176, 220, 248},
      {8, 44, 64, 100, 152, 188, 208, 244},
      {12, 40, 68, 96, 156, 184, 212, 240},
      {16, 52, 88, 124, 128, 164, 200, 236},
      {20, 48, 92, 120, 132, 160, 204, 232},
      {24, 60, 80, 116, 136, 172, 192, 228},
      {28, 56, 84, 112, 140, 168, 196, 224}}},
    // 169: LBO 256 B, SBO 128 B (wgmma_spec.txt:641-676)
    {169, "async-warpgroup-k-no-swizzle-tf32.png", false, 0, 32, 4, 256, 128, 16, 4,
     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {0, 4, 8, 12},
     {{0, 64, 128, 192},   {4, 68, 132, 196},   {8, 72, 136, 200},   {12, 76, 140, 204},
      {16, 80, 144, 208},  {20, 84, 148, 212},  {24, 88, 152, 216},  {28, 92, 156, 220},
      {32, 96, 160, 224},  {36, 100, 164, 228}, {40, 104, 168, 232}, {44, 108, 172, 236},
      {48, 112, 176, 240}, {52, 116, 180, 244}, {56, 120, 184, 248}, {60, 124, 188, 252}}},
    // 170: SBO 256 B (wgmma_spec.txt:678-713)
    {170, "async-warpgroup-k-32B-swizzle-tf32.png", false, 32, 32, 4, 0, 256, 16, 2,
     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {0, 4},
     {{0, 4},    {8, 12},   {16, 20},  {24, 28},  {36, 32},  {44, 40},   {52, 48},   {60, 56},
      {64, 68},  {72, 76},  {80, 84},  {88, 92},  {100, 96}, {108, 104}, {116, 112}, {124, 120}}},
    // 171: LBO 256 B, SBO 128 B (wgmma_spec.txt:715-750)
    {171, "async-warpgroup-mn-no-swizzle-bf16.png", true, 0, 16, 2, 256, 128, 2, 16,
     {0, 8}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
     {{0, 8, 16, 24, 32, 40, 48, 56, 128, 136, 144, 152, 160, 168, 176, 184},
      {64, 72, 80, 88, 96, 104, 112, 120, 192, 200, 208, 216, 224, 232, 240, 248}}},
    // 172: LBO 256 B, SBO 512 B (wgmma_spec.txt:752-787)
    {172, "async-warpgroup-mn-32B-swizzle-bf16.png", true, 32, 16, 2, 256, 512, 4, 16,
     {0, 8, 16, 24}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
     {{0, 16, 32, 48, 72, 88, 104, 120, 256, 272, 288, 304, 328, 344, 360, 376},
      {8, 24, 40, 56, 64, 80, 96, 112, 264, 280, 296, 312, 320, 336, 352, 368},
      {128, 144, 160, 176, 200, 216, 232, 248, 384, 400, 416, 432, 456, 472, 488, 504},
      {136, 152, 168, 184, 192, 208, 224, 240, 392, 408, 424, 440, 448, 464, 480, 496}}},
    // 173: LBO 512 B, SBO 1024 B (wgmma_spec.txt:789-824)
    {173, "async-warpgroup-mn-64B-swizzle-bf16.png", true, 64, 16, 2, 512, 1024, 8, 16,
     {0, 8, 16, 24, 32, 40, 48, 56}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
     {{0, 32, 72, 104, 144, 176, 216, 248, 512, 544, 584, 616, 656, 688, 728, 760},
      {8, 40, 64, 96, 152, 184, 208, 240, 520, 552, 576, 608, 664, 696, 720, 752},
      {16, 48, 88, 120, 128, 160, 200, 232, 528, 560, 600, 632, 640, 672, 712, 744},
      {24, 56, 80, 112, 136, 168, 192, 224, 536, 568, 592, 624, 648, 680, 704, 736},
      {256, 288, 328, 360, 400, 432, 472, 504, 768, 800, 840, 872, 912, 944, 984, 1016},
      {264, 296, 320, 352, 408, 440, 464, 496, 776, 808, 832, 864, 920, 952, 976, 1008},
      {272, 304, 344, 376, 384, 416, 456, 488, 784, 816, 856, 888, 896, 928, 968, 1000},
      {280, 312, 336, 368, 392, 424, 448, 480, 792, 824, 848, 880, 904, 936, 960, 992}}},
};
static const int kNumSmemFigs = sizeof(kSmemFigs) / sizeof(kSmemFigs[0]);

// "Exact layout" lines of the examples (wgmma_spec.txt:644, 681, 718, 755, 792):
// Swizzle<B,4,3> o (row modes):(row strides),(col modes):(col strides), in
// elements; mode 0 is mn, mode 1 is k. The swizzle applies to byte addresses.
struct ExactLayout {
  int fig;
  bool mn_major;
  int swizzle_bytes;
  int elem_bits;
  uint32_t lbo;
  uint32_t sbo;
  int nr;
  int rshape[3];
  int rstride[3];
  int nc;
  int cshape[3];
  int cstride[3];
};
static const ExactLayout kExactLayouts[] = {
    {169, false, 0, 32, 256, 128, 2, {8, 2, 1}, {4, 32, 0}, 2, {4, 4, 1}, {1, 64, 0}},
    {170, false, 32, 32, 0, 256, 2, {8, 2, 1}, {8, 64, 0}, 2, {4, 2, 1}, {1, 4, 0}},
    {171, true, 0, 16, 256, 128, 3, {8, 1, 2}, {1, 8, 64}, 2, {8, 2, 1}, {8, 128, 0}},
    {172, true, 32, 16, 256, 512, 3, {8, 2, 2}, {1, 8, 128}, 2, {8, 2, 1}, {16, 256, 0}},
    {173, true, 64, 16, 512, 1024, 3, {8, 4, 2}, {1, 8, 256}, 2, {8, 2, 1}, {32, 512, 0}},
};
static const int kNumExactLayouts = sizeof(kExactLayouts) / sizeof(kExactLayouts[0]);

// clang-format on

}  // namespace figtab
}  // namespace wgemu
