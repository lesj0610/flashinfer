// Host driver for include/wgmma_emu/cvt.h (fork-only). Owner: T2.
//
//   cvt_driver <op> [params...]  < inputs  > outputs
//
// Every stdin line holds the operands of one case as hexadecimal words separated by blanks;
// every stdout line holds the result as one hexadecimal word. Parameters are decimal.
//
//   f32_to_fp8 e5m2 relu sat        x            f16_to_fp8 e5m2 relu sat     h
//   fp8_to_f16 e5m2 relu            v            f32x2_to_fp8x2 e5m2 relu sat a b
//   f16x2_to_fp8x2 e5m2 relu sat    a            fp8x2_to_f16x2 e5m2 relu     a
//   f32_to_tf32 rnd relu sat        x            f32_to_bf16 rnd              x
//   f16_to_bf16 rnd                 h            f64_to_bf16 rnd              d
//   int_to_bf16 signed bits rnd     v            f32_to_f16 rnd               x
//   bf16_to_f16 rnd                 b            f16_to_f32 / bf16_to_f32     h / b
//   f32_ftz                         x            f32_to_f64                   x
//   round_to_fmt fmt rnd sat        neg mag exp2 (exp2 as a signed decimal)
//
// rnd: 0 rn, 1 rz, 2 rm, 3 rp, 4 rna. fmt: tf32, f16, bf16, e4m3, e5m2.
// Driven by tests/misc/test_cvt_cpu.py against tests/misc/ref_cvt.py.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wgmma_emu/cvt.h"

namespace {

using namespace wgemu::cvt;

[[noreturn]] void die(const char* msg, const char* arg) {
  fprintf(stderr, "cvt_driver: %s%s\n", msg, arg ? arg : "");
  exit(2);
}

int param(int argc, char** argv, int i) {
  if (i >= argc) die("missing parameter for ", argv[1]);
  return atoi(argv[i]);
}

Rnd rnd(int v) {
  if (v < 0 || v > 4) die("bad rounding mode", nullptr);
  return static_cast<Rnd>(v);
}

Fmt fmt_by_name(const char* s) {
  if (!strcmp(s, "tf32")) return fmt_tf32();
  if (!strcmp(s, "f16")) return fmt_f16();
  if (!strcmp(s, "bf16")) return fmt_bf16();
  if (!strcmp(s, "e4m3")) return fmt_e4m3();
  if (!strcmp(s, "e5m2")) return fmt_e5m2();
  die("unknown format ", s);
}

// Reads up to n hexadecimal words from one line; returns the count read, -1 at end of input.
int read_words(uint64_t* w, int n, long long* signed_last) {
  char line[256];
  if (!fgets(line, sizeof line, stdin)) return -1;
  char* p = line;
  int k = 0;
  while (k < n) {
    char* end;
    if (signed_last && k == n - 1) {
      *signed_last = strtoll(p, &end, 10);
    } else {
      w[k] = strtoull(p, &end, 16);
    }
    if (end == p) break;
    p = end;
    ++k;
  }
  return k;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: cvt_driver <op> [params...]", nullptr);
  const char* op = argv[1];
  uint64_t w[3];
  long long e2 = 0;
  const bool rtf = !strcmp(op, "round_to_fmt");
  const int nin = rtf ? 3 : (!strcmp(op, "f32x2_to_fp8x2") ? 2 : 1);
  for (;;) {
    const int got = read_words(w, nin, rtf ? &e2 : nullptr);
    if (got < 0) break;
    if (got != nin) die("malformed input line for ", op);
    const uint32_t x = (uint32_t)w[0];
    uint64_t r;
    if (!strcmp(op, "f32_to_fp8")) {
      r = f32_to_fp8(x, param(argc, argv, 2), param(argc, argv, 3), param(argc, argv, 4));
    } else if (!strcmp(op, "f16_to_fp8")) {
      r = f16_to_fp8(x, param(argc, argv, 2), param(argc, argv, 3), param(argc, argv, 4));
    } else if (!strcmp(op, "fp8_to_f16")) {
      r = fp8_to_f16(x, param(argc, argv, 2), param(argc, argv, 3));
    } else if (!strcmp(op, "f32x2_to_fp8x2")) {
      r = f32x2_to_fp8x2(x, (uint32_t)w[1], param(argc, argv, 2), param(argc, argv, 3),
                         param(argc, argv, 4));
    } else if (!strcmp(op, "f16x2_to_fp8x2")) {
      r = f16x2_to_fp8x2(x, param(argc, argv, 2), param(argc, argv, 3), param(argc, argv, 4));
    } else if (!strcmp(op, "fp8x2_to_f16x2")) {
      r = fp8x2_to_f16x2(x, param(argc, argv, 2), param(argc, argv, 3));
    } else if (!strcmp(op, "f32_to_tf32")) {
      r = f32_to_tf32(x, rnd(param(argc, argv, 2)), param(argc, argv, 3), param(argc, argv, 4));
    } else if (!strcmp(op, "f32_to_bf16")) {
      r = f32_to_bf16(x, rnd(param(argc, argv, 2)));
    } else if (!strcmp(op, "f16_to_bf16")) {
      r = f16_to_bf16(x, rnd(param(argc, argv, 2)));
    } else if (!strcmp(op, "f64_to_bf16")) {
      r = f64_to_bf16(w[0], rnd(param(argc, argv, 2)));
    } else if (!strcmp(op, "int_to_bf16")) {
      r = int_to_bf16(w[0], param(argc, argv, 2), param(argc, argv, 3), rnd(param(argc, argv, 4)));
    } else if (!strcmp(op, "f32_to_f16")) {
      r = f32_to_f16(x, rnd(param(argc, argv, 2)));
    } else if (!strcmp(op, "bf16_to_f16")) {
      r = bf16_to_f16(x, rnd(param(argc, argv, 2)));
    } else if (!strcmp(op, "f16_to_f32")) {
      r = f16_to_f32(x);
    } else if (!strcmp(op, "bf16_to_f32")) {
      r = bf16_to_f32(x);
    } else if (!strcmp(op, "f32_ftz")) {
      r = f32_ftz(x);
    } else if (!strcmp(op, "f32_to_f64")) {
      r = f32_to_f64(x);
    } else if (rtf) {
      if (argc < 5) die("usage: round_to_fmt <fmt> <rnd> <sat>", nullptr);
      r = round_to_fmt(fmt_by_name(argv[2]), w[0] != 0, w[1], (int)e2, rnd(atoi(argv[3])),
                       atoi(argv[4]) != 0);
    } else {
      die("unknown op ", op);
    }
    printf("%llx\n", (unsigned long long)r);
  }
  return 0;
}
