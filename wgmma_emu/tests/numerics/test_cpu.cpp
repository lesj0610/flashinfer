// Acceptance criterion 2(a) (PLAN section 1): numerics.h against MMA-Sim vectors on the CPU.
//
// Reads the files written by gen_vectors.py and recomputes every dot product with numerics.h:
//   float  dot_float with wgmma_params (wgmma.*) or mma_params (mma.*); bit-exact compare
//   f64    dot_f64_fma; bit-exact compare except NaN vs NaN, which counts as a match (MMA-Sim's
//          own equivalence tests use allow_different_nan for f64; the NaN payload of its
//          reference is the host CPU's std::fma, not the GPU's) and is reported separately
//   int    dot_int; b1: dot_b1_and_popc (exact Python references)
// The expected combination list is compiled in (the MMA-Sim instruction lists, see
// gen_vectors.py), so a missing file is reported as cases=0, which fails (PLAN section 11).
// Every file header is checked against the combination it claims to be before any compare, and
// for float combinations the (F, L_max, e_zero, rho) MMA-Sim used is compared with numerics.h.
//
// Build: g++ -std=c++17 -O2 -pthread -I include tests/numerics/test_cpu.cpp -o test_cpu
// Run:   ./test_cpu <vector_dir> [--threads N] [--max-print N] [--only SUBSTRING]
// Output: per combination "WGEMU-TEST name=<id> cases=<n> mismatches=<m>" (plus
// "name=header:<id>" and "name=params:<id>" lines), WGEMU-MISMATCH details for the first
// mismatches, then WGEMU-SUMMARY and WGEMU-TEST-END. Exit status 0 only if every line passed.
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "wgmma_emu/numerics.h"

namespace {

using wgemu::Arch;
using wgemu::EType;
using wgemu::FdpaParams;
using wgemu::ParamSource;
using wgemu::Rho;

// Must match gen_vectors.py (HEADER_FMT "<8s13I3iIQ12x", KIND_*, FLAG_*, ARCH_CODE, API_CODE).
constexpr size_t kHeaderSize = 96;
constexpr uint32_t kVersion = 1;
constexpr uint32_t kKindFloat = 0, kKindF64 = 1, kKindInt = 2, kKindB1 = 3;
constexpr uint32_t kFlagSat = 1, kFlagASigned = 2, kFlagBSigned = 4, kFlagInferred = 8;
constexpr uint32_t kArchAmpere = 0, kArchAda = 1, kArchHopper = 2, kArchNone = 3;
constexpr uint32_t kApiMma = 0, kApiWgmma = 1, kApiNone = 2;
constexpr uint32_t kCodeF64 = 10;  // EType has no f64; only the vector files use this code

struct Header {
  uint32_t version, kind, atype, btype, d_f16, arch, api, K, units, a_bytes, b_bytes, cd_bytes,
      flags;
  int32_t F, L_max, e_zero;
  uint32_t rho;
  uint64_t count;
};

struct Expect {
  std::string name;
  uint32_t kind, atype, btype, d_f16, arch, api, K, units, flags;
};

uint64_t load_le(const uint8_t* p, int nbytes) {
  uint64_t v = 0;
  for (int i = nbytes - 1; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

uint32_t type_code(const std::string& t) {
  static const char* const kNames[] = {"f16", "bf16", "tf32", "e4m3", "e5m2", "s8",
                                       "u8",  "b1",   "f32",  "s32",  "f64"};
  for (uint32_t i = 0; i < 11; ++i)
    if (t == kNames[i]) return i;
  fprintf(stderr, "test_cpu: unknown type %s\n", t.c_str());
  exit(2);
}

uint32_t type_bytes(uint32_t code) {
  switch (code) {
    case 0:
    case 1:
      return 2;
    case 2:
    case 8:
    case 9:
      return 4;
    case 3:
    case 4:
    case 5:
    case 6:
      return 1;
    case 7:
      return 4;  // b1 is stored as 32-bit words
    case kCodeF64:
      return 8;
    default:
      return 0;
  }
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  for (;;) {
    const size_t pos = s.find(sep, start);
    out.push_back(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
    if (pos == std::string::npos) return out;
    start = pos + 1;
  }
}

// mma "m16n8k16.f32.f16.f16.f32" (nv_ptx/isa_mma.py) for one architecture.
Expect mma_entry(const std::string& prefix, uint32_t arch, const std::string& st, uint32_t flags) {
  const std::vector<std::string> q = split(st, '.');
  const uint32_t K = static_cast<uint32_t>(atoi(q[0].c_str() + q[0].find('k') + 1));
  const uint32_t a = type_code(q[2]);
  const uint32_t kind = a == kCodeF64 ? kKindF64 : kKindFloat;
  Expect e{prefix + st, kind, a, type_code(q[3]), 0, arch, kApiMma, K, K, flags};
  e.d_f16 = q[1] == "f16" ? 1u : 0u;
  return e;
}

// The combinations gen_vectors.py enumerates from MMA-Sim b953d74 (isa_mma.py, isa_wgmma.py).
std::vector<Expect> expected_list() {
  std::vector<Expect> out;
  const char* const kWgmma[] = {"k8.f32.tf32.tf32", "k16.f32.f16.f16", "k16.f16.f16.f16",
                                "k16.f32.bf16.bf16"};
  std::vector<std::string> wg(kWgmma, kWgmma + 4);
  const char* const kFp8[] = {"e5m2", "e4m3"};
  for (const char* d : {"f32", "f16"})
    for (const char* a : kFp8)
      for (const char* b : kFp8) wg.push_back(std::string("k32.") + d + "." + a + "." + b);
  for (const std::string& s : wg) {
    const std::vector<std::string> q = split(s, '.');
    const uint32_t K = static_cast<uint32_t>(atoi(q[0].c_str() + 1));
    out.push_back({"wgmma.hopper." + s, kKindFloat, type_code(q[2]), type_code(q[3]),
                   q[1] == "f16" ? 1u : 0u, kArchHopper, kApiWgmma, K, K, 0});
  }
  const std::vector<std::string> sm75 = {"m16n8k8.f32.f16.f16.f32", "m16n8k8.f16.f16.f16.f16"};
  const std::vector<std::string> sm80 = {"m8n8k4.f64.f64.f64.f64",    "m16n8k8.f32.tf32.tf32.f32",
                                         "m16n8k4.f32.tf32.tf32.f32", "m16n8k16.f32.f16.f16.f32",
                                         "m16n8k16.f16.f16.f16.f16",  "m16n8k16.f32.bf16.bf16.f32",
                                         "m16n8k8.f32.bf16.bf16.f32"};
  std::vector<std::string> sm89;
  for (int k : {32, 16})
    for (const char* d : {"f32", "f16"})
      for (const char* a : kFp8)
        for (const char* b : kFp8)
          sm89.push_back("m16n8k" + std::to_string(k) + "." + d + "." + a + "." + b + "." + d);
  const std::vector<std::string> sm90 = {"m16n8k16.f64.f64.f64.f64", "m16n8k8.f64.f64.f64.f64",
                                         "m16n8k4.f64.f64.f64.f64"};
  auto add = [&](const std::string& prefix, uint32_t arch,
                 std::initializer_list<const std::vector<std::string>*> lists, uint32_t flags) {
    for (const std::vector<std::string>* l : lists)
      for (const std::string& st : *l) out.push_back(mma_entry(prefix, arch, st, flags));
  };
  add("mma.ampere.", kArchAmpere, {&sm80, &sm75}, 0);
  add("mma.ada.", kArchAda, {&sm89, &sm80, &sm75}, 0);
  add("mma.hopper.", kArchHopper, {&sm90, &sm80, &sm75}, 0);
  add("mma.hopper-inferred.", kArchHopper, {&sm89}, kFlagInferred);
  for (uint32_t k : {16u, 32u})
    for (const char* a : {"s8", "u8"})
      for (const char* b : {"s8", "u8"})
        for (bool sat : {false, true}) {
          const uint32_t flags = (sat ? kFlagSat : 0) | (a[0] == 's' ? kFlagASigned : 0) |
                                 (b[0] == 's' ? kFlagBSigned : 0);
          out.push_back(
              {"int.k" + std::to_string(k) + "." + a + "." + b + (sat ? ".satfinite" : ".wrap"),
               kKindInt, type_code(a), type_code(b), 0, kArchNone, kApiNone, k, k, flags});
        }
  for (uint32_t k : {128u, 256u})
    out.push_back({"b1.k" + std::to_string(k) + ".and.popc", kKindB1, 7, 7, 0, kArchNone, kApiNone,
                   k, k / 32, 0});
  return out;
}

bool g_failed_any = false;

void emit(const std::string& name, uint64_t cases, uint64_t mismatches) {
  printf("WGEMU-TEST name=%s cases=%llu mismatches=%llu\n", name.c_str(),
         static_cast<unsigned long long>(cases), static_cast<unsigned long long>(mismatches));
  if (cases == 0 || mismatches != 0) g_failed_any = true;
}

bool read_file(const std::string& path, std::vector<uint8_t>& data) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size < 0) {
    fclose(f);
    return false;
  }
  data.resize(static_cast<size_t>(size));
  const size_t got = data.empty() ? 0 : fread(data.data(), 1, data.size(), f);
  fclose(f);
  return got == data.size();
}

Header parse_header(const uint8_t* p) {
  Header h;
  uint32_t* const u[] = {&h.version, &h.kind,  &h.atype,   &h.btype,   &h.d_f16,    &h.arch, &h.api,
                         &h.K,       &h.units, &h.a_bytes, &h.b_bytes, &h.cd_bytes, &h.flags};
  for (int i = 0; i < 13; ++i) *u[i] = static_cast<uint32_t>(load_le(p + 8 + 4 * i, 4));
  h.F = static_cast<int32_t>(load_le(p + 60, 4));
  h.L_max = static_cast<int32_t>(load_le(p + 64, 4));
  h.e_zero = static_cast<int32_t>(load_le(p + 68, 4));
  h.rho = static_cast<uint32_t>(load_le(p + 72, 4));
  h.count = load_le(p + 76, 8);
  return h;
}

// Header fields that disagree with the expected combination; empty when consistent.
std::string header_problems(const Header& h, const Expect& e, size_t file_size) {
  std::string bad;
  auto chk = [&](const char* field, uint64_t got, uint64_t want) {
    if (got != want)
      bad += std::string(" ") + field + "=" + std::to_string(got) + "/" + std::to_string(want);
  };
  chk("version", h.version, kVersion);
  chk("kind", h.kind, e.kind);
  chk("atype", h.atype, e.atype);
  chk("btype", h.btype, e.btype);
  chk("d_f16", h.d_f16, e.d_f16);
  chk("arch", h.arch, e.arch);
  chk("api", h.api, e.api);
  chk("K", h.K, e.K);
  chk("units", h.units, e.units);
  chk("flags", h.flags, e.flags);
  chk("a_bytes", h.a_bytes, type_bytes(e.atype));
  chk("b_bytes", h.b_bytes, type_bytes(e.btype));
  const uint32_t cd = (e.kind == kKindF64) ? 8u : (e.kind == kKindFloat && e.d_f16) ? 2u : 4u;
  chk("cd_bytes", h.cd_bytes, cd);
  if (h.count == 0) bad += " count=0";
  const uint64_t want_size =
      kHeaderSize + h.count * (uint64_t(e.units) * (h.a_bytes + h.b_bytes) + 2ull * h.cd_bytes);
  chk("file_size", file_size, want_size);
  return bad;
}

std::string hex_list(const uint8_t* p, int n, int nbytes) {
  std::string s;
  char buf[24];
  for (int i = 0; i < n; ++i) {
    snprintf(buf, sizeof(buf), "%s%0*llx", i ? "," : "", 2 * nbytes,
             static_cast<unsigned long long>(load_le(p + size_t(i) * nbytes, nbytes)));
    s += buf;
  }
  return s;
}

struct Job {
  const Expect* e;
  const Header* h;
  const uint8_t* A;
  const uint8_t* B;
  const uint8_t* C;
  const uint8_t* D;
  EType at, bt;
  FdpaParams p;
};

struct Partial {
  uint64_t mismatches = 0;
  uint64_t nan_class = 0;
  std::vector<uint64_t> first;  // case indices of the first mismatches in this range
};

// Recomputes D for one case; returns the numerics.h result in the low cd_bytes.
uint64_t compute(const Job& j, uint64_t i) {
  const uint32_t units = j.h->units;
  const uint32_t ab = j.h->a_bytes, bb = j.h->b_bytes, cb = j.h->cd_bytes;
  const uint8_t* a = j.A + i * units * ab;
  const uint8_t* b = j.B + i * units * bb;
  const uint64_t c = load_le(j.C + i * cb, int(cb));
  if (j.e->kind == kKindF64) {
    uint64_t av[64], bv[64];
    for (uint32_t k = 0; k < units; ++k) {
      av[k] = load_le(a + k * 8, 8);
      bv[k] = load_le(b + k * 8, 8);
    }
    return wgemu::dot_f64_fma(av, bv, int(units), c);
  }
  uint32_t av[256], bv[256];
  for (uint32_t k = 0; k < units; ++k) {
    av[k] = static_cast<uint32_t>(load_le(a + k * ab, int(ab)));
    bv[k] = static_cast<uint32_t>(load_le(b + k * bb, int(bb)));
  }
  if (j.e->kind == kKindFloat)
    return wgemu::dot_float(av, bv, int(units), j.at, j.bt, static_cast<uint32_t>(c),
                            j.e->d_f16 != 0, j.p);
  if (j.e->kind == kKindInt)
    return static_cast<uint32_t>(wgemu::dot_int(
        av, bv, int(units), (j.e->flags & kFlagASigned) != 0, (j.e->flags & kFlagBSigned) != 0,
        static_cast<int32_t>(static_cast<uint32_t>(c)), (j.e->flags & kFlagSat) != 0));
  return static_cast<uint32_t>(
      wgemu::dot_b1_and_popc(av, bv, int(units), static_cast<int32_t>(static_cast<uint32_t>(c))));
}

bool f64_nan(uint64_t x) { return (x & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull; }

void run_range(const Job& j, uint64_t lo, uint64_t hi, size_t max_keep, Partial* out) {
  const int cb = int(j.h->cd_bytes);
  for (uint64_t i = lo; i < hi; ++i) {
    const uint64_t want = load_le(j.D + i * cb, cb);
    const uint64_t got = compute(j, i);
    if (got == want) continue;
    if (j.e->kind == kKindF64 && f64_nan(got) && f64_nan(want)) {
      ++out->nan_class;
      continue;
    }
    ++out->mismatches;
    if (out->first.size() < max_keep) out->first.push_back(i);
  }
}

int affinity_cpus() {
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) return CPU_COUNT(&set);
  const unsigned n = std::thread::hardware_concurrency();
  return n ? int(n) : 1;
}

void run_one(const std::string& dir, const Expect& e, int threads, int max_print) {
  std::vector<uint8_t> data;
  const std::string path = dir + "/" + e.name + ".bin";
  if (!read_file(path, data) || data.size() < kHeaderSize ||
      memcmp(data.data(), "WGEMUNV1", 8) != 0) {
    printf("WGEMU-ERROR name=%s missing or unreadable file %s\n", e.name.c_str(), path.c_str());
    emit(e.name, 0, 0);
    return;
  }
  const Header h = parse_header(data.data());
  const std::string bad = header_problems(h, e, data.size());
  emit("header:" + e.name, 1, bad.empty() ? 0 : 1);
  if (!bad.empty()) {
    printf("WGEMU-MISMATCH name=header:%s got/want:%s\n", e.name.c_str(), bad.c_str());
    emit(e.name, 0, 0);
    return;
  }
  Job j{
      &e, &h, nullptr, nullptr, nullptr, nullptr, EType::F16, EType::F16, {0, 0, 0, Rho::RZ_FP32}};
  j.A = data.data() + kHeaderSize;
  j.B = j.A + h.count * h.units * h.a_bytes;
  j.C = j.B + h.count * h.units * h.b_bytes;
  j.D = j.C + h.count * h.cd_bytes;
  if (e.kind == kKindFloat) {
    j.at = static_cast<EType>(e.atype);
    j.bt = static_cast<EType>(e.btype);
    const bool d16 = e.d_f16 != 0;
    const Arch arch = e.arch == kArchAmpere ? Arch::Ampere
                      : e.arch == kArchAda  ? Arch::Ada
                                            : Arch::Hopper;
    const ParamSource want_src =
        (e.flags & kFlagInferred) ? ParamSource::Inferred : ParamSource::MmaSim;
    const ParamSource src = e.api == kApiWgmma ? wgemu::wgmma_params_source(j.at, j.bt, d16)
                                               : wgemu::mma_params_source(arch, j.at, j.bt, d16);
    if (src != want_src) {
      emit("params:" + e.name, 1, 1);
      printf("WGEMU-MISMATCH name=params:%s source=%d want=%d\n", e.name.c_str(), int(src),
             int(want_src));
      emit(e.name, 0, 0);
      return;
    }
    j.p = e.api == kApiWgmma ? wgemu::wgmma_params(j.at, j.bt, d16)
                             : wgemu::mma_params(arch, j.at, j.bt, d16);
    const bool same = j.p.F == h.F && j.p.L_max == h.L_max && j.p.e_zero == h.e_zero &&
                      uint32_t(j.p.rho) == h.rho;
    emit("params:" + e.name, 1, same ? 0 : 1);
    if (!same)
      printf(
          "WGEMU-MISMATCH name=params:%s numerics=(F=%d,L_max=%d,e_zero=%d,rho=%u) "
          "mmasim=(F=%d,L_max=%d,e_zero=%d,rho=%u)\n",
          e.name.c_str(), j.p.F, j.p.L_max, j.p.e_zero, unsigned(j.p.rho), h.F, h.L_max, h.e_zero,
          h.rho);
  }
  const auto t0 = std::chrono::steady_clock::now();
  const int nt = int(std::min<uint64_t>(uint64_t(threads), h.count));
  std::vector<Partial> parts(nt);
  std::vector<std::thread> pool;
  for (int t = 0; t < nt; ++t) {
    const uint64_t lo = h.count * t / nt, hi = h.count * (t + 1) / nt;
    pool.emplace_back(run_range, std::cref(j), lo, hi, size_t(max_print), &parts[t]);
  }
  for (std::thread& th : pool) th.join();
  uint64_t mism = 0, nan_class = 0;
  std::vector<uint64_t> first;
  for (const Partial& p : parts) {
    mism += p.mismatches;
    nan_class += p.nan_class;
    first.insert(first.end(), p.first.begin(), p.first.end());
  }
  std::sort(first.begin(), first.end());
  if (first.size() > size_t(max_print)) first.resize(size_t(max_print));
  for (uint64_t i : first) {
    const int cb = int(h.cd_bytes);
    uint8_t got[8];
    const uint64_t g = compute(j, i);
    for (int k = 0; k < 8; ++k) got[k] = uint8_t(g >> (8 * k));
    printf("WGEMU-MISMATCH name=%s case=%llu K=%u a=[%s] b=[%s] c=%s want=%s got=%s\n",
           e.name.c_str(), static_cast<unsigned long long>(i), h.K,
           hex_list(j.A + i * h.units * h.a_bytes, int(h.units), int(h.a_bytes)).c_str(),
           hex_list(j.B + i * h.units * h.b_bytes, int(h.units), int(h.b_bytes)).c_str(),
           hex_list(j.C + i * cb, 1, cb).c_str(), hex_list(j.D + i * cb, 1, cb).c_str(),
           hex_list(got, 1, cb).c_str());
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (e.kind == kKindF64)
    printf("WGEMU-INFO name=%s nan_class_matches=%llu (NaN vs NaN with different payloads)\n",
           e.name.c_str(), static_cast<unsigned long long>(nan_class));
  printf("WGEMU-INFO name=%s seconds=%.2f threads=%d\n", e.name.c_str(), sec, nt);
  emit(e.name, h.count, mism);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir, only;
  int threads = affinity_cpus(), max_print = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) {
      threads = atoi(argv[++i]);
    } else if (a == "--max-print" && i + 1 < argc) {
      max_print = atoi(argv[++i]);
    } else if (a == "--only" && i + 1 < argc) {
      only = argv[++i];
    } else if (!a.empty() && a[0] != '-' && dir.empty()) {
      dir = a;
    } else {
      fprintf(stderr, "usage: %s <vector_dir> [--threads N] [--max-print N] [--only SUBSTRING]\n",
              argv[0]);
      return 2;
    }
  }
  const uint16_t probe = 1;
  uint8_t first_byte;
  memcpy(&first_byte, &probe, 1);
  if (dir.empty() || threads < 1 || max_print < 0 || first_byte != 1) {
    fprintf(stderr, "test_cpu: need <vector_dir>, --threads >= 1, --max-print >= 0, LE host\n");
    return 2;
  }
  const std::vector<Expect> list = expected_list();
  if (list.size() != 92) {  // 12 wgmma + 9 + 25 + 12 mma + 16 inferred + 16 int + 2 b1
    fprintf(stderr, "test_cpu: expected list has %zu entries, not 92\n", list.size());
    return 2;
  }
  std::set<std::string> known;
  int items = 0;
  for (const Expect& e : list) {
    known.insert(e.name);
    if (!only.empty() && e.name.find(only) == std::string::npos) continue;
    run_one(dir, e, threads, max_print);
    fflush(stdout);
    ++items;
  }
  if (items == 0) emit("selection:" + only, 0, 0);
  if (FILE* m = fopen((dir + "/manifest.txt").c_str(), "r")) {
    char line[512];
    while (fgets(line, sizeof(line), m)) {
      std::string s(line);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
      if (!s.empty() && !known.count(s))
        printf("WGEMU-INFO unexpected_file=%s (not tested)\n", s.c_str());
    }
    fclose(m);
  }
  printf("WGEMU-SUMMARY combinations=%d result=%s\n", items, g_failed_any ? "FAIL" : "PASS");
  printf("WGEMU-TEST-END\n");
  return g_failed_any ? 1 : 0;
}
