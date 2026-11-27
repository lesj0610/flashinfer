# wgmma_emu — SM80에서 SM90 wgmma·비동기 명령 실행 (포크 전용)

작성 2026-11-25 KST. 브랜치 `lesj/wgmma-emu-sm80`(origin/main b37daecc1 기반). 상류 PR 없음.
참고 자료(읽기 전용):
- PTX ISA 9.5: `/nvme512g/wgmma_emu_0fff1f87/ext/ptxisa/ptx.txt`(전체), `wgmma_spec.txt`(§9.7.17 wgmma 1650줄), `img/`(wgmma 그림 33개)
- MMA-Sim(MIT, b953d74): `/nvme512g/wgmma_emu_0fff1f87/ext/mmasim` — `src/mmasim/mmasim/nv_ptx/sim.py`, `arithmetic/fdpa.py`, `arithmetic/helper.py`, `nv_ptx/isa_*.py`
- CUTLASS b46b16d0 고정 사본: `/nvme512g/wgmma_emu_0fff1f87/cutlass` (수정 금지)
- 파이썬: `/home/lesj0610/.uvenvs/vLLM_dev/bin/python` (3.12, torch 2.14+cu130, numpy 2.3.5)
- 툴체인: nvcc 13.0 V13.0.88 `/usr/local/cuda/bin`. GPU: SM80 2장(블록당 smem opt-in 166912B, SM 74개, 레지스터 65536/블록).

## 0. 목표와 범위

SM90(`sm_90`, `sm_90a`)용으로 컴파일한 CUDA 코드를 SM80에서 실행한다. PTX 번역기가 sm_89·sm_90 전용 명령을 에뮬레이션 함수 호출이나 sm_80 명령열로 바꾸고, ptxas·fatbinary는 sm_80으로 돈다. 장치 코드는 `__CUDA_ARCH__=900`(+`__CUDA_ARCH_FEAT_SM90_ALL`)으로 컴파일되므로 SM90 경로를 그대로 탄다.

재현 대상 밖은 실기 시간 하나뿐이다(시간은 SM80 하드웨어 시간). 명령 의미, 레지스터·smem 배치, 반올림, 비동기 규약은 전부 구현한다.

1단계 범위(전부 구현, 누락 금지):
- **wgmma 전부:** `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group`, `wgmma.mma_async`(dense), `wgmma.mma_async.sp`(sparse). 모든 N, atype/btype f16·bf16·tf32·e4m3·e5m2·s8·u8·b1, D f16·f32·s32, SS·RS, imm-trans-a/b, imm-scale-a/b, scale-d, `.satfinite`, b1 `.and.popc`.
- **비동기·TMA:** mbarrier 전 명령(소프트웨어 구현), `cp.async.bulk`·`cp.async.bulk.tensor`(1~5D, tile·im2col, load·store·prefetch, multicast), `cp.reduce.async.bulk(.tensor)`, bulk_group commit/wait, `st.async`, `red.async`, `tensormap.replace`, `tensormap.cp_fenceproxy`, `prefetch.tensormap`, `fence.proxy.*`, `fence.mbarrier_init`, `cp.async.mbarrier.arrive(.noinc)`.
- **그 밖의 sm_89·sm_90·sm_90a 명령 전부:** 인벤토리(`tools/ptxemu/inventory.tsv`)로 확정한다. 예: `elect.sync`, `setmaxnreg`, `griddepcontrol`, `stmatrix`, `mapa`, `getctarank`, `barrier.cluster`, 클러스터 특수 레지스터, fp8·bf16·tf32 `cvt`, bf16 산술·비교, 벡터 `atom`/`red`, `mma.sync` fp8(sm_89), 근사 명령.
- **호스트:** `cuTensorMapEncodeTiled`/`Im2col`(SM80 드라이버는 801 반환, 11-25 확인) 대체, `cuGetProcAddress`·`cudaGetDriverEntryPoint*`·`dlsym` 경유 포함. 실행 속성(cluster 1×1×1, PDL) 제거. 선택적으로 CC 9.0 보고.
- **모드:** exact(소프트웨어 Hopper 수치), fast(SM80 `mma.sync`), check(exact + 지연 완료와 규약 위반 검출).

2단계(1단계 합격 직후 착수): cluster>1 가상화, smem>166912B 커널(CUTLASS 용량 상수 그림자 헤더 + 번역기 smem→global 모드), NVRTC·cuLink 경로, FI JIT·Humming 연결, CUTLASS SM90 단위 시험 전체, 서빙과 SM80↔SM90 커널 대응표.

## 1. 합격 기준 (착수 전 고정, 이후 바꾸지 않음)

모든 단계는 "실제로 실행된 사례 수"를 출력하고, 판정은 그 수와 불일치 수로 한다. rc=0만으로 통과시키지 않는다.

1. **명세(CPU):** `spec.h`를 (a) PTX 그림을 손으로 옮긴 표, (b) CuTe `mma_traits_sm90_gmma*.hpp`의 ALayout/BLayout/CLayout(희소는 ELayout 포함), (c) CuTe `make_gmma_desc`로 만든 서술자와 CuTe 레이아웃 주소와 비교한다. 대상: 모든 GMMA atom 형식, 모든 swizzle(none·32·64·128B) × K/MN major × 원소 폭, 타일 내 모든 (mn,k), CuTe가 k-block마다 서술자를 옮기는 방식 그대로. **합격: 불일치 0.**
2. **수치:**
   - (a) CPU: C++ 이식 vs MMA-Sim 파이썬. Hopper wgmma 전 형식 조합, Ampere·Ada·Hopper `mma.sync` 전 형식 조합. 조합마다 무작위 100만 벡터 이상 + 경계 사례(±0, 아정규, inf, NaN, 지수 폭 최대, 상쇄, 넘침). **합격: 불일치 0.**
   - (b) GPU(SM80): Ampere 파라미터 모델 vs SM80 `mma.sync`. f16→f32, f16→f16, bf16→f32, tf32→f32, m16n8k16·m16n8k8. 조합당 1000만 내적 이상, 경계 사례 포함. s8/u8 혼합(±satfinite), b1 and.popc. **합격: 불일치 0.**
3. **wgmma 장치 함수 직접 시험(GPU):** 모든 (atype, btype, dtype) × 모든 N × SS/RS × 허용 trans × imm-scale ±1 × scale-d 0/1 × satfinite × sparse(모든 sp-sel, 모든 유효 메타데이터) × swizzle 4종 × major. N은 형식 조합마다 전부, 플래그 값은 형식 조합마다 작은 N과 큰 N에서 각각 한 번 이상. exact 모드는 CPU 참조(numerics.h + 논리 행렬)와 비트 일치, fast 모드는 CPU의 Ampere 파라미터 분해 모델과 비트 일치. **합격: 불일치 0.**
4. **번역기 종단(GPU):** inline asm SM90 프로그램을 `nvcc_emu`로 컴파일(sm_90a → sm_80 SASS)해 실행. 3단계와 같은 wgmma 변형 전부, 모든 비동기 명령 형식, 인벤토리의 모든 대체 명령. CuTe 튜토리얼 `examples/cute/tutorial/hopper/wgmma_sm90.cu`, `wgmma_tma_sm90.cu` 포함. **합격: ptxas 성공, 남은 sm_89+ 명령 0, 결과 불일치 0.**
5. **check 모드 규약:** 양성(3·4단계 프로그램 전부)은 위반 0이고 결과가 맞아야 한다. 음성(wait 전 누산기 읽기, wait 전 smem 덮어쓰기, 누산기 수정 뒤 fence 누락, wait N 과소, RS A 레지스터 wait 전 수정)은 전부 검출돼야 한다. **합격: 양성 오검출 0, 음성 미검출 0.**
6. **결함 주입:** swizzle 비트, LBO↔SBO 교환, transpose 무시, scale-d 무시, imm-scale 무시, sp-sel 무시, 메타데이터 해석 뒤바꿈, check 모드 wait no-op, T-FDPA F±1, e_zero±1, 출력 반올림 RZ↔RNE, TMA swizzle 비트, OOB 채움 값, mbarrier phase 반전. **합격: 모든 변이가 1~5단계 중 하나 이상에서 실패.** 살아남은 변이는 시험을 보강한다.
7. **CUTLASS SM90 단위 시험(2단계):** `test/unit/gemm/device/sm90_*` 63개 파일과 `test/unit/cute/hopper/*`을 `nvcc_emu`로 빌드해 SM80에서 실행. 바이너리별로 컴파일 여부, 실행 여부, gtest 실행·통과·실패·건너뜀 수를 표로 남긴다. **합격: 전부 통과.** cluster>1 사례는 2단계 가상화 뒤 같은 기준.

수치 근거 수준을 구분해 보고한다. 2(b)는 SM80 실기 비트 일치(사실). Hopper 파라미터는 MMA-Sim의 H100 검증 결과에 기대며 2(a)는 그 이식 일치다. 둘을 합쳐 "SM90 실기와 일치"라고 쓰지 않는다. 희소 수치 가정(§8)은 [추론]이다.

## 2. 구조와 파일 소유권

남의 파일은 고치지 않는다. 필요한 변경은 최종 보고에 diff로 적고, 통합 스레드가 합친다.

```
wgmma_emu/
  PLAN.md, include/wgmma_emu/types.h, runtime/emu_lib.cu            통합 스레드
  include/wgmma_emu/numerics.h, tests/numerics/                     N
  include/wgmma_emu/spec.h, tests/spec/                             S
  include/wgmma_emu/wgmma_device.cuh, runtime/lib_wgmma.cu,
    tools/ptxemu/rules_wgmma.py, tests/wgmma/                       W
  include/wgmma_emu/async_device.cuh, include/wgmma_emu/tensormap.h,
    runtime/lib_async.cu, tools/ptxemu/rules_async.py,
    host/shim.c, tests/async/                                       A
  tools/ptxemu/{__init__,lexer,core,cli,check}.py, tools/nvcc_emu,
    tests/translator/, tests/e2e/common/                            T1
  tools/ptxemu/rules_misc.py, tools/ptxemu/inventory.tsv,
    include/wgmma_emu/cvt.h, include/wgmma_emu/misc_device.cuh,
    runtime/lib_misc.cu, tests/misc/                                T2
```

구현 에이전트는 코드만 쓴다. 빌드·pytest·GPU 실행·commit은 하지 않는다. 형식 검사(`ruff check`, `ruff format --check`, `clang-format --dry-run`)만 허용한다. 빌드·시험·측정은 통합 뒤 통합 스레드가 한 번에 한다. 최종 보고에는 쓴 파일, 빌드 명령, 시험 실행 명령, 기대 출력, 가정([추론] 표시), 다른 소유자 파일에 필요한 diff를 적는다.

## 3. 빌드 흐름

- 에뮬레이션 라이브러리: `runtime/emu_lib.cu`가 `lib_wgmma.cu`, `lib_async.cu`, `lib_misc.cu`를 include한다. `nvcc -arch=compute_80 -rdc=true --ptx -O3 -std=c++17 -I include`로 PTX 하나(`emu_lib.ptx`)를 만든다. 진입점은 전부 `extern "C" __device__ __noinline__`이라 `.visible .func`로 남는다.
- `tools/nvcc_emu`(nvcc와 같은 CLI):
  1. 아키텍처 인자를 `-gencode arch=compute_90a,code=sm_90a` 하나로 정규화한다(`WGEMU_SRC_ARCH`로 90/90a 선택). 다른 arch 이미지는 넣지 않는다(같은 sm_80 이미지 중복 방지).
  2. `nvcc -dryrun -keep -keep-dir <dir> ...`의 `#$` 명령열을 받아 환경 변수 줄은 반영하고 나머지를 차례로 실행한다.
  3. cicc가 만든 `.ptx`마다 `python -m ptxemu`로 번역한다(라이브러리 PTX를 모듈 머리 바로 뒤에 붙임, `.target sm_90a`→`sm_80`).
  4. `ptxas -arch=sm_90a`→`sm_80`, `nvlink -arch=sm_90a`→`sm_80`, `fatbinary --image3=kind=elf,sm=90a`→`sm=80`, PTX 이미지(kind=ptx) 제거.
  5. 모드는 `WGEMU_MODE=exact|fast|check`(기본 exact). 번역 시점에 고정된다.
- 호스트 shim: `gcc -O2 -shared -fPIC host/shim.c -ldl -o libwgemu_shim.so`, 실행 시 `LD_PRELOAD`.
- 시험 빌드: CPU 시험은 `g++ -std=c++17`(또는 nvcc 호스트 전용), 직접 장치 시험은 `nvcc -arch=sm_80`, 종단 시험은 `nvcc_emu`.

## 4. 공통 형식 (`include/wgmma_emu/types.h`, 수정 금지)

`wgemu::EType { F16=0, BF16=1, TF32=2, E4M3=3, E5M2=4, S8=5, U8=6, B1=7, F32=8, S32=9 }`, `etype_bits`, `etype_is_fp8`, `etype_is_int8`, `etype_is_float_input`, 매크로 `WGEMU_HD`(`__host__ __device__ inline`), `WGEMU_HDF`(forceinline), `WGEMU_DEV`.

## 5. 번역기 API (T1 구현, W·A·T2 사용)

`tools/ptxemu/core.py`:

```python
@dataclass
class Stmt:
    guard: str | None      # '@%p1', '@!%p1' 또는 None
    opcode: str            # 점으로 이은 전체 opcode. 예 'wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16'
    operands: list[str]    # 최상위 쉼표로 나눈 피연산자. '{%f1, %f2}'와 '[%rd1+16]', '[tmap, {%r1, %r2}]'는 한 덩어리
    text: str              # 원문(끝 ';' 제외)
    lineno: int

def parse_vector(op: str) -> list[str]                 # '{%f1, %f2}' -> ['%f1', '%f2']
def parse_address(op: str) -> tuple[str, int]          # '[%rd1+16]' -> ('%rd1', 16), '[sym]' -> ('sym', 0)
def split_dests(op: str) -> list[str]                  # '%r1|%p1' -> ['%r1', '%p1']

class Func:   # name, is_entry, params(list[(type, name)]), body(list[Stmt | str])
class Ctx:
    mode: str                      # 'exact' | 'fast' | 'check'
    src_target: str                # 'sm_90a' 등
    func: Func
    def new_reg(self, ty: str) -> str              # 'pred','b16','b32','b64','u32','u64','f32' 등. 함수 범위 .reg 선언
    def local_buf(self, key: str, nbytes: int, align: int = 16) -> str   # 함수 범위 .local 배열, key당 1회. 심볼 반환
    def shared_var(self, key: str, nbytes: int, align: int = 16) -> str  # 모듈 범위 .shared 변수, key당 1회
    def generic_addr_of_local(self, sym: str) -> tuple[list[str], str]   # (PTX 줄, b64 레지스터)
    def to_shared_u32(self, addr_operand: str, space: str | None) -> tuple[list[str], str]
        # 주소 피연산자(심볼+오프셋·레지스터+오프셋)와 상태 공간('shared','shared::cta','shared::cluster',None=generic)을
        # 32비트 shared 창 주소 레지스터로 바꾼다
    def to_generic_u64(self, addr_operand: str, space: str | None) -> tuple[list[str], str]
    def call(self, func: str, args: list[tuple[str, str]], ret: tuple[str, str] | None = None) -> list[str]
        # args: (PTX 형 'b32'|'b64', 피연산자: 레지스터 또는 정수 즉시값). ret: (형, 목적 레지스터).
        # nvcc가 만드는 callseq 블록({ .param 선언; st.param; call.uni; ld.param })을 그대로 생성
    def func_state(self) -> dict                   # 규칙별 함수 단위 상태
    def warn(self, msg: str) -> None
    def error(self, msg: str) -> None              # 번역 실패로 끝냄(조용한 무시 금지)
```

규칙 모듈(`rules_wgmma.py`, `rules_async.py`, `rules_misc.py`) 규약:

```python
def scan(func: Func, ctx: Ctx) -> None                       # 선택. 함수 재작성 전 1회
def function_prologue(ctx: Ctx) -> list[str]                 # 선택. 함수 본문 선언 뒤에 넣을 줄
def match(stmt: Stmt) -> bool
def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]              # stmt 대신 넣을 PTX 줄. guard는 core가 감싼다
def rewrite_operand(op: str, stmt: Stmt, ctx: Ctx) -> tuple[list[str], str] | None   # 선택. 특수 레지스터 등
```

core는 규칙을 wgmma → async → misc 순으로 적용하고 처음 맞는 규칙을 쓴다. guard가 있으면 `@!p bra $WGEMU_SKIP_n; ...; $WGEMU_SKIP_n:`로 감싼다. 번역 뒤 `check.py`가 `inventory.tsv`로 남은 sm_89+ 명령을 찾아 있으면 실패시킨다(최종 관문은 `ptxas -arch=sm_80`).

`inventory.tsv` 형식(T2 작성, T1 사용): 탭 구분 `opcode_regex  min_sm  handling  ptx_section  note`. handling은 `native`(sm_80에 있음) | `lib:<func>` | `inline` | `noop` | `strip` | `rewrite:<rule>` 중 하나. `min_sm`이 89 이상인 줄은 handling이 `native`일 수 없다.

## 6. 라이브러리 호출 규약

- 진입점은 `extern "C" __device__ __noinline__`, 이름은 `wgemu_` 접두사. 인자는 u32·u64 스칼라만(포인터는 generic u64). 반환은 u32 하나 또는 없음.
- 호출 사이에 남는 상태는 호출자가 넘기는 포인터에 둔다(번역기가 함수별 `.local`·모듈별 `.shared` 버퍼를 선언). 라이브러리 전역 변수 금지.
- libdevice·libm 호출 금지(수치는 정수 비트 연산). 외부 심볼은 `vprintf`만 허용(check 모드 경고). 장치 측 경고 문자열은 `WGEMU-` 접두사로 시작한다.
- 각 소유자는 자기 진입점 목록과 시그니처를 자기 헤더 머리 주석에 적는다.

## 7. 명세 사실 (S가 원문·그림·CuTe로 재확인한다. 다르면 고치지 말고 보고)

표기: t = tid % 128, w = t / 32(워프), l = lane, g = l / 4, q = l % 4, r = 16w + g.

- **D(전 형식):** 원소 i ∈ [0, N/2) → row = r + 8·((i % 4) / 2), col = 8·(i / 4) + 2q + (i % 2). f32·s32 D는 원소 i가 레지스터 i. f16 D는 레지스터 i가 원소 2i(하위 16비트), 2i+1(상위).
  - b1 k256 그림 158과 충돌함(15행 7열 `T31:d1`). 균일 배치를 채택하고 그 칸은 `d3` 오타로 봄 [추론]. 근거: 그림 152/154/156, CuTe CLayout_64xN, `wgmma_spec.txt:1209-1212`(D 배치는 모든 형상이 같다는 문장). `-DWGEMU_SPEC_STRICT_FIG158`이면 충돌을 실패로 셈(11-25 S 정정). b1은 FI MoE 경로에서 쓰지 않음.
- **dense RS A:** 레지스터 j ∈ 0..3 → row = r + 8·(j & 1), col 시작 = (j >> 1)·K/2 + q·K/8, 레지스터 안 원소는 하위부터 +0, +1, …. k8 tf32, k16 16비트, k32 8비트, k256 b1 그림과 모두 맞다.
- **sparse RS A:** 압축 열 kc ∈ [0, K/2)에 대해 dense 식에서 K를 K/2로 바꾼 것과 같다(f16 k32: 압축 쌍 (2q,2q+1)이 논리 4q..4q+3 청크, a4..a7은 +16; tf32 k16: 레지스터 j가 논리 (j>>1)·8 + 2q의 2칸 청크; 8비트 k64: a0..a3 row r 논리 8q..8q+7, a4..a7 row r+8, a8..a15 +32).
- **sparse 메타데이터:** 워프 w가 행 16w..16w+15. rr = m % 16, g = rr % 8, hi = rr / 8, 청크 c.
  - f16/bf16(2:4, 32열당 8청크)·tf32(1:2, 16열당 8청크): lane = 4g + 2·sel + (c ≥ 4), bit = 16·hi + 4·(c % 4), sel ∈ {0,1}.
  - f16 니블: bits[1:0] = 첫 비영 원소 위치, bits[3:2] = 둘째. tf32 니블: 유효값 0b0100(원소 0), 0b1110(원소 1), 원소 = (nibble & 3) >> 1.
  - 8비트(2:4, 64열당 16청크): lane = 4g + hi + 2·(c ≥ 8), bit = 4·(c % 8), sel = 0만.
  - 2:4에서 0b0000·0b0101·0b1010·0b1111은 정의되지 않음.
- **서술자(64비트):** bits 13–0 시작 주소>>4, 29–16 LBO>>4, 45–32 SBO>>4, 51–49 base offset, 63–62 swizzle(0 none, 1 128B, 2 64B, 3 32B). 인코딩 `(x & 0x3FFFF) >> 4`. smem 행렬은 16B 정렬.
- **정준 레이아웃(T = 128 / elem_bits, eb = 원소 바이트, mn = M 또는 N 쪽 행):**
  - K-major, swizzle 없음: mn = r0 + 8·r1, k = kk + T·k1 → r0·16 + r1·SBO + kk·eb + k1·LBO.
  - K-major 32B/64B/128B(W바이트): r0·W + r1·SBO + k·eb (LBO 미사용, 1로 가정).
  - MN-major, swizzle 없음: mn = mm + T·m1, k = k0 + 8·k1 → mm·eb + m1·SBO + k0·16 + k1·LBO.
  - MN-major, W바이트 swizzle: W바이트 행 안 위치 + m1·LBO + k0·W + k1·SBO.
  - K 방향 행당 읽는 바이트: dense A·B와 sparse A(압축 기준)는 32바이트(f16 k16, tf32 k8, fp8 k32, b1 k256). **sparse B는 논리 K 전체라 64바이트임.** (11-25 S 정정, 근거 CuTe `mma_traits_sm90_gmma.hpp:236-238`(MN: dense 256, sparse A 128, sparse B 512 비트), 272(K-major: dense 2, sparse 4 uint128).)
  - K-major W바이트 swizzle 식(155줄)은 k 방향 바이트 kb ≥ W를 정의하지 않음. spec.h는 선형 연장으로 두고 시험에서는 제외함(11-25 S 정정).
    - 해당 조합은 sparse B K-major 32B swizzle뿐임. PTX 정준 표(`wgmma_spec.txt` 32B 행 `((8,m),(T,2)):((2T,SBO),(1,T))`)는 K를 2단위(32B)만 정의하고, K-major swizzle에서 LBO는 "not used, assumed to be 1"(같은 파일 540)이라 두 번째 32B 블록 위치를 지정할 필드가 없음. 선형 연장은 (r, kb≥32)와 (r+1, kb−32)가 겹치는 비단사임(W 보고).
    - 결정(11-25 메인): check 모드는 이 조합을 위반으로 보고함(통합 단계 구현). exact·fast는 spec.h 연장식을 유지하되 결과를 근거로 쓰지 않음. 시험은 SW64로 바꿔 만든 현 방식 유지. FI MoE·dense 경로는 sparse를 쓰지 않음.
  - swizzle: 바이트 주소에 Swizzle<B,4,3>, 즉 bits[4, 4+B) ^= bits[7, 7+B), B = 1(32B)·2(64B)·3(128B). 행 위상은 ((A >> 7) − base_offset) & mask로 본다. base_offset = 0이면 절대 주소 swizzle과 같고 CuTe는 항상 0을 쓴다. 0이 아닌 해석은 [추론].
  - base_offset 적용 범위: swizzle 없음을 뺀 모든 모드(32B 포함)임. 근거 PTX `ext/ptxisa/wgmma_spec.txt:860` "valid for all swizzling modes except the no-swizzle mode". CuTe 주석 `mma_sm90_desc.hpp:121`과 다르며 spec.h는 PTX를 따름(11-25 S 정정).
- **형상·형식(PTX 9.5):** dense f16/bf16 k16, tf32 k8, e4m3/e5m2 k32(4조합), N = 8..256(8단위). 정수 s8/u8(혼합 부호 허용) k32와 b1 k256(`.and.popc`): N ∈ {8,16,24,32,48,64,80,…,256}(32 초과는 16단위). 문법 절은 정수를 n224까지만 적지만 형상 표에 n240·n256이 있으므로 상위집합을 지원한다. sparse: f16/bf16 k32, tf32 k16, fp8·정수 k64. transpose는 f16/bf16 SS만. imm-scale ±1은 부동소수 형식만. D 형식: f16 입력은 f16·f32, bf16·tf32는 f32, fp8은 f16·f32, 정수·b1은 s32.

## 8. 수치 모델 (MMA-Sim T-FDPA, N이 `sim.py`·`fdpa.py`에서 직접 확인)

- 입력마다 s ∈ [1,2)와 e로 정규화한다(아정규 지수: f32 −126, f16 −14, bf16 −126, e5m2 −14, e4m3 −6). 곱은 정확하고 지수는 e_a + e_b. c도 항에 넣는다. 0인 항은 e = e_zero. e_max는 모든 항의 최댓값.
- 각 항은 trunc(s·2^(e − e_max + F))·2^(−F)로 자르고 정확히 더한다.
- 출력 반올림 rho: RZ-FP32(아정규 −126, 23비트 절삭), RZ-E8M13(13비트 절삭), RNE-FP16(아정규 −14, 10비트 RNE). 넘침은 inf, NaN은 정규 NaN(f32 0x7FFFFFFF, f16 0x7FFF). +0을 더하므로 −0은 +0이 된다.
- K는 L = min(K, L_max)씩 나눠 c에 순차 누적한다. tf32 입력은 `x >> 13 << 13`.
- Hopper wgmma: f16/bf16/tf32는 F=25, rho = RZ-FP32(D f16이면 RNE-FP16). fp8은 F=13, rho = RZ-E8M13(f32 출력)·RNE-FP16(f16 출력). L_max = 32바이트/원소 크기(f16/bf16 16, tf32 8, fp8 32), dense wgmma 하나가 정확히 한 블록. e_zero는 f32 D −133, f16 D −21.
- Ampere `mma.sync`: F=24, L_max = 16바이트/원소 크기(f16 8, tf32 4), e_zero −132 / −20. Ada·Hopper `mma.sync`는 `sim.py` 표를 따른다.
- 정수: int64로 정확히 더한 뒤 `.satfinite`면 s32로 포화, 아니면 2^32로 감싼다. b1: c + Σ popc(a AND b), 감쌈.
- scale-d = 0은 c = +0으로 본다 [추론]. imm-scale-a/b = −1은 해당 입력의 부호 비트 반전.
- sparse: 선택된 K/2개 비영 곱에 대한 dense T-FDPA 한 번으로 본다 [추론]. MMA-Sim은 sp를 모델링하지 않는다.

## 9. 실행 모드

- **exact:** 발행 시점에 Hopper 수치로 계산해 바로 레지스터에 쓴다(비동기 연산이 wait 전 아무 때나 끝나도 되므로 유효한 실행).
- **fast:** SM80 `mma.sync`로 계산한다(fp8은 f16으로 정확 변환 뒤 f16 mma, sparse는 dense로 펼침). 처리량용이고 반올림은 SM80 하드웨어 반올림이다. CPU 시험은 W가 정한 분해를 Ampere 파라미터로 재현해 비트 비교한다.
- **check:** exact 계산 결과를 레지스터 대신 함수별 대기표(`.local`)에 넣고 누산기 레지스터에는 독 값(f32 NaN 페이로드 0x7FC0DEAD, f16x2 0x7E017E01, s32 0xDEADBEEF)을 쓴다.
  - 대기표는 정적 누산기 id(서로 다른 d 레지스터 목록마다 하나)로 찾는다. 같은 id의 연쇄 wgmma는 저장된 참값을 C로 쓴다.
  - 연산마다 seq = 현재 그룹 번호. commit에서 그룹 번호 +1. `wait N`은 seq < cur − N인 연산을 완료시키고, 번역기는 wait 뒤에 누산기 id마다 fetch 호출과 조건부 레지스터 복원을 넣는다.
  - 발행 시와 완료 시 smem 피연산자 해시를 비교해 wait 전 덮어쓰기를 잡는다. RS A 레지스터 변경 검사, fence 누락 경고 범주를 둔다.
  - 위반은 `WGEMU-VIOLATION <종류> ...`를 printf한다. 번역 옵션으로 trap도 고른다.

## 10. 비동기·TMA·호스트

- **mbarrier:** 64비트 word를 우리 형식으로 쓴다(예상 도착 수 20비트, 남은 도착 20비트, 부호 있는 tx 21비트, phase 1비트). `atom.shared.cas.b64`로 갱신하고 release/acquire 순서를 지킨다. arrive·arrive_drop·expect_tx·complete_tx·test_wait·try_wait(parity 포함)·pending_count·inval·`cp.async.mbarrier.arrive`(cp.async.wait_all 뒤 arrive) 전부.
- **TMA·bulk:** 발행 스레드가 동기적으로 복사하고 complete_tx 한다. tile·im2col, 1~5D, OOB 채움(0, NaN 요청), swizzle(smem 절대 주소), multicast(cluster 1이면 자기 CTA), reduce(add/min/max/inc/dec/and/or/xor), store, prefetch(no-op).
- **tensormap:** 128B CUtensorMap 안에 우리 형식을 쓴다(`tensormap.h`). 호스트 shim의 encode와 장치의 `tensormap.replace`가 같은 형식을 쓴다.
- **클러스터 크기 1:** `%cluster_ctarank` = 0 등 특수 레지스터 치환, `mapa`는 항등, `barrier.cluster`는 번역기가 추가한 `.shared` 소프트웨어 장벽, `.reqnctapercluster` 등 지시어 제거. cluster>1은 2단계.
- **호스트 shim:** `cuTensorMapEncodeTiled`·`cuTensorMapEncodeIm2col`·`cuTensorMapReplaceAddress`, `cuGetProcAddress(_v2)`, `cudaGetDriverEntryPoint(ByVersion)`, libcuda 핸들에 대한 `dlsym`(정적 cudart 대응), `cuLaunchKernelEx`·`cudaLaunchKernelEx*`의 cluster 1×1×1·PDL 속성 제거, `WGEMU_SPOOF_CC=1`이면 CC 9.0 보고.

## 11. 시험 출력·목록 규약 (모든 시험 공통)

- 시험 프로그램은 항목마다 `WGEMU-TEST name=<id> cases=<n> mismatches=<m>` 한 줄을 찍고, 끝에 `WGEMU-TEST-END`를 찍는다. END가 없거나 cases=0인 항목은 실패다. 불일치가 있으면 처음 몇 건만 `WGEMU-MISMATCH name=<id> ...`로 자세히 찍는다.
- 종단(nvcc_emu) 시험 목록: `tests/<영역>/e2e_manifest.json` = `[{"name", "source"(영역 폴더 기준 상대 경로), "nvcc_args"(추가 인자 목록), "modes"(["exact","fast","check"] 부분집합), "expect_violation"(bool, check 모드 음성 시험), "shim"(bool, LD_PRELOAD 필요), "covers"(이 시험이 다루는 opcode 정규식 목록)}]`. T1의 `tests/e2e/common/build_and_run.py`가 읽어 빌드·실행·집계한다.
- 음성 시험은 출력에 `WGEMU-VIOLATION`이 있거나 불일치가 있어야 통과한다.

## 12. 모듈 사이 C++ API (구현 소유자 외에는 이 시그니처만 믿고 쓴다)

`numerics.h` (N), `namespace wgemu`:

```cpp
enum class Rho : uint8_t { RZ_FP32, RZ_E8M13, RNE_FP16 };
enum class Arch : uint8_t { Ampere, Ada, Hopper };
struct FdpaParams { int F; int L_max; int e_zero; Rho rho; };
WGEMU_HD FdpaParams wgmma_params(EType a, EType b, bool d_f16);                 // Hopper wgmma
WGEMU_HD FdpaParams mma_params(Arch arch, EType a, EType b, bool d_f16);        // mma.sync
// 원소 비트: f16/bf16은 하위 16비트, tf32는 f32 비트(내부에서 하위 13비트 절삭), fp8은 하위 8비트.
// c: d_f16이면 하위 16비트 f16, 아니면 f32 비트. 반환도 같은 형식.
WGEMU_HD uint32_t tfdpa(const uint32_t* a, const uint32_t* b, int n, EType at, EType bt,
                        uint32_t c, bool d_f16, const FdpaParams& p);          // 블록 하나(n <= L_max)
WGEMU_HD uint32_t dot_float(const uint32_t* a, const uint32_t* b, int K, EType at, EType bt,
                            uint32_t c, bool d_f16, const FdpaParams& p);      // L_max 블록으로 나눠 순차 누적
WGEMU_HD int32_t dot_int(const uint32_t* a, const uint32_t* b, int K, bool a_signed, bool b_signed,
                         int32_t c, bool satfinite);                           // 8비트 값은 하위 바이트
WGEMU_HD int32_t dot_b1_and_popc(const uint32_t* a_words, const uint32_t* b_words, int nwords, int32_t c);
```

`spec.h` (S), `namespace wgemu::spec`:

```cpp
WGEMU_HD int k_of(EType atype, bool sparse);            // 논리 K: f16/bf16 16(sp 32), tf32 8(16), 8비트 32(64), b1 256
WGEMU_HD bool valid_n(EType atype, int n);
WGEMU_HD void d_frag_coord(int t, int i, int* m, int* n);                       // t: 0..127, i: 원소
WGEMU_HD void a_frag_coord(EType at, int t, int j, int e, int* m, int* k);      // dense RS A. b1은 e=0..31 비트
WGEMU_HD void sp_a_frag_coord(EType at, int t, int j, int e, int* m, int* kc);  // sparse RS A, 압축 열
WGEMU_HD void sp_meta_loc(EType at, int sel, int m, int chunk, int* lane, int* bit);
WGEMU_HD void sp_meta_decode(EType at, uint32_t field, int* idx0, int* idx1);   // tf32는 idx1 = -1
struct Desc { uint32_t start, lbo, sbo, base_off; int swizzle_bytes; /* 0, 32, 64, 128 */ };
WGEMU_HD Desc decode_desc(uint64_t d);
WGEMU_HD uint64_t encode_desc(const Desc& d);
// 논리 원소 (mn, k)의 smem 바이트 주소. elem_bits는 8·16·32. b1은 32비트 단어(k = 단어 번호, 행당 8단어)로 다룬다.
WGEMU_HD uint32_t smem_addr(const Desc& d, bool mn_major, int elem_bits, int mn, int k);
WGEMU_HD uint32_t swizzle_addr(uint32_t addr, int swizzle_bytes, uint32_t base_off);
```
