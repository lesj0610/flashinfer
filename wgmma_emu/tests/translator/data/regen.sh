#!/usr/bin/env bash
# Regenerate the PTX fixtures of the translator tests. Only PTX text is produced; nothing runs
# on a GPU. Line-info fixtures embed the absolute source path, which the tests do not inspect.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
sm90=(-gencode arch=compute_90a,code=sm_90a)

"$NVCC" --ptx "${sm90[@]}" -O3 -std=c++17 sample_sm90.cu -o sample_sm90.ptx
"$NVCC" --ptx "${sm90[@]}" -O3 -std=c++17 -rdc=true -lineinfo sample_sm90.cu -o sample_sm90_rdc_lineinfo.ptx
"$NVCC" --ptx "${sm90[@]}" -O3 -std=c++17 -rdc=true callseq_sample.cu -o callseq_sample.ptx
"$NVCC" -arch=compute_80 -rdc=true --ptx -O3 -std=c++17 lib_sample.cu -o lib_sample.ptx
