#!/usr/bin/env bash
# Create a CUDA_HOME shadow directory whose bin/nvcc is nvcc_emu.
#
#   make_cuda_home_shadow.sh <shadow_dir> [real_cuda_home]
#
# real_cuda_home defaults to $CUDA_HOME, then /usr/local/cuda. Every top-level entry of the real
# CUDA home except bin/ is symlinked; bin/ is a real directory whose entries are symlinks, except
# bin/nvcc -> tools/nvcc_emu. The real home is recorded in <shadow_dir>/.wgemu_real_cuda_home,
# which nvcc_emu reads to find the real nvcc. Re-running on an existing shadow of the same real
# home refreshes the links; any other existing non-empty directory is refused.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <shadow_dir> [real_cuda_home]" >&2
  exit 2
fi

shadow=$1
real=${2:-${CUDA_HOME:-/usr/local/cuda}}
real=$(readlink -f -- "$real")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
emu=$script_dir/nvcc_emu
marker_name=.wgemu_real_cuda_home

if [[ ! -x $real/bin/nvcc ]]; then
  echo "error: $real/bin/nvcc is not an executable nvcc" >&2
  exit 1
fi
if [[ -f $real/$marker_name ]]; then
  echo "error: $real is itself a wgemu shadow; pass the real CUDA home" >&2
  exit 1
fi
if [[ ! -x $emu ]]; then
  echo "error: $emu is missing or not executable" >&2
  exit 1
fi

if [[ -e $shadow ]]; then
  if [[ ! -d $shadow ]]; then
    echo "error: $shadow exists and is not a directory" >&2
    exit 1
  fi
  if [[ -f $shadow/$marker_name ]]; then
    old=$(<"$shadow/$marker_name")
    if [[ $old != "$real" ]]; then
      echo "error: $shadow shadows $old, not $real; use a new directory" >&2
      exit 1
    fi
  elif [[ -n $(ls -A -- "$shadow") ]]; then
    echo "error: $shadow is a non-empty directory that is not a wgemu shadow" >&2
    exit 1
  fi
fi
mkdir -p -- "$shadow/bin"
shadow=$(cd -- "$shadow" && pwd -P)

shopt -s nullglob dotglob
for entry in "$real"/*; do
  name=${entry##*/}
  [[ $name == bin || $name == "$marker_name" ]] && continue
  if [[ -e $shadow/$name && ! -L $shadow/$name ]]; then
    echo "error: $shadow/$name exists and is not a symlink" >&2
    exit 1
  fi
  ln -sfn -- "$entry" "$shadow/$name"
done
for entry in "$real"/bin/*; do
  name=${entry##*/}
  [[ $name == nvcc ]] && continue
  ln -sfn -- "$entry" "$shadow/bin/$name"
done
ln -sfn -- "$emu" "$shadow/bin/nvcc"
printf '%s\n' "$real" >"$shadow/$marker_name"

echo "shadow CUDA home: $shadow (real: $real)"
echo "export CUDA_HOME=$shadow PATH=$shadow/bin:\$PATH"
