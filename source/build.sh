#!/bin/bash
# Build the engine: target/engine (one executable, every NEFF embedded, links libnrt only).
#  1. tools/build_neffs.py compiles the NKI kernels (kernels/) to NEFFs through NKI's documented route; that needs
#     free NeuronCores: ENGINE_BUILD_CORES (comma-separated 4-core ranges, one build lane each), default the run's
#     cores (ENGINESMITH_NEURON_CORE.., two lanes when ENGINESMITH_NEURON_CORES >= 8). Cached in target/neff/cache.
#  2. target/neffs.S embeds them (.incbin); host/engine.cpp is the host program.
set -euo pipefail
cd "$(dirname "$0")"
V=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
export PATH=$V/bin:/opt/aws/neuron/bin:/usr/bin:/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
if [ -n "${ENGINE_BUILD_CORES:-}" ]; then
  LANES=$ENGINE_BUILD_CORES
else
  B=${ENGINESMITH_NEURON_CORE:-0}; N=${ENGINESMITH_NEURON_CORES:-4}
  if [ "$N" -ge 8 ]; then LANES="$B-$((B + 3)),$((B + 4))-$((B + 7))"; else LANES="$B-$((B + 3))"; fi
fi
mkdir -p target
"$V/bin/python" tools/build_neffs.py target/neff "$LANES"
{
  echo '.section .rodata'
  echo '.global neff_table_names'
  echo 'neff_table_names:'
  while read -r n; do echo "  .asciz \"$n\""; done < target/neff/neffs.txt
  echo '  .byte 0'
  i=0
  while read -r n; do
    echo '.balign 64'
    echo "neff_$i:"
    echo "  .incbin \"$PWD/target/neff/$n.neff\""
    echo "neff_${i}_end:"
    i=$((i + 1))
  done < target/neff/neffs.txt
  echo '.section .data.rel.ro'
  echo '.balign 8'
  echo '.global neff_table_ptr'
  echo 'neff_table_ptr:'
  i=0
  while read -r n; do echo "  .quad neff_$i, neff_${i}_end"; i=$((i + 1)); done < target/neff/neffs.txt
  echo '.section .note.GNU-stack,"",@progbits'
} > target/neffs.S
NRT=/opt/aws/neuron
g++ -O2 -std=c++17 -pthread -I$NRT/include host/engine.cpp target/neffs.S -o target/engine \
  -L$NRT/lib -lnrt -Wl,-rpath,$NRT/lib
echo "built target/engine ($(du -h target/engine | cut -f1))"
