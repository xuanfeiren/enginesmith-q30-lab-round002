#!/bin/bash
# Build + self-check lab 3 (mixture of experts: Qwen3-30B-A3B in ~/models/Qwen3-30B-A3B). Lab cores 0-15 only.
#   ./build.sh host      C++ drivers (host/ccrun, host/tpchain, host/tpgen)
#   ./build.sh refs      HF transformers fp32 captures on the CPU (build/hf_ref_*.npz: ~6 min, ~10 GB)
#   ./build.sh quick     drivers + router/top-8/gather probe + 2-layer MoE block on 1 core vs HF (~3 min; needs refs)
#   ./build.sh moe       problem 1: 48-layer MoE block TP=4 vs HF + chained benchmark, shard alone (~6 min)
#   ./build.sh decode    problem 2: whole decode step TP=4 (b512, b4608, 470-token prompt), TP=8, TP=16 vs HF (~35 min)
#   ./build.sh prefill   problem 3: MoE prefill TP=4 S=256/1024/4096 vs HF, whole-model prefill + decode (~30 min)
#   ./build.sh nxdi      problem 4: NxD Inference Qwen3-MoE TP=4 on cores 12-15 (~10 min, ~/es-pr2/.venv)
#   ./build.sh all       everything above
# Big weight files go to /dev/shm/nki-lab3 (LAB3_SCRATCH). Every run prints PASS / FAIL and its timings.
set -e
cd "$(dirname "$0")"
source ./env.sh
NRT=/opt/aws/neuron
mkdir -p neff build/logs
for t in ccrun tpchain tpgen; do
  g++ -O2 -std=c++17 -I$NRT/include host/$t.cpp -o host/$t -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
done
echo "host tools built: host/ccrun host/tpchain host/tpgen"
[ "$1" = "host" ] && exit 0
export LAB3_COMM_ID=${LAB3_COMM_ID:-localhost:54301}
LP=$HOME/nki-lab/build/longprompt.txt
run() { echo "== $*"; "$@" 2>&1 | grep -v "Warn\|Loading weights" | grep -E "PASS|FAIL|wrote /|CHECK|BENCH|STEP|GEN rep [12]|agree|identical|differs|mismatch|err|rror|RESULT|HF bf16" ; }
want() { [ "$1" = "$MODE" ] || [ "$MODE" = "all" ]; }
MODE=${1:-all}
if want refs; then
  run python tests/hf_ref.py 24
  TAG=long4k PROMPT_FILE=$LP PROMPT_TOKENS=4096 run python tests/hf_ref.py 1
  TAG=p1000 PROMPT_FILE=$LP PROMPT_TOKENS=1000 run python tests/hf_ref.py 16
  TAG=p4000 PROMPT_FILE=$LP PROMPT_TOKENS=4000 run python tests/hf_ref.py 8
  TAG=p470 PROMPT_FILE=$LP PROMPT_TOKENS=470 run python tests/hf_ref.py 16
  run python tests/hf_bf16_agree.py p470
fi
if want quick || [ "$MODE" = "moe" ]; then
  run python -c "import nb; print('p1 router/top-8/gather', nb.build('p1_rg', 'p1_router_gather:run', '0')[0])"
  run python -c "import nb; print('p1 ties', nb.build('p1_rg_tie', 'p1_router_gather:run', '0', 128, 16, 512, 2, 1, 1)[0])"
  run python -c "import nb; print('p1 tie across 8/9', nb.build('p1_rg_tie2', 'p1_router_gather:run', '0', 128, 16, 512, 2, 1, 2)[0])"
  BENCH=20 run python tests/test_moe.py 1 2 4
fi
if want moe; then
  BENCH=20 run python tests/test_moe.py 4 48 6
  NOCC=1 NOCHECK=1 BENCH=20 CORES=4-7 WTAG=_n run python tests/test_moe.py 4 48
  run python tools/placement.py
fi
if want decode; then
  REPS=3 run python tests/test_step_moe.py 4 24
  TB=4608 REPS=3 run python tests/test_step_moe.py 4 24
  CORES=4-7 NOCC=1 REPS=3 WTAG=_n run python tests/test_step_moe.py 4 24
  TAG=p470 REPS=2 run python tests/test_step_moe.py 4 16
  REPS=3 run python tests/test_step_moe.py 8 24
  SL=2 SLO=8 REPS=3 run python tests/test_step_moe.py 8 24
  REPS=3 run python tests/test_step_moe.py 16 24
fi
if want prefill; then
  for S in 256 1024 4096; do run python tests/test_moe_prefill.py 4 $S 0,24,47; done
  REPS=3 run python tests/test_prefill_tp.py 4 256 24
  TAG=p1000 DEC=smoe_w4_L48_b4608 REPS=3 run python tests/test_prefill_tp.py 4 1024 16
  TAG=p4000 DEC=smoe_w4_L48_b4608 REPS=3 run python tests/test_prefill_tp.py 4 4096 8
  REPS=3 run python tests/test_prefill_tp.py 8 256 24
fi
if want nxdi; then
  (cd build && mkdir -p nxdi && cd nxdi && NEURON_RT_VISIBLE_CORES=12-15 run ~/es-pr2/.venv/bin/python ../../tools/ref_nxdi_moe.py 4 4608 \
     '{"blockwise_matmul_config": {"use_shard_on_intermediate_dynamic_while": true}}')
fi
