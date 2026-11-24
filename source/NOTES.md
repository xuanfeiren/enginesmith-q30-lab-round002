# Qwen3-30B-A3B engine on Trainium2 (TP=4) — notes

## Round 001: the kernel lab 3 kernels behind the engine protocol

What the engine is:
- `build.sh` -> `tools/build_neffs.py` compiles 11 NEFFs through NKI's documented route (NKI_ARTIFACTS_DIR, kernel
  called with zero numpy arrays of the per-rank shapes, one list entry per rank, run once on 4 free cores):
  decode step `dec_b512`, `dec_b4608` (kernels/step_moe.py, lab defaults SL=1 hw=2) and the layered prefill
  `pf{256,1024,4096}_{emb,layer,head}` (kernels/prefill_tp.py). Cached in `target/neff/cache/` keyed by the kernel
  sources + shapes + parameters. Build cores: `ENGINE_BUILD_CORES` (e.g. `8-11,12-15` for experiments), default the
  run's cores (ENGINESMITH_NEURON_CORE.., two lanes of 4 when ENGINESMITH_NEURON_CORES >= 8). Cold build ~5 min
  with two lanes (decode 65-76 s each, pf4096 layer 79 s).
- `target/neffs.S` embeds them with `.incbin`; `host/engine.cpp` (C++, libnrt only) is the whole host side:
  safetensors mmap, the lab's weight re-layouts ported from `kernels/moe_weights.py` / `tests/test_step_moe.py`
  (`split_step`) to C++ (values unchanged), 4 ranks, one scheduling thread per rank, protocol loop.
- The engine restricts `NEURON_RT_VISIBLE_CORES` to the first 4 visible cores before `nrt_init` (the harness lists
  8; the last 4 are the reference judge's).
- Request flow: reset -> prefill bucket = smallest of 256/1,024/4,096 >= P (ids padded with 0), emb + 48 layer
  executions (per-layer slices of the decode tensors) + head (writes toks[P], pos = P); prompts longer than 4,096
  prefill the first 4,096 and feed the rest through the decode step (it reads prompt[pos] while pos < plen); then
  n-1 decode steps, each at bucket 512 if pos < 512 else 4,608, chained on the device (pos buffers X/Y swap).
  Continue -> n decode steps. Tokens read once per request from toks.
- One-token prompts reproduce the reference's quirk (NxD pads to [t, 0] and predicts at the pad position): prefill
  [t, 0], then the host moves that token to toks[1] and sets pos = 1.
- Load (hello): ~26 s (NEFF loads 16 s, weights 9 s for 64 GB over 4 ranks, warm-up of every bucket).
- Host waits: rank threads and stdin spin for 10 s after the last activity, then back off.

Verified:
- `tools/drive.py 8-15 <requests>`: the lab's 12-token prompt gives HF fp32's 24 tokens (" Berlin. The capital of
  Italy is Rome. ... Lisbon. The"), identical as generate(P,24) and generate(P,1) + continue(23); 1-token and
  4-token prompts give sensible text. Times there: generate(12, 24) 182 ms, generate(12,1) 68 ms, continue(23)
  114 ms (~5.0 ms/token).
- dev-eval: see below.

Ideas not done yet (from FINDINGS.md / the 0.6B example): decode buckets at the timed positions (e.g. 1,152,
4,224: b4608 costs 5.77 vs 4.97 ms at b512), TP=8 is not allowed (4 cores); prefill pipelining; o_proj prefetch
variants were measured as no gain in the lab.

dev-eval 001 (1 round, public prompts, lengths 256 / 1,024 / 4,096): **PASSED, score 2.715** (prefill 3.44x,
decode 2.14x). Every timed stream identical to the reference.

| case | prefill ref / ours ms | decode (64 tokens) ref / ours ms |
|---|---|---|
| 256 | 336.9 / 72.1 (4.67x) | 692.1 / 317.7 (2.18x) |
| 1,024 | 455.0 / 113.5 (4.01x) | 813.2 / 368.2 (2.21x) |
| 4,096 | 691.9 / 318.9 (2.17x) | 753.5 / 367.9 (2.05x) |

Control prompts (17 lengths, 1 to 4,097) all passed, 4 identical; streams that differ from the reference differ at near-ties (the
reference's 2nd to 4th choice; e.g. log-prob margins 0.125-0.375 at length 1), at most 2 mismatching steps out of 65 (13 allowed).

Next steps (not tried): decode buckets matching the timed positions (1,024 + 64 and 4,096 + 64 now use b4608,
5.77 ms vs 4.97 at b512 in the lab), the lab's DMA / prefetch variants re-measured on the full step, prefill MoE block
loop (sets the prefill: 1.47 / 2.16 / 4.84 ms per layer).

## Round 002: decode attention buckets / DMAs, prefill without intra-rank HBM round trips

Kept (each verified with `tools/ab4.py`: both builds loaded at once on cores 0-3 / 4-7, requests alternating, same
random prompts, every token compared; all of them token-identical to their parent on every stream):

| change | where | measured (ab4, ms, parent -> new) |
|---|---|---|
| decode bucket 1,152 (buckets now 512 / 1,152 / 4,608; the host takes every embedded `dec_b<TB>` NEFF, warm-up runs each) | `tools/build_neffs.py` DEC_BUCKETS, `host/engine.cpp` | decode 64 tokens after 1,024: 367.9 -> 318.8 (with VD) |
| V cache of a decode step in one DMA instead of one per 128 positions (`VD=1`) | `kernels/step_moe.py` `_attn_tp` | after 4,096: 367.8 -> 346.5; after 256: 318.0 -> 316.1 |
| all score tiles of a decode step in one PSUM tile, mask added by one instruction (`ATT=1`, same fp32 adds) | same | after 4,096: 346.5 -> 331.1; 1,024: 319.2 -> 316.3 |
| prefill NEFFs loaded before the decode NEFFs | `host/engine.cpp` startup | prefill 4,096: 325.3 -> 320.2 (device memory placement: with the bigger VD/ATT decode NEFFs loaded first the 4,096 prefill was 1.8% slower; reproducible, A/A within 0.3%) |
| o_proj: each physical core computes its half of the output tiles from both cores' heads (`OPX=1`), MoE combine: each core combines its token tiles from both cores' `y_slot` (`CMB=1`); the two partials are still added in the same order, so bit-identical, but `s_p` / `y_part` (32 MB per core at 4,096) no longer go through HBM | `kernels/prefill_tp.py` | prefill 72.3 / 113.3 / 320.3 -> 70.3 / 108.8 / 301.2 |
| MoE dispatch compaction split over the two physical cores (8 `nonzero_with_count` rounds each into the shared `stag[0]`, `CSPLIT=1`) | same | prefill 70.5 / 108.5 / 300.8 -> 69.2 / 106.6 / 293.3 (A/A bias of the B arm +0.3-0.6%) |

| 4,096 only: each physical core routes its own token tiles (norm, x_tm, router, top-8; h += the attention all-reduce result fused into that loop), the other core's one-hot / ids / weights come through `y_part` after a barrier (`RSPLIT`, per bucket in `PF_PER`) | same | 4,096 prefill 292.5 -> 284.9; at 1,024 it was slower (105.9 -> 108.1), so off there; 256: +0.8% (within the A/A bias), off |

dev-eval 003 (1 round, public prompts, before RSPLIT): **PASSED, score 2.855** (prefill 3.647, decode 2.235; round 001's dev-eval
2.715), every timed stream identical to the reference:

| case | prefill ref / ours ms | decode ref / ours ms |
|---|---|---|
| 256 | 337.0 / 69.9 (4.82x) | 679.3 / 315.5 (2.15x) |
| 1,024 | 455.6 / 106.8 (4.27x) | 747.2 / 316.5 (2.36x) |
| 4,096 | 690.1 / 292.6 (2.36x) | 727.6 / 331.4 (2.20x) |

dev-eval 004 (2 rounds, the final engine of this round): **PASSED, score 2.865** (prefill 3.686, decode 2.227), 0
mismatching steps: prefill 70.3/68.8, 107.1/105.8, 284.9/285.0 ms (4.84x / 4.28x / 2.42x), decode 315.5/315.7,
316.6/316.3, 331.0/331.0 ms (2.15x / 2.37x / 2.17x).

Next steps (not tried): bf16 prefill all-reduces (numerics change); the MoE expert block loop is PE-bound
(transposes + 32 gate/up matmuls of N=128 + 4 down of N=512 per block, the PE throttled to ~50% under sustained
load per the profiler's throttle metrics) and sets ~40% of the prefill; the combine (16 software-DGE gathers per pair
of token tiles) ~10% at 4,096; decode is the per-layer chain (two ~8-10 us all-reduces + router before the expert
slabs) that the lab's prefetch variants did not shorten.

Where the prefill time goes now (multi-rank profile of the layer NEFF: `neuron-explorer capture -n pf4096_layer.neff
-r 4 --collectives-worker-count 4 -i 0` on cores 0-3 with random inputs, rank 0, sg00; layer 6.15 ms at 4,096):
QKV + RoPE 0.41 ms, attention_cte 0.65, o_proj 0.15, attention all-reduce (32 MB fp32) 0.38 + read-back 0.13, router
+ x_tm 0.29, dispatch 0.5 (before CSPLIT), expert blocks 2.54 (384 blocks, PE-bound: transposes 16 + gate/up 32 +
down 4 matmuls per block, Tensor queue ~100% busy at 1,024), combine 0.65, MoE all-reduce 0.25 + read-back 0.13.
Ablations (timing only, wrong results; `ABL`, `NOAR`): prefill without the MoE half 9.8 / 23.2 / 92.7 ms, without
attention 68.8 / 105.5 / 242.3 ms, without both all-reduces 68.8 / 107.1 / 296.4 (vs 71.7 / 112.7 / 320.0).
Decode step profile (b512, 5.19 ms with random inputs): 98 all-reduces 0.88 ms in all, DMA active 72%, ~100 us per
layer of which ~30 us with no weight DMA in flight (attention compute, the two all-reduces, router before the expert
slabs).

Tried and rejected (numbers from ab4 unless noted; all token-identical):
- KV cache loads on the scalar engine's DMA queue (`KVQ=1`): decode after 4,096 331.3 -> 338.0 (slower).
- MoE all-reduce in 2 / 4 chunks, each chunk's all-reduce issued as soon as it is combined (`MCH`): 4,096 prefill
  320.5 -> 323.3, 1,024 113.1 -> 113.4 (no overlap gained).
- Smaller MoE blocks for short prompts (`BS` 32 at 256, 64 at 1,024): 71.8 -> 76.7 and 113.5 -> 140.7 (per-block cost
  is fixed, more blocks = slower).
- Blocks past 8S/128 computed inside `fori_loop(0, flag)` so empty ones skip their compute (`TAIL=1`): 72 -> 178 ms
  (~17 us per fori_loop instance).
- Static block loop capped at NCAP, the rest in one `fori_loop(NCAP, n_real)` (`NCAP`, correct: tokens identical
  with 64-112 dynamic blocks per layer, ~9.5 us per dynamic block vs ~4-6.6 static): caps 136 / 144 / 336 gave
  69.0 / 109.4 / 317.7 vs 70.3 / 108.5 / 300.3: with the real model the routing is skewed and the real block count
  is close to the static worst case 8S/128 + 128 (the device profile with random weights is misleading here: it showed
  1.38 vs 1.52 ms per 256-layer).
- Token-major gate/up (x tile stationary, the 192 gate + up columns moving, `GU=1`; also with the act transpose
  moved into the down step): 1,024 layer 2.39 vs 2.14 ms device, prefill 113.1 -> 119.9 ms (N=192 matmuls ran at
  ~1.1 columns/ns vs 2.1 for N=128).
- Block DMA prefetch depth 1 / 3 (`PD`): no gain (device 2.15 / 2.35 vs 2.14 ms at 1,024).
- Next layer's QKV (+ o_proj) prefetched on the weight queue right after this layer's expert slab DMAs (`PFX=1/3`):
  decode unchanged (315.2 vs 315.4 ms after 256), as the lab found for its prefetch variants.
- More decode buckets (384, 4,224 in addition): decode 315.0 -> 316.0 after 256, 331.0 -> 332.7 after 4,096 (no
  gain; the per-tile attention cost after VD / ATT is small).
- Not tried: bf16 all-reduces in prefill (would save at most ~12 / 3 / 1.5 ms at 4,096 / 1,024 / 256, changes the
  numerics).

Dev tools added: `tools/ab4.py` (paired timing + token check of two builds), `tools/xbuild.sh NAME` (variant build in
/tmp/q30x/NAME sharing the NEFF cache; variants through `ENGINE_DEC_KW`, `ENGINE_DEC_BUCKETS`, `ENGINE_PF_KW`,
`ENGINE_PF_MCH/BS/NCAP`). The NEFF cache key now hashes only the kernels a target imports.
