# NKI lab 3 findings: mixture-of-experts decode and prefill (Qwen3-30B-A3B on Trainium2)

Toolchain: Neuron runtime 2.34.10, aws-neuronx-collectives 2.34, neuronx-cc 2.27.5334, nki 0.6.0, logical NeuronCore
config 2 (LNC=2). Lab cores: logical NeuronCores 0-15 only (devices 0-3; `neuron-ls`: device 0 is a NeuronLink
neighbour of 1 and 3, so 0-1-2-3 form a ring). Every process sets `NEURON_RT_VISIBLE_CORES`. Every number is
measured unless it says *estimate*. "queued" = `nrta_execute_schedule` with many executions in flight, wall time /
count, from the C++ drivers in `host/` (one scheduling thread per rank). Model: Qwen3-30B-A3B bf16 (48 layers, hidden
2048, 32 q / 4 kv heads of 128, 128 experts with intermediate 768, top-8 with renormalised weights, vocab 151,936;
61 GB). Reference for correctness: HF transformers 5.15 fp32 on the CPU (`tests/hf_ref.py`, 26 s to load, all
captures in one run).

## Summary

| | ours | reference (NxD Inference, same TP degree) |
|---|---|---|
| Qwen3-30B-A3B decode, TP=4 (cores 0-3, one device), b512 attention bucket | **4.97 ms/step** (11/11 teacher-forced, 24/24 generated = HF fp32) | 10.2-11.5 ms/token (TP=4, cores 12-15) |
| same, TP=8 (cores 0-7, two devices), 1 MB QKV / 2 o_proj DMAs per layer | **4.22 ms/step** (11/11, 24/24) | 9.3-11.0 ms/token (TP=8, cores 8-15) |
| same, TP=16 (cores 0-15, four devices) | 4.61 ms/step (11/11, 24/24): slower than TP=8 | - |
| MoE half-layer alone (router + top-8 + 8 gathered experts + combine + all-reduce), TP=4 | 60.3 us per layer | - |
| prefill 256 / 1,024 / 4,096 tokens, whole model, TP=4 (then decode, tokens = HF fp32) | **68.9 / 113.6 / 319.2 ms** | 337 / 445 / 688 ms (TP=4) |
| same, TP=8 | **58.6 / 104.3 / 294.5 ms** | 396 / 506 / 718 ms (TP=8) |
| MoE prefill half-layer alone, TP=4, 256 / 1,024 / 4,096 tokens | 1.47 / 2.16 / 4.84 ms | - |

What an MoE engine should do (details and numbers in the sections):
1. Router, top-8 and the weights on the device (`max8` + `nc_match_replace8`, not `nc_find_index8`); fetch each
   selected expert's slab by a DMA whose offset is the expert id read from SBUF: no bandwidth penalty (827 GB/s per
   logical core, same as static DMAs). Keep every dynamically indexed tensor below 4 GiB (beyond, wrong data or
   `NRT_EXEC_OOB`).
2. Tensor parallelism inside every expert (2W virtual cores x IM/2W rows), not expert parallelism: on real routing EP
   makes the slowest rank stream 1.25-3.9x more bytes.
3. Batch-1 decode is a per-layer dependency chain (router before the expert DMAs, two all-reduces per layer) and
   collectives slow down when weight DMAs are in flight: measure every change on the multi-rank step.
4. Prefill with static shapes: dispatch by stream compaction (`nonzero_with_count`), a static worst-case block
   schedule (8S/128 + E blocks of 128 slots, empty blocks' DMAs skipped by `oob_mode=skip`), layered prefill NEFFs on
   slices of the decode step's tensors.

Status: all four problems solved with measured numbers. Build / test: `./build.sh` (end of this file). A full
`./build.sh moe; ./build.sh decode; ./build.sh prefill` run (08:30-09:12 UTC, `build/logs/build_verify.log`) reproduced
every number here within 1.5% (decode TP=4 4.971 ms, b4608 5.770, TP=8 4.272 / 4.224, TP=16 4.596; 470-token prompt
460/469; MoE prefill 1.484 / 2.184 / 4.790 ms per layer; whole prefill 68.2 / 113.1 / 319.3 ms TP=4, 58.7 ms TP=8
S = 256) with 16 PASS and 0 FAIL.

---------------------------------------------------------------------------------------------------------

## 1. The MoE decode step

Kernel `kernels/moe_tp.py` (`moe_block`; `_router`, `_experts`, `_moe_layer` are used by the whole step too). Host
re-layout `kernels/moe_weights.py`. Test `python tests/test_moe.py <W> [L=48] [NTOK=6]` (env `NOCC=1`: rank 0's
shard alone, `KW='{"OPT": n}'` variants, `MOD=moe_x` the experimental copy).

### What one MoE half-layer does (one token, all on the device)
1. `x = RMSNorm(h) * ln2` (column layout `[128, 16]`, lab 1).
2. Router logits as a **row** `[1, 128]` fp32 in PSUM: `x[:, kt]` (one column) is the stationary operand and the
   router tile `Wr^T [128 k, 128 e]` the moving one, 16 accumulating matmuls.
3. Top-8: `nisa.max8` (the 8 largest, descending) then `nisa.nc_match_replace8(dst_idx=...)` for their positions.
   Do not use `nc_find_index8` for this: it returns the *first* occurrence of each value, so two experts with the same
   logit would both map to the same index (documented in the ISA docstring). `nc_match_replace8` pairs duplicate values
   with distinct positions.
4. Weights `w_k = exp(v_k - v_0) / sum_j exp(v_j - v_0)` (ScalarE exp with fused sum, then reciprocal). This equals
   HF's `softmax(logits, fp32)` over all 128, `topk(8)`, `/= sum`, because the softmax denominator cancels in the
   renormalisation (`norm_topk_prob = true` for this model). Measured against a numpy fp32 implementation of HF's
   rule on the same bf16 inputs: weights within 6e-7, logits within 4e-7 (`probes/p1_router_gather.py`).
5. For each slot k, the slab of expert `idx[k]` owned by this virtual core is DMA'd by a token-dependent offset:
   `wgu[l, :, pid].select(0, idx.ap(pattern=[[8, 1], [1, 1]], offset=k))` (an SBUF element as the scalar offset,
   nkilib's `selective_expert_impl.py` pattern), hardware DGE from the sync engine.
6. gate/up GEMV (weight-stationary, `[96, 1]` outputs), `silu(g) * u * w_k` (fp32, cast to bf16), down GEMV
   accumulated over the 8 experts in one PSUM tile (the weighted combine is the accumulation), then the TP
   all-reduce of the partial hidden vector (lab 2's sendrecv + HBM all_reduce) and the residual add.

### Tie and normalisation rules (exact)
- HF `Qwen3MoeTopKRouter`: `softmax(logits, dtype=float32)`, `torch.topk(probs, 8)`, renormalise, cast to the
  hidden dtype. NxD Inference `RouterTopK`: `topk` on the **logits** (same order as on the probabilities except where
  distinct logits round to the same fp32 probability), affinities = softmax cast to the hidden dtype (bf16 in a
  bf16 model) and renormalised later in the expert MLP.
- Ties: `torch.topk` on the CPU does **not** prefer the lowest index. With 10 experts tied at the top it returned
  `[80, 77, 90, 60, 9, 100, 70, 50]` and left out experts 3 and 5 (measured; the order is implementation-defined).
  Ours: `nc_match_replace8` pairs tied values with their positions lowest first, so a tie at the 8/9 boundary goes to
  the lower expert id. Measured (`probes/p1_router_gather.py`, `tie=1/2`): 4 experts with identical router rows at
  the top -> 4 distinct indices, weights 0.25 each; **10** experts tied at the top -> the kernel selects the 8 lowest
  ids [3, 5, 9, 50, 60, 70, 77, 80] (weights 0.125), where `torch.topk` on the CPU selected
  [80, 77, 90, 60, 9, 100, 70, 50] for the same pattern. So for an exact tie across the 8/9 boundary the sets differ
  from HF's by construction (torch's choice is an implementation detail, not a rule); with real activations no such
  tie occurred, the 2-37 differences per 288-4,096 decisions are near-ties (below).
- The weights are fp32 throughout on the device (HF fp32 reference: fp32; HF bf16 and NxD: bf16).

### Correctness on real tokens (HF fp32, Qwen3-30B-A3B)
`tests/hf_ref.py` hooks every layer: the residual entering the MoE half-layer, the MoE output, the router logits and
HF's top-8 indices / weights, for the 12-token prompt "The capital of France is Paris. The capital of Germany is"
plus 24 greedy tokens (36 positions). `test_moe.py` feeds 6 of these positions to all 48 layers (each layer gets HF's
own input) and compares:

| | TP=1, 2 layers | TP=4, 48 layers (cores 0-3) |
|---|---|---|
| top-8 set identical to HF | 8/8 | **286/288** |
| the 2 differences | - | near-ties at the 8/9 boundary: HF's 8th-9th logit gap 2.7e-4 (token 14, layer 34) and 6.4e-4 (token 35, layer 44); our router input is the bf16-rounded normalised hidden state |
| routing weights (same set) | max abs err 2.0e-4 | 9.9e-4 |
| MoE output error / max abs MoE output (same set) | 2.4e-3 | (see section 2: end-to-end tokens identical) |
| ranks | - | identical outputs and indices on all 4 ranks |

### Placement: tensor parallel inside every expert, not expert parallel
Measured routing (HF's decisions on the 36 x 48 = 1,728 real (token, layer) pairs, `tools/placement.py`): with pure
TP every rank streams exactly 8/W experts' worth of bytes per token; with expert parallelism (EP degree D, experts
assigned contiguously or strided, TP W/D inside a group) the slowest rank streams on average:

| W | pure TP | EP 2 x TP W/2 | EP 4 x TP W/4 | EP 8 x TP W/8 | EP 16 |
|---|---|---|---|---|---|
| 4 | 2.00 experts | 2.51 (1.25x) | 3.43 (1.71x) | - | - |
| 8 | 1.00 | 1.25 (1.25x) | 1.71 (1.71x) | 2.49 (2.49x) | - |
| 16 | 0.50 | 0.63 (1.25x) | 0.86 (1.71x) | 1.24 (2.49x) | 1.94 (3.88x) |

(p99 is worse still: EP 4 at W=4 streams 5-6 experts on the slowest rank.) For batch-1 decode, bytes on the slowest
rank set the step, so **TP inside every expert** (each of the 2W virtual cores owns 768/2W rows of every expert's
gate/up and the same columns of down) is 1.25-3.9x better than any EP split. EP would also need a variable number of
experts per rank inside a static NEFF (masking or device loops). TP costs one all-reduce per MoE layer, which dense
TP pays anyway.

The 48 layers: every rank holds a slice of every layer. Splitting the layers over devices (pipeline parallelism, e.g.
layers 0-23 on device 0 and 24-47 on device 1, TP=4 inside each) holds the model as well but does not shorten a
batch-1 step: the token still crosses all 48 layers in sequence, each at TP=4 speed, plus a device-to-device hop
(*estimate*: >= the 4.97 ms of TP=4 on one device); it only helps throughput with several sequences in flight.
TP over both devices (TP=8) is what made the step faster (4.22 ms).

Per virtual core slices: W=1: 3 x 128 rows; W=2: 2 x 96; **W=4: 96**; W=8: 48; W=16: 24 (`moe_weights.il_tiles`).
HBM per rank for the experts: 58 GB / W (14.5 GB at W=4).

### The bandwidth floor
Bytes per token per logical core (TP=W): experts `48 x 8 x 9.44 MB / W` + router (each physical core reads the
whole 0.5 MB router of the layer) = 956 MB at W=4 for the MoE half-layers alone.

Streaming rate of token-dependent DMAs (`probes/p2_gather_bw.py`, 48 x 8 gathered slabs of [128, F] per physical
core, queued, `probes/bench_p2*.py`). Correction: the first sweep (`bench_p2.py`, `bench_p2b.py`, `bench_p2c.py`)
compiled the probe without the `[2]` grid, i.e. as an LNC=1 NEFF on ONE physical core, and counted the bytes of two;
the per-logical-core figures it printed were 2x the real ones. The rows below are per physical core for those, and
the LNC=2 rows (`bench_p2d.py`, both physical cores streaming their own halves) give the per-logical-core rates:

| DMAs (core 15 unless noted) | one physical core (LNC=1 NEFF) | logical core (LNC=2 NEFF) |
|---|---|---|
| dynamic offset (index read from SBUF), hardware DGE on the sync engine, 9 KB rows (1.18 MB slabs) | 413-414 GB/s | **826 GB/s** |
| same, 6 KB rows (786 KB slabs) | 401 GB/s | 808 GB/s |
| dynamic, software DGE (GpSimd descriptors) | 414 GB/s | - |
| static offsets (same slabs, indices baked in) | 414 GB/s | - |
| dynamic, hardware DGE alternating sync / scalar queues | 416 GB/s | - |
| one physical core on each of the 4 logical cores of device 0 at once (4 processes, LNC=1) | 412-414 GB/s each | - |
| dynamic + every 128-column tile consumed by a matmul (a real GEMV's PE work), one PSUM tile per layer | 372-381 GB/s | **742-761 GB/s** |
| same, but one PSUM tile accumulating all 48 x 8 x 36 = 13.8k matmuls | 133 GB/s | - |
| row size sweep (LNC=1): 2 / 4 / 6 / 9 / 16 KB rows | 198 / 390 / 405 / 414 / 423 GB/s | - |

The two physical cores of a logical core stream without slowing each other (404-413 GB/s each).
So a dynamic offset costs nothing in bandwidth. Floor used below: bytes per logical core / 743 GB/s (the measured
LNC=2 rate with PE consumption; an *estimate* when applied to a whole step).

| 48 chained MoE half-layers (`tests/test_moe.py <W> 48`, CHAIN NEFF, queued) | bytes per logical core | floor at 743 GB/s | measured | per layer | vs HF fp32 (6 tokens x 48 layers) |
|---|---|---|---|---|---|
| TP=4 (cores 0-3) | 956 MB | 1.29 ms | **2.90 ms** | 60.3 us | 286/288 sets, weights 9.9e-4 |
| TP=4 rank 0's shard alone (`NOCC=1`) | 956 MB | 1.29 ms | 2.16 ms | 44.9 us | - |
| TP=8 (cores 0-7) | 503 MB | 0.68 ms | **2.44 ms** | 50.9 us | 286/288 sets (the same 2 near-ties) |

At TP=8 the MoE half-layer is 4x its byte floor: the router-before-experts chain and one all-reduce per layer
(~17 us over two devices) dominate. Spreading the slab DMAs over two DMA queues (down slabs on the scalar engine's
queue, `OPT=1024`) made it slower (56.0 us per layer), one explicit register per expert (`OPT=2048`) changed nothing
(51.0 us).

### Pitfall: a dynamically indexed DMA addresses at most 4 GiB of its tensor
`probes/p2_gather_bw.py` with a growing expert count E (tensor `[48, E, 2, 128, 4608]` bf16), gathering from all 48
layers: E = 20 (2.26 GB), 30 (3.40 GB), 36 (**4,076,863,488 B**): exact. E = 38 (**4,303,355,904 B**, > 2^32): wrong
data, no error (max abs error 37.9 vs 2e-6). In the MoE kernel at W=4 (one 9.66 GB gate/up tensor) the first build
run failed instead: `TDRV:exec_process_custom_notification ... scatter/gather (indirect memory copy via scalar DGE)
out-of-bound access` then `NRT_EXEC_OOB(1006)` on every rank. Fix: the expert weights are passed as 4 layer groups
(`wgu0..3`, `wd0..3`, each < 4 GiB; asserted in `moe_weights.py`). Static DMAs from the same big tensors are fine.

### Where the MoE time goes (profile of rank 0's shard, W=4, 8 layers, `tools/prof.py`, `tools/tl2.py`)
47.4 us per layer (queued, no communication). One physical core, one layer:
- expert slab DMAs stream at ~300 GB/s per physical core for ~24 us; each dynamic DMA costs the sync engine
  ~1.25 us to issue (`TENSOR_LOAD` of the index + 3 `ALU_OP` + `MOVE` + `DMA_DIRECT2D` 0.6 us), 16 per layer;
- a ~14 us gap with no DMA: the last down matmuls (4 us), the partial-sum exchange, the next RMSNorm (2.3 us),
  router matmuls (1.9 us), max8 / match_replace8 / exp / reciprocal (1.8 us), the first dynamic DMA issue (1.3 us).
  The next layer's expert weights cannot be fetched before its router has run: this chain is inherent to MoE decode.
- With the all-reduce (W=4): 60.3 us per layer, i.e. 15.4 us per all-reduce on the critical path.

TensorE rates of the decode shapes (`probes/p3_pe_rate.py`, weights resident, ns per matmul with a one-column moving
operand): `[128, 128]` stationary 34 ns (also at unaligned column offsets), `[128, 96]` **80 ns**, `[128, 64]` 53 ns,
down `[96 k, 128]` 34 ns, x-stationary `[128, 1]` x `[128, 192]` moving 82 ns, `[96, 1]` x `[96, 512]` 54 ns. The
gate/up GEMV now uses 128-wide stationary windows (the 96 real columns + 32 columns of the next slab rows, whose
outputs land on unused partitions; the slab tile has 32 padding columns): `W128`, default. MoE chain at W=4 L=48:
60.3 us per layer either way, shard 44.7 vs 44.9 us: the expert stream is DMA-bound, not TensorE-bound.

Expert-major down projection (each expert's 16 down matmuls right after its slabs arrive, all experts accumulating
into the columns of one PSUM tile in interleaved order with explicit `accumulate=(k > 0)` flags, `MOEOPT=128`):
**wrong results** (teacher-forced 7/11, generated 0/24) and only 1% faster (4.948 ms, shard 3.689). Interleaving the
accumulation groups of different columns of one PSUM tile is not safe; keep each column's accumulation group
contiguous, or one PSUM tile per expert.

Variants tried (all slower or equal, shard alone, 8 layers): per-expert down GEMV + fp32 combine as each expert's
slabs arrive (51.3 us), plus down slabs on the scalar engine's DGE queue (57.4), plus router with the weights
stationary + PE transpose (56.8), plus next layer's router weights prefetched (57.0), explicit register for the expert
id shared by both slab DMAs (47.7), down slabs on software DGE (54.6). Baseline 47.4 us.

---------------------------------------------------------------------------------------------------------

## 2. The whole Qwen3-30B-A3B decode step

Kernel `kernels/step_moe.py` (`step_moe`): embedding -> 48 x {attention half-layer (lab 2's TP attention, unchanged:
QKV N-split by heads, the 4 kv heads replicated over the 2W virtual cores, KV cache at the device-held position,
o_proj K-split + all-reduce), MoE half-layer (section 1)} -> final norm -> LM head (vocab N-split) -> argmax (ties to
the lowest id, lab 2) -> `toks[pos + 1]`, `pos + 1`. One NEFF per step, SPMD on W ranks, chained from C++
(`host/tpchain`, two tensor sets swapping `pos` / `output_0`, one scheduling thread per rank).
Test: `python tests/test_step_moe.py <W> [G=24] [L=48]` (env `CORES`, `REPS`, `NOCC=1` rank 0's shard alone,
`SL SLO SLH` n-tiles per weight DMA of QKV / o_proj / LM head, `PF` `PFO` prefetch, `WTAG` scratch dir per concurrent
run). Weight files are written to `/dev/shm/nki-lab3` (the root disk writes at ~130 MB/s: the first split of 58 GB
to `build/` took 525 s, to `/dev/shm` 12-24 s).

### Fewest cores: 4 (one device)
61 GB of bf16 weights; per logical core 24 GiB. TP=4: 16.03 GB per rank (experts 14.5 GB, attention 0.45 GB,
embedding 0.62 GB (every rank), LM head shard 0.16 GB, KV cache 4,608 positions 0.23 GB). TP=2 cannot hold it
(~31 GB per rank, *estimate* from the shapes). TP=3 would (~21.5 GB per rank, *estimate*), and its MoE split is even
(768 / 6 virtual cores = 128 rows, one full tile), but its attention is not: 32 q heads over 6 virtual cores means
padding to 36 heads (zero heads with zero o_proj columns add nothing) and virtual cores whose q heads span two of
the 4 kv heads, which lab 2's attention code (G consecutive q heads per local kv head) does not handle; not built.
TP=4 per virtual core: 4 q heads, 1 kv head (each kv head on 2 virtual cores), 96 rows of every expert.

### Measured (b512 attention bucket, real weights, 12-token prompt + 24 greedy tokens, REPS=3, min chain ms/step)

| | NEFF | compile + 1 run | nrt_load | upload | ms/step | check vs HF fp32 |
|---|---|---|---|---|---|---|
| **TP=4** (cores 0-3) | 2.55 MB | 62 s | 1.1 s | 14.3 s (16.0 GB per rank) | **4.974** | 11/11 teacher-forced, 24/24 generated, same tokens on all ranks |
| TP=4 rank 0's shard alone, no communication (`NOCC=1`, core 4) | 2.48 MB | 56 s | | | 3.758 | - |
| **TP=8** (cores 0-7, devices 0 + 1) | 2.29 MB | 63 s | | (8.5 GB per rank) | **4.303** | 11/11, 24/24 |
| TP=8 rank 0's shard alone (`NOCC=1`, core 8) | | | | | 2.619 | - |
| **TP=8, 1 MB QKV DMAs and 2 o_proj DMAs per layer (`SL=2 SLO=8`)** | | | | | **4.220** | 11/11, 24/24 |
| TP=16 (cores 0-15, devices 0-3; per virtual core 1 q head, 24 expert rows, kv heads replicated 8x) | 2.17 MB | 62 s | | (4.74 GB per rank) | 4.610 | 11/11, 24/24 |
| TP=4, b4608 attention bucket | 2.95 MB | 77 s | | | 5.767 | 11/11, 24/24 |
| TP=4 baseline NEFF re-run (`NOBUILD=1 REPS=5`) | | | | | 4.980 (reps 4.980-4.984) | - |

Variants (each changes one thing against the TP=4 or TP=8 rows above; all keep the tokens identical except where
noted):

| | NEFF | compile + 1 run | nrt_load | upload | ms/step | check vs HF fp32 |
|---|---|---|---|---|---|---|
| TP=4, next layer's QKV prefetched during the MoE (`PF=6`) | 2.55 MB | 67 s | | | 4.958 | 11/11, 24/24 |
| TP=4, 1 MB weight DMAs (`SL=2 SLO=4 SLH=2`) | | | | | 5.102 | 11/11, 24/24 |
| same, shard alone | | | | | **3.333** | - |
| TP=4, `SL=2 SLO=4 SLH=2 PF=3 PFO=1` (QKV and o_proj of the next layer prefetched) | 2.46 MB | 64 s | | | 5.058 | 11/11, 24/24 |
| same, shard alone | | | | | 3.325 | - |
| TP=8 `SL=2 SLO=8` + the expert id in one explicit register for both slab DMAs (`MOEOPT=2048`, cores 8-15) | | | | | 4.213 | 11/11, 24/24 |
| TP=4, o_proj DMAs of 512 KB only (`SLO=4`) | | | | | 5.079 | - |
| TP=4, down projection in 4 expert groups, each group's down GEMVs right after its gate/up (`MOEOPT=256`, cores 8-11) | | | | | 4.998 | 11/11, 24/24 |
| same, shard alone (core 12) | | | | | 3.782 | - |
| TP=4, next layer's o_proj prefetched during the MoE, 512 KB o_proj DMAs (`PFO=1 SLO=4`, cores 8-11) | | | | | 4.999 | 11/11, 24/24 |
| same, shard alone (core 12) | | | | | **3.295** | - |
| TP=4, router with the weights stationary + one PE transpose of the logits (`MOEOPT=4`, not checked vs HF) | | | | | 4.957 | - |
| TP=4, QKV slabs on the scalar engine's DMA queue, issued when ScalarE gets there (`QKVQ=1`) | | | | | 5.059 | 11/11, 24/24 |
| TP=4, next layer's o_proj prefetched on the scalar engine's queue after the router, i.e. with the expert slabs and not during an all-reduce (`PFO=1 PFOQ=1 SLO=4`, cores 8-11) | | | | | 5.031 | 11/11, 24/24 |
| TP=4, no sendrecv before the all-reduce (`XCH=1`: each physical core writes its half of a [2, 128, 16] HBM buffer, `core_barrier`, one all-reduce, both halves added; cores 8-11) | 2.58 MB | 71 s | | | 5.021 | 11/11, 24/24 |

Text: " Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The"
(HF fp32, NxD Inference TP=4 and TP=8 give the same 24 tokens).

Long prompt (`TAG=p470`: the first 470 tokens of `~/nki-lab/build/longprompt.txt`, fed token by token through the
decode chain, then 16 greedy tokens; 485 chained steps):

| | teacher-forced argmax = HF fp32 | the mismatches | generated |
|---|---|---|---|
| TP=4 (4.955 ms/step) | **460/469** | near-ties: HF fp32 top-1 - top-2 logit gap median 0.030, max 0.31 (median over all positions 1.03) | 10/16, then a near-tie (gap 0.030) |
| TP=8 `SL=2 SLO=8` (4.196 ms/step) | **460/469** | gap median 0.028, max 0.16 | 10/16, the same near-tie |
| HF transformers **bf16** on the CPU, same sequence (`tests/hf_bf16_agree.py p470`) | 445/469 | gap median 0.088 | - |

So the engine is closer to HF fp32 than HF's own bf16 model (residual stream and routing weights in fp32, fp32 PSUM
accumulation); MoE models flip more argmaxes than the dense lab-2 models (8B TP=2: 467/469) because a near-tie in a
router changes which expert runs.

Decomposition at TP=4: communication (98 all-reduces: 2 per layer + 2 for the argmax) = 4.974 - 3.758 = 1.22 ms,
12.4 us per all-reduce, the in-isolation latency of a 4-rank HBM all-reduce (lab 2: 11.3 us). Bytes per rank per
step: experts 906 MB + router 50 MB + attention 453 MB + LM head 156 MB + KV 25 MB = 1.59 GB -> floor 2.14 ms at
743 GB/s (*estimate*). The shard alone streams at 423 GB/s (3.76 ms) or 477 GB/s (3.33 ms with 1 MB DMAs).

TP=16 needed the argmax all-reduce tiles widened from [128, 8] to [128, 16]: with 8 columns `nrt_load` asserted
`tdrv/instr_collectives.c:538: get_tensor_mr_size: Assertion 'tensor->num_elem[0] != 0 && tensor->num_elem[1] != 0'`
(lab 2's rule: free size >= W).

At TP=8 the sync engine issuing the weight DMAs is close to saturation: in rank 0's shard profile it is busy 1,970 of
2,673 us, 1,586 us of it in 2,091 `DMA_DIRECT2D` (0.76 us each, about the same for a 64 KB o_proj slab and a 590 KB
expert slab; dynamic ones add `TENSOR_LOAD` + 3 `ALU_OP` + `MOVE`, ~1.26 us in all). Per layer 44 DMAs: 16 expert
slabs, 16 o_proj slabs of 64 KB, 4 QKV, 5 KV cache, 1 router. Fewer, bigger DMAs (`SL=2 SLO=8`) took the step from
4.303 to 4.220 ms.

TP=8: communication = 4.303 - 2.619 = 1.68 ms (17 us per all-reduce over two devices); the shard streams 0.83 GB in
2.62 ms (317 GB/s): at 55 us per layer for 17 MB per rank the step is the per-layer dependency chain plus the
all-reduces.

Every change that made rank 0's shard faster (3.76 -> 3.30 ms with the o_proj prefetch) left the full TP=4 step at
4.98-5.10 ms: the communication share grew by the same amount (1.22 -> 1.70 ms). Down slabs on the scalar engine's
DMA queue (`OPT=1024`, a second descriptor stream): MoE shard 51.9 vs 47.4 us per layer (slower).

**Why: weight DMAs in flight slow the all-reduces down** (`probes/p4_cc_contention.py`, `probes/bench_p4.py`, TP=4 on
cores 8-11, queued): a chain of 33 dependent HBM all-reduces of a [128, 16] fp32 partial (with the SBUF -> HBM -> SBUF
copies of the decode step) takes 577.3 us (15.4 us per all-reduce; 1 all-reduce: 85.6 us), 128 MB per physical core
of independent 1 MB weight DMAs alone take 234.8 us, both together **683.2 us** with the weights on the sync engine's
hardware DGE queue (680.1 on the scalar engine's queue, 726.5 on GpSimd software DGE) instead of max(577, 235) = 577:
while the weights stream, the all-reduces are ~45% slower. A decode step that prefetches weights during its
all-reduces pays for it there.

Pitfall: bigger weight DMAs and prefetching made the shard 11% faster but the full TP step 3% slower (5.06-5.10 vs
4.97 ms): with the weight queues busier, the all-reduces (which use the same DMA engines) took longer
(1.73 ms vs 1.22 ms per step). Measure changes on the full multi-rank step, not only on the shard.

---------------------------------------------------------------------------------------------------------

## 3. MoE prefill: grouped expert GEMMs over the tokens routed to each expert

Kernel `kernels/moe_prefill.py` (`moe_prefill`, one MoE half-layer for S tokens, TP over W ranks with the decode
kernel's expert split: virtual core g owns rows [96g, 96g + 96) of every expert at W=4). Test
`python tests/test_moe_prefill.py <W> <S> [layers]` (HF fp32 capture of a 4,096-token text, `TAG=long4k`:
`PROMPT_FILE=~/nki-lab/build/longprompt.txt PROMPT_TOKENS=4096 python tests/hf_ref.py 1`, 60 s for the
teacher-forced pass; the first S tokens of it serve S = 256 / 1,024 / 4,096 since the model is causal).

### Design (all static shapes, no device loops)
1. Router for every 128-token tile (`[128 tok, 128 e]` logits: the x tile stationary, Wr^T moving), `max8` +
   `nc_match_replace8` per partition (= per token), weights as in decode. One-hot `M_j [128 tok, 128 e]`.
2. Rank of each token inside its expert `R_j = Lstrict^T M_j + sum_{i<j} ones^T M_i` and counts (exact small integers
   in fp32 PSUM); blocks of B = 128 slots, `nb_e = ceil(cnt_e / 128)` computed as `sum_m (cnt_e > 128 m)` with m on
   the partitions (`tensor_scalar` has no `mod` / floor: `unsupported operator 'mod'`); block starts by a matmul with
   a strictly upper-triangular ones matrix; `slot(t, k) = 128 bs_e + R[t, e]` for the combine.
3. Per-expert token lists by **stream compaction**: M^T (experts on partitions), 16 rounds that move experts 16c + r
   to partitions 16c with a 0/1 matmul, `nisa.nonzero_with_count` (GpSimd; it only reads / writes partitions
   0, 16, ..., 112) writes each expert's token ids in increasing order padded with S (a zero row), 8 rows per round to
   HBM. ~200 us at S = 1,024.
4. NMAX = 8S/128 + 128 blocks (the static worst case). Block b: expert id e_b (E for blocks past the last real one:
   their slab DMAs are skipped with `oob_mode=skip`), DMA its two slabs (dynamic offset), its 128 ids (dynamic
   offset into the lists), gather the 128 token rows (`vector_offset` DMA, software DGE) from a token-major copy of
   x, 16 PE transposes to feature-major, gate/up (weights stationary in 128-wide windows, tokens moving), silu * up,
   down token-major (act stationary [96, 128 tok], down moving [96, 512]), write the block's 128 rows (bf16).
   DMAs of block b + 3 and the transposes / gate-up of block b + 1 are issued before block b's down projection.
5. Combine: `y[t] = sum_k w[t, k] * y_slot[slot(t, k)]` (8 gathers per token tile, routing weights applied here in
   fp32), sum of the two physical cores (shared `y_part` + `core_barrier`), HBM `all_reduce` over the ranks.

### Correctness (TP=4, cores 0-3, layers 0 / 24 / 47, every token of the first S of the 4,096-token text)
| S | top-8 set = HF's | the differing tokens | error on the others (max err / max abs MoE output) |
|---|---|---|---|
| 256 | 255, 252, 255 / 256 | HF 8th-9th router logit gap <= 2.7e-3 (median gap 0.05-0.14) | 7e-4, 2.3e-3, 1.9e-3 |
| 1,024 | 1021, 1013, 1023 / 1024 | gap <= 3.1e-3 | 7e-4, 1.6e-3, 2.3e-3 |
| 4,096 | 4071, 4059, 4081 / 4096 | gap <= 4.3e-3 | 7e-4, 1.9e-3, 1.5e-3 |
Identical results on all 4 ranks. The routing differences are near-ties flipped by the bf16 rounding of the
normalised activations (the same effect as in decode, section 1).

### Measured (TP=4, cores 0-3, queued, ms per MoE half-layer; x 48 for the model)
| version | S = 256 | 1,024 | 4,096 |
|---|---|---|---|
| v1: (token, weight) scattered to slots with per-partition-offset DMAs, fp32 slot outputs | 1.89 | 2.78 | 6.64 |
| **v6: compaction (no scatter), empty blocks skipped, bf16 slot outputs, pipelined block DMAs** | **1.47** | **2.16** | **4.84** |

Rank 0's shard alone (W=1 build, core 5) at S = 1,024 through the versions: 2.65 (v1) -> 2.65 (128-wide gate/up
windows) -> 2.69 (block DMAs pipelined) -> 2.50 (compaction) -> 2.53 (compute pipelined) -> **2.01** (bf16 slot
outputs + skipped empty blocks) -> 1.98 (prefetch depth 4).

Where the time goes (profile of the v6 shard, S = 1,024): dispatch (router .. compaction) ~250 us, 192 blocks in
1.37 ms (7.1 us per block), combine 0.24 ms. In the block loop no engine is saturated (TensorE 66%, ScalarE 40%,
VectorE 33%, sync 40% busy); each block evacuates 8 PSUM tiles of [128, 512] (4 for the transposed x, 4 for the
output) at 0.7-1.0 us each on ScalarE / VectorE (GpSimd cannot read PSUM: `tensor_copy src must be in [sbuf], got
psum`).

PSUM evacuation cost (`probes/p5_evac.py`, `probes/bench_p5.py`, one physical core, profiled instruction durations):
VectorE fp32 PSUM [128, 512] -> bf16 SBUF 680 ns, ScalarE (`activation(copy)`) 752 ns, VectorE [128, 2048] (4 PSUM
banks) in one instruction 2,282 ns (vs 4 x 680 = 2,720): ~0.9 elements per ns per partition. A block evacuates
2 x 2,048 elements per partition (the transposed x and the output), ~4.6 us of VectorE + ScalarE time against
~7 us per block measured.

What did not work (S = 1,024 shard): feature-major gather on the chip with `nisa.nc_n_gather` from an SBUF-resident
x^T (one gather over HT x S elements fails ISA validation, `PoolBufferLoad ... 'valid_pool_buffer_load_element_count'`;
16 gathers of S elements per block work but take 6.70 ms in all); gather + transpose by the DMA
(`nisa.dma_transpose` with `vector_offset`, software DGE, 16 per block): 5.26 ms. PE transposes after a row gather
(2.0 ms) stay.

### The whole model: layered TP prefill + decode, end to end from C++
Kernel `kernels/prefill_tp.py` (`pf_tp`, three stages = three NEFFs per bucket S: emb, layer, head), driver
`host/tpgen.cpp`, test `python tests/test_prefill_tp.py <W> <S> [G]` (env `TAG` = HF capture whose prompt is fed,
`DEC` = decode NEFF, `CORES`, `REPS`). Per layer: lab 1's prefill attention (norm, QKV with this virtual core's
heads, q/k norm, RoPE, KV cache rows written in the decode layout, nkilib `attention_cte`, o_proj K-split) with an HBM
all-reduce of the o_proj partial over the ranks, then the MoE half-layer above on RMSNorm(h), its result transposed
back (fp32 `nc_transpose`, bit-accurate on trn2) and added to h. The layer NEFF runs 48 times on per-layer slices of
the decode step's tensors (`nrt_tensor_allocate_slice`; the 4 expert-weight layer groups are sliced as group l / 12,
row l % 12), the head writes the first token and `pos = plen`, then the decode chain continues on the same tensors.
One scheduling thread per rank queues all 50 prefill executions and the decode steps.

| TP=4 (cores 0-3 / 4-7) | prompt | prefill (all 50 executions queued, min of the warm reps) | decode after it | tokens vs HF fp32 greedy | NxD Inference TP=4 prefill |
|---|---|---|---|---|---|
| S = 256 | 12 tokens | **68.9 ms** | 4.96 ms/step (b512) | 24/24 identical | 336.7 ms |
| S = 1,024 | 1,000 tokens | **113.6 ms** | 5.73 ms/step (b4608) | 16/16 identical | 445.3 ms |
| S = 4,096 | 4,000 tokens | **319.2 ms** | 5.74-5.88 ms/step (b4608) | 8/8 identical | 688.4 ms |
| TP=8 (cores 0-7), S = 256 | 12 tokens | **58.6 ms** | 4.26 ms/step (b512) | 24/24 identical | 396.2 ms (TP=8) |
| TP=8, S = 1,024 | 1,000 tokens | **104.3 ms** | 5.09 ms/step (b4608) | 16/16 identical | 506.5 ms (TP=8) |
| TP=8, S = 4,096 | 4,000 tokens | **294.5 ms** | 5.06 ms/step (b4608) | 8/8 identical | 717.7 ms (TP=8) |
Per layer (TP=4): 1.43 / 2.37 / 6.65 ms; the MoE half-layer kernel alone (its own NEFF, with its input / output
copies) takes 1.47 / 2.16 / 4.84 ms: the MoE dispatch and block loop, not attention, set the prefill. TP=8 gains only 8-15% (the per-block latency does not shrink
with W, and the [S, 2048] fp32 all-reduces cross two devices).
NEFFs: emb 0.02 MB, layer 0.99 MB (S = 256), head 0.20 MB; compile + one run 14 / 39 / 20 s (S = 256). The first
repetition after loading is slower (155 / 294 / 374 ms): warm-up, as lab 2 found for decode.
Decode with the b4608 bucket (`TB=4608 python tests/test_step_moe.py 4`): 5.767 ms/step (11/11, 24/24).

---------------------------------------------------------------------------------------------------------

## 4. The reference: NxD Inference Qwen3-MoE

`tools/ref_nxdi_moe.py <TP>` (venv `~/es-pr2/.venv`, `NEURON_RT_VISIBLE_CORES=12-15` / `8-15`): NxD Inference's
`NeuronQwen3MoeForCausalLM` (`MODEL_TYPES['qwen3_moe']`), `MoENeuronConfig` with the settings of
`enginesmith.references.nxdi` (batch 1, bf16, seq_len 4,608, context and token buckets 128 ... 4,608, on-device greedy
sampling, async mode, fused QKV), driven like the harness reference (whole prompt through context encoding, then one
token-generation call per token). Prefill = one context-encoding call (min of 3), decode = mean over 128 tokens after
the prompt (min of 3); prompts are pseudo-random token ids of the given length.

| TP (cores) | compile | load | prefill 256 / 1,024 / 4,096 | decode ms/token after 256 / 1,024 / 4,096 | 24 greedy tokens |
|---|---|---|---|---|---|
| 4 (12-15) | 441 s | 78 s | 336.7 / 445.3 / 688.4 ms | 10.24 / 11.52 / 11.37 | = HF fp32 |
| 8 (8-15) | 426 s | 80 s | 396.2 / 506.5 / 717.7 ms | 9.31 / 10.98 / 9.72 | = HF fp32 |

Against ours (sections 2 and 3): decode TP=4 4.97 ms vs 10.2-11.5 (2.1-2.3x faster), TP=8 4.22 vs 9.3-11.0
(2.2-2.6x); prefill TP=4 68.9 / 113.6 / 319.2 vs 336.7 / 445.3 / 688.4 ms (4.9x / 3.9x / 2.2x), TP=8 58.6 / 104.3 /
294.5 vs 396.2 / 506.5 / 717.7 ms (6.8x / 4.9x / 2.4x). Caveats: the reference's times are wall times of its Python
`model(...)` calls (as the harness's reference measures them), ours are queued executions from C++; its prompts are
pseudo-random ids, ours real text (our MoE prefill cost depends a little on how evenly the tokens route: blocks of
unused experts skip their weight DMAs).

What did not work: the default MoE configuration fails while tracing the context-encoding model:
`NotImplementedError: _call_shard_hidden_kernel is not available - kernel not imported from nkilib` (the LNC=2
blockwise path defaults to a shard-on-hidden kernel that this neuronx-distributed build does not ship; log
`build/logs/nxdi_tp4_default_fail.log`). Setting `blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while":
true}` (the shard-on-intermediate nkilib kernel) compiles and runs; these are the numbers above. With it NxD pads the
expert intermediate per rank to a multiple of 256 (`maybe_pad_intermediate`: 768 / 4 = 192 -> 256 at TP=4, 33% more
expert work in context encoding). NxD's fused MoE token-generation kernel (`moe_fused_nki_kernel_enabled`) needs the
per-rank intermediate to be a multiple of 128 (`MOE_TKG_MK_INTERMEDIATE_PER_TP`): 192 (TP=4) and 96 (TP=8) are not,
so the decode numbers above use NxD's non-fused path.

---------------------------------------------------------------------------------------------------------

## Files, commands, reproduction

All in ~/nki-lab3 (`source env.sh`: venv on PATH, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, PYTHONPATH,
`NEURON_RT_DISABLE_EXECUTION_BARRIER=1`, default cores 0-3). One script: `./build.sh [host|refs|quick|moe|decode|prefill|nxdi|all]`.

| file | what |
|---|---|
| `kernels/moe_tp.py` | decode MoE half-layer: `_router` (logits row, max8, match_replace8, weights), `_experts` (gathered slabs, W128 windows, column-major combine; variants `OPT`), `_moe_layer`, the problem-1 kernel `moe_block` |
| `kernels/moe_weights.py` | host re-layout of the experts per virtual core (`il_tiles`, `moe_split`: all experts of a layer in one transpose, 4 layer groups < 4 GiB, memmaps into `/dev/shm`) |
| `kernels/step_moe.py` | the whole decode step (`step_moe`: lab 2's TP attention half-layer `_attn_tp` + `_moe_layer` + LM head / argmax) |
| `kernels/moe_prefill.py` | MoE prefill half-layer (`moe_prefill`: dispatch by compaction, static block schedule, `_issue` / `_front` / `_back`) |
| `kernels/prefill_tp.py` | whole-model layered TP prefill (`pf_tp`, STAGE 1 emb / 2 layer / 3 head) |
| `kernels/step_tp.py`, `layer_decode.py`, `step_decode.py`, `qwen3_weights.py`, `ref_qwen3.py`, `nb.py` | from labs 1 / 2 (argmax tiles widened for TP=16, `nb.py` checks cores 0-15, scratch in `/dev/shm`) |
| `kernels/kbuild.py` | build target for `nb.build`: any kernel, zero arrays of the per-rank shapes, lists per rank for W > 1 |
| `kernels/moe_x.py` | experimental copy of moe_tp (register / software-DGE / padded single-DMA variants, all slower) |
| `host/ccrun.cpp` | any (collective) NEFF on W ranks: parallel upload, `CC_NIN` multi-input mode (one input set per execution), queued benchmark |
| `host/tpchain.cpp` | TP decode chain (lab 2) |
| `host/tpgen.cpp` | TP layered prefill (emb, L x layer on slices, head) + decode chain, one thread per rank |
| `tests/hf_ref.py` | HF fp32 CPU reference and per-layer MoE captures (`TAG`, `PROMPT_FILE`, `PROMPT_TOKENS`) |
| `tests/test_moe.py` | problem 1 (`<W> [L] [NTOK]`, `NOCC`, `KW`, `MOD`, `WTAG`) |
| `tests/test_step_moe.py` | problem 2 (`<W> [G] [L]`, `TB`, `SL SLO SLH PF PFO MOEOPT XCH NOCC`) |
| `tests/test_moe_prefill.py`, `tests/test_prefill_tp.py` | problem 3 |
| `tools/ref_nxdi_moe.py` | problem 4 (NxD Inference, `~/es-pr2/.venv`) |
| `tools/placement.py` | TP vs EP bytes on the slowest rank from HF's routing |
| `tools/prof.py`, `tools/tl2.py`, `tools/busy.py`, `tools/timeline.py` | neuron-explorer profile of one NEFF on one core (zero inputs), per-engine runs per source line, union busy time |
| `probes/p1_router_gather.py`, `p2_gather_bw.py`, `p3_pe_rate.py`, `p4_cc_contention.py`, `p5_evac.py` (+ `bench_p*.py`) | router / top-8 / gather check (ties), dynamic DMA rates and the 4 GiB limit, TensorE rates by shape, all-reduce vs concurrent weight DMAs, PSUM evacuation cost |
| `tests/hf_bf16_agree.py` | HF bf16 vs HF fp32 teacher-forced agreement (the reference's own rounding error) |
| `skill-drafts/moe.md` | the backend skill draft |
| `build/logs/*.log` | raw logs of every run quoted here |

Environment for multi-rank runs: `NEURON_RT_VISIBLE_CORES=<contiguous range in 0-15>`,
`NEURON_RT_ROOT_COMM_ID=localhost:<port>` (`LAB3_COMM_ID`; one port per concurrent job), `WTAG` (one scratch
directory per concurrent job: two jobs splitting into the same `/dev/shm` directory corrupted each other once,
`ValueError: mmap length is greater than file size`). `/dev/shm` is RAM: after a few hours of per-variant scratch
directories (16-75 GB each) it was 100% full (1000 GB) and every new Python process dumped core at start; the tests
now delete their scratch directory on exit (`KEEP=1` keeps it).
