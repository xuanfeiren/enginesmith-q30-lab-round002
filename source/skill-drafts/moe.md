# Mixture-of-experts models on Trainium2: decode and prefill with token-dependent expert weights

Purpose: how to run a mixture-of-experts (MoE) model (router, top-k, a few active experts out of many per token)
as NKI kernels driven from C++ on libnrt: the router and top-k on the device, fetching only the selected experts'
weights by indices computed on the device, the expert placement over logical NeuronCores, the prefill dispatch with
static shapes, the measured costs and the pitfalls. Measured by kernel lab 3 on one trn2.48xlarge (nki 0.6.0,
neuronx-cc 2.27.5334, Neuron runtime 2.34.10, logical NeuronCore config 2, 2026-11-04) on Qwen3-30B-A3B (bf16,
48 layers, 128 experts, 8 active, moe_intermediate 768, 61 GB) with its own C++ drivers and a CPU reference (HF
transformers fp32). Not run through an EngineSmith evaluation yet. Public NKI API only. Builds on
`tensor-parallel.md` (collectives, SPMD load, one thread per rank) and `kernels-fusion.md` (one NEFF per step).

## Result

| Qwen3-30B-A3B, bf16, b512 attention bucket, one NEFF per step, queued | ms/step | reference (NxD Inference, same TP) |
|---|---|---|
| decode, TP=4 (one device, the fewest power-of-two cores that hold it) | 4.97 | 10.2-11.5 ms/token |
| decode, TP=8 (two devices) | 4.22 | 9.3-11.0 ms/token |
| decode, TP=16 (four devices) | 4.61 (slower than TP=8: collectives over 4 devices) | - |
| whole prefill (48 layered executions + emb + head), TP=4, S = 256 / 1,024 / 4,096 | 68.9 / 113.6 / 319.2 ms | 337 / 445 / 688 ms |
| whole prefill, TP=8, S = 256 / 1,024 / 4,096 | 58.6 / 104.3 / 294.5 ms | 396 / 507 / 718 ms |
| MoE prefill half-layer alone, TP=4, S = 256 / 1,024 / 4,096 | 1.47 / 2.16 / 4.84 ms per layer | - |

Greedy tokens identical to HF fp32 (decode: 11/11 teacher-forced, 24/24 generated; prefill + decode: 24/24, 16/16,
8/8 for 12-, 1,000- and 4,000-token prompts). On a 470-token prompt fed through the decode chain, 460/469 teacher-forced
argmaxes equal HF fp32's (mismatches at near-ties, median logit gap 0.03), where HF's own bf16 model gets 445/469:
judge an MoE engine against the reference's bf16 error, not against exact agreement. The top-8 expert set equals HF's for
99.3-99.7% of (token, layer) pairs; every difference is a near-tie at the 8/9 boundary (HF's own 8th-9th router
logit gap <= 4.3e-3 against a median of ~0.05) caused by the bf16 rounding of the router input.

## The decode MoE half-layer, all on the device

```python
lps = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)        # router logits as a ROW on partition 0
for kt in range(HT):                                               # x column stationary, Wr^T tile [128 k, E] moving
    nisa.nc_matmul(dst=lps, stationary=x[:, kt:kt + 1], moving=wr_sb[:, kt * E:(kt + 1) * E])
nisa.max8(dst=v8, src=lg)                                          # 8 largest, descending
nisa.nc_match_replace8(dst=junk, data=lg, vals=v8, imm=-3e38, dst_idx=idx)   # their positions (uint32 [1, 8])
# w_k = exp(v_k - v_0) / sum_j exp(v_j - v_0): ScalarE exp with fused sum, reciprocal, multiply
for k in range(8):
    ix = idx.ap(pattern=[[8, 1], [1, 1]], offset=k)                # an SBUF element as the scalar offset
    nisa.dma_copy(dst=slab, src=wgu[l, :, pid].select(0, ix),      # this core's slice of expert idx[k]
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
```
- `w_k` equals HF's rule (softmax over all experts in fp32, top-k, renormalise when `norm_topk_prob`): the softmax
  denominator cancels. Weights within 6e-7 of an fp32 implementation of HF's rule on the same inputs.
- Use `nc_match_replace8` for the indices, not `nc_find_index8`: the latter returns the first occurrence of each
  value, so two experts with equal logits map to one index. `nc_match_replace8` pairs duplicates with distinct
  positions, lowest first, so a tie at the k/k+1 boundary goes to the lower expert id. `torch.topk` on the CPU (the
  HF reference) does not prefer the lowest index (10 tied experts: ours keeps the 8 lowest ids, torch dropped ids 3
  and 5); an exact tie at
  the boundary did not occur with real activations.
- Dynamic-offset DMAs stream as fast as static ones: 826 GB/s per logical core (413 per physical core, the two do not
  slow each other) for 1.18 MB slabs with 9 KB partition rows, 742-761 GB/s when every tile is also consumed by a
  matmul; 4 KB rows 390 and 2 KB rows 198 GB/s per physical core. Measure microbenchmarks with the `[2]` grid: a
  kernel called without it compiles for one physical core (`--lnc 1`), and counting both cores' bytes doubles the rate.
- Fold the combine into the down projection: accumulate the 8 experts' down GEMVs into one PSUM tile with the routing
  weight applied to the activations, or keep one PSUM tile per expert and accumulate `w_k * down_k` in fp32.

## Placement: tensor parallel inside every expert, not expert parallel

Every one of the 2W virtual cores (lab 2's split) owns IM/2W rows of every expert's gate/up and the same columns of
down (TP=4: 96 rows; TP=8: 48). Measured on HF's real routing decisions (1,728 token-layer pairs): pure TP streams
exactly 8/W experts per rank; any expert-parallel split makes the slowest rank stream more: EP=2 x TP=W/2 1.25x, EP=4
1.71x, EP=8 2.49x, EP=16 3.9x the pure-TP bytes (p99 worse). Batch-1 decode is set by the slowest rank's bytes, and
EP also needs a variable expert count per rank in a static NEFF. TP costs one all-reduce per MoE layer.

## Costs and floors (TP=4, Qwen3-30B-A3B)

- Bytes per rank per token: experts 906 MB + router 50 MB (each physical core reads the whole router) + attention
  453 MB + LM head 156 MB + KV 25 MB = 1.59 GB; floor at 743 GB/s ~2.1 ms (estimate). Measured 4.97 ms: 1.22 ms of it
  is 98 all-reduces (12.4 us each, the in-isolation latency), the rest is the per-layer dependency chain.
- The MoE chain cannot be prefetched: the next experts are known only after the router. Per layer (shard alone):
  RMSNorm 2.3 us, router 1.9 us, top-8 + weights 1.8 us, first dynamic DMA 1.3 us, then ~24-37 us of expert slabs at
  ~300 GB/s per physical core (all 16 DMA engines busy, 4-6 KB descriptors at 17-19 GB/s each), down tail 4 us.
- **Weight DMAs in flight slow the all-reduces.** 33 dependent 4-rank HBM all-reduces: 577 us alone, 683 us with
  128 MB per physical core of independent weight DMAs streaming next to them (any queue: sync 683, scalar 680,
  GpSimd 727), where the stream alone takes 235 us. So prefetching the next layer's attention weights during the MoE
  or bigger weight DMAs made the shard (one rank alone) up to 12% faster (3.76 -> 3.30 ms) and the full TP=4 step no
  faster or 3% slower (4.98 -> 5.00-5.10): the communication share grew by the same amount. Measure on the
  multi-rank step, never on the shard alone.
- Each DMA costs its issuing engine ~0.6-0.8 us (`DMA_DIRECT2D`), plus ~0.5 us for a dynamic offset (index load and
  address arithmetic), about the same for 64 KB as for 1 MB. At TP=8 the sync engine was busy 74% of the step issuing
  44 DMAs per layer; fewer, bigger attention-weight DMAs took the step from 4.30 to 4.22 ms.

## Prefill: grouped expert GEMMs with static shapes

1. Router per 128-token tile, `max8` / `nc_match_replace8` per partition (= token), one-hot `M [128 tok, E]`.
2. Ranks inside each expert by matmuls with a strictly lower-triangular ones matrix plus `ones^T M` of earlier
   tiles (exact small integers); blocks of 128 slots per expert, `ceil` as `sum_m (cnt > 128 m)` (no `mod` in
   `tensor_scalar`).
3. Per-expert token lists by stream compaction: `nisa.nonzero_with_count` on M^T (it only reads partitions 0, 16, ...,
   112: 16 rounds with a 0/1 matmul moving 8 experts at a time to those partitions). Much cheaper than scattering
   (token, weight) pairs with per-partition-offset DMAs (0.7 ms at S = 1,024 for 128 scatters).
4. NMAX = 8S/128 + E blocks (static worst case); blocks past the last real one get expert id E and their slab DMAs
   are skipped with `oob_mode=nisa.oob_mode.skip`. Per block: slabs by dynamic offset, 128 token rows by a
   `vector_offset` gather, 16 PE transposes, gate/up with weights stationary, down token-major (act stationary
   [ILP, 128 tok]), bf16 rows to a slot buffer. Issue the next blocks' DMAs before this block's compute.
5. Combine with 8 gathers per token tile, routing weights applied in fp32, then the cores and ranks.

## Whole-model prefill: layered, on the decode step's tensors

One layer NEFF (attention half from `prefill.md`, MoE half above) executed 48 times with every per-layer tensor bound
to a slice of the decode step's stacked tensor (`nrt_tensor_allocate_slice`; layer-grouped expert tensors: group
l / LG, row l % LG), plus an embedding NEFF and a head NEFF that writes the first token and `pos = plen`; then the
decode chain continues on the same tensors (KV cache in the decode layout). All executions of a rank queued from its
own thread. The o_proj partial and the MoE output each need one HBM all-reduce of the whole [S, H] fp32 activation
over the ranks (each physical core writes its half of the hidden tiles / tokens into the internal buffer, then
`core_barrier`, then the collective: half-written internal `shared_hbm` buffers worked here).

## Other MoE models (not measured; what changes)

- top-k < 8: use the first k of `max8` / `nc_match_replace8`. top-k > 8: `nc_match_replace8` already replaces the
  found values with `imm` in its output, so run `max8` + `nc_match_replace8` again on that output for the next 8.
- `norm_topk_prob = false` (e.g. Qwen2-MoE): the weights are the softmax probabilities themselves, so the full
  denominator is needed: one more ScalarE `exp` over all E logits with the fused sum.
- Shared experts: a dense MLP of the same token, added before the all-reduce (same TP split as the routed experts).
- Expert count E > 128 or a different intermediate size: the per-virtual-core slab shapes change (`il_tiles`); keep
  every dynamically indexed tensor below 4 GiB (layer groups) and the stationary tiles 128 wide.

## Pitfalls (measured)

- **A dynamically indexed DMA addresses at most 4 GiB of its tensor.** Gathering from a 4,076,863,488-byte tensor was
  exact; from 4,303,355,904 bytes it returned wrong data with no error; a 9.66 GB tensor faulted:
  `scatter/gather (indirect memory copy via scalar DGE) out-of-bound access`, `NRT_EXEC_OOB`. Split expert weights
  into tensors below 4 GiB (layer groups).
- **TensorE stationary width**: a `[128, 96]` stationary tile with a one-column moving operand costs 80 ns, `[128,
  64]` 53 ns, `[128, 128]` 34 ns (at any offset). Use 128-wide stationary windows even when only 96 columns are
  needed (the extra outputs land on unused partitions). It did not change the decode step (DMA-bound there).
- Interleaving the accumulation groups of different columns of one PSUM tile (expert-major down projection with
  explicit `accumulate=(k > 0)` flags) gave **wrong results** (0/24 tokens). Keep each column's accumulation group
  contiguous (column-major over experts), or one PSUM tile per expert group.
- An exact top-8 tie across the boundary is resolved differently from HF: 10 experts with identical router rows ->
  ours picks the 8 lowest ids, `torch.topk` (CPU) dropped ids 3 and 5. Real activations never tie exactly; the
  0.3-0.9% of differing sets are near-ties flipped by bf16 rounding of the router input.
- `nisa.nc_n_gather` (one gather over 16 x S elements per partition) fails ISA validation
  (`valid_pool_buffer_load_element_count`); 16 smaller gathers per block work but were 3x slower than DMA gather +
  PE transposes. `nisa.dma_transpose` with `vector_offset` (gather + transpose by the DMA): 2.6x slower.
- GpSimd cannot read PSUM; PSUM evacuation of `[128, 512]` tiles costs 0.7-1.0 us on ScalarE or VectorE: in prefill
  it, not TensorE, limits the block rate.
- Write big weight files to `/dev/shm` (the root disk wrote at ~130 MB/s: 525 s for 58 GB vs 12-24 s).
