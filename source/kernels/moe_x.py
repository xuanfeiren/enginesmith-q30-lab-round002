"""Qwen3-MoE decode MoE block (router + top-8 + gathered expert GEMVs + weighted combine), tensor-parallel over W
logical NeuronCores (LNC=2 each: 2W "virtual cores" g = 2*rank + pid), one token.

Per layer l (column layout, lab 1/2: a vector v[K] is [128, K/128] with v[kt*128 + p] at [p, kt]):
  x      = RMSNorm(h) * ln2                                   (redundant on every core, fp32 -> bf16)
  logits = Wr x  as a ROW [1, E] fp32: x[:, kt] (one column) is the stationary operand, Wr^T tile [128 k, E] moving
  top-8  : nisa.max8 (8 largest, descending) + nisa.nc_match_replace8 (their positions; duplicates of a value are
           paired with distinct positions, lowest positions first => ties at the 8/9 boundary go to the lowest id)
  w_k    = exp(v_k - v_0) / sum_j exp(v_j - v_0)   (= HF: softmax over all E in fp32, top-8, renormalised; the
           softmax denominator cancels in the renormalisation)
  experts: virtual core g owns rows [g*IL, (g+1)*IL) of every expert's gate/up (IL = 768 / 2W) and the same columns of
           down. For slot k the core DMAs its slab of expert idx[k] by a dynamic scalar offset read from SBUF
           (wgu[l, :, pid].select(0, idx[k]) on a hardware DGE queue of the sync engine), then
           g_k, u_k = slab^T x (weight-stationary, [ILP, 1] columns), a_k = silu(g_k) * u_k * w_k (fp32 -> bf16),
           partial[:, nt] = sum_k sum_it Wd_k(it, nt)^T a_k(it)   (the weighted combine is the PSUM accumulation)
  h     += all-reduce over all 2W virtual cores of partial (sendrecv between the two physical cores, then one HBM
           all_reduce over the ranks: lab 2's _xchg_add_tp_hbm)

Per-rank weight tensors (host: kernels/moe_weights.py, values unchanged, layout only); wgu / wd are passed as
NG = 4 layer groups wgu0..3 / wd0..3 of LG = ceil(L / 4) layers each (each tensor < 4 GiB, see _moe_layer):
  wr  [L, 128, HT*E]                         wr[l, p, kt*E + e] = Wr[e, kt*128 + p]
  wgu [L, E, 2, 128, HT*2*ILT*ILP]           [.., p, ((kt*2 + gu)*ILT + it)*ILP + j] = (gate|up)[g*IL + it*ILP + j, kt*128 + p]
  wd  [L, E, 2, ILP, ILT*HT*128]             [.., i, (it*HT + nt)*128 + j] = down[nt*128 + j, g*IL + it*ILP + i]
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from layer_decode import _rms_col, _dma
from step_tp import _xchg_add_tp, _xchg_add_tp_hbm

TOPK = 8


def _router(x, wr_l, HT, E, hw, qi, RWS=0, wr_sb=None):
    """x [128, HT] bf16 SBUF, wr_l HBM [128, HT*E] -> idx [1, 8] uint32, w8 [1, 8] fp32 (both on partition 0).
    RWS=0: x[:, kt] stationary, Wr^T tile moving -> logits row [1, E] directly (16 matmuls with 128-wide moving);
    RWS=1: Wr^T tile stationary, x[:, kt] moving -> logits column [E, 1], then one PE transpose to the row."""
    if wr_sb is None:
        wr_sb = nl.ndarray((128, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
        _dma(wr_sb, wr_l, qi, hw)
    lg = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    if RWS:
        cps = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=cps, stationary=wr_sb[:, kt * E:(kt + 1) * E], moving=x[:, kt:kt + 1])
        col = nl.ndarray((E, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=col, src=cps)
        lps = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=lps, data=col)
        nisa.tensor_copy(dst=lg, src=lps)
    else:
        lps = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
        for kt in range(HT):
            nisa.nc_matmul(dst=lps, stationary=x[:, kt:kt + 1], moving=wr_sb[:, kt * E:(kt + 1) * E])
        nisa.tensor_copy(dst=lg, src=lps)
    v8 = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=v8, src=lg)
    junk = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((1, TOPK), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=junk, data=lg, vals=v8, imm=-3.0e38, dst_idx=idx)
    nb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb, data=v8[:, 0:1], op0=nl.multiply, operand0=-1.0)
    e8 = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=e8, op=nl.exp, data=v8, bias=nb, scale=1.0, reduce_op=nl.add, reduce_res=s,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    r = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=r, data=s)
    w8 = nl.ndarray((1, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w8, data=e8, op0=nl.multiply, operand0=r)
    return idx, w8, qi + 1


def _experts(x, l, pid, idx, w8, wgu, wd, ones_row, HT, ILT, ILP, hw, qi, REG=0, SWD=0, PAD=0):
    """the 8 selected experts' slices on this virtual core -> partial hidden vector d_ps [128, HT] fp32 PSUM.
    REG: the expert id goes into an explicit register once per expert, both slab DMAs use it as scalar_offset.
    SWD: the down slabs go through software DGE (GpSimd descriptors) instead of the sync engine's hardware DGE."""
    F1 = HT * 2 * ILT * ILP
    F2 = ILT * HT * 128
    gu_ps = nl.ndarray((ILP, 2 * ILT * TOPK), dtype=nl.float32, buffer=nl.psum)
    slabs_d = []
    for k in range(TOPK):
        if PAD:   # one DMA per expert: [128, F1 + F2], down rows zero-padded from ILP to 128 partitions
            ix = idx.ap(pattern=[[TOPK, 1], [1, 1]], offset=k)
            sgd = nl.ndarray((128, F1 + F2), dtype=nl.bfloat16, buffer=nl.sbuf)
            _dma(sgd, wgu[l, :, pid].select(0, ix), qi, hw)
            qi += 1
            slabs_d.append(sgd)
            for gu in range(2):
                for it in range(ILT):
                    c = (gu * ILT + it) * TOPK + k
                    for kt in range(HT):
                        o = ((kt * 2 + gu) * ILT + it) * ILP
                        nisa.nc_matmul(dst=gu_ps[:, c:c + 1], stationary=sgd[:, o:o + ILP], moving=x[:, kt:kt + 1])
            continue
        sg = nl.ndarray((128, F1), dtype=nl.bfloat16, buffer=nl.sbuf)
        sd = nl.ndarray((ILP, F2), dtype=nl.bfloat16, buffer=nl.sbuf)
        if REG:
            reg = nisa.register_alloc()
            nisa.register_load(reg, idx[0:1, k:k + 1])
            src_g = wgu[l, :, pid].ap(pattern=[[F1, 128], [1, F1]], scalar_offset=reg, indirect_dim=0)
            src_d = wd[l, :, pid].ap(pattern=[[F2, ILP], [1, F2]], scalar_offset=reg, indirect_dim=0)
        else:
            ix = idx.ap(pattern=[[TOPK, 1], [1, 1]], offset=k)
            src_g = wgu[l, :, pid].select(0, ix)
            src_d = wd[l, :, pid].select(0, ix)
        _dma(sg, src_g, qi, hw)
        if SWD:
            nisa.dma_copy(dst=sd, src=src_d, dge_mode=nisa.dge_mode.swdge)
        else:
            _dma(sd, src_d, qi + 1, hw)
        qi += 2
        slabs_d.append(sd)
        for gu in range(2):
            for it in range(ILT):
                c = (gu * ILT + it) * TOPK + k
                for kt in range(HT):
                    o = ((kt * 2 + gu) * ILT + it) * ILP
                    nisa.nc_matmul(dst=gu_ps[:, c:c + 1], stationary=sg[:, o:o + ILP], moving=x[:, kt:kt + 1])
    # routing weights broadcast to all partitions: ones[1, 128]^T @ w8[1, 8]
    wb = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=wb, stationary=ones_row, moving=w8)
    wbs = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wbs, src=wb)
    G = ILT * TOPK
    sgm = nl.ndarray((ILP, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sgm, op=nl.silu, data=gu_ps[:, 0:G])
    gup = nl.ndarray((ILP, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gup, data1=sgm, data2=gu_ps[:, G:2 * G], op=nl.multiply)
    act = nl.ndarray((ILP, G), dtype=nl.bfloat16, buffer=nl.sbuf)
    for it in range(ILT):
        nisa.tensor_tensor(dst=act[:, it * TOPK:(it + 1) * TOPK], data1=gup[:, it * TOPK:(it + 1) * TOPK],
                           data2=wbs[0:ILP, :], op=nl.multiply)
    d_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    for nt in range(HT):
        for k in range(TOPK):
            for it in range(ILT):
                o = (it * HT + nt) * 128
                if PAD:
                    o += F1
                nisa.nc_matmul(dst=d_ps[:, nt:nt + 1], stationary=slabs_d[k][0:ILP, o:o + 128],
                               moving=act[:, it * TOPK + k:it * TOPK + k + 1])
    return d_ps, qi


def _experts_pipe(x, l, pid, idx, w8, wgu, wd, ones_row, HT, ILT, ILP, hw, qi, DQ=1):
    """OPT & 1: every expert finishes as soon as its slabs are in: g/u GEMV, a = silu(g) * u (fp32 -> bf16), its own
    down GEMV into its own PSUM tile, then acc += w_k * down_k in fp32 (HF's order: weight applied to the expert
    output). All 16 slab DMAs are issued first; DQ=1 puts the down slabs on the scalar engine's hardware DGE queue
    so two sequencers issue the descriptors in parallel (sync: gate/up, scalar: down)."""
    F1 = HT * 2 * ILT * ILP
    F2 = ILT * HT * 128
    sgs, sds = [], []
    for k in range(TOPK):
        ix = idx.ap(pattern=[[TOPK, 1], [1, 1]], offset=k)
        sg = nl.ndarray((128, F1), dtype=nl.bfloat16, buffer=nl.sbuf)
        _dma(sg, wgu[l, :, pid].select(0, ix), 0, hw)
        sd = nl.ndarray((ILP, F2), dtype=nl.bfloat16, buffer=nl.sbuf)
        if DQ:
            nisa.dma_copy(dst=sd, src=wd[l, :, pid].select(0, ix), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        else:
            _dma(sd, wd[l, :, pid].select(0, ix), 0, hw)
        sgs.append(sg)
        sds.append(sd)
    wb = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=wb, stationary=ones_row, moving=w8)
    wbs = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wbs, src=wb)
    acc = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    for k in range(TOPK):
        gu_ps = nl.ndarray((ILP, 2 * ILT), dtype=nl.float32, buffer=nl.psum)
        for gu in range(2):
            for it in range(ILT):
                c = gu * ILT + it
                for kt in range(HT):
                    o = ((kt * 2 + gu) * ILT + it) * ILP
                    nisa.nc_matmul(dst=gu_ps[:, c:c + 1], stationary=sgs[k][:, o:o + ILP], moving=x[:, kt:kt + 1])
        sgm = nl.ndarray((ILP, ILT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sgm, op=nl.silu, data=gu_ps[:, 0:ILT])
        act = nl.ndarray((ILP, ILT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=act, data1=sgm, data2=gu_ps[:, ILT:2 * ILT], op=nl.multiply)
        d_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
        for nt in range(HT):
            for it in range(ILT):
                o = (it * HT + nt) * 128
                nisa.nc_matmul(dst=d_ps[:, nt:nt + 1], stationary=sds[k][:, o:o + 128], moving=act[:, it:it + 1])
        if k == 0:
            nisa.tensor_scalar(dst=acc, data=d_ps, op0=nl.multiply, operand0=wbs[:, 0:1])
        else:
            nisa.scalar_tensor_tensor(dst=acc, data=d_ps, op0=nl.multiply, operand0=wbs[:, k:k + 1], op1=nl.add,
                                      operand1=acc)
    return acc, qi


def _moe_layer(h, l, pid, w2, wr, wgus, wds, ones, ones_row, HT, E, ILT, ILP, hw, rg, W, hbuf, idx_out=None, w_out=None,
               OPT=0, wr_sb=None):
    """h [128, HT] fp32 residual -> h + MoE(RMSNorm(h) * w2). Optionally writes idx / w of the layer to HBM rows.
    wgus / wds: the expert weights as NG layer groups (LG layers each). A dynamically indexed DMA addresses its tensor
    with 32-bit byte offsets: beyond 4 GiB it reads wrong data or faults (FINDINGS 1), so no tensor may exceed it."""
    LG = wgus[0].shape[0]
    wgu, wd, lo = wgus[l // LG], wds[l // LG], l % LG
    qi = 0
    x = _rms_col(h, w2, HT, HT * 128, ones, nl.bfloat16)
    idx, w8, qi = _router(x, wr[l], HT, E, hw, qi, RWS=OPT & 4, wr_sb=wr_sb)
    if idx_out is not None:
        ii = nl.ndarray((1, TOPK), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ii, src=idx)
        nisa.dma_copy(dst=idx_out[l:l + 1, :], src=ii)
        nisa.dma_copy(dst=w_out[l:l + 1, :], src=w8)
    if OPT & 1:
        d_ps, qi = _experts_pipe(x, lo, pid, idx, w8, wgu, wd, ones_row, HT, ILT, ILP, hw, qi, DQ=OPT & 2)
    else:
        d_ps, qi = _experts(x, lo, pid, idx, w8, wgu, wd, ones_row, HT, ILT, ILP, hw, qi, REG=OPT & 8, SWD=OPT & 16, PAD=OPT & 32)
    if hbuf is None:
        d = _xchg_add_tp(d_ps, pid, rg, W)
    else:
        d = _xchg_add_tp_hbm(d_ps, pid, rg, W, hbuf[0], hbuf[1])
    h2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=d, op=nl.add)
    return h2


def _moe_consts():
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ones_row = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)
    return ones, ones_row


@nki.jit
def moe_block(hin, ln2, wr, wgu0, wgu1, wgu2, wgu3, wd0, wd1, wd2, wd3, ILT, ILP, W, rg, CHAIN=0, HBMCC=1, hw=2, OPT=0,
              WRPF=0):
    """Problem 1 kernel. hin [L, 128, HT] fp32: CHAIN=0: layer l's MoE half-layer applied to its own input hin[l]
    (per-layer check against HF on real hidden states); CHAIN=1: h_{l+1} = h_l + MoE_l(h_l) starting from hin[0]
    (the timing of 48 dependent MoE layers). Outputs hout [L, 128, HT] fp32, idx [L, 8] int32, w [L, 8] fp32."""
    L = hin.shape[0]
    HT = hin.shape[2]
    E = wgu0.shape[1]
    wgus, wds = (wgu0, wgu1, wgu2, wgu3), (wd0, wd1, wd2, wd3)
    pid = nl.program_id(0)
    ones, ones_row = _moe_consts()
    hbuf = None
    if HBMCC and W > 1:
        hb = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_m")
        hr = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_m")
        hbuf = (hb, hr)
    hout = nl.ndarray((L, 128, HT), dtype=nl.float32, buffer=nl.shared_hbm)
    idx_o = nl.ndarray((L, TOPK), dtype=nl.int32, buffer=nl.shared_hbm)
    w_o = nl.ndarray((L, TOPK), dtype=nl.float32, buffer=nl.shared_hbm)
    h = None
    wr_next = None
    for l in range(L):
        if CHAIN == 0 or l == 0:
            h = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=h, src=hin[l])
        w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w2, src=ln2[l])
        wr_sb = wr_next
        if WRPF and l + 1 < L:   # next layer's router weights: issued now, before this layer's expert DMAs
            if wr_sb is None:
                wr_sb = nl.ndarray((128, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
                _dma(wr_sb, wr[l], 0, hw)
            wr_next = nl.ndarray((128, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
            _dma(wr_next, wr[l + 1], 0, hw)
        h = _moe_layer(h, l, pid, w2, wr, wgus, wds, ones, ones_row, HT, E, ILT, ILP, hw, rg, W, hbuf, idx_o, w_o,
                       OPT=OPT, wr_sb=wr_sb)
        nisa.dma_copy(dst=hout[l], src=h)
    return hout, idx_o, w_o
