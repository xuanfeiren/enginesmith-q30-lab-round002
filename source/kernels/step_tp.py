"""Tensor-parallel Qwen3 decode step: one NEFF, executed SPMD on W logical NeuronCores (LNC=2 each), one execution
per step per core. Derived from nki-lab/kernels/step_g.py (lab 1's single-logical-core step).

Split: 2W "virtual cores" g = 2*rank + pid (pid = physical core of the logical core). Every weight is split over
the 2W virtual cores exactly as lab 1 split it over 2:
  QKV     N-split by heads (NQL q heads, NKL kv heads per virtual core); attention local, local KV cache slice
  o_proj  K-split (the core's heads) -> partial h [128, HT] fp32
  gate/up N-split (ITL intermediate tiles per core); down K-split (same tiles) -> partial h
  LM head N-split over vocab (NVL tiles per core), local argmax with ties to the lowest id.
Reductions: partial h -> nki.collectives.all_reduce (SBUF -> SBUF, add) over the W ranks, done by each physical
core for its own partial (probe p2: an SBUF all_reduce under LNC=2 reduces each physical core's buffer with the same
physical core of the other ranks), then lab 1's nisa.sendrecv exchange + add between the two physical cores.
=> 2 all_reduces + 2 sendrecvs per layer. Argmax: (max, BIGC - id) combined over the 2 physical cores (sendrecv) and
then over the ranks with two all_reduce(max) of [1, 1] values.
Embedding, norms, residual, token selection run redundantly on every core.

Per-rank inputs (host: tests/test_tp.py split_tp): wqkv [L, 2, 128, (NQL+2NKL)*HT*128], wo [L, 2, 128, HT*NQL*128],
wgu [L, 2, 128, 2*ITL*HT*128], wd [L, 2, 128, HT*ITL*128], wlm [2, 128, NVL*HT*128], kcache [L, 2, NKL, 128, TMAX],
vcache [L, 2, TMAX, NKL*128], rkb [128, 1] fp32 = rank * 2 * NVL * 128 (vocab id base of the rank).
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from layer_decode import _consts, _gemv, _rms_col, _dma, D, EPS, NEG, SCALE
from step_decode import _token_in, BIGC

VOCAB = 151936


def _embed_g(embed, t_sb, HT):
    """embedding row of token t (register) -> column layout h [128, HT] fp32 (lab 1 step_g._embed_g)"""
    treg = nisa.register_alloc()
    nisa.register_load(treg, t_sb)
    eb = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=eb, src=embed.ap(pattern=[[1, 128], [128, HT]], offset=0, scalar_offset=treg, indirect_dim=0))
    h = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=h, src=eb)
    return h


def _xchg_add_tp(part, pid, rg, W):
    """part [128, n] fp32 on each physical core of each rank -> sum over all 2W cores (on every core)."""
    n = part.shape[1]
    mine = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mine, src=part)
    red = mine
    if W > 1:
        red = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_reduce(srcs=[mine], dsts=[red], op=nl.add, replica_group=rg)
    other = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=red, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    s = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s, data1=red, data2=other, op=nl.add)
    return s


def _prefetch(W, KT, SL, n_slabs, hw, qi):
    """issue the DMAs of the first n_slabs weight slabs (SL n-tiles each) of W [128, NT*KT*128] into SBUF now"""
    tiles = []
    for i in range(n_slabs):
        wt = nl.ndarray((128, SL * KT * 128), dtype=W.dtype, buffer=nl.sbuf)
        _dma(wt, W[:, i * SL * KT * 128:(i + 1) * SL * KT * 128], qi, hw)
        qi += 1
        tiles.append(wt)
    return tiles, qi


def _gemv_pf(xT, W, KT, NT, ps, SL, hw, qi, pre):
    """ps[:, nt] = W_tile(nt, :)^T @ xT for nt in [0, NT); the first len(pre) slabs come from pre-loaded tiles"""
    for i in range(len(pre)):
        for t in range(SL):
            nt = i * SL + t
            for k in range(KT):
                nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=pre[i][:, (t * KT + k) * 128:(t * KT + k + 1) * 128],
                               moving=xT[:, k:k + 1])
    return _gemv(xT, W, KT, len(pre) * SL, NT, ps, len(pre) * SL, SL, hw, qi)


def _xchg_add_tp_hbm(part, pid, rg, W, hb, hr):
    """HBM variant: sum the two physical cores first (sendrecv), write the logical core's partial to an internal
    HBM buffer hb (both cores write the same bytes), all_reduce hb -> hr in HBM over the ranks (an HBM collective
    runs once per logical core), read hr back on both cores."""
    n = part.shape[1]
    mine = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mine, src=part)
    other = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=mine, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    s2 = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s2, data1=mine, data2=other, op=nl.add)
    nisa.dma_copy(dst=hb, src=s2)
    ncc.all_reduce(srcs=[hb], dsts=[hr], op=nl.add, replica_group=rg)
    s = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=hr)
    return s


def _layer_tp(h, l, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
              preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, HT, NQL, NKL, ITL, rg, W, PF=0, pre_qkv=None, hbuf=None):
    HID = HT * 128
    G = NQL // NKL
    TMAX = vcache.shape[2]
    NT = TB // 128
    NS = NT + 1
    qi = 0
    kall = nl.ndarray((D, NKL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
    vall = nl.ndarray((128, NT, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    for hh in range(NKL):
        _dma(kall[:, hh, :], kcache[l, pid, hh, :, 0:TB], qi, hw)
        qi += 1
    for j in range(NT):
        _dma(vall[:, j, :], vcache[l, pid, j * 128:(j + 1) * 128, :], qi, hw)
        qi += 1
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1, src=ln1[l])
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2, src=ln2[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=knc, src=kn[l])

    x1 = _rms_col(h, w1, HT, HID, ones, nl.bfloat16)
    NQKV = NQL + 2 * NKL
    qkv_ps = nl.ndarray((128, NQKV), dtype=nl.float32, buffer=nl.psum)
    if pre_qkv is None:
        qi = _gemv(x1, wqkv[l, pid], HT, 0, NQKV, qkv_ps, 0, SL, hw, qi)
    else:
        qi = _gemv_pf(x1, wqkv[l, pid], HT, NQKV, qkv_ps, SL, hw, qi, pre_qkv)
    qkv = nl.ndarray((128, NQKV), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkv, src=qkv_ps)
    NQK = NQL + NKL
    sq = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=qkv[:, 0:NQK], data2=qkv[:, 0:NQK], op=nl.multiply)
    ss = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss, stationary=ones, moving=sq)
    rstd = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn, data1=qkv[:, 0:NQK], data2=rstd, op=nl.multiply)
    qkn = nl.ndarray((128, NQK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qkn[:, 0:NQL], data=xn[:, 0:NQL], op0=nl.multiply, operand0=qnc)
    nisa.tensor_scalar(dst=qkn[:, NQL:NQK], data=xn[:, NQL:NQK], op0=nl.multiply, operand0=knc)
    rps = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rps, stationary=rot_sb, moving=qkn)
    xc = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xc, data=qkn, op0=nl.multiply, operand0=cs_col)
    qkT = nl.ndarray((128, NQK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=qkT, data=rps, op0=nl.multiply, operand0=sn_col, op1=nl.add, operand1=xc)
    vT = nl.ndarray((128, NKL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vT, src=qkv[:, NQK:NQKV])
    vr_ps = nl.ndarray((1, NKL * D), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        nisa.nc_matmul(dst=vr_ps[0:1, hh * D:(hh + 1) * D], stationary=vT[:, hh:hh + 1], moving=ident)
    vrow = nl.ndarray((1, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vrow, src=vr_ps)
    nisa.dma_copy(dst=kcache[l, pid].ap(pattern=[[TMAX, D], [D * TMAX, NKL]], offset=0, scalar_offset=preg, indirect_dim=2),
                  src=qkT[:, NQL:NQK])
    nisa.dma_copy(dst=vcache[l, pid].ap(pattern=[[NKL * D, 1], [1, NKL * D]], offset=0, scalar_offset=preg, indirect_dim=0),
                  src=vrow)

    io = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[128, NT]], offset=0, channel_multiplier=1)
    madd = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=madd, data=io, op0=nl.greater_equal, operand0=pb, op1=nl.multiply, operand1=NEG)
    S = nl.ndarray((128, NS, NQL), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(NT):
        ps = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NKL):
            nisa.nc_matmul(dst=ps[:, G * hh:G * hh + G], stationary=kall[:, hh, j * 128:(j + 1) * 128],
                           moving=qkT[:, G * hh:G * hh + G])
        nisa.tensor_scalar(dst=S[:, j, :], data=ps, op0=nl.add, operand0=madd[:, j:j + 1])
    nisa.memset(dst=S[:, NT, :], value=NEG)
    psc = nl.ndarray((1, NQL), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        nisa.nc_matmul(dst=psc[0:1, G * hh:G * hh + G], stationary=qkT[:, NQL + hh:NQL + hh + 1], moving=qkT[:, G * hh:G * hh + G])
    nisa.tensor_copy(dst=S[0:1, NT, :], src=psc)
    pm = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm, op=nl.maximum, data=S.ap(pattern=[[NS * NQL, 128], [1, NQL], [NQL, NS]], offset=0), axis=2)
    pmT = nl.ndarray((NQL, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pmT, data=pm)
    m = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=pmT, axis=1)
    dg = nl.ndarray((NQL, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dg, data=idf[0:NQL, 0:NQL], op0=nl.multiply, operand0=m)
    mb = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=mb, stationary=ones16[0:NQL, :], moving=dg)
    mbs = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mbs, src=mb)
    Sm = nl.ndarray((128, NS, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Sm, data1=S, data2=mbs.ap(pattern=[[NQL, 128], [0, NS], [1, NQL]], offset=0), op=nl.subtract)
    E = nl.ndarray((128, NS, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=E, op=nl.exp, data=Sm, scale=SCALE)
    es = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=es, op=nl.add, data=E.ap(pattern=[[NS * NQL, 128], [1, NQL], [NQL, NS]], offset=0), axis=2)
    den = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=den, stationary=ones, moving=es)
    rden = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    ops_ = nl.ndarray((D, NQL), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        for j in range(NT):
            nisa.nc_matmul(dst=ops_[:, G * hh:G * hh + G], stationary=vall[:, j, hh * D:(hh + 1) * D],
                           moving=E[:, j, G * hh:G * hh + G])
        nisa.nc_matmul(dst=ops_[:, G * hh:G * hh + G], stationary=vrow[0:1, hh * D:(hh + 1) * D],
                       moving=E[0:1, NT, G * hh:G * hh + G])
    oT = nl.ndarray((D, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=oT, data1=ops_, data2=rden, op=nl.multiply)

    y_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(oT, wo[l, pid], NQL, 0, HT, y_ps, 0, SL, hw, qi)
    pre_gu = []
    if PF > 0:   # gate/up slabs stream in while the o_proj all-reduce is in flight
        pre_gu, qi = _prefetch(wgu[l, pid], HT, SL, PF, hw, qi)
    if hbuf is None:
        y = _xchg_add_tp(y_ps, pid, rg, W)
    else:
        y = _xchg_add_tp_hbm(y_ps, pid, rg, W, hbuf[0], hbuf[1])
    h2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=y, op=nl.add)

    x2 = _rms_col(h2, w2, HT, HID, ones, nl.bfloat16)
    gu_ps = nl.ndarray((128, 2 * ITL), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv_pf(x2, wgu[l, pid], HT, 2 * ITL, gu_ps, SL, hw, qi, pre_gu)
    sg = nl.ndarray((128, ITL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=gu_ps[:, 0:ITL])
    act = nl.ndarray((128, ITL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=sg, data2=gu_ps[:, ITL:2 * ITL], op=nl.multiply)
    d_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(act, wd[l, pid], ITL, 0, HT, d_ps, 0, SL, hw, qi)
    nxt = None
    if PF > 0 and l + 1 < wqkv.shape[0]:   # next layer's QKV slabs stream in during the down all-reduce
        nxt, qi = _prefetch(wqkv[l + 1, pid], HT, SL, PF, hw, qi)
    if hbuf is None:
        d = _xchg_add_tp(d_ps, pid, rg, W)
    else:
        d = _xchg_add_tp_hbm(d_ps, pid, rg, W, hbuf[2], hbuf[3])
    h3 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h3, data1=h2, data2=d, op=nl.add)
    return h3, nxt


def _lm_argmax_tp(xf, wlm_c, pid, rkb, SL, hw, HT, NVL, rg, W):
    CH = 8192 // HT     # at most 8,192 accumulating matmuls per PSUM tile (lab 1, p11)
    if CH > 512:
        CH = 512
    lg = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    qi = 0
    c0 = 0
    while c0 < NVL:
        n = CH
        if NVL - c0 < CH:
            n = NVL - c0
        ps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
        qi = _gemv(xf, wlm_c, HT, c0, c0 + n, ps, 0, SL, hw, qi)
        nisa.tensor_copy(dst=lg[:, c0:c0 + n], src=ps[:, 0:n])
        c0 += n
    # global ids: rank base + (pid*NVL + col)*128 + p; padded ids (>= VOCAB) masked out
    gid = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=gid, pattern=[[128, NVL]], offset=0, channel_multiplier=1)
    rb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rb, src=rkb)
    base = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base, data=rb, op0=nl.add, operand0=pid * (NVL * 128))
    nisa.tensor_scalar(dst=gid, data=gid, op0=nl.add, operand0=base)
    msk = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msk, data=gid, op0=nl.greater_equal, operand0=float(VOCAB), op1=nl.multiply, operand1=-3.0e38)
    nisa.tensor_tensor(dst=lg, data1=lg, data2=msk, op=nl.add)
    pm = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm[:, 0:1], op=nl.maximum, data=lg, axis=1)
    nisa.tensor_scalar(dst=gid, data=gid, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    eq = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=pm[:, 0:1])
    nisa.tensor_tensor(dst=eq, data1=eq, data2=gid, op=nl.multiply)
    nisa.tensor_reduce(dst=pm[:, 1:2], op=nl.maximum, data=eq, axis=1)
    rows = nl.ndarray((1, 2, 128), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(2):
        tp = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=pm[:, c:c + 1])
        nisa.tensor_copy(dst=rows[:, c, :], src=tp)
    loc = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)      # (max, BIGC - lowest id) of this core
    nisa.tensor_reduce(dst=loc[:, 0:1], op=nl.maximum, data=rows[:, 0, :], axis=1)
    e2 = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e2, data=rows[:, 0, :], op0=nl.equal, operand0=loc[:, 0:1])
    nisa.tensor_tensor(dst=e2, data1=e2, data2=rows[:, 1, :], op=nl.multiply)
    nisa.tensor_reduce(dst=loc[:, 1:2], op=nl.maximum, data=e2, axis=1)
    oth = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=loc, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    both = nl.ndarray((1, 2, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=both[:, 0, :], src=loc)
    nisa.tensor_copy(dst=both[:, 1, :], src=oth)
    gm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gm, data1=loc[:, 0:1], data2=oth[:, 0:1], op=nl.maximum)
    e3 = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e3, data=both[:, :, 0], op0=nl.equal, operand0=gm)
    nisa.tensor_tensor(dst=e3, data1=e3, data2=both[:, :, 1], op=nl.multiply)
    bs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=bs, op=nl.maximum, data=e3, axis=1)
    if W > 1:
        # over the ranks: M = max_r gm_r; best = max_r (gm_r == M) * bs_r   (exact: equality of identical values)
        # SBUF all_reduce tiles need a free size >= max(2, W) (smaller asserts in nrt_load:
        # "get_enc_tensor_from_collectives_tensor: Assertion `coll_tensor->num_elem[0] > 0 && ...num_elem[1] > 0'"),
        # and a 128-partition tile is faster than a 1-partition one (W=4: 14.9 us for [128, 8] vs 35.7 us for
        # [1, 16]): the value sits in row 0 of a [128, 8] tile whose other rows hold the identity of the op
        CW = 8 if W <= 8 else W     # collective tiles need a free size >= W (lab 2; W=16 asserted in nrt_load)
        gm2 = nl.ndarray((128, CW), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gm2, value=-3.0e38)
        nisa.memset(dst=gm2[0:1, :], value=0.0)
        nisa.tensor_scalar(dst=gm2[0:1, :], data=gm2[0:1, :], op0=nl.add, operand0=gm)
        gM = nl.ndarray((128, CW), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_reduce(srcs=[gm2], dsts=[gM], op=nl.maximum, replica_group=rg)
        e4 = nl.ndarray((128, CW), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=e4, value=0.0)
        nisa.tensor_tensor(dst=e4[0:1, :], data1=gm2[0:1, :], data2=gM[0:1, :], op=nl.equal)
        nisa.tensor_scalar(dst=e4[0:1, :], data=e4[0:1, :], op0=nl.multiply, operand0=bs)
        bsa = nl.ndarray((128, CW), dtype=nl.float32, buffer=nl.sbuf)
        ncc.all_reduce(srcs=[e4], dsts=[bsa], op=nl.maximum, replica_group=rg)
        bs = bsa[0:1, 0:1]
    tokf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tokf, data=bs, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    return tokf


@nki.jit
def step_tp(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
            kcache, vcache, rkb, TB, SL, hw, NQL, NKL, ITL, W, rg, PF=0, HBMCC=0):
    L = wqkv.shape[0]
    HT = embed.shape[1] // 128
    HID = HT * 128
    NVL = wlm.shape[2] // (HT * 128)
    pid = nl.program_id(0)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)
    cs_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sn_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        nisa.dma_copy(dst=cs_col[half * 64:(half + 1) * 64, :],
                      src=cos.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        nisa.dma_copy(dst=sn_col[half * 64:(half + 1) * 64, :],
                      src=sin.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    ones, ident, idf, ones16 = _consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    t_sb = _token_in(pos, plen, prompt, toks, p_sb)
    h = _embed_g(embed, t_sb, HT)
    pre = None
    hbuf = None
    if HBMCC:
        # internal HBM collective buffers (collectives may not touch kernel I/O tensors; nkilib: name= required)
        hb_a = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_a")
        hr_a = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_a")
        hb_m = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_m")
        hr_m = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_m")
        hbuf = (hb_a, hr_a, hb_m, hr_m)
    for l in range(L):
        h, pre = _layer_tp(h, l, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
                           preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, HT, NQL, NKL, ITL, rg, W,
                           PF, pre, hbuf)
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(h, wf, HT, HID, ones, nl.bfloat16)
    tokf = _lm_argmax_tp(xf, wlm[pid], pid, rkb, SL, hw, HT, NVL, rg, W)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok_i)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    return pos_out
