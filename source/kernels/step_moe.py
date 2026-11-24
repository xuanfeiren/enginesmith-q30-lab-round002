"""Whole Qwen3-30B-A3B (Qwen3-MoE) decode step in one NEFF, tensor-parallel over W logical NeuronCores (LNC=2).

embedding -> 48 x {attention half-layer (lab 2's TP attention: QKV N-split by heads, kv heads replicated when there
are fewer than 2W, local KV cache at the device-held position, o_proj K-split + all-reduce), MoE half-layer
(kernels/moe_tp.py: router + top-8 on the device, gathered expert slices, weighted combine, all-reduce)}
-> final RMSNorm -> LM head (vocab N-split) -> argmax (ties to the lowest id) -> toks[pos + 1], pos + 1.
Per-rank inputs: see tests/test_step_moe.py (split_step).
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from layer_decode import _consts, _gemv, _rms_col, _dma, D, EPS, NEG, SCALE
from step_decode import _token_in
from step_tp import _embed_g, _xchg_add_tp, _xchg_add_tp_hbm, _gemv_pf, _prefetch, _lm_argmax_tp
from moe_tp import _moe_layer, _moe_consts, _xchg_add_hbm2


def _attn_tp(h, l, pid, ln1, qn, kn, wqkv, wo, rot_sb, kcache, vcache,
             preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, HT, NQL, NKL, rg, W, pre_qkv=None, hbuf=None,
             SLO=1, pre_o=None, hwq=None, VD=0, ATT=0, KVQ=0):
    """attention half-layer (lab 2 step_tp._layer_tp, unchanged): h -> h + o_proj(attn(RMSNorm(h) * ln1))"""
    HID = HT * 128
    G = NQL // NKL
    TMAX = vcache.shape[2]
    NT = TB // 128
    NS = NT + 1
    qi = 0
    kall = nl.ndarray((D, NKL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
    vall = nl.ndarray((128, NT, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    hkv = 3 if KVQ else hw
    for hh in range(NKL):
        _dma(kall[:, hh, :], kcache[l, pid, hh, :, 0:TB], qi, hkv)
        qi += 1
    if VD == 0:
        for j in range(NT):
            _dma(vall[:, j, :], vcache[l, pid, j * 128:(j + 1) * 128, :], qi, hw)
            qi += 1
    else:   # VD DMAs of NT / VD position tiles each (vall[p, j, :] = vcache[128 j + p, :])
        RW = NKL * D
        j0 = 0
        for c in range(VD):
            j1 = (NT * (c + 1)) // VD
            if j1 > j0:
                _dma(vall[:, j0:j1, :], vcache[l, pid].ap(pattern=[[RW, 128], [128 * RW, j1 - j0], [1, RW]],
                                                         offset=j0 * 128 * RW), qi, hkv)
                qi += 1
            j0 = j1
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1, src=ln1[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=knc, src=kn[l])

    x1 = _rms_col(h, w1, HT, HID, ones, nl.bfloat16)
    NQKV = NQL + 2 * NKL
    qkv_ps = nl.ndarray((128, NQKV), dtype=nl.float32, buffer=nl.psum)
    if pre_qkv is None:
        qi = _gemv(x1, wqkv[l, pid], HT, 0, NQKV, qkv_ps, 0, SL, hw if hwq is None else hwq, qi)
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
    if ATT == 0:
        for j in range(NT):
            ps = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
            for hh in range(NKL):
                nisa.nc_matmul(dst=ps[:, G * hh:G * hh + G], stationary=kall[:, hh, j * 128:(j + 1) * 128],
                               moving=qkT[:, G * hh:G * hh + G])
            nisa.tensor_scalar(dst=S[:, j, :], data=ps, op0=nl.add, operand0=madd[:, j:j + 1])
    else:   # all score tiles in one PSUM tile, the mask added by one instruction (same fp32 adds)
        psa = nl.ndarray((128, NT, NQL), dtype=nl.float32, buffer=nl.psum)
        for j in range(NT):
            for hh in range(NKL):
                nisa.nc_matmul(dst=psa[:, j, G * hh:G * hh + G], stationary=kall[:, hh, j * 128:(j + 1) * 128],
                               moving=qkT[:, G * hh:G * hh + G])
        nisa.tensor_tensor(dst=S[:, 0:NT, :], data1=psa, data2=madd.ap(pattern=[[NT, 128], [1, NT], [0, NQL]], offset=0),
                           op=nl.add)
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
    if pre_o is None:
        qi = _gemv(oT, wo[l, pid], NQL, 0, HT, y_ps, 0, SLO, hw, qi)
    else:
        qi = _gemv_pf(oT, wo[l, pid], NQL, HT, y_ps, SLO, hw, qi, pre_o)
    if hbuf is None:
        y = _xchg_add_tp(y_ps, pid, rg, W)
    else:
        if len(hbuf[0].shape) == 3:
            y = _xchg_add_hbm2(y_ps, pid, rg, W, hbuf[0], hbuf[1])
        else:
            y = _xchg_add_tp_hbm(y_ps, pid, rg, W, hbuf[0], hbuf[1])
    h2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=y, op=nl.add)

    return h2


@nki.jit
def step_moe(pos, plen, prompt, toks, embed, ln1, qn, kn, wqkv, wo, ln2, wr, wgu0, wgu1, wgu2, wgu3, wd0, wd1, wd2, wd3,
             normf, wlm, rot, cos, sin,
             kcache, vcache, rkb, TB, SL, hw, NQL, NKL, ILT, ILP, W, rg, PF=0, HBMCC=1, NOMOE=0, SLO=1, PFO=0, SLH=1,
             MOEOPT=0, XCH=0, QKVQ=0, PFOQ=0, VD=0, ATT=0, KVQ=0, PFX=0):
    """SL: n-tiles per weight DMA of QKV, SLO of o_proj, SLH of the LM head. PF: QKV slabs of the next layer
    prefetched before this layer's MoE (they do not depend on the routing); PFO=1: all of the next layer's o_proj
    too (its own SBUF tiles, so its DMAs wait for nothing)."""
    L = wqkv.shape[0]
    HT = embed.shape[1] // 128
    HID = HT * 128
    E = wgu0.shape[1]
    wgus, wds = (wgu0, wgu1, wgu2, wgu3), (wd0, wd1, wd2, wd3)
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
    _, ones_row = _moe_consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    t_sb = _token_in(pos, plen, prompt, toks, p_sb)
    h = _embed_g(embed, t_sb, HT)
    hbuf = None
    if HBMCC and W > 1:
        if XCH:   # [2, 128, HT]: one half per physical core (moe_tp._xchg_add_hbm2)
            hb_a = nl.ndarray((2, 128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_a")
            hr_a = nl.ndarray((2, 128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_a")
            hb_m = nl.ndarray((2, 128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_m")
            hr_m = nl.ndarray((2, 128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_m")
        else:
            hb_a = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_a")
            hr_a = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_a")
            hb_m = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb_m")
            hr_m = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr_m")
        hbuf = (hb_a, hr_a, hb_m, hr_m)
    pre = None
    pre_o = None
    for l in range(L):
        h = _attn_tp(h, l, pid, ln1, qn, kn, wqkv, wo, rot_sb, kcache, vcache, preg, pb, cs_col, sn_col, ones, ident,
                     idf, ones16, TB, SL, hw, HT, NQL, NKL, rg, W, pre, None if hbuf is None else hbuf[0:2],
                     SLO=SLO, pre_o=pre_o, hwq=3 if QKVQ else None, VD=VD, ATT=ATT, KVQ=KVQ)
        pre = None
        pre_o = None
        if PF > 0 and l + 1 < L:   # next layer's first QKV slabs: independent of the routing, stream during the MoE
            pre, _ = _prefetch(wqkv[l + 1, pid], HT, SL, PF, hw, 0)
        if PFO and l + 1 < L and not PFOQ:
            pre_o, _ = _prefetch(wo[l + 1, pid], NQL, SLO, HT // SLO, hw, 0)
        if not NOMOE:
            w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w2, src=ln2[l])
            pfx = None
            if PFX and l + 1 < L:   # next layer's QKV (PFX & 1) / o_proj (PFX & 2) after this layer's expert slabs
                pfx = []
                if PFX & 1:
                    pfx.append((wqkv[l + 1, pid], HT, SL, (NQL + 2 * NKL) // SL))
                if PFX & 2:
                    pfx.append((wo[l + 1, pid], NQL, SLO, HT // SLO))
            h = _moe_layer(h, l, pid, w2, wr, wgus, wds, ones, ones_row, HT, E, ILT, ILP, hw, rg, W,
                           None if hbuf is None else hbuf[2:4], OPT=MOEOPT,
                           pf=(wo[l + 1, pid], NQL, SLO) if (PFO and PFOQ and l + 1 < L) else None, pfx=pfx)
            if pfx is not None:
                k = 0
                if PFX & 1:
                    pre = h[1][k]
                    k += 1
                if PFX & 2:
                    pre_o = h[1][k]
                h = h[0]
            if PFO and PFOQ and l + 1 < L:
                pre_o = h[1]
                h = h[0]
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(h, wf, HT, HID, ones, nl.bfloat16)
    tokf = _lm_argmax_tp(xf, wlm[pid], pid, rkb, SLH, hw, HT, NVL, rg, W)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok_i)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    return pos_out
