"""Fused Qwen3-0.6B decoder layer(s) for one decode token, everything in "column layout".

Column layout: a vector v[K] lives in SBUF as [128, K/128] with v_col[p, kt] = v[kt*128 + p]. Every GEMV is
weight-stationary: tile (nt, kt) of W^T, [128 k, 128 n], is the stationary operand and x_col[:, kt] (one
column) the moving one, so y lands in PSUM as y_col[:, nt] -- already the layout of the next op. TensorE does
one 128x128 tile per ~30 ns this way (probes/p6_pe_rate.py), twice the x-stationary rate.

Inputs (per layer l, stacked on a leading L dim; all bf16 unless noted; layouts from qwen3_weights.py):
  h_in   [128, 8]  fp32       residual stream, column layout (hidden 1024)
  ln1, ln2 [L, 128, 8]        input / post-attention RMSNorm weights, column layout
  qn, kn [L, 128, 1]          q_norm / k_norm weights as columns (d on partitions)
  wqkv   [L, 128, 32*8*128]   ws_layout(concat(q_proj, k_proj, v_proj))   N=4096, K=1024
  wo     [L, 128, 8*16*128]   ws_layout(o_proj)                           N=1024, K=2048
  wgu    [L, 128, 48*8*128]   ws_layout(concat(gate_proj, up_proj))       N=6144, K=1024
  wd     [L, 128, 8*24*128]   ws_layout(down_proj)                        N=1024, K=3072
  rot    [128, 128]           rope_rot_matrix() (rotate-half as a matmul)
  cos, sin [TMAX, 64] fp32    RoPE tables
  kcache [L, 8, 128, TMAX], vcache [L, TMAX, 1024]  (see attn_decode.py), pos [1, 1] int32
Output: h_out [128, 8] fp32 (column layout); caches updated in place at pos.
"""
import nki
import nki.language as nl
import nki.isa as nisa

D = 128
NQ = 16
NKV = 8
EPS = 1e-6
NEG = -30000.0
SCALE = 0.08838834764831845
HID = 1024
HT = HID // 128          # 8 hidden column tiles
INTER = 3072
IT = INTER // 128        # 24


def _dma(dst, src, i, hw):
    """hw 0: default (software DGE, descriptors issued by GpSimd); 1: hardware DGE alternating sync / scalar
    engines; 2: hardware DGE from the sync engine only (it runs no compute, so a waiting compute instruction
    never blocks the weight stream)."""
    if hw == 0:
        nisa.dma_copy(dst=dst, src=src)
    elif hw == 3:   # hardware DGE from the scalar engine (issued when ScalarE reaches it in its in-order stream)
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    elif hw == 2:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync if i % 2 == 0 else nisa.engine.scalar)


def _gemv(xT, W, KT, nt0, nt1, ps, c0, SL, hw, qi):
    """ps[:, c0 + (nt - nt0)] = sum_k W_tile(nt, k)^T @ xT[:, k] for nt in [nt0, nt1). W: HBM [128, NT*KT*128]."""
    nt = nt0
    c = c0
    while nt < nt1:
        nb = SL
        if nt1 - nt < SL:
            nb = nt1 - nt
        wt = nl.ndarray((128, nb * KT * 128), dtype=W.dtype, buffer=nl.sbuf)
        _dma(wt, W[:, nt * KT * 128:(nt + nb) * KT * 128], qi, hw)
        qi += 1
        for t in range(nb):
            for k in range(KT):
                nisa.nc_matmul(dst=ps[:, c:c + 1], stationary=wt[:, (t * KT + k) * 128:(t * KT + k + 1) * 128],
                               moving=xT[:, k:k + 1])
            c += 1
        nt += nb
    return qi


def _rms_col(h, w_col, nt, n, ones, out_dtype):
    """RMSNorm of a column-layout vector h [128, nt] fp32 (length n) times weight w_col [128, nt] fp32."""
    sq = nl.ndarray((128, nt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=h, data2=h, op=nl.multiply)
    ss = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1)
    tot = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tot, stationary=ones, moving=ss)            # every partition: sum over all n
    rstd = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=tot, op0=nl.multiply, operand0=1.0 / n, op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, nt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xn, data=h, op0=nl.multiply, operand0=rstd)
    xo = nl.ndarray((128, nt), dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xo, data1=xn, data2=w_col, op=nl.multiply)
    return xo


def _consts():
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    idf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=idf, pattern=[[-1, 128]], offset=0, channel_multiplier=1)
    nisa.tensor_scalar(dst=idf, data=idf, op0=nl.equal, operand0=0.0)
    ident = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ident, src=idf)
    ones16 = nl.ndarray((NQ, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones16, value=1.0)
    return ones, ident, idf, ones16


def _layer(h, l, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, cos, sin, kcache, vcache,
           preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw):
    TMAX = vcache.shape[1]
    NT = TB // 128
    NS = NT + 1
    qi = 0
    # ---- attention cache bucket loads first (independent of this layer's activations)
    kall = nl.ndarray((D, NKV, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
    vall = nl.ndarray((128, NT, NKV * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    for hh in range(NKV):
        _dma(kall[:, hh, :], kcache[l, hh, :, 0:TB], qi, hw)
        qi += 1
    for j in range(NT):
        _dma(vall[:, j, :], vcache[l, j * 128:(j + 1) * 128, :], qi, hw)
        qi += 1
    # ---- small per-layer weights
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1, src=ln1[l])
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2, src=ln2[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=knc, src=kn[l])

    # ---- input RMSNorm + fused QKV (weight-stationary) -> qkv_ps [128 d, 32 heads]
    x1 = _rms_col(h, w1, HT, HID, ones, nl.bfloat16)
    qkv_ps = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(x1, wqkv[l], HT, 0, 32, qkv_ps, 0, SL, hw, qi)
    qkv = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkv, src=qkv_ps)

    # ---- q/k RMSNorm over d (partitions): sums of squares via ones-matmul
    sq = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=qkv[:, 0:24], data2=qkv[:, 0:24], op=nl.multiply)
    ss = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss, stationary=ones, moving=sq)
    rstd = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn, data1=qkv[:, 0:24], data2=rstd, op=nl.multiply)
    qkn = nl.ndarray((128, 24), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qkn[:, 0:NQ], data=xn[:, 0:NQ], op0=nl.multiply, operand0=qnc)
    nisa.tensor_scalar(dst=qkn[:, NQ:24], data=xn[:, NQ:24], op0=nl.multiply, operand0=knc)
    # ---- RoPE: x*cos + rotate_half(x)*sin, rotate_half as a permutation matmul
    rps = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rps, stationary=rot_sb, moving=qkn)
    xc = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xc, data=qkn, op0=nl.multiply, operand0=cs_col)
    qkT = nl.ndarray((128, 24), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=qkT, data=rps, op0=nl.multiply, operand0=sn_col, op1=nl.add, operand1=xc)
    vT = nl.ndarray((128, NKV), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vT, src=qkv[:, 24:32])
    # v as a row [1, 1024] on partition 0: out[0, n] = sum_k vT[k, h] I[k, n]
    vr_ps = nl.ndarray((1, NKV * D), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKV):
        nisa.nc_matmul(dst=vr_ps[0:1, hh * D:(hh + 1) * D], stationary=vT[:, hh:hh + 1], moving=ident)
    vrow = nl.ndarray((1, NKV * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vrow, src=vr_ps)
    # ---- cache writes at pos
    nisa.dma_copy(dst=kcache[l].ap(pattern=[[TMAX, D], [D * TMAX, NKV]], offset=0, scalar_offset=preg, indirect_dim=2),
                  src=qkT[:, NQ:24])
    nisa.dma_copy(dst=vcache[l].ap(pattern=[[NKV * D, 1], [1, NKV * D]], offset=0, scalar_offset=preg, indirect_dim=0),
                  src=vrow)

    # ---- scores [128 t, NS, 16]
    io = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[128, NT]], offset=0, channel_multiplier=1)
    madd = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=madd, data=io, op0=nl.greater_equal, operand0=pb, op1=nl.multiply, operand1=NEG)
    S = nl.ndarray((128, NS, NQ), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(NT):
        ps = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NKV):
            nisa.nc_matmul(dst=ps[:, 2 * hh:2 * hh + 2], stationary=kall[:, hh, j * 128:(j + 1) * 128],
                           moving=qkT[:, 2 * hh:2 * hh + 2])
        nisa.tensor_scalar(dst=S[:, j, :], data=ps, op0=nl.add, operand0=madd[:, j:j + 1])
    nisa.memset(dst=S[:, NT, :], value=NEG)
    psc = nl.ndarray((1, NQ), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKV):
        nisa.nc_matmul(dst=psc[0:1, 2 * hh:2 * hh + 2], stationary=qkT[:, NQ + hh:NQ + hh + 1], moving=qkT[:, 2 * hh:2 * hh + 2])
    nisa.tensor_copy(dst=S[0:1, NT, :], src=psc)
    # ---- softmax
    pm = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm, op=nl.maximum, data=S.ap(pattern=[[NS * NQ, 128], [1, NQ], [NQ, NS]], offset=0), axis=2)
    pmT = nl.ndarray((NQ, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pmT, data=pm)
    m = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=pmT, axis=1)
    dg = nl.ndarray((NQ, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dg, data=idf[0:NQ, 0:NQ], op0=nl.multiply, operand0=m)
    mb = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=mb, stationary=ones16, moving=dg)
    mbs = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mbs, src=mb)
    Sm = nl.ndarray((128, NS, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Sm, data1=S, data2=mbs.ap(pattern=[[NQ, 128], [0, NS], [1, NQ]], offset=0), op=nl.subtract)
    E = nl.ndarray((128, NS, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=E, op=nl.exp, data=Sm, scale=SCALE)
    es = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=es, op=nl.add, data=E.ap(pattern=[[NS * NQ, 128], [1, NQ], [NQ, NS]], offset=0), axis=2)
    den = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=den, stationary=ones, moving=es)
    rden = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    # ---- PV -> O^T [128 d, 16 heads]
    ops_ = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKV):
        for j in range(NT):
            nisa.nc_matmul(dst=ops_[:, 2 * hh:2 * hh + 2], stationary=vall[:, j, hh * D:(hh + 1) * D],
                           moving=E[:, j, 2 * hh:2 * hh + 2])
        nisa.nc_matmul(dst=ops_[:, 2 * hh:2 * hh + 2], stationary=vrow[0:1, hh * D:(hh + 1) * D],
                       moving=E[0:1, NT, 2 * hh:2 * hh + 2])
    oT = nl.ndarray((D, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=oT, data1=ops_, data2=rden, op=nl.multiply)

    # ---- o_proj + residual
    y_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(oT, wo[l], NQ, 0, HT, y_ps, 0, SL, hw, qi)
    h2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=y_ps, op=nl.add)

    # ---- post-attention RMSNorm + gate/up + SiLU*up + down + residual
    x2 = _rms_col(h2, w2, HT, HID, ones, nl.bfloat16)
    gu_ps = nl.ndarray((128, 2 * IT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(x2, wgu[l], HT, 0, 2 * IT, gu_ps, 0, SL, hw, qi)
    sg = nl.ndarray((128, IT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=gu_ps[:, 0:IT])
    act = nl.ndarray((128, IT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=sg, data2=gu_ps[:, IT:2 * IT], op=nl.multiply)
    d_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    qi = _gemv(act, wd[l], IT, 0, HT, d_ps, 0, SL, hw, qi)
    h3 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h3, data1=h2, data2=d_ps, op=nl.add)
    return h3


def _pos_setup(cos, sin, pos):
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)
    # cos/sin of position pos as columns: partition d gets freq index d mod 64
    cs_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sn_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        nisa.dma_copy(dst=cs_col[half * 64:(half + 1) * 64, :],
                      src=cos.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        nisa.dma_copy(dst=sn_col[half * 64:(half + 1) * 64, :],
                      src=sin.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    return p_sb, preg, pb, cs_col, sn_col


def _layers(h_in, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cos, sin, kcache, vcache, pos, TB, SL, hw):
    L = wqkv.shape[0]
    p_sb, preg, pb, cs_col, sn_col = _pos_setup(cos, sin, pos)
    ones, ident, idf, ones16 = _consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    h = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h, src=h_in)
    for l in range(L):
        h = _layer(h, l, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, cos, sin, kcache, vcache,
                   preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw)
    h_out = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=h_out, src=h)
    return h_out


@nki.jit
def layers_b512(h_in, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cos, sin, kcache, vcache, pos):
    return _layers(h_in, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cos, sin, kcache, vcache, pos, 512, 4, 0)


@nki.jit
def layers_b512_hw(h_in, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cos, sin, kcache, vcache, pos):
    return _layers(h_in, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot, cos, sin, kcache, vcache, pos, 512, 4, 1)
