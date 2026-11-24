"""Whole-model Qwen3-30B-A3B prefill, tensor-parallel over W logical NeuronCores (LNC=2), layered: three NEFFs
(emb, layer, head) as in lab 1's prefill_g.py; the layer NEFF is executed once per layer on per-layer slices of the
decode step's stacked tensors (host/tpgen.cpp), so the prefill writes the KV cache the decode NEFF reads.

layer (S tokens, feature-major residual h = s_h [128, HT, S] fp32 shared by the two cores of a rank):
  A  norm1 + QKV (this virtual core's NQL q / NKL kv heads; decode weight layout) + q/k norm + RoPE -> s_q, s_k, s_v,
     and the KV cache rows 0..S-1 (lab 1 code, head counts as parameters)
  B  attention: nkilib attention_cte (causal), the logical core's 2*NQL q heads / 2*NKL kv heads split over its two
     physical cores by the library exactly like our QKV split
  C  o_proj K-split (own heads) -> partial h for all hidden tiles -> s_p[pid] -> core_barrier -> each core sums both
     partials for its half of the hidden tiles -> (W > 1) HBM all_reduce over the ranks -> h += ...
  D  MoE (kernels/moe_prefill.py, v6): per token tile x2 = RMSNorm(h) * ln2 -> router / top-8 / dispatch by
     compaction / blocks of 128 slots / combine -> sum over the two cores and the ranks -> transposed back to
     feature-major and added to h.
head: final norm of token plen-1, LM head (vocab split over the 2W virtual cores) + argmax (lab 2) -> toks[plen],
      output_0 = plen (the decode chain's first pos).
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from layer_decode import _consts, _rms_col, D, EPS
from step_tp import _lm_argmax_tp
from moe_prefill import _issue, _front, _back, TOPK, B
from nkilib.core.attention.attention_cte import _attention_cte

SCALE = 0.08838834764831845


def _hd(dst, src, eng=None):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync if eng is None else eng)


def _norm_blk(h, t0, n, w_sb, onesb, HT, xo, xo0, addsrc=None):
    """RMSNorm of tokens [t0, t0+n) of h [128, HT, S] fp32 (HBM) into xo[:, :, xo0:xo0+n] bf16 (SBUF). (lab 1)
    addsrc (same layout as h): h += addsrc first (written back to h), the attention all-reduce's read-back fused in."""
    hc = nl.ndarray((128, HT, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(hc, h[:, :, nl.ds(t0, n)])
    if addsrc is not None:
        rr = nl.ndarray((128, HT, n), dtype=nl.float32, buffer=nl.sbuf)
        _hd(rr, addsrc[:, :, nl.ds(t0, n)], nisa.engine.scalar)
        nisa.tensor_tensor(dst=hc, data1=hc, data2=rr, op=nl.add)
        _hd(h[:, :, nl.ds(t0, n)], hc)
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hc)
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xo[:, kt, xo0:xo0 + n], data=hc[:, kt, :], op0=nl.multiply,
                                  operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


def _slab(W, nt, KT):
    t = nl.ndarray((128, KT * 128), dtype=W.dtype, buffer=nl.sbuf)
    _hd(t, W[:, nl.ds(nt * KT * 128, KT * 128)])
    return t


def _attn_part(h, ln1, qn, kn, wqkv, wo, rot, cosT, sinT, kcache, vcache, s_q, s_k, s_v, s_o, s_p, ar_a, ar_b,
               pid, S, HT, NQL, NKL, W, rg, ident, onesb, NOAR=0, OPX=0, RSPLIT=0):
    """A + B + C of the layer (attention half-layer), h updated in place (HBM s_h)."""
    l = 0
    TB = 512 if S >= 512 else S
    NSUB = 256 if TB >= 256 else TB
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1, src=ln1[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=knc, src=kn[l])
    NQKV = NQL + 2 * NKL
    for blk in range(S // TB):
        t0 = blk * TB
        x1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
        for sb in range(TB // NSUB):
            _norm_blk(h, t0 + sb * NSUB, NSUB, w1, onesb, HT, x1, sb * NSUB)
        cs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
        _hd(cs, cosT[:, nl.ds(t0, TB)])
        sn = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
        _hd(sn, sinT[:, nl.ds(t0, TB)])
        for nt in range(NQKV):
            wt = _slab(wqkv[l, pid], nt, HT)
            ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
            for kt in range(HT):
                nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=x1[:, kt, :])
            if nt < NQL + NKL:
                sq = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=sq, op=nl.square, data=ps)
                ss = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
                rs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
                yb = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                if nt < NQL:
                    nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=qnc, op1=nl.multiply, operand1=rs)
                else:
                    nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=knc, op1=nl.multiply, operand1=rs)
                rp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
                xc = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
                xs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
                ro = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
                if nt < NQL:
                    _hd(s_q[pid * NQL + nt, :, nl.ds(t0, TB)], ro)
                else:
                    j = nt - NQL
                    _hd(s_k[pid * NKL + j, :, nl.ds(t0, TB)], ro)
                    _hd(kcache[l, pid, j, :, nl.ds(t0, TB)], ro)
            else:
                j = nt - NQL - NKL
                vb = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=vb, op=nl.copy, data=ps)
                vr = nl.ndarray((128, TB // 128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                for tb in range(TB // 128):
                    tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
                    nisa.tensor_copy(dst=vr[:, tb, :], src=tp)
                for tb in range(TB // 128):
                    _hd(s_v[pid * NKL + j, nl.ds(t0 + tb * 128, 128), :], vr[:, tb, :])
                    _hd(vcache[l, pid, nl.ds(t0 + tb * 128, 128), j * D:(j + 1) * D], vr[:, tb, :])
    nisa.core_barrier(data=s_q, cores=(0, 1))
    nisa.core_barrier(data=s_k, cores=(0, 1))
    nisa.core_barrier(data=s_v, cores=(0, 1))
    _attention_cte(q=s_q, k=s_k, v=s_v, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                   out_o_hbm=s_o)
    nisa.core_barrier(data=s_o, cores=(0, 1))
    HH = HT // 2
    if OPX and W > 1:
        # OPX: each physical core computes its half of the output tiles from both cores' heads (the other core's o_proj
        # slabs and attention output read from HBM); the two head-group partials stay separate PSUM accumulations and
        # are added in the same order as before (core 0's + core 1's), so the result is bit-identical, without the
        # s_p round trip through HBM
        for blk in range(S // TB):
            t0 = blk * TB
            oc = nl.ndarray((128, 2 * NQL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in range(2 * NQL):
                _hd(oc[:, hh, :], s_o[hh, :, nl.ds(t0, TB)])
            b0 = nl.ndarray((128, HH, TB), dtype=nl.float32, buffer=nl.sbuf)
            for i2 in range(HH):
                nt = pid * HH + i2
                pos_ = []
                for src in range(2):
                    wt = _slab(wo[l, src], nt, NQL)
                    ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(NQL):
                        nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=oc[:, src * NQL + kt, :])
                    pos_.append(ps)
                p0 = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=p0, op=nl.copy, data=pos_[0])
                nisa.tensor_tensor(dst=b0[:, i2, :], data1=p0, data2=pos_[1], op=nl.add)
            _hd(ar_a[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)], b0)
    else:
        for blk in range(S // TB):
            t0 = blk * TB
            oc = nl.ndarray((128, NQL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in range(NQL):
                _hd(oc[:, hh, :], s_o[pid * NQL + hh, :, nl.ds(t0, TB)])
            for nt in range(HT):
                wt = _slab(wo[l, pid], nt, NQL)
                ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                for kt in range(NQL):
                    nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=oc[:, kt, :])
                po = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=po, op=nl.copy, data=ps)
                _hd(s_p[pid, :, nt, nl.ds(t0, TB)], po)
        nisa.core_barrier(data=s_p, cores=(0, 1))
        for blk in range(S // TB):
            t0 = blk * TB
            b0 = nl.ndarray((128, HH, TB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(b0, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(t0, TB)])
            b1 = nl.ndarray((128, HH, TB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(b1, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(t0, TB)], nisa.engine.scalar)
            nisa.tensor_tensor(dst=b0, data1=b0, data2=b1, op=nl.add)
            if W > 1:
                _hd(ar_a[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)], b0)
            else:
                hc = nl.ndarray((128, HH, TB), dtype=nl.float32, buffer=nl.sbuf)
                _hd(hc, h[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)])
                nisa.tensor_tensor(dst=hc, data1=hc, data2=b0, op=nl.add)
                _hd(h[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)], hc)
    if W > 1:
        nisa.core_barrier(data=ar_a, cores=(0, 1))
        if NOAR:   # timing experiment only (wrong results): no all-reduce
            ar_b = ar_a
        else:
            ncc.all_reduce(srcs=[ar_a], dsts=[ar_b], op=nl.add, replica_group=rg)
        if not RSPLIT:   # (RSPLIT: h += ar_b fused into the MoE router loop)
            for blk in range(S // TB):
                t0 = blk * TB
                hc = nl.ndarray((128, HH, TB), dtype=nl.float32, buffer=nl.sbuf)
                _hd(hc, h[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)])
                rr = nl.ndarray((128, HH, TB), dtype=nl.float32, buffer=nl.sbuf)
                _hd(rr, ar_b[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)], nisa.engine.scalar)
                nisa.tensor_tensor(dst=hc, data1=hc, data2=rr, op=nl.add)
                _hd(h[:, nl.ds(pid * HH, HH), nl.ds(t0, TB)], hc)
    if not (RSPLIT and W > 1):
        nisa.core_barrier(data=h, cores=(0, 1))
    return ar_b


def _route_tile(h, j, w2, onesb, HT, H, E, x_tm, XP, wr_sb, idx_all, w_all, M_all, iota_e, hw, sync, arb):
    """norm + token-major copy + router / top-8 / weights / one-hot of token tile j (the lab's loop body)"""
    xt = nl.ndarray((128, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    _norm_blk(h, j * 128, 128, w2, onesb, HT, xt, 0, addsrc=arb)
    xtm = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    for q in range(HT // 4):
        tp = nl.ndarray((128, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
        for kk in range(4):
            nisa.nc_transpose(dst=tp[:, kk, :], data=xt[:, q * 4 + kk, :])
        nisa.tensor_copy(dst=xtm[:, q * 512:(q + 1) * 512], src=tp)
    nisa.dma_copy(dst=x_tm[XP, j * 128:(j + 1) * 128, :], src=xtm, dge_mode=hw, engine=sync)
    lps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lps, stationary=xt[:, kt, :], moving=wr_sb[:, kt * E:(kt + 1) * E])
    lg = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lg, src=lps)
    v8 = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=v8, src=lg)
    junk = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    ix = nl.ndarray((128, TOPK), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=junk, data=lg, vals=v8, imm=-3.0e38, dst_idx=ix)
    nisa.tensor_copy(dst=idx_all[:, j, :], src=ix)
    nb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb, data=v8[:, 0:1], op0=nl.multiply, operand0=-1.0)
    e8 = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    ssum = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=e8, op=nl.exp, data=v8, bias=nb, scale=1.0, reduce_op=nl.add, reduce_res=ssum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=ssum)
    nisa.tensor_scalar(dst=w_all[:, j, :], data=e8, op0=nl.multiply, operand0=rs)
    Mf = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Mf, data=iota_e, op0=nl.equal, operand0=idx_all[:, j, 0:1])
    for k in range(1, TOPK):
        nisa.scalar_tensor_tensor(dst=Mf, data=iota_e, op0=nl.equal, operand0=idx_all[:, j, k:k + 1],
                                  op1=nl.add, operand1=Mf)
    nisa.tensor_copy(dst=M_all[:, j, :], src=Mf)


def _moe_part(h, ln2, wr, wgu, wd, x_tm, stag, y_slot, y_part, ar_m, ar_n, pid, S, HT, E, ILT, ILP, W, rg,
              onesb, identf, PD=2, NOAR=0, MCH=1, BS=B, TAIL=0, GU=0, CMB=0, NCAP=0, CSPLIT=0, RSPLIT=0,
              arb=None):
    """D: the MoE half-layer of kernels/moe_prefill.py (v6) on RMSNorm(h) * ln2, h updated in place."""
    H = HT * 128
    NT = S // 128
    NMAX = (TOPK * S) // BS + E
    NBT = NMAX // 128 if NMAX % 128 == 0 else NMAX // 128 + 1
    F1 = HT * 2 * ILT * ILP
    F2 = ILT * HT * 128
    SW = 128
    hw = nisa.dge_mode.hwdge
    sync = nisa.engine.sync
    idf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=idf, pattern=[[-1, 128]], offset=0, channel_multiplier=1)
    lstr = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lstr, data=idf, op0=nl.less, operand0=0.0)
    ustr = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ustr, data=idf, op0=nl.less, operand0=0.0)
    pme = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)                  # [e, p] = e - p
    nisa.tensor_copy(dst=pme, src=idf)
    nisa.tensor_scalar(dst=idf, data=idf, op0=nl.equal, operand0=0.0)
    onesf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesf, value=1.0)
    iota_e = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=iota_e, pattern=[[1, E]], offset=0, channel_multiplier=0)
    iota_p = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=iota_p, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2, src=ln2[0])
    wr_sb = nl.ndarray((128, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr[0], dge_mode=hw, engine=sync)
    M_all = nl.ndarray((128, NT, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx_all = nl.ndarray((128, NT, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    w_all = nl.ndarray((128, NT, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    # RSPLIT: each physical core routes its token tiles (j % 2 == pid; h += the attention all-reduce result on the way,
    # x_tm written once into the shared x_tm[0]), the one-hot / ids / weights of the other core's tiles come through
    # y_part (unused with CMB) after a barrier
    XP = 0 if RSPLIT else pid
    for j in range(NT):
        if RSPLIT and j % 2 != pid:
            pass
        else:
            _route_tile(h, j, w2, onesb, HT, H, E, x_tm, XP, wr_sb, idx_all, w_all, M_all, iota_e, hw, sync,
                        arb if RSPLIT else None)
            if RSPLIT:
                ex = nl.ndarray((128, E + 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ex[:, 0:E], src=M_all[:, j, :])
                nisa.tensor_copy(dst=ex[:, E:E + TOPK], src=idx_all[:, j, :])
                nisa.tensor_copy(dst=ex[:, E + TOPK:E + 2 * TOPK], src=w_all[:, j, :])
                _hd(y_part[0, j * 128:(j + 1) * 128, 0:E + 2 * TOPK], ex)
    if RSPLIT:
        nisa.core_barrier(data=y_part, cores=(0, 1))
        for j in range(NT):
            if j % 2 != pid:
                ex = nl.ndarray((128, E + 2 * TOPK), dtype=nl.float32, buffer=nl.sbuf)
                _hd(ex, y_part[0, j * 128:(j + 1) * 128, 0:E + 2 * TOPK])
                nisa.tensor_copy(dst=M_all[:, j, :], src=ex[:, 0:E])
                nisa.tensor_copy(dst=idx_all[:, j, :], src=ex[:, E:E + TOPK])
                nisa.tensor_copy(dst=w_all[:, j, :], src=ex[:, E + TOPK:E + 2 * TOPK])
    # counts, blocks
    cps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    for j in range(NT):
        nisa.nc_matmul(dst=cps, stationary=onesb, moving=M_all[:, j, :])
    cnt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cnt, src=cps)
    mB = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mB, data=iota_p, op0=nl.multiply, operand0=float(BS))
    cmpm = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cmpm, data=cnt, op0=nl.greater, operand0=mB)
    nps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=nps, stationary=onesf, moving=cmpm)
    nbk = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nbk, src=nps)
    dg = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dg, data1=nbk, data2=idf, op=nl.multiply)
    nbc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=nbc, op=nl.add, data=dg, axis=1)
    nbd = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nbd, data=onesf, op0=nl.multiply, operand0=nbc)
    bps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=bps, stationary=nbd, moving=ustr)
    bsB = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bsB, data=bps, op0=nl.multiply, operand0=float(BS))
    endb = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=endb, data=bps, op0=nl.add, operand0=0.0, op1=nl.add, operand1=nbk)
    slot_all = nl.ndarray((128, NT, TOPK), dtype=nl.int32, buffer=nl.sbuf)
    for j in range(NT):
        rps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rps, stationary=lstr, moving=M_all[:, j, :])
        for i in range(j):
            nisa.nc_matmul(dst=rps, stationary=onesb, moving=M_all[:, i, :])
        V = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=V, data1=rps, data2=bsB, op=nl.add)
        slotf = nl.ndarray((128, TOPK), dtype=nl.float32, buffer=nl.sbuf)
        for k in range(TOPK):
            oh = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=oh, data=iota_e, op0=nl.equal, operand0=idx_all[:, j, k:k + 1],
                                      op1=nl.multiply, operand1=V)
            nisa.tensor_reduce(dst=slotf[:, k:k + 1], op=nl.add, data=oh, axis=1)
        nisa.tensor_copy(dst=slot_all[:, j, :], src=slotf)
    # per-expert token lists by compaction
    MT = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(NT):
        tpm = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpm, data=M_all[:, j, :])
        nisa.tensor_copy(dst=MT[:, j * 128:(j + 1) * 128], src=tpm)
    m16 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=m16, value=0.0)
    nisa.memset(dst=m16.ap(pattern=[[128, 128], [16, 8]], offset=0), value=1.0)
    CH = 512 if S >= 512 else S
    # CSPLIT: the two physical cores compact 8 rounds each into the shared stag[0] (both read it after a barrier)
    SP = 0 if CSPLIT else pid
    r0 = 0
    r1 = 16
    if CSPLIT:
        r0 = 8 * pid
        r1 = 8 * pid + 8
    for r in range(r0, r1):
        pr = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=pr, data=pme, op0=nl.equal, operand0=float(r), op1=nl.multiply, operand1=m16)
        sel = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(S // CH):
            sps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=sps, stationary=pr, moving=MT[:, c * CH:(c + 1) * CH])
            nisa.tensor_copy(dst=sel[:, c * CH:(c + 1) * CH], src=sps)
        lists = nl.ndarray((128, S + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nonzero_with_count(dst=lists, src=sel, index_offset=0, padding_val=S)
        for c in range(8):
            nisa.dma_copy(dst=stag[SP, 16 * c + r:16 * c + r + 1, :], src=lists[16 * c:16 * c + 1, :],
                          dge_mode=hw, engine=nisa.engine.sync if c % 2 == 0 else nisa.engine.scalar)
    if CSPLIT:
        nisa.core_barrier(data=stag, cores=(0, 1))
    blk_row = nl.ndarray((1, NBT * 128), dtype=nl.int32, buffer=nl.sbuf)
    off_row = nl.ndarray((1, NBT * 128), dtype=nl.int32, buffer=nl.sbuf)
    for q in range(NBT):
        bcol = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=bcol, data=iota_p, op0=nl.add, operand0=float(q * 128))
        le = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=le, data=endb, op0=nl.less_equal, operand0=bcol)
        cntb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cntb, op=nl.add, data=le, axis=1)
        tps = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tps, data=cntb)
        nisa.tensor_copy(dst=blk_row[:, q * 128:(q + 1) * 128], src=tps)
        nisa.tensor_scalar(dst=cntb, data=cntb, op0=nl.minimum, operand0=float(E - 1))
        ohb = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=ohb, data=iota_e, op0=nl.equal, operand0=cntb, op1=nl.multiply, operand1=bsB)
        bsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=bsb, op=nl.add, data=ohb, axis=1)
        offc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=offc, data=bcol, op0=nl.multiply, operand0=float(BS), op1=nl.subtract, operand1=bsb)
        nisa.scalar_tensor_tensor(dst=offc, data=cntb, op0=nl.multiply, operand0=float(S + 1), op1=nl.add, operand1=offc)
        nisa.tensor_scalar(dst=offc, data=offc, op0=nl.maximum, operand0=0.0, op1=nl.minimum,
                           operand1=float(E * (S + 1) - BS))
        tpo = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tpo, data=offc)
        nisa.tensor_copy(dst=off_row[:, q * 128:(q + 1) * 128], src=tpo)
    # blocks
    # TAIL=1: blocks past the 8S/BS that are always real run their compute inside nl.fori_loop(0, flag) with flag =
    # 1 if the block is real, so the empty ones (DMAs already skipped) cost no compute
    NG = NMAX if TAIL == 0 else (TOPK * S) // BS
    NSTAT = NMAX
    if NCAP > 0 and NCAP < NMAX:
        # NCAP: only blocks [0, NCAP) in the static pipelined loop; the real blocks past NCAP (none unless the routing
        # is skewed) run one at a time in a device loop fori_loop(NCAP, n_real) (same arithmetic per block)
        NSTAT = NCAP
        flf2 = nl.ndarray((1, NBT * 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=flf2, data=blk_row, op0=nl.less, operand0=float(E))
        nrf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=nrf, op=nl.add, data=flf2, axis=1)
        nri = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nri, src=nrf)
        nreg = nisa.register_alloc()
        nisa.register_load(nreg, nri)
    if NG > NSTAT:
        NG = NSTAT
    if TAIL:
        flf = nl.ndarray((1, NBT * 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=flf, data=blk_row, op0=nl.less, operand0=float(E))
        fli = nl.ndarray((1, NBT * 128), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=fli, src=flf)
    pend = []
    for b in range(PD + 1):
        if b < NSTAT:
            pend.append(_issue(b, blk_row, off_row, NBT, wgu[0], wd[0], x_tm, stag, pid, F1, F2, SW, ILP, H, BB=BS, spid=SP, xpid=XP))
    nxt = _front(pend.pop(0), HT, ILT, ILP, SW, BS, GU)
    for b in range(NSTAT):
        if b < NG:
            this = nxt
            if b + 1 + PD < NSTAT:
                pend.append(_issue(b + 1 + PD, blk_row, off_row, NBT, wgu[0], wd[0], x_tm, stag, pid, F1, F2, SW, ILP, H,
                                   BB=BS, spid=SP, xpid=XP))
            if b + 1 < NG:
                nxt = _front(pend.pop(0), HT, ILT, ILP, SW, BS, GU)
            _back(this, b, y_slot, pid, H, HT, ILT, BS)
        else:
            if b + 1 + PD < NSTAT:
                pend.append(_issue(b + 1 + PD, blk_row, off_row, NBT, wgu[0], wd[0], x_tm, stag, pid, F1, F2, SW, ILP, H,
                                   BB=BS, spid=SP, xpid=XP))
            cur = pend.pop(0)
            freg = nisa.register_alloc()
            nisa.register_load(freg, fli[0:1, b:b + 1])

            def _blk(i):
                _back(_front(cur, HT, ILT, ILP, SW, BS, GU), b, y_slot, pid, H, HT, ILT, BS)

            nl.fori_loop(0, freg, _blk)
    if NSTAT < NMAX:
        br3 = blk_row.reshape((1, NBT * 128, 1))
        or3 = off_row.reshape((1, NBT * 128, 1))
        y4 = y_slot.reshape((2, y_slot.shape[1] // BS, BS, H))

        def _dynblk(i):
            e_t = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=e_t, src=br3.ap(pattern=[[NBT * 128, 1], [1, 1]], offset=0, scalar_offset=i, indirect_dim=1))
            o_t = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_t, src=or3.ap(pattern=[[NBT * 128, 1], [1, 1]], offset=0, scalar_offset=i, indirect_dim=1))
            cur = _issue(0, blk_row, off_row, NBT, wgu[0], wd[0], x_tm, stag, pid, F1, F2, SW, ILP, H, BB=BS, spid=SP, xpid=XP,
                         e_ap=e_t, o_ap=o_t)
            ydst = y4[pid].ap(pattern=[[H, BS], [1, H]], offset=0, scalar_offset=i, indirect_dim=0)
            _back(_front(cur, HT, ILT, ILP, SW, BS, GU), 0, y_slot, pid, H, HT, ILT, BS, ydst=ydst)

        nl.fori_loop(NSTAT, nreg, _dynblk)
    # combine over the 8 slots (with the routing weights), the two cores, the ranks; back to feature-major, h +=
    # MCH > 1 (W > 1): in MCH chunks of token tiles, each chunk's all-reduce (its own HBM buffers ar_m[c] -> ar_n[c])
    # issued as soon as the chunk is combined, its result added to h after the next chunk's combine
    if MCH == 1 or W == 1:
        ar_ms = [ar_m]
        ar_ns = [ar_n]
        NCH = 1
    else:
        ar_ms = ar_m
        ar_ns = ar_n
        NCH = MCH
    CT = NT // NCH

    if CMB and W > 1:
        # CMB: each physical core combines its token tiles (j % 2 == pid) from both cores' y_slot (the two weighted
        # sums kept separate and added in the same order as before: bit-identical), without the y_part round trip
        nisa.core_barrier(data=y_slot, cores=(0, 1))
    for c in range(NCH):
        for jj in range(CT):
            j = c * CT + jj
            if CMB and W > 1:
                if j % 2 == pid:
                    accs = []
                    for src in range(2):
                        acc = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
                        for k in range(TOPK):
                            g = nl.ndarray((128, H), dtype=y_slot.dtype, buffer=nl.sbuf)
                            nisa.dma_copy(dst=g, src=y_slot[src].ap(pattern=[[H, 128], [1, H]], offset=0,
                                                                     vector_offset=slot_all[:, j, k:k + 1], indirect_dim=0),
                                          dge_mode=nisa.dge_mode.swdge)
                            if k == 0:
                                nisa.tensor_scalar(dst=acc, data=g, op0=nl.multiply, operand0=w_all[:, j, 0:1])
                            else:
                                nisa.scalar_tensor_tensor(dst=acc, data=g, op0=nl.multiply, operand0=w_all[:, j, k:k + 1],
                                                          op1=nl.add, operand1=acc)
                        accs.append(acc)
                    nisa.tensor_tensor(dst=accs[0], data1=accs[0], data2=accs[1], op=nl.add)
                    _hd(ar_ms[c][jj * 128:(jj + 1) * 128, :], accs[0])
            else:
                acc = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
                for k in range(TOPK):
                    g = nl.ndarray((128, H), dtype=y_slot.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=g, src=y_slot[pid].ap(pattern=[[H, 128], [1, H]], offset=0,
                                                             vector_offset=slot_all[:, j, k:k + 1], indirect_dim=0),
                                  dge_mode=nisa.dge_mode.swdge)
                    if k == 0:
                        nisa.tensor_scalar(dst=acc, data=g, op0=nl.multiply, operand0=w_all[:, j, 0:1])
                    else:
                        nisa.scalar_tensor_tensor(dst=acc, data=g, op0=nl.multiply, operand0=w_all[:, j, k:k + 1],
                                                  op1=nl.add, operand1=acc)
                _hd(y_part[pid, j * 128:(j + 1) * 128, :], acc)
        if not (CMB and W > 1):
            nisa.core_barrier(data=y_part, cores=(0, 1))
        for jj in range(CT):
            j = c * CT + jj
            if j % 2 == pid and not (CMB and W > 1):
                a = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
                _hd(a, y_part[0, j * 128:(j + 1) * 128, :])
                b2 = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
                _hd(b2, y_part[1, j * 128:(j + 1) * 128, :], nisa.engine.scalar)
                nisa.tensor_tensor(dst=a, data1=a, data2=b2, op=nl.add)
                if W > 1:
                    _hd(ar_ms[c][jj * 128:(jj + 1) * 128, :], a)
                else:
                    _add_tm_to_h(h, a, j, HT, identf)
        if W > 1:
            nisa.core_barrier(data=ar_ms[c], cores=(0, 1))
            if NOAR == 1:   # timing experiment only
                ar_ns[c] = ar_ms[c]
            else:
                ncc.all_reduce(srcs=[ar_ms[c]], dsts=[ar_ns[c]], op=nl.add, replica_group=rg)
            if c > 0:
                _readback(h, ar_ns[c - 1], c - 1, CT, pid, H, HT, identf)
    if W > 1:
        _readback(h, ar_ns[NCH - 1], NCH - 1, CT, pid, H, HT, identf)
    nisa.core_barrier(data=h, cores=(0, 1))


def _readback(h, arn, c, CT, pid, H, HT, identf):
    """h += the all-reduced MoE output of chunk c (this core's token tiles)"""
    for jj in range(CT):
        j = c * CT + jj
        if j % 2 == pid:
            a = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            _hd(a, arn[jj * 128:(jj + 1) * 128, :])
            _add_tm_to_h(h, a, j, HT, identf)


def _add_tm_to_h(h, a, j, HT, identf):
    """h[:, :, tile j] += a^T: a [128 tok, H] fp32 token-major -> feature-major by fp32 PE transposes"""
    hc = nl.ndarray((128, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
    _hd(hc, h[:, :, nl.ds(j * 128, 128)])
    for q in range(HT // 4):
        tp = nl.ndarray((128, 4, 128), dtype=nl.float32, buffer=nl.psum)
        for kk in range(4):
            kt = q * 4 + kk
            nisa.nc_transpose(dst=tp[:, kk, :], data=a[:, kt * 128:(kt + 1) * 128])     # bit-accurate (trn2)
        nisa.tensor_tensor(dst=hc[:, q * 4:(q + 1) * 4, :], data1=hc[:, q * 4:(q + 1) * 4, :], data2=tp, op=nl.add)
    _hd(h[:, :, nl.ds(j * 128, 128)], hc)


@nki.jit
def pf_tp(ids, plen, toks, embed, ln1, qn, kn, wqkv, wo, ln2, wr, wgu, wd, normf, wlm, rot, cosT, sinT,
          kcache, vcache, rkb, s_h, s_q, s_k, s_v, s_o, s_p, x_tm, stag, y_slot, y_part,
          STAGE, NQL, NKL, ILT, ILP, W, rg, NOAR=0, MCH=1, ABL=0, BS=128, TAIL=0, GU=0, OPX=0, CMB=0, PD=2, NCAP=0, CSPLIT=0, RSPLIT=0):
    S = ids.shape[0]
    HT = embed.shape[1] // 128
    HID = HT * 128
    E = wgu.shape[1]
    NVL = wlm.shape[2] // (HT * 128)
    pid = nl.program_id(0)
    ones, ident, idf, ones16 = _consts()
    onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesb, value=1.0)
    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    h = s_h
    if STAGE == 1:
        NB = S // 128
        for bi in range(NB // 2):
            bb = 2 * bi + pid
            idx = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=ids[nl.ds(bb * 128, 128), :])
            er = nl.ndarray((128, HID), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=er, src=embed.ap(pattern=[[HID, 128], [1, HID]], offset=0, vector_offset=idx, indirect_dim=0))
            et = nl.ndarray((128, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
            for kt in range(HT):
                tp = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=tp, stationary=er[:, kt * 128:(kt + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=et[:, kt, :], src=tp)
            _hd(h[:, :, nl.ds(bb * 128, 128)], et)
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out
    if STAGE == 2:
        ar_a = None
        ar_b = None
        ar_m = None
        ar_n = None
        if W > 1:
            if ABL != 1:
                ar_a = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.shared_hbm, name="pf_ar_a")
            if not NOAR and ABL != 1:
                ar_b = nl.ndarray((128, HT, S), dtype=nl.float32, buffer=nl.shared_hbm, name="pf_ar_b")
            if ABL == 2:
                pass
            elif MCH == 1:
                ar_m = nl.ndarray((S, HID), dtype=nl.float32, buffer=nl.shared_hbm, name="pf_ar_m")
                if NOAR != 1:
                    ar_n = nl.ndarray((S, HID), dtype=nl.float32, buffer=nl.shared_hbm, name="pf_ar_n")
            else:
                ar_m = []
                ar_n = []
                for c in range(MCH):
                    ar_m.append(nl.ndarray((S // MCH, HID), dtype=nl.float32, buffer=nl.shared_hbm,
                                           name=("pf_ar_m0", "pf_ar_m1", "pf_ar_m2", "pf_ar_m3", "pf_ar_m4", "pf_ar_m5",
                                                 "pf_ar_m6", "pf_ar_m7")[c]))
                    ar_n.append(nl.ndarray((S // MCH, HID), dtype=nl.float32, buffer=nl.shared_hbm,
                                           name=("pf_ar_n0", "pf_ar_n1", "pf_ar_n2", "pf_ar_n3", "pf_ar_n4", "pf_ar_n5",
                                                 "pf_ar_n6", "pf_ar_n7")[c]))
        if ABL != 1:   # ABL (timing experiments only, wrong results): 1 = no attention half, 2 = no MoE half
          arb = _attn_part(h, ln1, qn, kn, wqkv, wo, rot, cosT, sinT, kcache, vcache, s_q, s_k, s_v, s_o, s_p, ar_a, ar_b,
                   pid, S, HT, NQL, NKL, W, rg, ident, onesb, NOAR=NOAR, OPX=OPX, RSPLIT=RSPLIT)
        if ABL != 2:
          _moe_part(h, ln2, wr, wgu, wd, x_tm, stag, y_slot, y_part, ar_m, ar_n, pid, S, HT, E, ILT, ILP, W, rg,
                  onesb, idf, NOAR=NOAR, MCH=MCH, BS=BS, TAIL=TAIL, GU=GU, CMB=CMB, PD=PD, NCAP=NCAP, CSPLIT=CSPLIT, RSPLIT=RSPLIT, arb=arb)
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out
    # STAGE 3: LM head on token plen - 1
    lm1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lm1, data=pl_sb, op0=nl.add, operand0=-1, op1=nl.maximum, operand1=0)   # plen >= 1
    lreg = nisa.register_alloc()
    nisa.register_load(lreg, lm1)
    hl = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hl, src=h.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=lreg, indirect_dim=2))
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(hl, wf, HT, HID, ones, nl.bfloat16)
    tokf = _lm_argmax_tp(xf, wlm[pid], pid, rkb, 2, 2, HT, NVL, rg, W)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    preg = nisa.register_alloc()
    nisa.register_load(preg, pl_sb)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0), src=tok_i)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    return pos_out
