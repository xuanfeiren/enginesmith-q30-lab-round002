"""Probe 1: MoE router on the device + gather of the selected experts' weights by token-dependent indices.

router: logits row [1, E] = x^T Wr^T (x column [128, HT] bf16 is the *stationary* operand, one column per k-tile;
Wr^T tile [128 k, E] moving) -> fp32 PSUM row on partition 0. top-8: nisa.max8 (8 largest, descending) +
nisa.nc_match_replace8 (indices of the matched positions; duplicates get distinct positions, lowest index first).
weights: w_k = exp(v_k - v_0) / sum_j exp(v_j - v_0)  (== softmax over all E then renormalised over the top 8).
gather: for each k, dma_copy(dst=sbuf slab, src=wexp[l].select(0, idx[k]))  (dynamic scalar offset from SBUF).

run(mode, E, F, L, ...) builds + runs once (nki's runner) and returns outputs; timing through host/ccrun.
"""
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


def router_top8(x_col, wr, HT, E):
    """x_col [128, HT] bf16 SBUF; wr HBM [128, HT*E] bf16 (wr[p, kt*E + e] = Wr[e, kt*128 + p]).
    returns (lg [1, E] fp32 SBUF, v8 [1, 8] fp32, idx [1, 8] uint32, w8 [1, 8] fp32)"""
    wr_sb = nl.ndarray((128, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    lps = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=lps, stationary=x_col[:, kt:kt + 1], moving=wr_sb[:, kt * E:(kt + 1) * E])
    lg = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lg, src=lps)
    v8 = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=v8, src=lg)
    junk = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((1, 8), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_match_replace8(dst=junk, data=lg, vals=v8, imm=-3.0e38, dst_idx=idx)
    nb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb, data=v8[:, 0:1], op0=nl.multiply, operand0=-1.0)
    e8 = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=e8, op=nl.exp, data=v8, bias=nb, scale=1.0, reduce_op=nl.add, reduce_res=s,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    r = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=r, data=s)
    w8 = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w8, data=e8, op0=nl.multiply, operand0=r)
    return lg, v8, idx, w8


@nki.jit
def k_router_gather(x, wr, wexp, HW=2):
    """x [128, HT] bf16 HBM; wr [128, HT*E]; wexp [E, 128, F] bf16. outputs: lg, w8, idx, gathered [8, 128, F]"""
    HT = x.shape[1]
    E = wexp.shape[0]
    F = wexp.shape[2]
    x_sb = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    lg, v8, idx, w8 = router_top8(x_sb, wr, HT, E)
    o_lg = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.shared_hbm)
    o_w = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.shared_hbm)
    o_i = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.shared_hbm)
    o_g = nl.ndarray((8, 128, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=o_lg, src=lg)
    nisa.dma_copy(dst=o_w, src=w8)
    ii = nl.ndarray((1, 8), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ii, src=idx)
    nisa.dma_copy(dst=o_i, src=ii)
    for k in range(8):
        g = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf)
        src = wexp.select(0, idx.ap(pattern=[[8, 1], [1, 1]], offset=k))
        if HW == 2:
            nisa.dma_copy(dst=g, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        elif HW == 1:
            nisa.dma_copy(dst=g, src=src, dge_mode=nisa.dge_mode.swdge)
        else:
            nisa.dma_copy(dst=g, src=src)
        nisa.dma_copy(dst=o_g[k], src=g)
    return o_lg, o_w, o_i, o_g


def ref_router(x_col, wr_full):
    """numpy fp32: HF Qwen3MoeTopKRouter on the same bf16 inputs (softmax fp32, topk 8, renormalise)"""
    x = x_col.astype(np.float32).T.reshape(-1)
    lg = wr_full.astype(np.float32) @ x
    p = np.exp(lg - lg.max()); p /= p.sum()
    order = np.argsort(-p, kind="stable")[:8]
    w = p[order] / p[order].sum()
    return lg, order, w


def run(E=128, HT=16, F=512, HW=2, seed=0, tie=0):
    from qwen3_weights import bf16
    rng = np.random.default_rng(seed)
    xv = rng.standard_normal(HT * 128).astype(np.float32)
    x_col = np.ascontiguousarray(xv.reshape(HT, 128).T).astype(bf16)
    Wr = (rng.standard_normal((E, HT * 128)) * 0.02).astype(bf16)
    if tie == 2:   # 10 experts with identical rows at the top: a tie across the 8/9 boundary
        big = (np.sign(xv) * 0.05).astype(bf16)
        for e in (3, 5, 9, 50, 60, 70, 77, 80, 90, 100):
            Wr[e] = big
    elif tie:   # experts 3 and 9 identical rows, and 70 too: exact ties of the logits
        Wr[9] = Wr[3]; Wr[70] = Wr[3]; Wr[3] *= bf16(1.0)
        Wr[3] = (Wr[3].astype(np.float32) + 0).astype(bf16)
        # make them the top: scale them up
        big = (np.sign(xv) * 0.05).astype(bf16)
        Wr[3] = big; Wr[9] = big; Wr[70] = big; Wr[100] = big
    wr = np.ascontiguousarray(Wr.reshape(E, HT, 128).transpose(2, 1, 0)).reshape(128, HT * E)
    wexp = (np.arange(E)[:, None, None] * 1.0 + rng.standard_normal((E, 128, F)) * 0.01).astype(bf16)
    o_lg, o_w, o_i, o_g = k_router_gather(x_col, wr, wexp, HW=HW)
    lg, order, w = ref_router(x_col, Wr)
    o_i = np.asarray(o_i).reshape(-1)
    res = dict(lg_err=float(np.abs(np.asarray(o_lg).reshape(-1) - lg).max()), lg_max=float(np.abs(lg).max()),
               idx=o_i.tolist(), ref_idx=order.tolist(), w=np.asarray(o_w).reshape(-1).tolist(), ref_w=w.tolist(),
               w_err=float(np.abs(np.asarray(o_w).reshape(-1) - w).max()),
               gather_ok=bool(all((np.asarray(o_g)[k] == wexp[o_i[k]]).all() for k in range(8))))
    print(res)
    return res
