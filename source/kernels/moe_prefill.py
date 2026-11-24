"""Qwen3-MoE prefill MoE half-layer for S tokens (grouped expert GEMMs over the tokens routed to each expert),
tensor-parallel over W logical NeuronCores (LNC=2: 2W virtual cores; virtual core g owns rows [g*IL, (g+1)*IL) of
every expert's gate/up and the same columns of down, exactly as the decode kernel moe_tp.py). One NEFF per layer.

Everything on the device, static shapes, no device loops:
  0. x token-major copy for gathers: PE transposes of the feature-major input xT [128, HT, S] -> x_tm[pid] (HBM).
  1. router logits per 128-token tile ([128 tok, 128 e], x tile stationary, Wr^T moving), max8 + match_replace8
     -> top-8 ids, weights w = exp(v - v0) / sum (HF rule, renormalised), one-hot M_j [128 tok, 128 e].
  2. rank of each token inside its expert: R_j = Lstrict^T M_j + sum_{i<j} ones^T M_i (exact small integers, fp32
     PSUM); counts C = sum_i ones^T M_i (broadcast on every partition).
  3. blocks of B = 128 slots: nb_e = ceil(C_e / B), bs_e = exclusive prefix of nb over e (matmul with a strictly upper
     triangular ones matrix), slot(t, k) = B * bs_e + R[t, e], e = idx[t, k].
  4. per-expert token lists by stream compaction (nonzero_with_count over M^T), written to stag[pid, e, :]
     (ids in increasing order, padded with S = a zero row of x_tm). Block b reads its 128 ids at
     e_b * (S + 1) + (b - bs_e) * B. (v3 scattered (token, weight) to slot(t, k) with per-partition-offset DMAs
     instead: 128 software-DGE scatters, 0.7 ms at S = 1,024.)
  5. block -> expert: blk[b] = #{e : end_e <= b}, clamped to E-1 (blocks past the last real one load a real expert
     and compute garbage that is never read). NMAX = 8S/B + E blocks: the static worst case.
  6. for every block b (static loop): DMA the slabs of expert blk[b] (dynamic offset, as in decode), gather the 128
     token rows (vector_offset DMA), 16 PE transposes, gate/up GEMMs (weights stationary [128, ILP], tokens moving
     [128, 128]), silu * up, down GEMM token-major (act stationary [ILP, 128 tok], down moving [ILP, 512]),
     write y_slot[pid][block] (fp32, contiguous).
  7. combine: y[t] = sum_k w[t, k] * y_slot[slot(t, k)] (8 gathers per token tile), then the sum over the two physical cores
     (y_part shared HBM + core_barrier) and over the ranks (HBM all_reduce).
Caller-allocated scratch (inputs; lab 1: internal shared_hbm scratch is not shared between the two cores):
  x_tm [2, S + 1, H] bf16 (row S stays zero), stag [2, E, S + 1] int32, y_slot [2, NMAX*B, H] bf16 (or fp32),
  y_part [2, S, H] fp32. Output y [S, H] fp32 (the MoE output, token-major).
"""
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc

TOPK = 8
B = 128


def _issue(b, blk_row, off_row, NBT, wgu, wd, x_tm, stag, pid, F1, F2, SW, ILP, H, xres=None, koff=None, BB=B,
           e_ap=None, o_ap=None, spid=None, xpid=None):
    """the DMAs of block b: its expert's two slabs (dynamic offset, sync queue), its 128 token ids (dynamic offset
    into the compacted per-expert lists) and its 128 token rows (gather)"""
    if e_ap is None:   # (given: the dynamic-block path, fixed [1, 1] tiles holding block b's values)
        e_ap = blk_row.ap(pattern=[[NBT * 128, 1], [1, 1]], offset=b)
        o_ap = off_row.ap(pattern=[[NBT * 128, 1], [1, 1]], offset=b)
    if xres is not None:
        # feature-major gather on the chip: the block's 128 ids on every partition (broadcast DMA), + kt * S, then
        # nisa.nc_n_gather from the SBUF-resident x^T [128, HT * S] (GpSimd): no transposes, no PSUM evacuation
        HT = H // 128
        S = xres.shape[2]
        idb = nl.ndarray((128, 128), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idb, src=stag[pid].ap(pattern=[[0, 128], [1, 128]], offset=0, scalar_offset=o_ap, indirect_dim=1),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        xgT = nl.ndarray((128, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):     # one gather per 128 features (a single one over HT * S elements fails ISA validation)
            nisa.nc_n_gather(dst=xgT[:, kt, :], data=xres[:, kt, :], indices=idb)
    sg = nl.ndarray((128, F1 + SW - ILP), dtype=nl.bfloat16, buffer=nl.sbuf)    # + window padding (moe_tp W128)
    # blocks past the last real one have expert id E: out of bounds -> the slab DMAs are skipped (oob_mode.skip),
    # the block computes on stale SBUF data and its output is never read
    nisa.dma_copy(dst=sg[:, 0:F1], src=wgu[:, pid].select(0, e_ap), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync,
                  oob_mode=nisa.oob_mode.skip)
    sd = nl.ndarray((ILP, F2), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=sd, src=wd[:, pid].select(0, e_ap), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync,
                  oob_mode=nisa.oob_mode.skip)
    if xres is not None:
        return [sg, sd, xgT, 1]
    ids = nl.ndarray((BB, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids, src=stag[pid if spid is None else spid].ap(pattern=[[1, BB], [1, 1]], offset=0, scalar_offset=o_ap, indirect_dim=1),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    if koff is not None:   # XRES=2: gather + transpose by the DMA (software DGE dma_transpose), one per 128 features
        HT = H // 128
        idu = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idu, src=ids)
        xgT = nl.ndarray((128, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kt in range(HT):
            nisa.dma_transpose(dst=xgT[:, kt, :], src=x_tm[pid].ap(pattern=[[H, 128], [1, 128]], offset=kt * 128,
                                                                   vector_offset=idu, indirect_dim=0), axes=(1, 0))
        return [sg, sd, xgT, 1]
    xg = nl.ndarray((BB, H), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xg, src=x_tm[pid if xpid is None else xpid].ap(pattern=[[H, BB], [1, H]], offset=0, vector_offset=ids,
                                           indirect_dim=0), dge_mode=nisa.dge_mode.swdge)
    return [sg, sd, xg, 0]


def _front(cur, HT, ILT, ILP, SW, BB=B, GU=0):
    """block part 1: token rows -> feature-major (PE transposes), gate/up GEMMs, silu(g) * u -> act [ILP, ILT, 128]"""
    sg = cur[0]
    sd = cur[1]
    xg = cur[2]
    if cur[3] == 1:
        xgT = xg
    else:
        xgT = _transpose_rows(xg, HT, BB)
    if GU:
        return _gate_up_tm(sg, sd, xgT, HT, ILP, BB)
    return _gate_up(sg, sd, xgT, HT, ILT, ILP, SW, BB)


def _gate_up_tm(sg, sd, xgT, HT, ILP, BB):
    """GU=1 (ILT = 1): token-major gate/up, the x tile stationary and the 2 * ILP gate + up columns of the slab moving
    (16 matmuls of N = 2 * ILP instead of 32 of N = 128), silu(g) * u token-major, one PE transpose to [ILP, BB]"""
    gps = nl.ndarray((BB, 2 * ILP), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        o = kt * 2 * ILP
        nisa.nc_matmul(dst=gps, stationary=xgT[:, kt, :], moving=sg[:, o:o + 2 * ILP])
    sgm = nl.ndarray((BB, ILP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sgm, op=nl.silu, data=gps[:, 0:ILP])
    atm = nl.ndarray((BB, ILP), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=atm, data1=sgm, data2=gps[:, ILP:2 * ILP], op=nl.multiply)
    return [atm, sd, 1]   # transposed at the start of _back (by then silu * up is long done)


def _transpose_rows(xg, HT, BB=B):
    xgT = nl.ndarray((128, HT, BB), dtype=nl.bfloat16, buffer=nl.sbuf)
    for q in range(HT // 4):
        tp = nl.ndarray((128, 4, BB), dtype=nl.bfloat16, buffer=nl.psum)
        for kk in range(4):
            nisa.nc_transpose(dst=tp[:, kk, :], data=xg[:, (q * 4 + kk) * 128:(q * 4 + kk + 1) * 128])
        if q % 2 == 0:
            nisa.tensor_copy(dst=xgT[:, q * 4:(q + 1) * 4, :], src=tp, engine=nisa.engine.vector)
        else:
            nisa.activation(dst=xgT[:, q * 4:(q + 1) * 4, :], op=nl.copy, data=tp)
    return xgT


def _gate_up(sg, sd, xgT, HT, ILT, ILP, SW, BB=B):
    gps = nl.ndarray((SW, 2 * ILT, BB), dtype=nl.float32, buffer=nl.psum)
    for gu in range(2):
        for it in range(ILT):
            for kt in range(HT):
                o = ((kt * 2 + gu) * ILT + it) * ILP
                nisa.nc_matmul(dst=gps[:, gu * ILT + it, :], stationary=sg[:, o:o + SW], moving=xgT[:, kt, :])
    sgm = nl.ndarray((ILP, ILT, BB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sgm, op=nl.silu, data=gps[0:ILP, 0:ILT, :])
    act = nl.ndarray((ILP, ILT, BB), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=sgm, data2=gps[0:ILP, ILT:2 * ILT, :], op=nl.multiply)
    return [act, sd]


def _back(this, b, y_slot, pid, H, HT, ILT, BB=B, ydst=None):
    """block part 2: down GEMM token-major [128 tok, H] fp32, PSUM evacuation, write the block's rows of y_slot"""
    act = this[0]
    sd = this[1]
    if len(this) > 2:   # GU=1: token-major act [BB, ILP] -> [ILP, 1, BB]
        ILP = act.shape[1]
        tp = nl.ndarray((ILP, BB), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=act)
        act = nl.ndarray((ILP, 1, BB), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=act[:, 0, :], src=tp, engine=nisa.engine.vector)
    yb = nl.ndarray((BB, H), dtype=y_slot.dtype, buffer=nl.sbuf)
    for c in range(H // 512):
        yps = nl.ndarray((BB, 512), dtype=nl.float32, buffer=nl.psum)
        for it in range(ILT):
            o = (it * HT + c * 4) * 128
            nisa.nc_matmul(dst=yps, stationary=act[:, it, :], moving=sd[:, o:o + 512])
        if c % 2 == 0:   # PSUM evacuation alternately on ScalarE and VectorE (weights: in the combine)
            nisa.activation(dst=yb[:, c * 512:(c + 1) * 512], op=nl.copy, data=yps)
        else:
            nisa.tensor_copy(dst=yb[:, c * 512:(c + 1) * 512], src=yps, engine=nisa.engine.vector)
    nisa.dma_copy(dst=y_slot[pid, b * BB:(b + 1) * BB, :] if ydst is None else ydst, src=yb, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)


@nki.jit
def moe_prefill(xT, wr, wgu, wd, x_tm, stag, y_slot, y_part, ILT, ILP, W, rg, PD=2, XRES=0):
    HT = xT.shape[1]
    S = xT.shape[2]
    H = HT * 128
    E = wgu.shape[0]
    NT = S // 128
    NMAX = (TOPK * S) // B + E
    NBT = NMAX // 128 if NMAX % 128 == 0 else NMAX // 128 + 1
    F1 = HT * 2 * ILT * ILP
    F2 = ILT * HT * 128
    SW = 128          # gate/up stationary window width (128: fast TensorE path, see moe_tp._experts)
    pid = nl.program_id(0)
    hw = nisa.dge_mode.hwdge
    sync = nisa.engine.sync

    # ---- constants
    identb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    idf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=idf, pattern=[[-1, 128]], offset=0, channel_multiplier=1)          # p - f
    lstr = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)                 # [t', t] = 1 if t' < t
    nisa.tensor_scalar(dst=lstr, data=idf, op0=nl.less, operand0=0.0)
    ustr = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)                  # [e', e] = 1 if e' < e
    nisa.tensor_scalar(dst=ustr, data=idf, op0=nl.less, operand0=0.0)
    nisa.tensor_scalar(dst=idf, data=idf, op0=nl.equal, operand0=0.0)                # identity fp32
    nisa.tensor_copy(dst=identb, src=idf)
    onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesb, value=1.0)
    onesf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesf, value=1.0)
    iota_e = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)                 # value e along the free axis
    nisa.iota(dst=iota_e, pattern=[[1, E]], offset=0, channel_multiplier=0)
    iota_p = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)                 # partition index
    nisa.iota(dst=iota_p, pattern=[[0, 1]], offset=0, channel_multiplier=1)

    idx_o = nl.ndarray((S, TOPK), dtype=nl.int32, buffer=nl.shared_hbm)      # the top-8 ids (for the checks)
    # ---- 0 + 1: token-major copy, router, top-8, one-hot (every core redundantly, all tokens)
    wr_sb = nl.ndarray((128, HT * E), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wr_sb, src=wr, dge_mode=hw, engine=sync)
    M_all = nl.ndarray((128, NT, E), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx_all = nl.ndarray((128, NT, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    w_all = nl.ndarray((128, NT, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    xres = None
    koff = None
    if XRES == 2:
        koff = 1
    if XRES == 1:   # x^T resident in SBUF for the on-chip gathers (S * HT * 2 bytes per partition)
        xres = nl.ndarray((128, HT, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xres, src=xT, dge_mode=hw, engine=sync)
        koff = nl.ndarray((128, HT, 128), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(dst=koff, pattern=[[S, HT], [0, 128]], offset=0, channel_multiplier=0)
    for j in range(NT):
        xt = nl.ndarray((128, HT, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=xT[:, :, j * 128:(j + 1) * 128], dge_mode=hw, engine=sync)
        xtm = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        for q in range(HT // 4):
            tp = nl.ndarray((128, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
            for kk in range(4):
                nisa.nc_transpose(dst=tp[:, kk, :], data=xt[:, q * 4 + kk, :])
            nisa.tensor_copy(dst=xtm[:, q * 512:(q + 1) * 512], src=tp)
        nisa.dma_copy(dst=x_tm[pid, j * 128:(j + 1) * 128, :], src=xtm, dge_mode=hw, engine=sync)
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
        ixo = nl.ndarray((128, TOPK), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ixo, src=ix)
        nisa.dma_copy(dst=idx_o[j * 128:(j + 1) * 128, :], src=ixo)
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

    # ---- 2 + 3: ranks, counts, blocks
    cps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    for j in range(NT):
        nisa.nc_matmul(dst=cps, stationary=onesb, moving=M_all[:, j, :])
    cnt = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)                      # counts, every row equal
    nisa.tensor_copy(dst=cnt, src=cps)
    # nb_e = ceil(cnt_e / B) = #{m : cnt_e > m * B}, m on the partitions (up to 128 blocks per expert), summed by a
    # ones matmul (exact small integers; tensor_scalar has no mod / floor)
    mB = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mB, data=iota_p, op0=nl.multiply, operand0=float(B))
    cmpm = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cmpm, data=cnt, op0=nl.greater, operand0=mB)
    nps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=nps, stationary=onesf, moving=cmpm)
    nbk = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)                      # ceil(cnt / B), every row
    nisa.tensor_copy(dst=nbk, src=nps)
    # nb as a column (diagonal of the broadcast rows), then NBD[e', p] = nb[e'] and bs[p, e] = sum_e' NBD U
    dg = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dg, data1=nbk, data2=idf, op=nl.multiply)
    nbc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=nbc, op=nl.add, data=dg, axis=1)
    nbd = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nbd, data=onesf, op0=nl.multiply, operand0=nbc)
    bps = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=bps, stationary=nbd, moving=ustr)
    bsB = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)                      # B * block start, every row
    nisa.tensor_scalar(dst=bsB, data=bps, op0=nl.multiply, operand0=float(B))
    endb = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)                     # block end (exclusive)
    nisa.scalar_tensor_tensor(dst=endb, data=bps, op0=nl.add, operand0=0.0, op1=nl.add, operand1=nbk)

    # ---- 4: slot of every (t, k) (for the combine)
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
        sloti = slot_all[:, j, :]
        nisa.tensor_copy(dst=sloti, src=slotf)

    # ---- 4b: per-expert token lists by stream compaction (no scatter): M^T (experts on partitions); for round r the
    # experts 16c + r are moved to partitions 16c by a 0/1 matmul and nisa.nonzero_with_count (GpSimd: it reads and
    # writes partitions 0, 16, ..., 112 only) writes their token ids in increasing order, padded with S (a zero row
    # of x_tm), then the 8 rows go to stag[pid, e, :].
    MT = nl.ndarray((128, S), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(NT):
        tpm = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpm, data=M_all[:, j, :])
        nisa.tensor_copy(dst=MT[:, j * 128:(j + 1) * 128], src=tpm)
    m16 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)                  # column p kept iff p % 16 == 0
    nisa.memset(dst=m16, value=0.0)
    nisa.memset(dst=m16.ap(pattern=[[128, 128], [16, 8]], offset=0), value=1.0)
    pme = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)                  # [e, p] = e - p
    nisa.iota(dst=pme, pattern=[[-1, 128]], offset=0, channel_multiplier=1)
    CH = 512 if S >= 512 else S
    for r in range(16):
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
            nisa.dma_copy(dst=stag[pid, 16 * c + r:16 * c + r + 1, :], src=lists[16 * c:16 * c + 1, :],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync if c % 2 == 0 else nisa.engine.scalar)

    # ---- 5: block -> expert, as a row on partition 0 (the scalar offsets of the slab DMAs)
    blk_row = nl.ndarray((1, NBT * 128), dtype=nl.int32, buffer=nl.sbuf)
    off_row = nl.ndarray((1, NBT * 128), dtype=nl.int32, buffer=nl.sbuf)
    for q in range(NBT):
        bcol = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=bcol, data=iota_p, op0=nl.add, operand0=float(q * 128))
        le = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)                  # end_e <= b
        nisa.tensor_scalar(dst=le, data=endb, op0=nl.less_equal, operand0=bcol)
        cntb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cntb, op=nl.add, data=le, axis=1)
        tps = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tps, data=cntb)                                         # E for empty blocks
        nisa.tensor_copy(dst=blk_row[:, q * 128:(q + 1) * 128], src=tps)
        nisa.tensor_scalar(dst=cntb, data=cntb, op0=nl.minimum, operand0=float(E - 1))
        # offset of the block's 128 ids in the compacted lists: e_b * (S + 1) + (b - bs_{e_b}) * B, clamped
        ohb = nl.ndarray((128, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=ohb, data=iota_e, op0=nl.equal, operand0=cntb, op1=nl.multiply, operand1=bsB)
        bsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)                 # B * bs_{e_b}
        nisa.tensor_reduce(dst=bsb, op=nl.add, data=ohb, axis=1)
        offc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=offc, data=bcol, op0=nl.multiply, operand0=float(B), op1=nl.subtract, operand1=bsb,
                           reverse1=False)
        nisa.scalar_tensor_tensor(dst=offc, data=cntb, op0=nl.multiply, operand0=float(S + 1), op1=nl.add, operand1=offc)
        nisa.tensor_scalar(dst=offc, data=offc, op0=nl.maximum, operand0=0.0, op1=nl.minimum,
                           operand1=float(E * (S + 1) - B))
        tpo = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tpo, data=offc)
        nisa.tensor_copy(dst=off_row[:, q * 128:(q + 1) * 128], src=tpo)

    # ---- 6: the blocks, software-pipelined in program order: the slab DMAs (sync queue) and the token gather
    # (GpSimd queue) of block b + PD are issued before block b computes, so no in-order queue waits behind a
    # block's compute; the block's output goes out on the scalar engine's queue.

    pend = []
    for b in range(PD + 1):
        if b < NMAX:
            pend.append(_issue(b, blk_row, off_row, NBT, wgu, wd, x_tm, stag, pid, F1, F2, SW, ILP, H, xres, koff))
    # compute software-pipelined too: TensorE runs block b+1's transposes and gate/up before block b's down
    # projection, so it does not idle while ScalarE / VectorE compute block b's silu * up
    cur = pend.pop(0)
    nxt = _front(cur, HT, ILT, ILP, SW)
    for b in range(NMAX):
        this = nxt
        if b + 1 < NMAX:
            if b + 1 + PD < NMAX:
                pend.append(_issue(b + 1 + PD, blk_row, off_row, NBT, wgu, wd, x_tm, stag, pid, F1, F2, SW, ILP, H, xres, koff))
            nxt = _front(pend.pop(0), HT, ILT, ILP, SW)
        _back(this, b, y_slot, pid, H, HT, ILT)

    # ---- 7: combine the 8 slots of every token, then the two physical cores, then the ranks
    for j in range(NT):
        acc = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
        for k in range(TOPK):
            g = nl.ndarray((128, H), dtype=y_slot.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=g, src=y_slot[pid].ap(pattern=[[H, 128], [1, H]], offset=0,
                                                     vector_offset=slot_all[:, j, k:k + 1], indirect_dim=0),
                          dge_mode=nisa.dge_mode.swdge)
            if k == 0:
                nisa.tensor_scalar(dst=acc, data=g, op0=nl.multiply, operand0=w_all[:, j, 0:1])
            else:
                nisa.scalar_tensor_tensor(dst=acc, data=g, op0=nl.multiply, operand0=w_all[:, j, k:k + 1], op1=nl.add,
                                          operand1=acc)
        nisa.dma_copy(dst=y_part[pid, j * 128:(j + 1) * 128, :], src=acc, dge_mode=hw, engine=sync)
    nisa.core_barrier(data=y_part, cores=(0, 1))
    y = nl.ndarray((S, H), dtype=nl.float32, buffer=nl.shared_hbm)
    if W > 1:
        ar_in = nl.ndarray((S, H), dtype=nl.float32, buffer=nl.shared_hbm, name="pf_ar_in")
        ar_out = nl.ndarray((S, H), dtype=nl.float32, buffer=nl.shared_hbm, name="pf_ar_out")
    for j in range(NT):
        if j % 2 == pid:
            a = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=y_part[0, j * 128:(j + 1) * 128, :], dge_mode=hw, engine=sync)
            b2 = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=b2, src=y_part[1, j * 128:(j + 1) * 128, :], dge_mode=hw, engine=sync)
            nisa.tensor_tensor(dst=a, data1=a, data2=b2, op=nl.add)
            if W > 1:
                nisa.dma_copy(dst=ar_in[j * 128:(j + 1) * 128, :], src=a, dge_mode=hw, engine=sync)
            else:
                nisa.dma_copy(dst=y[j * 128:(j + 1) * 128, :], src=a, dge_mode=hw, engine=sync)
    if W > 1:
        nisa.core_barrier(data=ar_in, cores=(0, 1))
        ncc.all_reduce(srcs=[ar_in], dsts=[ar_out], op=nl.add, replica_group=rg)
        for j in range(NT):
            if j % 2 == pid:
                nisa.dma_copy(dst=y[j * 128:(j + 1) * 128, :], src=ar_out[j * 128:(j + 1) * 128, :])
    return y, idx_o
