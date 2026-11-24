"""Probe 2: streaming rate of token-dependent (dynamic-offset) expert-weight DMAs vs static DMAs of the same bytes.

wexp [L, E, 2, 128, F] bf16 (2 = physical core), idx [1, L*8] int32 in HBM. Each physical core, for every (l, k):
DMA wexp[l, idx[l, k], pid] ([128, F], F*256 bytes) into SBUF, then one tiny matmul on it (keeps it live).
MODE 0: dynamic offset, hwdge on the sync engine; 1: dynamic, swdge (GpSimd descriptors);
2: static offsets (the same indices baked in at trace time), hwdge sync; 3: dynamic, hwdge alternating sync/scalar;
SPLIT>1: each slab as SPLIT DMAs of F/SPLIT columns.
"""
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def k_gather_bw(idx, wexp, MODE=0, SPLIT=1, STATIC=None, FULL=0):
    L, E, _, P, F = wexp.shape
    pid = nl.program_id(0)
    idx_sb = nl.ndarray((1, L * 8), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=idx)
    ones = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    o = nl.ndarray((128, L * 8), dtype=nl.float32, buffer=nl.sbuf)
    FS = F // SPLIT
    q = 0
    for l in range(L):
        acc = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)   # one PSUM tile per layer (<= 8k matmuls per tile)
        for k in range(8):
            g = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf)
            for s in range(SPLIT):
                if MODE == 2:
                    e = STATIC[l * 8 + k]
                    src = wexp[l, e, pid, :, s * FS:(s + 1) * FS]
                else:
                    src = wexp[l, :, pid, :, s * FS:(s + 1) * FS].select(
                        0, idx_sb.ap(pattern=[[L * 8, 1], [1, 1]], offset=l * 8 + k))
                if MODE == 1:
                    nisa.dma_copy(dst=g[:, s * FS:(s + 1) * FS], src=src, dge_mode=nisa.dge_mode.swdge)
                elif MODE == 3:
                    nisa.dma_copy(dst=g[:, s * FS:(s + 1) * FS], src=src, dge_mode=nisa.dge_mode.hwdge,
                                  engine=nisa.engine.sync if q % 2 == 0 else nisa.engine.scalar)
                else:
                    nisa.dma_copy(dst=g[:, s * FS:(s + 1) * FS], src=src, dge_mode=nisa.dge_mode.hwdge,
                                  engine=nisa.engine.sync)
                q += 1
            if FULL:   # consume every 128-column tile (a real GEMV's worth of PE work)
                for t in range(F // 128):
                    nisa.nc_matmul(dst=acc[:, k:k + 1], stationary=g[:, t * 128:(t + 1) * 128], moving=ones)
            else:
                nisa.nc_matmul(dst=acc[:, k:k + 1], stationary=g[:, F - 128:F], moving=ones)
        nisa.tensor_copy(dst=o[:, l * 8:(l + 1) * 8], src=acc)
    out = nl.ndarray((128, L * 8), dtype=nl.float32, buffer=nl.shared_hbm)
    if pid == 0:
        nisa.dma_copy(dst=out, src=o)
    return out


def run(MODE=0, L=48, E=16, F=4608, SPLIT=1, seed=0, FULL=0, LNC=1):
    """LNC=1 (default, as first measured): the kernel runs on ONE physical core; LNC=2: grid [2], both physical
    cores of the logical core stream their own half (wexp[:, :, pid])."""
    from qwen3_weights import bf16
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(E, 8, replace=False) for _ in range(L)]).reshape(1, L * 8).astype(np.int32)
    wexp = rng.standard_normal((L, E, 2, 128, F)).astype(bf16)
    kern = k_gather_bw[2] if LNC == 2 else k_gather_bw
    out = kern(idx, wexp, MODE=MODE, SPLIT=SPLIT, STATIC=tuple(int(v) for v in idx.reshape(-1)), FULL=FULL)
    c0 = 0 if FULL else F - 128
    ref = np.stack([wexp[l, idx[0, l * 8 + k], 0, :, c0:].astype(np.float32).reshape(128, -1, 128).sum((0, 1))
                    for l in range(L) for k in range(8)], 1)
    err = float(np.abs(np.asarray(out) - ref).max())
    np.save("/tmp/p2_idx.npy", idx)
    return dict(err=err, bytes_per_core=L * 8 * 128 * F * 2 * LNC)   # bytes moved per logical core per execution
