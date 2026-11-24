"""Probe 4: does concurrent weight streaming slow down an in-NEFF HBM all-reduce?
A chain of N dependent all-reduces of a [128, 16] fp32 partial (the decode step's pattern: SBUF -> internal HBM
buffer, ncc.all_reduce, HBM -> SBUF, add), optionally with MB megabytes per physical core of independent weight DMAs
(1 MB slabs, each consumed by one matmul) issued before the chain on queue Q: 0 = sync hardware DGE, 1 = GpSimd
software DGE, 2 = scalar hardware DGE. Per-op latency = (t(N=33) - t(N=1)) / 32 from host/ccrun's queued benchmark.
"""
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc


@nki.jit
def k_cc(x, w, rg, W, N=33, MB=0, Q=0):
    pid = nl.program_id(0)
    out = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.shared_hbm)
    ones = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    acc = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.psum)
    for i in range(MB):   # independent weight stream: 1 MB per DMA per physical core
        t = nl.ndarray((128, 4096), dtype=nl.bfloat16, buffer=nl.sbuf)
        if Q == 1:
            nisa.dma_copy(dst=t, src=w[i % 8, pid], dge_mode=nisa.dge_mode.swdge)
        elif Q == 2:
            nisa.dma_copy(dst=t, src=w[i % 8, pid], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        else:
            nisa.dma_copy(dst=t, src=w[i % 8, pid], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.nc_matmul(dst=acc[:, i % 16:i % 16 + 1], stationary=t[:, 0:128], moving=ones)
    hb = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hb")
    hr = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_hr")
    h = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h, src=x)
    for i in range(N):
        nisa.dma_copy(dst=hb, src=h)
        ncc.all_reduce(srcs=[hb], dsts=[hr], op=nl.add, replica_group=rg)
        r = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=r, src=hr)
        h2 = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h2, data=r, op0=nl.multiply, operand0=1.0 / W)
        h = h2
    if MB > 0:
        a = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=a, src=acc)
        nisa.tensor_tensor(dst=h, data1=h, data2=a, op=nl.add)
    nisa.dma_copy(dst=out, src=h)
    return out


def run(W, N, MB, Q):
    from nki.collectives import ReplicaGroup
    from qwen3_weights import bf16
    xs = [np.full((128, 16), 1.0, np.float32) for r in range(W)]
    ws = [np.zeros((8, 2, 128, 4096), bf16) for r in range(W)]
    k_cc[2](xs, ws, rg=ReplicaGroup([list(range(W))]), W=W, N=N, MB=MB, Q=Q)
    return "ok"
