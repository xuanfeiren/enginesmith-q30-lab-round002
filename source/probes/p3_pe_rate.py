"""Probe 3: TensorE time of the decode GEMV shapes of the MoE kernel (weights resident in SBUF, no DMA in the loop).
MODE 0: stationary [K, N] = [96, 128] (down slab tile), moving [96, 1], accumulate over 8 "experts" per column,
        column-major order (for nt: for k)          -- the decode kernel's down projection
MODE 1: same shapes, expert-major order (for k: for nt), one PSUM tile per expert
MODE 2: stationary [128, 96] (gate/up tile), moving [128, 1], 16 accumulating k-tiles per column
MODE 3: stationary [128, 128], moving [128, 1], 16 k-tiles per column (lab 1's GEMV tile)
MODE 4: down as x-stationary: stationary act [96, 1], moving down slab [96, 512] -> [1, 512] rows
REP repetitions of 16 x 8 matmuls (MODE 0/1/4) or 8 x 2 x 16 (MODE 2/3)."""
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def k_pe(w, MODE=0, REP=50):
    out = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.shared_hbm)
    ws = nl.ndarray((128, 8 * 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    x = nl.ndarray((128, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=x, value=0.5)
    acc = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for r in range(REP):
        if MODE == 0:
            ps = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.psum)
            for nt in range(16):
                for k in range(8):
                    nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=ws[0:96, k * 2048 + nt * 128:k * 2048 + (nt + 1) * 128],
                                   moving=x[0:96, k:k + 1])
            nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
        elif MODE == 1:
            for k in range(8):
                ps = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.psum)
                for nt in range(16):
                    nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=ws[0:96, k * 2048 + nt * 128:k * 2048 + (nt + 1) * 128],
                                   moving=x[0:96, k:k + 1])
                nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)
        elif MODE == 4:
            ps = nl.ndarray((1, 2048), dtype=nl.float32, buffer=nl.psum)
            for c in range(4):
                for k in range(8):
                    nisa.nc_matmul(dst=ps[:, c * 512:(c + 1) * 512], stationary=x[0:96, k:k + 1],
                                   moving=ws[0:96, k * 2048 + c * 512:k * 2048 + (c + 1) * 512])
            nisa.tensor_tensor(dst=acc[0:1, :], data1=acc[0:1, :], data2=ps[:, 0:16], op=nl.add)
        elif MODE == 5:   # gate/up x-stationary: x[:, kt] [128, 1] stationary, [128, 192] weights moving -> [1, 192]
            for c in range(8):
                ps = nl.ndarray((1, 192), dtype=nl.float32, buffer=nl.psum)
                for kt in range(16):
                    o = c * 2048 + kt * 112
                    nisa.nc_matmul(dst=ps, stationary=x[:, kt:kt + 1], moving=ws[:, o:o + 192])
                nisa.tensor_tensor(dst=acc[0:1, 0:16], data1=acc[0:1, 0:16], data2=ps[:, 0:16], op=nl.add)
        else:
            N = 128
            if MODE == 2:
                N = 96
            if MODE == 6:
                N = 64
            ps = nl.ndarray((N, 16), dtype=nl.float32, buffer=nl.psum)
            st = 128
            if MODE == 7:
                st = 96
            for c in range(16):
                for kt in range(16):
                    o = (c % 8) * 2048 + kt * st
                    nisa.nc_matmul(dst=ps[:, c:c + 1], stationary=ws[:, o:o + N], moving=x[:, kt:kt + 1])
            nisa.tensor_tensor(dst=acc[0:N, :], data1=acc[0:N, :], data2=ps, op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


def run(MODE=0, REP=50):
    from qwen3_weights import bf16
    w = (np.random.default_rng(0).standard_normal((128, 8 * 2048)) * 0.01).astype(bf16)
    k_pe(w, MODE=MODE, REP=REP)
    return "ok"
