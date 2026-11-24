"""Probe 5: PSUM -> SBUF evacuation cost on VectorE / ScalarE (the prefill block loop's limiter).
MODE: 0 VectorE tensor_copy fp32 PSUM [128, 512] -> bf16 SBUF; 1 ScalarE activation(copy) same; 2 VectorE bf16 PSUM
[128, 512] -> bf16 SBUF; 3 ScalarE same; 4 VectorE fp32 PSUM [128, 2048] (4 banks) in one instruction;
5 VectorE fp32 PSUM [128, 512] -> fp32 SBUF. Each rep: one matmul writes the PSUM tile, then the copy."""
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def k_evac(w, MODE=0, REP=50):
    out = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.shared_hbm)
    ws = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    mv = nl.ndarray((128, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=mv, value=0.25)
    acc = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    N = 2048 if MODE == 4 else 512
    pdt = nl.bfloat16 if MODE in (2, 3) else nl.float32
    sdt = nl.float32 if MODE == 5 else nl.bfloat16
    for r in range(REP):
        ps = nl.ndarray((128, N), dtype=pdt, buffer=nl.psum)
        for c in range(N // 512):
            if MODE in (2, 3):
                for q in range(4):
                    nisa.nc_transpose(dst=ps[:, c * 512 + q * 128:c * 512 + (q + 1) * 128], data=mv[:, q * 128:(q + 1) * 128])
            else:
                nisa.nc_matmul(dst=ps[:, c * 512:(c + 1) * 512], stationary=ws, moving=mv[:, c * 512:(c + 1) * 512])
        sb = nl.ndarray((128, N), dtype=sdt, buffer=nl.sbuf)
        if MODE in (1, 3):
            nisa.activation(dst=sb, op=nl.copy, data=ps)
        else:
            nisa.tensor_copy(dst=sb, src=ps, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=sb[:, 0:16], op=nl.add, engine=nisa.engine.gpsimd)
    nisa.dma_copy(dst=out, src=acc)
    return out


def run(MODE=0, REP=50):
    from qwen3_weights import bf16
    k_evac(np.zeros((128, 128), bf16), MODE=MODE, REP=REP)
    return "ok"
