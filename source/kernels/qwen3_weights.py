"""Minimal safetensors reader (numpy only) + the re-layouts the lab kernels expect."""
import json, os, struct
import numpy as np
import ml_dtypes
bf16 = ml_dtypes.bfloat16


class SafeTensors:
    def __init__(self, model_dir):
        self.files = {}
        self.index = {}
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith(".safetensors"):
                p = os.path.join(model_dir, fn)
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(n))
                mm = np.memmap(p, dtype=np.uint8, mode="r")
                for k, v in hdr.items():
                    if k == "__metadata__":
                        continue
                    self.index[k] = (mm, 8 + n, v)

    def get(self, name):
        mm, base, v = self.index[name]
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16", v["dtype"]
        return np.frombuffer(mm[base + a:base + b], dtype=np.uint16).view(bf16).reshape(v["shape"])


def ws_layout(W):
    """HF weight W [N, K] (out, in) -> weight-stationary tile layout [128, NT*KT*128]:
    out[p, ((nt*KT)+kt)*128 + j] = W[nt*128 + j, kt*128 + p]. Each 128x128 tile (nt, kt) is the stationary
    operand [128 k, 128 n]; the tiles of one output tile nt are contiguous per partition."""
    N, K = W.shape
    NT, KT = N // 128, K // 128
    return np.ascontiguousarray(W.reshape(NT, 128, KT, 128).transpose(3, 0, 2, 1)).reshape(128, NT * KT * 128)


def col_layout(v):
    """vector [K] -> [128, K/128] with out[p, kt] = v[kt*128 + p]."""
    return np.ascontiguousarray(np.asarray(v).reshape(-1, 128).T)


def rope_rot_matrix():
    """R [128,128] with (R^T @ x)[m] = rotate_half(x)[m] = -x[m+64] (m<64), x[m-64] (m>=64)."""
    R = np.zeros((128, 128), np.float32)
    for m in range(64):
        R[m + 64, m] = -1.0
        R[m, m + 64] = 1.0
    return R
