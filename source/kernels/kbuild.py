"""Build target for nb.build: compile <module>.<kernel> for given per-rank input shapes (documented route: called
with numpy arrays, or lists of numpy arrays, one per rank, for W > 1; zero-filled, only shapes matter)."""
import importlib
import numpy as np
import ml_dtypes
from nki.collectives import ReplicaGroup

DT = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}


def build(module, kernel, order, spec, W, kw):
    k = getattr(importlib.import_module(module), kernel)
    mk = lambda: [np.zeros(spec[n][0], DT[spec[n][1]]) for n in order]
    if W == 1:
        k[2](*mk(), rg=None, W=1, **kw)
    else:
        per = [mk() for _ in range(W)]
        k[2](*[[per[r][i] for r in range(W)] for i in range(len(order))], rg=ReplicaGroup([list(range(W))]), W=W, **kw)
    return "ok"


DTN = {np.dtype(ml_dtypes.bfloat16): "bf16", np.dtype(np.float32): "f32", np.dtype(np.int32): "i32"}


def spec_of(d, order):
    return {k: (tuple(d[k].shape), DTN[np.dtype(d[k].dtype)]) for k in order}
