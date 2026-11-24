"""numpy fp32 references for Qwen3 pieces."""
import numpy as np
import ml_dtypes
bf16 = ml_dtypes.bfloat16


def rope_tables(tmax, d=128, theta=1e6):
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    f = np.arange(tmax, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)


def rmsnorm(x, w, eps=1e-6):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w.astype(np.float32)


def rope(x, c, s):  # x [..., 128], c/s [64]
    x1, x2 = x[..., :64], x[..., 64:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def attn_decode_ref(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, nq=16, nkv=8, d=128):
    """Returns (out [nq*d], k_new [nkv, d], v_new [nkv*d]) in fp32. kcache [nkv, d, T], vcache [T, nkv*d]."""
    x = qkv.astype(np.float32).reshape(-1)
    q = x[:nq * d].reshape(nq, d); k = x[nq * d:(nq + nkv) * d].reshape(nkv, d); v = x[(nq + nkv) * d:].reshape(nkv, d)
    q = rope(rmsnorm(q, qn_w.reshape(-1)), cos[pos], sin[pos])
    k = rope(rmsnorm(k, kn_w.reshape(-1)), cos[pos], sin[pos])
    qb = q.astype(bf16).astype(np.float32); kb = k.astype(bf16).astype(np.float32)
    K = kcache[:, :, :pos].astype(np.float32)          # [nkv, d, pos]
    V = vcache[:pos].astype(np.float32).reshape(pos, nkv, d)
    out = np.zeros((nq, d), np.float32)
    g = nq // nkv
    for hq in range(nq):
        h = hq // g
        s = np.concatenate([qb[hq] @ K[h], [qb[hq] @ kb[h]]]) / np.sqrt(d)
        p = np.exp(s - s.max()); p /= p.sum()
        vv = np.concatenate([V[:, h, :], v[h][None, :].astype(bf16).astype(np.float32)], 0)
        out[hq] = p @ vv
    return out.reshape(-1), kb, v.reshape(-1)


def bfr(x):
    return np.asarray(x, np.float32).astype(bf16).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def layer_ref(h, W, kc, vc, pos, cos, sin, nq=16, nkv=8, d=128):
    """One Qwen3 decoder layer for one token, fp32 math with bf16 rounding where the kernel rounds.
    h [1024] fp32; W: dict of fp32 numpy weights (HF [out, in]); kc [nkv, d, T], vc [T, nkv*d] (fp32 views).
    Returns (h_out, k_new [nkv, d] (bf16-rounded), v_new [nkv*d] (bf16-rounded))."""
    x1 = bfr(rmsnorm(h, W["ln1"]))
    q = x1 @ W["q"].T; k = x1 @ W["k"].T; v = x1 @ W["v"].T
    q = bfr(rmsnorm(q.reshape(nq, d), W["qn"])); k = bfr(rmsnorm(k.reshape(nkv, d), W["kn"]))
    q = bfr(rope(q, cos[pos], sin[pos])); k = bfr(rope(k, cos[pos], sin[pos])); v = bfr(v)
    K = kc[:, :, :pos]; V = vc[:pos].reshape(pos, nkv, d)
    o = np.zeros((nq, d), np.float32)
    for hq in range(nq):
        hk = hq // (nq // nkv)
        s = np.concatenate([q[hq] @ K[hk], [q[hq] @ k[hk]]]) / np.sqrt(d)
        p = np.exp(s - s.max()); p /= p.sum()
        o[hq] = p @ np.concatenate([V[:, hk, :], v.reshape(nkv, d)[hk][None]], 0)
    o = bfr(o.reshape(-1))
    h2 = h + o @ W["o"].T
    x2 = bfr(rmsnorm(h2, W["ln2"]))
    act = bfr(silu(x2 @ W["gate"].T) * (x2 @ W["up"].T))
    h3 = h2 + act @ W["down"].T
    return h3, k, v.reshape(-1)
