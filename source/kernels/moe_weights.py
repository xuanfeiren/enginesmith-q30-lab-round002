"""Host-side (build/load time) re-layout of Qwen3-MoE weights for kernels/moe_tp.py. Values unchanged.

moe_split(st, cfg, layers, W, out=None) -> per-rank dict {wr, wgu, wd, ln2}. With out=<dir>, each rank's big
arrays are written as raw files <dir>/r<r>/<name>.bin (np.memmap) instead of being kept in memory.
"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from qwen3_weights import bf16, col_layout


def il_tiles(IM, W):
    """per-virtual-core intermediate slice IL = IM / 2W, as ILT chunks of ILP <= 128 rows"""
    C = 2 * W
    assert IM % C == 0, (IM, C)
    IL = IM // C
    ILT = -(-IL // 128)
    assert IL % ILT == 0
    return IL, ILT, IL // ILT


def expert_slabs(gate, up, down, g, IL, ILT, ILP, HT):
    """one expert, one virtual core g -> (gu slab [128, HT*2*ILT*ILP], down slab [ILP, ILT*HT*128])"""
    rows = slice(g * IL, (g + 1) * IL)
    A = np.stack([gate[rows], up[rows]])                       # [2, IL, H]
    A = A.reshape(2, ILT, ILP, HT, 128).transpose(4, 3, 0, 1, 2)  # [p, kt, gu, it, j]
    D = down[:, rows].reshape(HT, 128, ILT, ILP).transpose(3, 2, 0, 1)   # [i, it, nt, j]
    return A.reshape(128, -1), D.reshape(ILP, -1)


def moe_split(st, cfg, layers, W, out=None, threads=12, pad=False):
    """pad=True: one tensor per layer group with each expert's slab [128, F1 + F2] (gate/up, then down zero-padded
    from ILP to 128 partitions) so one DMA fetches an expert (moe_x OPT & 32); wd{g} are dummies."""
    H = cfg["hidden_size"]; HT = H // 128; E = cfg["num_experts"]; IM = cfg["moe_intermediate_size"]
    IL, ILT, ILP = il_tiles(IM, W)
    L = len(layers)
    F1, F2 = HT * 2 * ILT * ILP, ILT * HT * 128
    shapes = dict(wr=((L, 128, HT * E), bf16), wgu=((L, E, 2, 128, F1 + (F2 if pad else 0)), bf16),
                  wd=((L, E, 2, ILP, F2) if not pad else (1, 1, 1, 1, 1), bf16), ln2=((L, 128, HT), np.float32))
    NG = 4                                  # expert weights in 4 layer groups: every tensor < 4 GiB (moe_tp._moe_layer)
    LG = -(-L // NG)
    ranks = []
    for r in range(W):
        d = {}
        for k, (sh, dt) in shapes.items():
            if k in ("wgu", "wd"):
                for gi in range(NG):
                    n = min(L, (gi + 1) * LG) - gi * LG
                    if k == "wd" and pad:
                        n = 0
                    shg = (n,) + sh[1:] if n > 0 else (1, 1, 1, 1, 1)     # empty groups: a dummy, never read
                    assert np.prod(shg) * 2 < 2 ** 32, (k, shg)
                    if out is not None and n > 0:
                        os.makedirs(os.path.join(out, f"r{r}"), exist_ok=True)
                        d[f"{k}{gi}"] = np.memmap(os.path.join(out, f"r{r}", f"{k}{gi}.bin"), dt, "w+", shape=shg)
                    else:
                        d[f"{k}{gi}"] = np.zeros(shg, dt)
            else:
                d[k] = np.empty(sh, dt)
        ranks.append(d)
    g = lambda l, n: st.get(f"model.layers.{l}.{n}")

    def one(li):
        l = layers[li]
        Wr = g(l, "mlp.gate.weight")
        wr = np.ascontiguousarray(Wr.reshape(E, HT, 128).transpose(2, 1, 0)).reshape(128, HT * E)
        ln2 = col_layout(g(l, "post_attention_layernorm.weight")).astype(np.float32)
        for r in range(W):
            ranks[r]["wr"][li] = wr
            ranks[r]["ln2"][li] = ln2
        # all experts of the layer at once (one big transpose per tensor; numpy releases the GIL for it)
        Gt = np.stack([g(l, f"mlp.experts.{e}.gate_proj.weight") for e in range(E)])
        Ut = np.stack([g(l, f"mlp.experts.{e}.up_proj.weight") for e in range(E)])
        A = np.stack([Gt, Ut], 1).reshape(E, 2, W, 2, ILT, ILP, HT, 128)      # [e, gu, r, c, it, j, kt, p]
        A = np.ascontiguousarray(A.transpose(2, 0, 3, 7, 6, 1, 4, 5))          # [r, e, c, p, kt, gu, it, j]
        del Gt, Ut
        Dn = np.stack([g(l, f"mlp.experts.{e}.down_proj.weight") for e in range(E)]).reshape(E, HT, 128, W, 2, ILT, ILP)
        Dn = np.ascontiguousarray(Dn.transpose(3, 0, 4, 6, 5, 1, 2))           # [r, e, c, i, it, nt, j]
        for r in range(W):
            if pad:
                t = ranks[r][f"wgu{li // LG}"]
                t[li % LG, :, :, :, :F1] = A[r].reshape(E, 2, 128, -1)
                t[li % LG, :, :, :ILP, F1:] = Dn[r].reshape(E, 2, ILP, -1)
                t[li % LG, :, :, ILP:, F1:] = 0
            else:
                ranks[r][f"wgu{li // LG}"][li % LG] = A[r].reshape(E, 2, 128, -1)
                ranks[r][f"wd{li // LG}"][li % LG] = Dn[r].reshape(E, 2, ILP, -1)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, range(L)))
    if out is not None:
        for d in ranks:
            for k, v in d.items():
                if isinstance(v, np.memmap):
                    v.flush()
    return ranks, (IL, ILT, ILP)
