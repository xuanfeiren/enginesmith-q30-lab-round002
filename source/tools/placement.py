"""Expert placement for batch-1 decode from real routing (HF fp32 router decisions captured by tests/hf_ref.py).
Per (token, layer): pure TP over W ranks streams 8/W experts' worth per rank (always balanced); EP degree D (experts
e -> group e*D//E, TP W/D inside a group) streams max_group(count) / (W/D) experts' worth on the slowest rank.
Prints the mean and p99 of the slowest rank's load in expert-equivalents (lower = faster; bytes of one expert
= 3 x 768 x 2048 x 2 B = 9.44 MB)."""
import sys, numpy as np
ref = np.load(sys.argv[1] if len(sys.argv) > 1 else "build/hf_ref_default.npz")
idx = ref["idx"].reshape(-1, 8)        # [L*T, 8]
E = 128
print(f"{idx.shape[0]} (layer, token) routing decisions")
for W in (4, 8, 16):
    print(f"W={W}: pure TP: {8 / W:.2f} experts per rank (every token)")
    D = 2
    while D <= W:
        for name, perm in (("contiguous", np.arange(E)), ("strided", (np.arange(E) % D) * (E // D) + np.arange(E) // D)):
            grp = (perm[idx] * D) // E
            mx = np.stack([(grp == g).sum(1) for g in range(D)], 1).max(1)
            load = mx / (W // D)
            print(f"   EP={D:2d} x TP={W // D:2d} ({name:10s}): slowest rank {load.mean():.2f} mean, "
                  f"{np.percentile(load, 99):.2f} p99 experts -> {load.mean() / (8 / W):.2f}x pure TP")
        D *= 2
