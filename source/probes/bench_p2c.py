# DMA rate vs bytes per partition row (descriptor size) for token-dependent expert-slab DMAs (probe p2, MODE 0)
import sys, numpy as np, nb
core = sys.argv[1]
for F in (1024, 2048, 3072, 4608, 8192):
    L = max(8, int(48 * 4608 / F)) if F < 4608 else 48
    name = f"p2_F{F}"
    r, dt = nb.build(name, "p2_gather_bw:run", core, 0, L, 16, F, 1)
    idx = np.load("/tmp/p2_idx.npy")
    _, so = nb.ccrun(name, [{"idx": idx}], {}, n_exec=1, bench=50, cores=core)
    us = float([l for l in so.splitlines() if "BENCH" in l][0].split()[2])
    print(f"row {F * 2} B, {L} x 8 slabs of {128 * F * 2 / 1e3:.0f} KB per physical core: err {r['err']:.2g} {us:.1f} us "
          f"{r['bytes_per_core'] / us / 1e3:.0f} GB/s per logical core", flush=True)
