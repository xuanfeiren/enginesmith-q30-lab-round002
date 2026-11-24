# DMA rate of token-dependent expert-slab DMAs, one physical core (LNC=1) vs both physical cores (LNC=2)
import sys, numpy as np, nb
core = sys.argv[1]
for LNC in (1, 2):
    for FULL in (0, 1):
        for F in (3072, 4608):
            name = f"p2d_l{LNC}_f{FULL}_F{F}"
            L = 48 if F == 4608 else 72
            r, dt = nb.build(name, "p2_gather_bw:run", core, 0, L, 16, F, 1, 0, FULL, LNC)
            idx = np.load("/tmp/p2_idx.npy")
            _, so = nb.ccrun(name, [{"idx": idx}], {}, n_exec=1, bench=50, cores=core)
            us = float([l for l in so.splitlines() if "BENCH" in l][0].split()[2])
            print(f"LNC={LNC} FULL={FULL} row {F*2} B: err {r['err']:.2g} {us:.1f} us -> {r['bytes_per_core'] / us / 1e3:.0f} GB/s "
                  f"per logical core ({r['bytes_per_core'] / us / 1e3 / LNC:.0f} per physical core)", flush=True)
