import sys, numpy as np, nb
modes = [(0, 1), (1, 1), (2, 1), (3, 1), (0, 4)]
for MODE, SPLIT in modes:
    name = f"p2_m{MODE}_s{SPLIT}"
    r, dt = nb.build(name, "p2_gather_bw:run", sys.argv[1], MODE, 48, 16, 4608, SPLIT)
    idx = np.load("/tmp/p2_idx.npy")
    _, so = nb.ccrun(name, [{"idx": idx}], {}, n_exec=1, bench=50, cores=sys.argv[1])
    us = float([l for l in so.splitlines() if "BENCH" in l][0].split()[2])
    print(f"MODE {MODE} SPLIT {SPLIT}: err {r['err']:.3g}  {us:.1f} us/exec  {r['bytes_per_core']/us/1e3:.1f} GB/s per logical core", flush=True)
