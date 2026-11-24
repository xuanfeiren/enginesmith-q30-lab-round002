# all-reduce chain vs concurrent weight streaming (probe p4), W=4 on the given cores
import sys, os, nb
cores = sys.argv[1]; W = 4
res = {}
for (N, MB, Q) in [(33, 0, 0), (1, 128, 0), (33, 128, 0), (1, 128, 1), (33, 128, 1), (1, 128, 2), (33, 128, 2), (1, 0, 0)]:
    name = f"p4_n{N}_m{MB}_q{Q}"
    nb.build(name, "p4_cc_contention:run", cores, W, N, MB, Q)
    _, so = nb.ccrun(name, [{} for _ in range(W)], {}, n_exec=1, bench=100, cores=cores)
    us = float([l for l in so.splitlines() if "BENCH" in l][0].split()[2])
    res[(N, MB, Q)] = us
    print(f"N={N:2d} all-reduces, stream {MB:3d} MB/core on queue {Q}: {us:8.1f} us/exec", flush=True)
