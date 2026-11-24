import sys, nb
core = sys.argv[1]
names = {0: "down [96,128] x [96,1], column-major (decode kernel)", 1: "down, expert-major, PSUM tile per expert",
         2: "gate/up [128,96] x [128,1]", 3: "[128,128] x [128,1] (lab 1 GEMV tile)", 4: "down x-stationary [96,1] x [96,512]",
         5: "gate/up x-stationary [128,1] x [128,192]", 6: "[128,64] x [128,1]", 7: "[128,128] at offsets k*96"}
NMM = {0: 128, 1: 128, 2: 256, 3: 256, 4: 32, 5: 128, 6: 256, 7: 256}
for MODE in [int(v) for v in sys.argv[2].split(",")]:
    nm = f"p3_m{MODE}"
    us = {}
    for REP in (10, 50):
        nb.build(nm + f"_r{REP}", "p3_pe_rate:run", core, MODE, REP)
        _, so = nb.ccrun(nm + f"_r{REP}", [{}], {}, n_exec=1, bench=50, cores=core)
        us[REP] = float([l for l in so.splitlines() if "BENCH" in l][0].split()[2])
    per = (us[50] - us[10]) / 40
    n = NMM[MODE]
    print(f"MODE {MODE} {names[MODE]}: {per:.2f} us per rep of {n} matmuls -> {per / n * 1e3:.1f} ns per matmul", flush=True)
