import sys, os, subprocess, numpy as np, nb
name = "p2_m0_full"
r, dt = nb.build(name, "p2_gather_bw:run", "1", 0, 48, 16, 4608, 1, 0, 1)
print("full-consumption err", r["err"])
idx = np.load("/tmp/p2_idx.npy")
_, so = nb.ccrun(name, [{"idx": idx}], {}, n_exec=1, bench=50, cores="1")
us = float([l for l in so.splitlines() if "BENCH" in l][0].split()[2]); print(f"alone on core 1: {us:.1f} us {r['bytes_per_core']/us/1e3:.1f} GB/s")
# 4 concurrent processes, cores 0-3 (one device)
import tempfile
d = tempfile.mkdtemp(dir=os.path.join(nb.LAB, "build")); os.makedirs(d + "/r0"); idx.tofile(d + "/r0/idx.bin")
for nm in [name, "p2_m0_s1"]:
    ps = [subprocess.Popen([nb.LAB + "/host/ccrun", nb.LAB + f"/neff/{nm}.neff", d, "1", "1", "200"], env={**os.environ, "NEURON_RT_VISIBLE_CORES": str(c), "NEURON_RT_ROOT_COMM_ID": f"localhost:{46100+c}"}, stdout=subprocess.PIPE, text=True) for c in range(4)]
    for c, p in enumerate(ps):
        out = p.communicate()[0]
        us = float([l for l in out.splitlines() if "BENCH" in l][0].split()[2])
        print(f"{nm} concurrent x4, core {c}: {us:.1f} us {r['bytes_per_core']/us/1e3:.1f} GB/s", flush=True)
