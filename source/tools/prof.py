"""Profile one NEFF on one core with neuron-explorer, all inputs zero (or from <iodir>/<name>.bin when present).
usage: python tools/prof.py <neff> <core> [iodir] -> build/prof/<name>.json (+ summary)"""
import os, sys, json, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import nb
neff, core = os.path.abspath(sys.argv[1]), sys.argv[2]
iodir = sys.argv[3] if len(sys.argv) > 3 else None
name = os.path.basename(neff).replace(".neff", "")
out = os.path.join(nb.LAB, "build", "prof"); os.makedirs(out, exist_ok=True)
tmp = os.path.join(nb.SCRATCH, "prof_" + name); os.makedirs(tmp, exist_ok=True)
info = json.load(open(os.path.join(nb.LAB, "build", name.replace("_nocc", "_nocc"), "info.json"))) if False else None
# tensor names / sizes from the NEFF via the C++ runner's listing
p = subprocess.run([os.path.join(nb.LAB, "host", "ccrun"), neff, tmp, "1", "0"], capture_output=True, text=True,
                   env={**os.environ, "NEURON_RT_VISIBLE_CORES": core, "NEURON_RT_ROOT_COMM_ID": "localhost:46991"})
args = []
for l in p.stdout.splitlines():
    f = l.split()
    if len(f) == 3 and f[0] == "in":
        n, sz = f[1], int(f[2])
        src = os.path.join(iodir, n + ".bin") if iodir else None
        a = np.fromfile(src, np.uint8) if src and os.path.exists(src) else np.zeros(sz, np.uint8)
        fn = os.path.join(tmp, n + ".npy"); np.save(fn, a); args += [n, fn]
env = {**os.environ, "NEURON_RT_VISIBLE_CORES": core}
s = os.path.join(out, name + ".ntff")
subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", s, "--num-exec", "3"] + args, env=env, check=True,
               capture_output=True)
e3 = s.replace(".ntff", "_exec_3.ntff")
with open(os.path.join(out, name + ".sum"), "w") as f:
    subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", e3, "--output-format", "summary-text"], stdout=f, env=env)
with open(os.path.join(out, name + ".json"), "w") as f:
    subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", e3, "--output-format", "json"], stdout=f, env=env, cwd=out)
os.replace(os.path.join(out, "ntff.json"), os.path.join(out, name + ".json"))
print("wrote", os.path.join(out, name + ".json"))
import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)   # /dev/shm is RAM
