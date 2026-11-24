"""Build-time helper: compile an NKI kernel to a NEFF through NKI's documented route only.

build(name, "module:func", cores, *args) starts a fresh Python process with NKI_ARTIFACTS_DIR=build/<name> set
before nki is imported and NEURON_RT_VISIBLE_CORES=<cores> (lab 3 cores 0-15 only); `module.func(*args)` must call a
@nki.jit kernel with numpy arrays (lists of arrays, one per rank, for a multi-rank collective kernel). nki compiles
it, runs it once on the cores (rank r on the r-th visible core) and leaves kernel.neff in the artifacts dir, which is
copied to neff/<name>.neff. Whatever func returns is pickled back (its own self-check results)."""
import os, pickle, shutil, subprocess, sys, time

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# big per-rank weight files (tens of GB): RAM-backed scratch, the root disk writes at ~130 MB/s
SCRATCH = os.environ.get("LAB3_SCRATCH", "/dev/shm/nki-lab3")


def parse_cores(cores):
    out = []
    for part in str(cores).split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def check_cores(cores):
    cs = parse_cores(cores)
    assert cs and all(0 <= c <= 15 for c in cs), f"lab 3 may only use cores 0-15, got {cores}"


def build(name, target, cores, *args, timeout=3600):
    check_cores(cores)
    art = os.path.join(LAB, "build", name)
    shutil.rmtree(art, ignore_errors=True)
    os.makedirs(art)
    env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
           "NEURON_RT_VISIBLE_CORES": str(cores),
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB3_COMM_ID", f"localhost:{47000 + os.getpid() % 2000}")}
    mod, fn = target.split(":")
    res = os.path.join(art, "result.pkl")
    code = (f"import pickle, sys; import {mod} as m; r = m.{fn}(*pickle.loads(bytes.fromhex(sys.argv[1]))); "
            f"pickle.dump(r, open({res!r}, 'wb'))")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code, pickle.dumps(args).hex()], env=env, cwd=art,
                       capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"build {name} failed ({dt:.1f} s):\n{p.stdout[-4000:]}\n{p.stderr[-6000:]}")
    os.makedirs(os.path.join(LAB, "neff"), exist_ok=True)
    dst = os.path.join(LAB, "neff", f"{name}.neff")
    shutil.copy(os.path.join(art, "kernel.neff"), dst)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e6:.2f} MB) in {dt:.1f} s (compile + one run)", flush=True)
    return pickle.load(open(res, "rb")) if os.path.exists(res) else None, dt


def ccrun(neff, per_rank_inputs, outputs, n_exec=1, bench=0, cores="56-57", first=0, stride=1, keep=False):
    """Run a collective NEFF with host/ccrun (C++/libnrt, one process, one thread per rank for the load).
    per_rank_inputs: list (one per rank) of {name: array}. outputs: {name: (shape, dtype)}.
    Returns (list of {name: array} per rank, stdout)."""
    import tempfile
    import numpy as np
    check_cores(cores)
    W = len(per_rank_inputs)
    d = tempfile.mkdtemp(prefix="cc_", dir=os.path.join(LAB, "build"))
    for r, ins in enumerate(per_rank_inputs):
        os.makedirs(os.path.join(d, f"r{r}"))
        for k, v in ins.items():
            np.ascontiguousarray(v).tofile(os.path.join(d, f"r{r}", k + ".bin"))
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": str(cores),
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB3_COMM_ID", f"localhost:{49000 + os.getpid() % 2000}")}
    if not os.path.isabs(neff):
        neff = os.path.join(LAB, "neff", neff + ".neff")
    p = subprocess.run([os.path.join(LAB, "host", "ccrun"), neff, d, str(W), str(n_exec), str(bench), str(first),
                        str(stride)], capture_output=True, text=True, env=env, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"ccrun failed: {p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    res = [{k: np.fromfile(os.path.join(d, f"r{r}", "out_" + k + ".bin"), dtype=dt).reshape(shape)
            for k, (shape, dt) in outputs.items()} for r in range(W)]
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return res, p.stdout
