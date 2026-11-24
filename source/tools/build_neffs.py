"""Build time only: compile every NEFF the engine embeds, through NKI's documented route (NKI_ARTIFACTS_DIR set
before `import nki`, the @nki.jit kernel called with zero numpy arrays of the per-rank shapes, one list entry per
rank; nki compiles it, runs it once on the first W visible cores and leaves kernel.neff).

usage: python tools/build_neffs.py OUT_DIR LANES
  LANES: comma-separated core ranges, each W=4 contiguous logical NeuronCores, e.g. "0-3,4-7"; one build per lane at
  a time. NEFFs are cached in OUT_DIR/cache keyed by the kernel sources, the shapes and the parameters.
Writes OUT_DIR/<name>.neff for every target and OUT_DIR/neffs.txt (one name per line).
"""
import hashlib, json, os, queue, shutil, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDIR = os.path.join(ROOT, "kernels")
PY = sys.executable

# model shapes (Qwen3-30B-A3B), TP=4
W = 4
L, H, E, NQ, NKV, IM, VOCAB = 48, 2048, 128, 32, 4, 768, 151936
HT = H // 128
C = 2 * W
NQL, NKL = NQ // C, max(1, NKV // C)
IL = IM // C
ILT = -(-IL // 128)
ILP = IL // ILT
F1, F2 = HT * 2 * ILT * ILP, ILT * HT * 128
TMAX = 4608
VT = -(-VOCAB // 128)
NVL = -(-VT // C)
NG, LG = 4, 12
DEC_BUCKETS = [512, 1152, 4608]
DEC_KW = dict(SL=1, hw=2, VD=1, ATT=1)
# experiments only: ENGINE_DEC_BUCKETS="512,4608", ENGINE_DEC_KW='{"VD": 1}' (merged into DEC_KW)
if os.environ.get("ENGINE_DEC_BUCKETS"):
    DEC_BUCKETS = [int(x) for x in os.environ["ENGINE_DEC_BUCKETS"].split(",")]
DEC_KW.update(json.loads(os.environ.get("ENGINE_DEC_KW", "{}")))
PF_KW = dict(OPX=1, CMB=1, CSPLIT=1)
PF_KW.update(json.loads(os.environ.get("ENGINE_PF_KW", "{}")))
# per prefill bucket: MCH = MoE all-reduce chunks, BS = MoE block size in slots (prefill_tp); 128 / 1 = the lab's
PF_PER = {"MCH": {256: 1, 1024: 1, 4096: 1}, "BS": {256: 128, 1024: 128, 4096: 128},
          "NCAP": {256: 0, 1024: 0, 4096: 0}, "RSPLIT": {256: 0, 1024: 0, 4096: 1}}
PF_DEFAULT = {"MCH": 1, "BS": 128, "NCAP": 0, "RSPLIT": 0}
for k in PF_PER:
    if os.environ.get("ENGINE_PF_" + k):
        PF_PER[k].update({int(a): int(b) for a, b in (x.split(":") for x in os.environ["ENGINE_PF_" + k].split(","))})


def pf_kw(S):
    kw = dict(PF_KW)
    for k, v in PF_PER.items():
        if v[S] != PF_DEFAULT[k]:
            kw[k] = v[S]
    return kw
PF_BUCKETS = [256, 1024, 4096]

DEC_ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr"] + \
    [f"wgu{i}" for i in range(4)] + [f"wd{i}" for i in range(4)] + \
    ["normf", "wlm", "rot", "cos", "sin", "kcache", "vcache", "rkb"]
PF_ORDER = ["ids", "plen", "toks", "embed", "ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr", "wgu", "wd", "normf", "wlm",
            "rot", "cosT", "sinT", "kcache", "vcache", "rkb", "s_h", "s_q", "s_k", "s_v", "s_o", "s_p", "x_tm", "stag",
            "y_slot", "y_part"]


def dec_spec():
    s = dict(pos=((1, 1), "i32"), plen=((1, 1), "i32"), prompt=((TMAX, 1), "i32"), toks=((TMAX + 1, 1), "i32"),
             embed=((VOCAB, H), "bf16"), ln1=((L, 128, HT), "f32"), qn=((L, 128, 1), "f32"), kn=((L, 128, 1), "f32"),
             wqkv=((L, 2, 128, (NQL + 2 * NKL) * HT * 128), "bf16"), wo=((L, 2, 128, HT * NQL * 128), "bf16"),
             ln2=((L, 128, HT), "f32"), wr=((L, 128, HT * E), "bf16"), normf=((128, HT), "f32"),
             wlm=((2, 128, NVL * HT * 128), "bf16"), rot=((128, 128), "bf16"), cos=((TMAX, 64), "f32"),
             sin=((TMAX, 64), "f32"), kcache=((L, 2, NKL, 128, TMAX), "bf16"),
             vcache=((L, 2, TMAX, NKL * 128), "bf16"), rkb=((128, 1), "f32"))
    for g in range(NG):
        s[f"wgu{g}"] = ((LG, E, 2, 128, F1), "bf16")
        s[f"wd{g}"] = ((LG, E, 2, ILP, F2), "bf16")
    return s


def pf_spec(S):
    d = dec_spec()
    NMAX = 8 * S // 128 + E
    s = dict(ids=((S, 1), "i32"))
    for k in PF_ORDER:
        if k in d:
            sh, dt = d[k]
            if k in ("ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr", "kcache", "vcache"):
                sh = (1,) + sh[1:]
            s[k] = (sh, dt)
    s["wgu"] = ((1, E, 2, 128, F1), "bf16")
    s["wd"] = ((1, E, 2, ILP, F2), "bf16")
    s.update(cosT=((128, TMAX), "f32"), sinT=((128, TMAX), "f32"),
             s_h=((128, HT, S), "f32"), s_q=((2 * NQL, 128, S), "bf16"), s_k=((2 * NKL, 128, S), "bf16"),
             s_v=((2 * NKL, S, 128), "bf16"), s_o=((2 * NQL, 128, S), "bf16"), s_p=((2, 128, HT, S), "f32"),
             x_tm=((2, S + 1, H), "bf16"), stag=((2, E, S + 1), "i32"), y_slot=((2, NMAX * 128, H), "bf16"),
             y_part=((2, S, H), "f32"))
    return s


def targets():
    t = []
    for TB in DEC_BUCKETS:
        t.append((f"dec_b{TB}", "step_moe", "step_moe", DEC_ORDER, dec_spec(),
                  dict(TB=TB, **DEC_KW, NQL=NQL, NKL=NKL, ILT=ILT, ILP=ILP)))
    for S in PF_BUCKETS:
        for stage, st in ((2, "layer"), (1, "emb"), (3, "head")):
            t.append((f"pf{S}_{st}", "prefill_tp", "pf_tp", PF_ORDER, pf_spec(S),
                      dict(STAGE=stage, **(pf_kw(S) if stage == 2 else {}), NQL=NQL, NKL=NKL, ILT=ILT, ILP=ILP)))
    return t


def deps(module):
    """the kernels/ modules `module` imports, transitively (itself included)"""
    import re
    seen, todo = set(), [module]
    while todo:
        m = todo.pop()
        if m in seen or not os.path.exists(os.path.join(KDIR, m + ".py")):
            continue
        seen.add(m)
        for x in re.findall(r"^(?:from|import)\s+(\w+)", open(os.path.join(KDIR, m + ".py")).read(), re.M):
            todo.append(x)
    return sorted(seen)


def src_hash(module):
    h = hashlib.sha256()
    for m in deps(module) + ["kbuild"]:
        h.update(m.encode()); h.update(open(os.path.join(KDIR, m + ".py"), "rb").read())
    for mod in ("nki", "neuronxcc"):
        try:
            import importlib.metadata as md
            h.update((mod + str(md.version(mod if mod != "neuronxcc" else "neuronx-cc"))).encode())
        except Exception:
            pass
    return h.hexdigest()


def build_one(tgt, cores, port, out, cache):
    name, module, kernel, order, spec, kw = tgt
    art = os.path.join(out, "art", name)
    shutil.rmtree(art, ignore_errors=True)
    os.makedirs(art)
    env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
           "NEURON_RT_VISIBLE_CORES": cores, "NEURON_RT_ROOT_COMM_ID": f"localhost:{port}",
           "NEURON_RT_DISABLE_EXECUTION_BARRIER": "1",
           "PYTHONPATH": KDIR + os.pathsep + os.environ.get("PYTHONPATH", "")}
    args = json.dumps([module, kernel, order, {k: [list(v[0]), v[1]] for k, v in spec.items()}, W, kw])
    code = ("import json, sys, kbuild; m, k, o, s, w, kw = json.loads(sys.argv[1]); "
            "kbuild.build(m, k, o, {n: (tuple(v[0]), v[1]) for n, v in s.items()}, w, kw)")
    t0 = time.time()
    p = subprocess.run([PY, "-c", code, args], env=env, cwd=art, capture_output=True, text=True, timeout=3000)
    dt = time.time() - t0
    neff = os.path.join(art, "kernel.neff")
    if p.returncode != 0 or not os.path.exists(neff):
        raise RuntimeError(f"build {name} failed ({dt:.0f} s) on cores {cores}:\n{p.stdout[-4000:]}\n{p.stderr[-6000:]}")
    shutil.copy(neff, cache)
    shutil.rmtree(art, ignore_errors=True)
    print(f"built {name}: {os.path.getsize(cache) / 1e6:.2f} MB in {dt:.0f} s (cores {cores})", flush=True)


def main():
    out, lanes = os.path.abspath(sys.argv[1]), sys.argv[2].split(",")
    os.makedirs(os.path.join(out, "cache"), exist_ok=True)
    todo = queue.Queue()
    names = []
    for tgt in targets():
        name = tgt[0]
        key = hashlib.sha256((src_hash(tgt[1]) + json.dumps([tgt[0], tgt[1], tgt[2], tgt[3], sorted((k, list(v[0]), v[1]) for k, v in tgt[4].items()), sorted(tgt[5].items())])).encode()).hexdigest()[:20]
        cache = os.path.join(out, "cache", f"{name}-{key}.neff")
        names.append((name, cache))
        if not os.path.exists(cache):
            todo.put((tgt, cache))
    print(f"NEFFs: {len(names)}, to build: {todo.qsize()}, lanes {lanes}", flush=True)
    errs = []

    def lane(i, cores):
        while True:
            try:
                tgt, cache = todo.get_nowait()
            except queue.Empty:
                return
            try:
                build_one(tgt, cores, 46000 + (os.getpid() % 1000) * 4 + i, out, cache + ".tmp")
                os.replace(cache + ".tmp", cache)
            except Exception as e:
                errs.append(str(e))
                print(str(e), file=sys.stderr, flush=True)
                return

    th = [threading.Thread(target=lane, args=(i, c)) for i, c in enumerate(lanes)]
    for x in th: x.start()
    for x in th: x.join()
    if errs:
        sys.exit(1)
    with open(os.path.join(out, "neffs.txt"), "w") as f:
        for name, cache in names:
            dst = os.path.join(out, name + ".neff")
            shutil.copy(cache, dst)
            f.write(name + "\n")
    print("all NEFFs ready", flush=True)


if __name__ == "__main__":
    main()
