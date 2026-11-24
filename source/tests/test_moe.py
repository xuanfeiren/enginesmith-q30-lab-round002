"""Problem 1: the MoE half-layer of Qwen3-30B-A3B decode (kernels/moe_tp.py moe_block) on W logical NeuronCores,
against HF transformers fp32 on the CPU on real tokens (tests/hf_ref.py captures every layer's MoE input/output).

usage: python tests/test_moe.py <W> [L=48] [NTOK=6]
  check : CHAIN=0 NEFF; for NTOK real token positions t: hin[l] = HF's residual entering layer l's MoE half;
          compare the kernel's h + MoE(h) with HF's, per layer; routing (top-8 set, weights) vs HF's router.
  bench : CHAIN=1 NEFF (48 dependent MoE layers), queued executions, us/exec.
env CORES (default 0..W-1), NOBUILD=1, NOCHECK=1, NOBENCH=1, TAG (hf ref), KEEP=1.
"""
import sys, os, json, time, shutil, subprocess, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import nb, kbuild
from qwen3_weights import SafeTensors, col_layout
from moe_weights import moe_split

ORDER = ["hin", "ln2", "wr"] + [f"wgu{i}" for i in range(4)] + [f"wd{i}" for i in range(4)]
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-30B-A3B"))


def write_inputs(d, ranks, extra):
    for r, ins in enumerate(ranks):
        os.makedirs(os.path.join(d, f"r{r}"), exist_ok=True)
        for k, v in {**ins, **extra}.items():
            p = os.path.join(d, f"r{r}", k + ".bin")
            if isinstance(v, np.memmap):
                if os.path.abspath(v.filename) != os.path.abspath(p):
                    os.symlink(v.filename, p)
            else:
                np.ascontiguousarray(v).tofile(p)


def run_cc(neff, d, W, cores, nin=0, bench=0):
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": cores, "CC_REPS": "3",
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB3_COMM_ID", f"localhost:{51000 + os.getpid() % 2000}")}
    if nin:
        env["CC_NIN"] = str(nin)
    first = 0
    p = subprocess.run([os.path.join(nb.LAB, "host", "ccrun"), neff, d, str(W), "1", str(bench), str(first)],
                       capture_output=True, text=True, env=env, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError(f"ccrun failed: {p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    return p.stdout


if __name__ == "__main__":
    W = int(sys.argv[1])
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    cores = os.environ.get("CORES", f"0-{W - 1}" if W > 1 else "0")
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    HT = cfg["hidden_size"] // 128
    t0 = time.time()
    wdir = os.path.join(nb.SCRATCH, f"moe_w{W}_L{L}" + os.environ.get("WTAG", ""))   # one per concurrent run
    import atexit
    if not os.environ.get("KEEP"):   # /dev/shm is RAM: 16-75 GB per run (it once filled all 1000 GB)
        atexit.register(shutil.rmtree, wdir, True)
    ranks, (IL, ILT, ILP) = moe_split(SafeTensors(MODEL), cfg, list(range(L)), W, out=wdir, pad=bool(os.environ.get("PAD")))
    print(f"split W={W} L={L}: IL={IL} as {ILT} x {ILP} rows per virtual core, per rank "
          f"{sum(v.nbytes for v in ranks[0].values()) / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    hin0 = np.zeros((L, 128, HT), np.float32)
    names = {}
    WB = W
    if os.environ.get("NOCC"):   # rank 0's shard built with W=1 and run alone: the MoE time without communication
        WB, ranks, cores = 1, ranks[:1], cores.split("-")[0]
        os.environ["NOCHECK"] = "1"
    for chain in (0, 1):
        if chain == 0 and os.environ.get("NOCHECK"): continue
        if chain == 1 and os.environ.get("NOBENCH"): continue
        name = f"moe_w{W}_L{L}_c{chain}" + os.environ.get("VARIANT", "") + ("_nocc" if WB != W else "")
        names[chain] = name
        if not os.environ.get("NOBUILD"):
            spec = kbuild.spec_of({**ranks[0], "hin": hin0}, ORDER)
            kw = dict(ILT=ILT, ILP=ILP, CHAIN=chain)
            kw.update(json.loads(os.environ.get("KW", "{}")))
            _, dt = nb.build(name, "kbuild:build", cores, os.environ.get("MOD", "moe_tp"), "moe_block", ORDER, spec, WB, kw)
    d = os.path.join(wdir, "io")
    shutil.rmtree(d, ignore_errors=True)
    write_inputs(d, ranks, {"hin": hin0})
    ok = True
    if 0 in names:
        ref = dict(np.load(os.path.join(nb.LAB, "build", "hf_ref_default.npz")))
        T = ref["hpre"].shape[1]
        toks = np.linspace(0, T - 1, NTOK).round().astype(int).tolist()
        for i, t in enumerate(toks):
            hin = np.stack([col_layout(ref["hpre"][l, t]) for l in range(L)]).astype(np.float32)
            for r in range(W):
                hin.tofile(os.path.join(d, f"r{r}", f"hin.{i}.bin"))
        so = run_cc(os.path.join(nb.LAB, "neff", names[0] + ".neff"), d, W, cores, nin=len(toks))
        print([l for l in so.splitlines() if "upload" in l or "nrt_load" in l])
        worst, worst_same, same_set, wmax, n_lay = 0.0, 0.0, 0, 0.0, 0
        for i, t in enumerate(toks):
            outs = [np.fromfile(os.path.join(d, f"r{r}", f"out_output_0.{i}.bin"), np.float32).reshape(L, 128, HT) for r in range(W)]
            idxs = [np.fromfile(os.path.join(d, f"r{r}", f"out_output_1.{i}.bin"), np.int32).reshape(L, 8) for r in range(W)]
            ws = np.fromfile(os.path.join(d, "r0", f"out_output_2.{i}.bin"), np.float32).reshape(L, 8)
            assert all((o == outs[0]).all() for o in outs) and all((x == idxs[0]).all() for x in idxs), "ranks differ"
            for l in range(L):
                got = outs[0][l].T.reshape(-1)          # column layout -> vector
                exp = ref["hpre"][l, t] + ref["moe"][l, t]
                moe_err = np.abs((got - ref["hpre"][l, t]) - ref["moe"][l, t]).max() / np.abs(ref["moe"][l, t]).max()
                worst = max(worst, float(moe_err))
                hs, ks = set(ref["idx"][l, t].tolist()), set(idxs[0][l].tolist())
                same = hs == ks
                same_set += same
                n_lay += 1
                if same:
                    worst_same = max(worst_same, float(moe_err))
                    wk = dict(zip(idxs[0][l].tolist(), ws[l].tolist()))
                    wmax = max(wmax, max(abs(wk[e] - w) for e, w in zip(ref["idx"][l, t].tolist(), ref["w"][l, t].tolist())))
                else:
                    lg = ref["logits"][l, t]; srt = np.sort(lg)[::-1]
                    print(f"  tok {t} layer {l}: top-8 set differs (HF 8th/9th logit gap {srt[7] - srt[8]:.2e}); "
                          f"HF-only {sorted(hs - ks)} kernel-only {sorted(ks - hs)}; MoE rel err {moe_err:.3e}")
        print(f"CHECK {len(toks)} tokens x {L} layers: top-8 set identical to HF in {same_set}/{n_lay}; routing weight "
              f"max abs err {wmax:.2e}; MoE output max err / max|MoE| = {worst_same:.3e} where the set is identical "
              f"({worst:.3e} over all)")
        ok = worst_same < 0.03 and same_set >= n_lay - max(2, n_lay // 50)
    if 1 in names:
        so = run_cc(os.path.join(nb.LAB, "neff", names[1] + ".neff"), d, WB, cores, bench=int(os.environ.get("BENCH", 30)))
        b = [l for l in so.splitlines() if "BENCH" in l][0]
        us = float(b.split()[2])
        act = L * 8 * 3 * cfg["moe_intermediate_size"] * cfg["hidden_size"] * 2 + L * 128 * cfg["hidden_size"] * 2 * 2 * W
        print(f"BENCH W={W}{' (shard alone, no communication)' if WB != W else ''} L={L} chain: {us:.1f} us/exec = {us / L:.1f} us per MoE layer; streamed per logical core "
              f"{act / W / 1e6:.1f} MB -> {act / W / us / 1e3:.0f} GB/s  ({b})")
    if not os.environ.get("KEEP"):
        shutil.rmtree(d, ignore_errors=True)
    print("PASS" if ok else "FAIL")
