"""Problem 3: MoE prefill half-layer (kernels/moe_prefill.py) for S tokens on W logical NeuronCores, against HF
transformers fp32 on the CPU on real tokens (tests/hf_ref.py TAG=long4k: 4,096-token capture of every layer).

usage: python tests/test_moe_prefill.py <W> <S> [layers=0,24,47]
  For each layer: x = RMSNorm(HF's residual entering the MoE half) * ln2 (host, fp32 -> bf16, feature-major), the
  kernel's MoE output vs HF's Qwen3MoeSparseMoeBlock output; then a queued benchmark (same inputs).
env CORES (default 0..W-1), NOBUILD=1, BENCH (executions, default 10), TAG (default long4k).
"""
import sys, os, json, time, shutil, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import nb, kbuild
from qwen3_weights import SafeTensors, bf16
from moe_weights import moe_split

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-30B-A3B"))
ORDER = ["xT", "wr", "wgu", "wd", "x_tm", "stag", "y_slot", "y_part"]

if __name__ == "__main__":
    W, S = int(sys.argv[1]), int(sys.argv[2])
    layers = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,24,47").split(",")]
    cores = os.environ.get("CORES", f"0-{W - 1}" if W > 1 else "0")
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    H = cfg["hidden_size"]; HT = H // 128; E = cfg["num_experts"]
    NMAX = 8 * S // 128 + E
    st = SafeTensors(MODEL)
    t0 = time.time()
    per_layer = []
    for l in layers:
        ranks, (IL, ILT, ILP) = moe_split(st, cfg, [l], W)
        per_layer.append([{"wr": r["wr"][0], "wgu": r["wgu0"][0], "wd": r["wd0"][0], "ln2": r["ln2"][0]} for r in ranks])
    print(f"split {len(layers)} layers for W={W} in {time.time() - t0:.0f} s", flush=True)
    scratch = dict(x_tm=np.zeros((2, S + 1, H), bf16), stag=np.zeros((2, E, S + 1), np.int32),
                   y_slot=np.zeros((2, NMAX * 128, H), np.float32 if os.environ.get('YF32') else bf16), y_part=np.zeros((2, S, H), np.float32))
    xT0 = np.zeros((128, HT, S), bf16)
    name = f"moepf_w{W}_s{S}" + os.environ.get("VARIANT", "")
    WB = W
    if os.environ.get("NOCC"):   # rank 0's shard built with W=1: timing / profiling without communication
        WB, cores, name = 1, cores.split("-")[0], name + "_nocc"
        per_layer = [pl[:1] for pl in per_layer]
    if not os.environ.get("NOBUILD"):
        spec = kbuild.spec_of({**per_layer[0][0], **scratch, "xT": xT0}, ORDER)
        kw = dict(ILT=ILT, ILP=ILP)
        kw.update(json.loads(os.environ.get("KW", "{}")))
        nb.build(name, "kbuild:build", cores, os.environ.get("MOD", "moe_prefill"), "moe_prefill", ORDER, spec, WB, kw)
    ref = np.load(os.path.join(nb.LAB, "build", f"hf_ref_{os.environ.get('TAG', 'long4k')}.npz"))
    d = os.path.join(nb.SCRATCH, name)   # one scratch dir per NEFF variant (concurrent runs)
    shutil.rmtree(d, ignore_errors=True)
    for r in range(WB):
        os.makedirs(os.path.join(d, f"r{r}"))
        for k, v in scratch.items():
            v.tofile(os.path.join(d, f"r{r}", k + ".bin"))
    for i, l in enumerate(layers):
        h = ref["hpre"][l, :S].astype(np.float32)
        x = h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-6) * per_layer[i][0]["ln2"].T.reshape(-1)
        xT = np.ascontiguousarray(x.astype(bf16).reshape(S, HT, 128).transpose(2, 1, 0))
        for r in range(WB):
            xT.tofile(os.path.join(d, f"r{r}", f"xT.{i}.bin"))
            for k in ("wr", "wgu", "wd"):
                np.ascontiguousarray(per_layer[i][r][k]).tofile(os.path.join(d, f"r{r}", f"{k}.{i}.bin"))
                if i == 0:
                    np.ascontiguousarray(per_layer[i][r][k]).tofile(os.path.join(d, f"r{r}", f"{k}.bin"))
        if i == 0:
            for r in range(WB):
                xT.tofile(os.path.join(d, f"r{r}", "xT.bin"))
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": cores, "CC_NIN": str(len(layers)), "CC_REPS": "3",
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB3_COMM_ID", f"localhost:{51000 + os.getpid() % 2000}")}
    p = subprocess.run([os.path.join(nb.LAB, "host", "ccrun"), os.path.join(nb.LAB, "neff", name + ".neff"), d, str(WB),
                        "1", str(int(os.environ.get("BENCH", 10)))], capture_output=True, text=True, env=env, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"ccrun failed: {p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    ok = True
    for i, l in enumerate(layers if WB == W else []):
        ys = [np.fromfile(os.path.join(d, f"r{r}", f"out_output_0.{i}.bin"), np.float32).reshape(S, H) for r in range(W)]
        same = all((y == ys[0]).all() for y in ys)
        exp = ref["moe"][l, :S]
        err = np.abs(ys[0] - exp)
        rel = float(err.max() / np.abs(exp).max())
        tok_rel = err.max(1) / np.abs(exp).max(1)
        kix = np.fromfile(os.path.join(d, "r0", f"out_output_1.{i}.bin"), np.int32).reshape(S, 8)
        same_set = np.array([set(kix[t].tolist()) == set(ref["idx"][l, t].tolist()) for t in range(S)])
        lg = ref["logits"][l, :S]; srt = -np.sort(-lg, 1); gap89 = srt[:, 7] - srt[:, 8]
        diff = np.nonzero(~same_set)[0]
        if len(diff):
            print(f"   top-8 set differs from HF for {len(diff)}/{S} tokens; their HF 8th-9th router logit gaps: max "
                  f"{gap89[diff].max():.2e} (median over all tokens {np.median(gap89):.2e})")
        bad = int((tok_rel[same_set] > 0.03).sum())     # same experts but a wrong output: a real error
        rel_same = float(err[same_set].max() / np.abs(exp).max())
        print(f"CHECK layer {l} S={S}: top-8 set = HF for {int(same_set.sum())}/{S} tokens; on those, max err / max|moe| "
              f"= {rel_same:.3e}, per-token rel err median {np.median(tok_rel[same_set]):.2e}, max "
              f"{tok_rel[same_set].max():.2e}; ranks identical: {same}")
        ok = ok and same and bad == 0 and same_set.sum() >= S - max(2, S // 50)
    b = [l for l in p.stdout.splitlines() if "BENCH" in l]
    if b:
        us = float(b[0].split()[2])
        print(f"BENCH W={W}{' shard alone' if WB != W else ''} S={S}: {us / 1e3:.3f} ms per MoE half-layer ({b[0]})")
    shutil.rmtree(d, ignore_errors=True)
    print("PASS" if ok else "FAIL")
