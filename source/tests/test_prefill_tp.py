"""Problem 3, whole model: layered TP prefill (kernels/prefill_tp.py: emb, layer x 48, head) + the TP decode step
(kernels/step_moe.py) end to end from C++ (host/tpgen), tokens vs HF transformers fp32 (tests/hf_ref.py captures).

usage: python tests/test_prefill_tp.py <W> <S bucket> [G=24]
env TAG (hf ref tag: its prompt is fed; default "default", 12 tokens), DEC (decode NEFF name, default
smoe_w<W>_L48_b512, built by tests/test_step_moe.py), CORES, NOBUILD=1, REPS (default 2).
"""
import sys, os, json, time, shutil, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb, kbuild
from qwen3_weights import SafeTensors, bf16
from test_step_moe import split_step, write_inputs, TMAX, MODEL

ORDER = ["ids", "plen", "toks", "embed", "ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr", "wgu", "wd", "normf", "wlm", "rot",
         "cosT", "sinT", "kcache", "vcache", "rkb", "s_h", "s_q", "s_k", "s_v", "s_o", "s_p", "x_tm", "stag", "y_slot",
         "y_part"]
PER_LAYER = ("ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr", "kcache", "vcache")

if __name__ == "__main__":
    W, S = int(sys.argv[1]), int(sys.argv[2])
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    cores = os.environ.get("CORES", f"0-{W - 1}" if W > 1 else "0")
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    L = cfg["num_hidden_layers"]; H = cfg["hidden_size"]; HT = H // 128; E = cfg["num_experts"]
    ref = dict(np.load(os.path.join(nb.LAB, "build", f"hf_ref_{os.environ.get('TAG', 'default')}.npz")))
    ids = ref["ids"].tolist()
    P = len(ids)
    assert P <= S, (P, S)
    G = min(G, len(ref["full"]) - P)
    t0 = time.time()
    wdir = os.path.join(nb.SCRATCH, f"step_w{W}_L{L}" + os.environ.get("WTAG", ""))
    import atexit
    if not os.environ.get("KEEP"):   # /dev/shm is RAM: 16-75 GB per run (it once filled all 1000 GB)
        atexit.register(shutil.rmtree, wdir, True)
    ranks, (nq, nk, ILT, ILP) = split_step(SafeTensors(MODEL), cfg, L, ids, W, wdir)
    print(f"split W={W} in {time.time() - t0:.0f} s", flush=True)
    cos, sin = ranks[0]["cos"], ranks[0]["sin"]
    cosT = np.ascontiguousarray(np.concatenate([cos, cos], 1).T).astype(np.float32)
    sinT = np.ascontiguousarray(np.concatenate([sin, sin], 1).T).astype(np.float32)
    idp = np.zeros((S, 1), np.int32); idp[:P, 0] = ids
    NMAX = 8 * S // 128 + E
    scr = dict(s_h=np.zeros((128, HT, S), np.float32), s_q=np.zeros((2 * nq, 128, S), bf16),
               s_k=np.zeros((2 * nk, 128, S), bf16), s_v=np.zeros((2 * nk, S, 128), bf16),
               s_o=np.zeros((2 * nq, 128, S), bf16), s_p=np.zeros((2, 128, HT, S), np.float32),
               x_tm=np.zeros((2, S + 1, H), bf16), stag=np.zeros((2, E, S + 1), np.int32),
               y_slot=np.zeros((2, NMAX * 128, H), bf16), y_part=np.zeros((2, S, H), np.float32))
    extra = dict(ids=idp, cosT=cosT, sinT=sinT)
    names = []
    for stage, st in ((1, "emb"), (2, "layer"), (3, "head")):
        nm = f"pftp_w{W}_s{S}_{st}"
        names.append(os.path.join(nb.LAB, "neff", nm + ".neff"))
        if os.environ.get("NOBUILD"):
            continue
        d0 = ranks[0]
        arr = {}
        for k in ORDER:
            if k in scr: arr[k] = scr[k]
            elif k in extra: arr[k] = extra[k]
            elif k == "wgu": arr[k] = d0["wgu0"][0:1]
            elif k == "wd": arr[k] = d0["wd0"][0:1]
            elif k in PER_LAYER: arr[k] = d0[k][0:1]
            else: arr[k] = d0[k]
        spec = kbuild.spec_of(arr, ORDER)
        t0 = time.time()
        nb.build(nm, "kbuild:build", cores, "prefill_tp", "pf_tp", ORDER, spec, W,
                 dict(STAGE=stage, NQL=nq, NKL=nk, ILT=ILT, ILP=ILP))
    dec = os.path.join(nb.LAB, "neff", os.environ.get("DEC", f"smoe_w{W}_L{L}_b512") + ".neff")
    d = os.path.join(wdir, "io")
    write_inputs(d, ranks)
    for r in range(W):
        for k, v in extra.items():
            np.ascontiguousarray(v).tofile(os.path.join(d, f"r{r}", k + ".bin"))
    del ranks
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": cores,
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB3_COMM_ID", f"localhost:{51000 + os.getpid() % 2000}")}
    p = subprocess.run([os.path.join(nb.LAB, "host", "tpgen"), ",".join(names), dec, d, str(W), str(G - 1),
                        os.environ.get("REPS", "2")], capture_output=True, text=True, env=env, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError(f"tpgen failed: {p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    for l in p.stdout.splitlines():
        print(l)
    toks = [np.fromfile(os.path.join(d, f"r{r}", "out_toks.bin"), np.int32) for r in range(W)]
    same_ranks = all((t == toks[0]).all() for t in toks)
    got = toks[0][P:P + G].tolist()
    full = ref["full"].tolist()
    exp = full[P:P + G]
    same = next((i for i in range(G) if got[i] != exp[i]), G)
    print(f"tokens identical on all ranks: {same_ranks}; first generated token {got[0]} (HF fp32 {exp[0]}); "
          f"generated: first {same}/{G} identical to HF fp32 greedy")
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    print("kernel text:", repr(tok.decode(got)))
    ok = same_ranks and same >= min(G, 8)
    if same < G:
        i = P - 1 + same
        print(f"   divergence at gen {same}: kernel {got[same]} ref {exp[same]}, HF top-2 logit gap {ref['top2gap'][i]:.4f}")
        ok = ok or (same_ranks and same > 0 and ref['top2gap'][i] < 0.1)
    print("PASS" if ok else "FAIL")
