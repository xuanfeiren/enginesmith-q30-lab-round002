"""Problem 2: the whole Qwen3-30B-A3B decode step (kernels/step_moe.py) on W logical NeuronCores, as a chain of
queued executions from C++ (host/tpchain), against HF transformers fp32 on the CPU (tests/hf_ref.py output).

usage: python tests/test_step_moe.py <W> [G=24] [L=48]
env CORES (default 0..W-1), TB (512 attention bucket), PF (QKV slabs prefetched across the MoE), NOBUILD=1, REPS,
    TAG (hf ref tag, default "default"), NOREF=1, NOMOE=1 (attention-only timing decomposition), NOCC=1 (rank 0's
    shard built with W=1: no communication; tokens meaningless).
"""
import sys, os, json, time, shutil, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import nb, kbuild
from ref_qwen3 import rope_tables
from qwen3_weights import SafeTensors, ws_layout, col_layout, rope_rot_matrix, bf16
from moe_weights import moe_split

TMAX = 4608
ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr"] + \
    [f"wgu{i}" for i in range(4)] + [f"wd{i}" for i in range(4)] + ["normf", "wlm", "rot", "cos", "sin", "kcache", "vcache", "rkb"]
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-30B-A3B"))


def split_step(st, cfg, L, ids, W, wdir):
    H = cfg["hidden_size"]; HT = H // 128; NQ = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
    C = 2 * W
    assert NQ % C == 0 and (NKV % C == 0 or C % NKV == 0), (NQ, NKV, C)
    nq, nk = NQ // C, max(1, NKV // C)
    g = lambda l, n: st.get(f"model.layers.{l}.{n}")
    VT = -(-cfg["vocab_size"] // 128)
    NVL = -(-VT // C)
    lm = ws_layout(st.get("lm_head.weight")).reshape(128, VT, HT, 128)
    lm = np.concatenate([lm, np.zeros((128, NVL * C - VT, HT, 128), bf16)], 1)
    cos, sin = rope_tables(TMAX)
    pr = np.zeros((TMAX, 1), np.int32); pr[:len(ids), 0] = ids
    common = dict(
        pos=np.zeros((1, 1), np.int32), plen=np.array([[len(ids)]], np.int32), prompt=pr,
        toks=np.zeros((TMAX + 1, 1), np.int32), embed=np.ascontiguousarray(st.get("model.embed_tokens.weight")),
        ln1=np.stack([col_layout(g(l, "input_layernorm.weight")).astype(np.float32) for l in range(L)]),
        qn=np.stack([g(l, "self_attn.q_norm.weight").reshape(128, 1).astype(np.float32) for l in range(L)]),
        kn=np.stack([g(l, "self_attn.k_norm.weight").reshape(128, 1).astype(np.float32) for l in range(L)]),
        normf=col_layout(st.get("model.norm.weight")).astype(np.float32),
        rot=rope_rot_matrix().astype(bf16), cos=cos, sin=sin)
    moe, (IL, ILT, ILP) = moe_split(st, cfg, list(range(L)), W, out=wdir)
    ranks = []
    for r in range(W):
        d = dict(common)
        d.update(moe[r])
        d["wqkv"] = np.empty((L, 2, 128, (nq + 2 * nk) * HT * 128), bf16)
        d["wo"] = np.empty((L, 2, 128, HT * nq * 128), bf16)
        d["wlm"] = np.ascontiguousarray(np.stack([lm[:, NVL * (2 * r + c):NVL * (2 * r + c + 1)] for c in range(2)])).reshape(2, 128, -1)
        d["kcache"] = np.zeros((L, 2, nk, 128, TMAX), bf16)
        d["vcache"] = np.zeros((L, 2, TMAX, nk * 128), bf16)
        d["rkb"] = np.full((128, 1), r * 2 * NVL * 128, np.float32)
        ranks.append(d)

    def one_layer(l):
        qkv = ws_layout(np.concatenate([g(l, "self_attn.q_proj.weight"), g(l, "self_attn.k_proj.weight"),
                                        g(l, "self_attn.v_proj.weight")])).reshape(128, NQ + 2 * NKV, HT * 128)
        wo = ws_layout(g(l, "self_attn.o_proj.weight")).reshape(128, HT, NQ, 128)
        for r in range(W):
            for c in range(2):
                v = 2 * r + c
                q0, k0 = v * nq, (v * NKV) // C if NKV < C else v * nk
                ranks[r]["wqkv"][l, c] = np.concatenate([qkv[:, q0:q0 + nq], qkv[:, NQ + k0:NQ + k0 + nk],
                                                         qkv[:, NQ + NKV + k0:NQ + NKV + k0 + nk]], 1).reshape(128, -1)
                ranks[r]["wo"][l, c] = wo[:, :, q0:q0 + nq].reshape(128, -1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one_layer, range(L)))
    return ranks, (nq, nk, ILT, ILP)


def write_inputs(d, ranks):
    for r, ins in enumerate(ranks):
        os.makedirs(os.path.join(d, f"r{r}"), exist_ok=True)
        for k, v in ins.items():
            p = os.path.join(d, f"r{r}", k + ".bin")
            if isinstance(v, np.memmap):
                if os.path.abspath(v.filename) != os.path.abspath(p):
                    if os.path.lexists(p): os.remove(p)
                    os.symlink(v.filename, p)
            else:
                np.ascontiguousarray(v).tofile(p)


def run_chain(neff, d, W, n_steps, reps, cores):
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": cores,
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB3_COMM_ID", f"localhost:{51000 + os.getpid() % 2000}")}
    p = subprocess.run([os.path.join(nb.LAB, "host", "tpchain"), neff, d, str(W), str(n_steps), "pos:output_0",
                        str(reps)], capture_output=True, text=True, env=env, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError(f"tpchain failed: {p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    toks = [np.fromfile(os.path.join(d, f"r{r}", "out_toks.bin"), np.int32) for r in range(W)]
    return toks, p.stdout


if __name__ == "__main__":
    W = int(sys.argv[1])
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    L = int(sys.argv[3]) if len(sys.argv) > 3 else cfg["num_hidden_layers"]
    TB, SL, HW = int(os.environ.get("TB", 512)), int(os.environ.get("SL", 1)), int(os.environ.get("HW", 2))
    PF = int(os.environ.get("PF", 0))
    SLO, PFO, SLH = int(os.environ.get("SLO", 1)), int(os.environ.get("PFO", 0)), int(os.environ.get("SLH", 1))
    MOEOPT = int(os.environ.get("MOEOPT", 0))
    XCH = int(os.environ.get("XCH", 0))
    QKVQ, PFOQ = int(os.environ.get("QKVQ", 0)), int(os.environ.get("PFOQ", 0))
    NOMOE = int(os.environ.get("NOMOE", 0))
    cores = os.environ.get("CORES", f"0-{W - 1}" if W > 1 else "0")
    tag = os.environ.get("TAG", "default")
    ref = dict(np.load(os.path.join(nb.LAB, "build", f"hf_ref_{tag}.npz")))
    ids = ref["ids"].tolist()
    P = len(ids)
    G = min(G, len(ref["full"]) - P)
    t0 = time.time()
    wdir = os.path.join(nb.SCRATCH, f"step_w{W}_L{L}" + os.environ.get("WTAG", ""))   # one per concurrent run
    import atexit
    if not os.environ.get("KEEP"):   # /dev/shm is RAM: 16-75 GB per run (it once filled all 1000 GB)
        atexit.register(shutil.rmtree, wdir, True)
    ranks, (nq, nk, ILT, ILP) = split_step(SafeTensors(MODEL), cfg, L, ids, W, wdir)
    print(f"split W={W} L={L} (per virtual core: {nq} q heads, {nk} kv heads, experts {ILT} x {ILP} rows) in "
          f"{time.time() - t0:.0f} s; per rank {sum(v.nbytes for v in ranks[0].values()) / 1e9:.2f} GB", flush=True)
    name = os.environ.get("NAME", f"smoe_w{W}_L{L}_b{TB}" + (f"_pf{PF}" if PF else "") + ("_nomoe" if NOMOE else "") +
                          (f"_s{SL}o{SLO}h{SLH}" if (SL, SLO, SLH) != (1, 1, 1) else "") + ("_pfo" if PFO else "") +
                          (f"_m{MOEOPT}" if MOEOPT else "") + ("_x" if XCH else "") + ("_qq" if QKVQ else "") + ("_pq" if PFOQ else ""))
    WB = W
    if os.environ.get("NOCC"):
        WB, ranks, name, cores = 1, ranks[:1], name + "_nocc", cores.split("-")[0]
        os.environ["NOREF"] = "1"
    if NOMOE:
        os.environ["NOREF"] = "1"
    if not os.environ.get("NOBUILD"):
        spec = kbuild.spec_of(ranks[0], ORDER)
        _, dt = nb.build(name, "kbuild:build", cores, "step_moe", "step_moe", ORDER, spec, WB,
                         dict(TB=TB, SL=SL, hw=HW, NQL=nq, NKL=nk, ILT=ILT, ILP=ILP, PF=PF, NOMOE=NOMOE, SLO=SLO, PFO=PFO, SLH=SLH, MOEOPT=MOEOPT, XCH=XCH, QKVQ=QKVQ, PFOQ=PFOQ))
    d = os.path.join(wdir, "io")
    write_inputs(d, ranks)
    del ranks
    neff = os.path.join(nb.LAB, "neff", name + ".neff")
    toks, so = run_chain(neff, d, WB, P + G - 1, int(os.environ.get("REPS", "1")), cores)
    for l in so.splitlines():
        if "CHAIN" in l or "nrt_load" in l or "upload" in l: print(l)
    reps = [float(l.split("=")[1].split()[0]) for l in so.splitlines() if l.startswith("CHAIN rep")]
    print(f"STEP W={W} L={L} b{TB}: {min(reps) / 1e3:.3f} ms/step (min over reps of chain wall time / steps)")
    same_ranks = all((t == toks[0]).all() for t in toks)
    print("token history identical on all ranks:", same_ranks)
    got = toks[0][1:P + G].tolist()
    if os.environ.get("NOREF"):
        sys.exit(0)
    ref_pred = ref["argmax"][:P + G - 1].tolist()
    full = ref["full"].tolist()
    tf = sum(int(a == b) for a, b in zip(got[:P - 1], ref_pred[:P - 1]))
    gen_k, gen_r = got[P - 1:], full[P:P + G]
    same = next((i for i in range(G) if gen_k[i] != gen_r[i]), G)
    print(f"[fp32] prompt teacher-forced argmax agree {tf}/{P - 1}; generated: first {same}/{G} tokens identical")
    mis = [i for i in range(P - 1) if got[i] != ref_pred[i]]
    if mis:
        gaps = ref["top2gap"][mis]
        print(f"   teacher-forced mismatches at {mis[:12]}: HF fp32 top-1 / top-2 logit gap there median "
              f"{float(np.median(gaps)):.3f}, max {float(gaps.max()):.3f} (median over all positions "
              f"{float(np.median(ref['top2gap'][:P - 1])):.3f})")
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    print("kernel text:", repr(tok.decode(gen_k)))
    ok = same_ranks and tf >= min(P - 2, int(0.98 * (P - 1))) and same >= min(G, 8)
    if same < G:
        i = P - 1 + same
        print(f"   divergence at gen {same}: kernel {gen_k[same]} ref {gen_r[same]}, HF top-2 logit gap {ref['top2gap'][i]:.4f}")
        ok = ok or (same_ranks and ref['top2gap'][i] < 0.1)
    print("PASS" if ok else "FAIL")
