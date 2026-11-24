"""Problem 4: the reference, NxD Inference's own Qwen3-MoE (MODEL_TYPES['qwen3_moe']) at tensor-parallel degree TP,
measured the way enginesmith.references.nxdi drives a model (whole prompt through context encoding, then one
token-generation call per token, on-device greedy sampling, async mode, fused QKV, bucketing).

Run with the venv at ~/es-pr2/.venv:
  NEURON_RT_VISIBLE_CORES=12-15 ~/es-pr2/.venv/bin/python tools/ref_nxdi_moe.py <TP> [max_len=4608] [json extra settings]
Compiled artifacts: ~/nki-lab3/build/nxdi/<key>. Prints RESULT {...} with prefill ms (256 / 1024 / 4096 tokens) and
decode ms/token (128 tokens after each prompt), min of 3 repetitions, plus the greedy tokens of the 12-token prompt.
"""
import hashlib, json, os, sys, time
from pathlib import Path

os.environ['PATH'] = f'{Path(sys.executable).parent}{os.pathsep}{os.environ.get("PATH", "")}'
LAB = Path(__file__).resolve().parent.parent
MODEL = Path(os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-30B-A3B")))


def buckets(limit):
    s, x = [], 128
    while x < limit:
        s.append(x); x *= 2
    return s + [limit]


def main():
    TP = int(sys.argv[1])
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 4608
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    assert os.environ.get("NEURON_RT_VISIBLE_CORES"), "set NEURON_RT_VISIBLE_CORES (lab 3: cores 0-15 only)"
    import torch
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
    cfg = json.loads((MODEL / "config.json").read_text())
    cls = MODEL_TYPES[cfg["model_type"]]["causal-lm"]
    seq_len = -(-max_len // 512) * 512
    settings = {"tp_degree": TP, "batch_size": 1, "seq_len": seq_len, "max_context_length": seq_len,
                "enable_bucketing": True, "context_encoding_buckets": buckets(seq_len),
                "token_generation_buckets": buckets(seq_len), "async_mode": True, "fused_qkv": True, **extra}
    key = hashlib.sha256(json.dumps({"cfg": cfg, "settings": settings, "cls": cls.__name__}, sort_keys=True).encode()).hexdigest()[:16]
    out = LAB / "build" / "nxdi" / key
    ncfg = cls.get_neuron_config_cls()(**settings, torch_dtype=torch.bfloat16,
                                        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False))
    os.environ["BASE_COMPILE_WORK_DIR"] = f"{out / 'work'}/"
    config = cls.get_config_cls()(ncfg, load_config=load_pretrained_config(str(MODEL)))
    model = cls(str(MODEL), config)
    t0 = time.time()
    if not (out / "done").exists():
        out.mkdir(parents=True, exist_ok=True)
        print(f"compiling {cls.__name__} TP={TP} into {out}", flush=True)
        model.compile(str(out))
        (out / "done").write_text("")
        print(f"compiled in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    model.load(str(out))
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f} s", flush=True)

    def call(tokens, start):
        end = start + len(tokens)
        return model(input_ids=torch.tensor([tokens], dtype=torch.int32),
                     attention_mask=torch.ones((1, end if start == 0 else start), dtype=torch.int32),
                     position_ids=torch.arange(start, end, dtype=torch.int32)[None],
                     seq_ids=torch.tensor([0], dtype=torch.int32), sampling_params=prepare_sampling_params(1))

    def generate(prompt, n):
        tok = int(call(prompt, 0).tokens.reshape(-1)[-1]); pos = len(prompt); toks = [tok]
        t1 = time.time()
        while len(toks) < n:
            tok = int(call([tok], pos).tokens.reshape(-1)[-1]); pos += 1; toks.append(tok)
        return toks, (time.time() - t1) / max(1, n - 1)

    import transformers
    tk = transformers.AutoTokenizer.from_pretrained(str(MODEL))
    ids = tk("The capital of France is Paris. The capital of Germany is")["input_ids"]
    toks, _ = generate(ids, 24)
    res = {"TP": TP, "load_s": round(load_s, 1), "greedy24": toks, "text": tk.decode(toks)}
    for L in (256, 1024, 4096):
        if L + 129 > seq_len:
            continue
        P = [(k * 2654435761 % 2 ** 32) % 151936 for k in range(L)]
        generate(P, 8)
        pre, dec = [], []
        for rep in range(3):
            P2 = [(x + rep + 1) % 151936 for x in P]
            t1 = time.time(); call(P2, 0); pre.append(time.time() - t1)
            _, d = generate(P2, 129); dec.append(d)
        res[f"p{L}"] = {"prefill_ms": round(min(pre) * 1e3, 2), "decode_ms": round(min(dec) * 1e3, 3)}
        print(res, flush=True)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
