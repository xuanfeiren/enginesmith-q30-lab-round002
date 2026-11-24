"""HF transformers fp32 reference for Qwen3-30B-A3B on the CPU (run once, cached in build/hf_ref_<tag>.npz).

Captures, for the prompt's tokens + G greedy tokens (teacher-forced pass over the full sequence):
  per layer l: hpre [T, H] = residual stream entering post_attention_layernorm (input of the MoE half-layer),
               moe [T, H] = Qwen3MoeSparseMoeBlock output, logits [T, E] router logits, idx [T, 8], w [T, 8]
  greedy tokens (generate, do_sample=False) and the argmax of the logits at every position.
usage: python tests/hf_ref.py [G=24] ; env MODEL, PROMPT, PROMPT_FILE, PROMPT_TOKENS, TAG
"""
import os, sys, time, json
import numpy as np


def load_ref(tag="default"):
    from nb import LAB
    return dict(np.load(os.path.join(LAB, "build", f"hf_ref_{tag}.npz")))


if __name__ == "__main__":
    import torch, transformers
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
    from nb import LAB
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-30B-A3B"))
    tag = os.environ.get("TAG", "default")
    torch.set_num_threads(int(os.environ.get("THREADS", "96")))
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = open(os.environ["PROMPT_FILE"]).read() if os.environ.get("PROMPT_FILE") else \
        os.environ.get("PROMPT", "The capital of France is Paris. The capital of Germany is")
    ids = tok(text)["input_ids"][:int(os.environ.get("PROMPT_TOKENS", "100000"))]
    t0 = time.time()
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    m.eval()
    print(f"loaded fp32 model in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    with torch.no_grad():
        full = m.generate(torch.tensor([ids]), max_new_tokens=G, do_sample=False, num_beams=1)[0].tolist()
    print(f"generate {G} in {time.time() - t0:.0f} s: {tok.decode(full[len(ids):])!r}", flush=True)
    L = m.config.num_hidden_layers
    cap = {k: [None] * L for k in ("hpre", "moe", "logits", "idx", "w")}
    hooks = []
    for l, layer in enumerate(m.model.layers):
        def pre_norm(mod, inp, l=l):
            cap["hpre"][l] = inp[0][0].float().numpy().copy()
        def moe_out(mod, inp, out, l=l):
            cap["moe"][l] = (out[0] if isinstance(out, tuple) else out)[0].float().numpy().copy()
        def gate_out(mod, inp, out, l=l):
            lg, sc, ix = out
            cap["logits"][l] = lg.float().numpy().copy(); cap["w"][l] = sc.float().numpy().copy()
            cap["idx"][l] = ix.numpy().astype(np.int32).copy()
        hooks += [layer.post_attention_layernorm.register_forward_pre_hook(pre_norm),
                  layer.mlp.register_forward_hook(moe_out), layer.mlp.gate.register_forward_hook(gate_out)]
    t0 = time.time()
    with torch.no_grad():
        lg = m(torch.tensor([full])).logits[0].float().numpy()
    print(f"teacher-forced pass over {len(full)} tokens in {time.time() - t0:.0f} s", flush=True)
    out = {k: np.stack(v) for k, v in cap.items()}
    out.update(ids=np.array(ids, np.int32), full=np.array(full, np.int32), argmax=lg.argmax(-1).astype(np.int32),
               top2gap=np.sort(lg, -1)[:, -1] - np.sort(lg, -1)[:, -2])
    # logits at every position are big (T x 151936 fp32); keep only what the checks need
    np.savez(os.path.join(LAB, "build", f"hf_ref_{tag}.npz"), **out)
    print("saved", {k: v.shape for k, v in out.items()})
