"""How close is HF transformers bf16 (CPU) to HF fp32 on the same token sequence? Teacher-forced argmax agreement
over the sequence of a tests/hf_ref.py capture (TAG), the baseline for judging the engine's own agreement.
usage: python tests/hf_bf16_agree.py <TAG>"""
import sys, os, numpy as np, torch, transformers
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import nb
tag = sys.argv[1]
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-30B-A3B"))
ref = np.load(os.path.join(nb.LAB, "build", f"hf_ref_{tag}.npz"))
full, P = ref["full"].tolist(), len(ref["ids"])
torch.set_num_threads(96)
m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
with torch.no_grad():
    lg = m(torch.tensor([full])).logits[0].float().numpy()
am = lg.argmax(-1)
tf = int((am[:P - 1] == ref["argmax"][:P - 1]).sum())
mis = np.nonzero(am[:P - 1] != ref["argmax"][:P - 1])[0]
print(f"HF bf16 vs HF fp32, {tag}: teacher-forced argmax agree {tf}/{P - 1} over the prompt; "
      f"over all {len(full) - 1} positions {int((am[:-1] == ref['argmax'][:len(full) - 1]).sum())}/{len(full) - 1}; "
      f"fp32 top-2 gaps at the mismatches median {float(np.median(ref['top2gap'][mis])) if len(mis) else 0:.3f}")
