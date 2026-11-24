"""Development driver (not part of the engine): start target/engine on given cores with the harness's clean
environment, send a list of requests, print tokens and wall times.

usage: python tools/drive.py CORES REQUESTS_JSON [ENGINE]
  REQUESTS_JSON: a file with a JSON list of {"prompt": [...] | "text": "...", "max_tokens": n, "reset": bool}
  ("text" is tokenized with the model's tokenizer, transformers from the Neuron venv).
"""
import json, os, subprocess, sys, tempfile, time

MODEL = "/home/ubuntu/models/Qwen3-30B-A3B"


def main():
    cores, reqf = sys.argv[1], sys.argv[2]
    eng = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "engine")
    reqs = json.load(open(reqf))
    tok = None
    for r in reqs:
        if "text" in r:
            if tok is None:
                import transformers
                tok = transformers.AutoTokenizer.from_pretrained(MODEL)
            r["prompt"] = tok.encode(r.pop("text"))
    scratch = tempfile.mkdtemp(prefix="q30drv_")
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": scratch, "TMPDIR": scratch, "HF_HUB_OFFLINE": "1",
           "NEURON_RT_VISIBLE_CORES": cores}
    for k in ("ENGINE_VERBOSE",):
        if k in os.environ: env[k] = os.environ[k]
    err = open(os.path.join(scratch, "stderr.log"), "w")
    p = subprocess.Popen([eng, "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, cwd=scratch,
                         env=env, text=True)
    def call(obj):
        t0 = time.time()
        p.stdin.write(json.dumps(obj) + "\n"); p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise SystemExit(f"engine died; stderr: {open(os.path.join(scratch, 'stderr.log')).read()[-5000:]}")
        return json.loads(line), (time.time() - t0) * 1e3
    rep, ms = call({"id": 0, "op": "hello"})
    print(f"hello {rep} {ms / 1e3:.1f} s", flush=True)
    out = []
    for i, r in enumerate(reqs):
        rep, ms = call({"id": i + 1, "op": "generate", "prompt": r["prompt"], "max_tokens": r["max_tokens"],
                        "reset": r.get("reset", True)})
        txt = ""
        if tok is not None and rep.get("ok"):
            txt = repr(tok.decode(rep["tokens"]))
        print(f"req {i}: P={len(r['prompt'])} n={r['max_tokens']} reset={r.get('reset', True)}: {ms:.1f} ms "
              f"{'ok' if rep.get('ok') else rep}: {rep.get('tokens', [])[:20]} {txt[:200]}", flush=True)
        out.append(rep)
    p.stdin.close()
    p.wait(timeout=120)
    print("stderr tail:\n" + open(os.path.join(scratch, "stderr.log")).read()[-3000:])
    if os.environ.get("OUT"):
        json.dump(out, open(os.environ["OUT"], "w"))


if __name__ == "__main__":
    main()
