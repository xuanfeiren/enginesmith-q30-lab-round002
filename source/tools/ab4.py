"""Development tool (not part of the engine): paired timing + token comparison of two engine builds, both loaded at
once on different 4-core groups, requests alternating (A then B on even rounds, B then A on odd), never concurrent.
Takes the GPU lock /tmp/engine-gpu.lock for its whole run.

usage: python tools/ab4.py ENGINE_A ENGINE_B [--rounds N] [--lengths 256,1024,4096] [--cores-a 0-3] [--cores-b 4-7]
  [--decode 64] [--seed S]
Per round and length: a fresh random prompt (uniform token ids), prefill = generate(P, 1), decode = generate([], 64).
Prints per-length medians and the median per-round ratio A/B (B's speedup), and every token difference.
"""
import argparse, fcntl, json, os, random, statistics, subprocess, sys, tempfile, time

MODEL = "/home/ubuntu/models/Qwen3-30B-A3B"
VOCAB = 151936


class Eng:
    def __init__(self, path, cores, tag, extra=""):
        self.scratch = tempfile.mkdtemp(prefix=f"q30ab_{tag}_")
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": self.scratch, "TMPDIR": self.scratch,
               "HF_HUB_OFFLINE": "1", "NEURON_RT_VISIBLE_CORES": cores}
        for kv in filter(None, extra.split(",")):
            env[kv.split("=")[0]] = kv.split("=", 1)[1]
        self.err = open(os.path.join(self.scratch, "stderr.log"), "w")
        self.p = subprocess.Popen([path, "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, cwd=self.scratch, env=env, text=True)
        self.i = 0

    def call(self, obj):
        self.i += 1
        obj = dict(obj, id=self.i)
        t0 = time.perf_counter()
        self.p.stdin.write(json.dumps(obj) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        dt = (time.perf_counter() - t0) * 1e3
        if not line:
            raise SystemExit(f"engine died: {open(os.path.join(self.scratch, 'stderr.log')).read()[-4000:]}")
        r = json.loads(line)
        if not r.get("ok"):
            raise SystemExit(f"error reply {r}")
        return r, dt

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lengths", default="256,1024,4096")
    ap.add_argument("--cores-a", default="0-3"); ap.add_argument("--cores-b", default="4-7")
    ap.add_argument("--env-a", default=""); ap.add_argument("--env-b", default="")
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    lens = [int(x) for x in a.lengths.split(",")]
    rng = random.Random(a.seed if a.seed is not None else time.time_ns())
    lock = open("/tmp/engine-gpu.lock", "w")
    print("waiting for the GPU lock", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    E = {"A": Eng(a.a, a.cores_a, "A", a.env_a), "B": Eng(a.b, a.cores_b, "B", a.env_b)}
    try:
        for k, e in E.items():
            _, dt = e.call({"op": "hello"})
            print(f"{k} hello {dt / 1e3:.1f} s", flush=True)
        for k, e in E.items():
            for _ in range(a.warmup):
                for P in lens:
                    pr = [rng.randrange(VOCAB) for _ in range(P)]
                    e.call({"op": "generate", "prompt": pr, "max_tokens": 1, "reset": True})
                    e.call({"op": "generate", "prompt": [], "max_tokens": a.decode, "reset": False})
        T = {(k, P, w): [] for k in E for P in lens for w in ("pf", "dec")}
        diffs = 0
        for r in range(a.rounds):
            for P in lens:
                pr = [rng.randrange(VOCAB) for _ in range(P)]
                toks = {}
                for k in (("A", "B") if r % 2 == 0 else ("B", "A")):
                    r1, t1 = E[k].call({"op": "generate", "prompt": pr, "max_tokens": 1, "reset": True})
                    r2, t2 = E[k].call({"op": "generate", "prompt": [], "max_tokens": a.decode, "reset": False})
                    T[(k, P, "pf")].append(t1); T[(k, P, "dec")].append(t2)
                    toks[k] = r1["tokens"] + r2["tokens"]
                if toks["A"] != toks["B"]:
                    diffs += 1
                    fi = next(i for i in range(len(toks["A"])) if toks["A"][i] != toks["B"][i])
                    print(f"  round {r} P={P}: tokens differ from step {fi} "
                          f"({sum(x != y for x, y in zip(toks['A'], toks['B']))} of {len(toks['A'])})", flush=True)
                print(f"round {r} P={P}: pf A {T[('A', P, 'pf')][-1]:.1f} B {T[('B', P, 'pf')][-1]:.1f}  "
                      f"dec A {T[('A', P, 'dec')][-1]:.1f} B {T[('B', P, 'dec')][-1]:.1f}", flush=True)
        print("summary (ms, median; ratio = median per-round A/B):")
        for P in lens:
            for w in ("pf", "dec"):
                ta, tb = T[("A", P, w)], T[("B", P, w)]
                rat = statistics.median([x / y for x, y in zip(ta, tb)])
                print(f"  P={P:5d} {w:3s}: A {statistics.median(ta):8.2f}  B {statistics.median(tb):8.2f}  A/B {rat:.4f}")
        print(f"streams that differ: {diffs} of {a.rounds * len(lens)}")
    finally:
        for e in E.values():
            e.close()


if __name__ == "__main__":
    main()
