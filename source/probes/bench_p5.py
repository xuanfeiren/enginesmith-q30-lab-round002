# per-instruction PSUM evacuation time from neuron-explorer (probe p5), one core
import sys, os, json, subprocess, collections, nb
core = sys.argv[1]
names = {0: "VectorE fp32 PSUM [128,512] -> bf16", 1: "ScalarE fp32 PSUM [128,512] -> bf16",
         2: "VectorE bf16 PSUM [128,512] -> bf16", 3: "ScalarE bf16 PSUM [128,512] -> bf16",
         4: "VectorE fp32 PSUM [128,2048] -> bf16 (one instruction)", 5: "VectorE fp32 PSUM [128,512] -> fp32"}
for MODE in [int(v) for v in sys.argv[2].split(',')]:
    nm = f"p5_m{MODE}"
    nb.build(nm, "p5_evac:run", core, MODE, 20)
    subprocess.run([sys.executable, os.path.join(nb.LAB, "tools", "prof.py"), os.path.join(nb.LAB, "neff", nm + ".neff"), core],
                   capture_output=True, text=True)
    d = json.load(open(os.path.join(nb.LAB, "build", "prof", nm + ".json")))
    c = collections.defaultdict(list)
    for i in d["instruction"]:
        if i["fully_qualified_subgraph"] == "/sg00" and "p5_evac.py:3" in i.get("nki_source_location", "") and i["label"] in ("Vector", "Scalar"):
            c[(i["label"], i["opcode"])].append(i["duration"])
    for k, v in c.items():
        if len(v) >= 10:
            print(f"MODE {MODE} {names[MODE]}: {k[0]} {k[1]} n={len(v)} avg {sum(v) / len(v):.0f} ns", flush=True)
