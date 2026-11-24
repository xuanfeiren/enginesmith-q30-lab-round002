"""Print a compact device timeline from `neuron-explorer view --output-format json` (ntff.json).
usage: timeline.py ntff.json [--instr] ; times in us relative to the first event."""
import json, sys, collections
d = json.load(open(sys.argv[1]))
ins = d["instruction"]; dma = d["dma"]
t0 = min([i["timestamp"] for i in ins] + [x["timestamp"] for x in dma])
print("engine instructions: first..last (us), count, busy us")
by = collections.defaultdict(list)
for i in ins: by[i["label"]].append(i)
for k, v in sorted(by.items()):
    s = min(i["timestamp"] for i in v) - t0; e = max(i["timestamp"] + i["duration"] for i in v) - t0
    print(f"  {k:12s} {s/1e3:8.2f} .. {e/1e3:8.2f}  n={len(v):4d}  busy={sum(i['duration'] for i in v)/1e3:7.2f}")
print("DMA by variable/queue: first..last (us), bytes")
g = collections.defaultdict(list)
for x in dma: g[(x.get("variable", ""), x.get("subgroup", "").split("-")[0])].append(x)
for k, v in sorted(g.items(), key=lambda kv: min(x["timestamp"] for x in kv[1])):
    s = min(x["timestamp"] for x in v) - t0; e = max(x["timestamp"] + x["duration"] for x in v) - t0
    b = sum(x["transfer_size"] for x in v)
    print(f"  {k[0][:28]:28s} {k[1][:18]:18s} {s/1e3:8.2f} .. {e/1e3:8.2f}  {b/1e3:9.1f} KB  n={len(v)}")
if "--instr" in sys.argv:
    for i in sorted(ins, key=lambda i: i["timestamp"]):
        print(f"{(i['timestamp']-t0)/1e3:8.3f} +{i['duration']:6d}ns {i['label'][:8]:8s} {i['opcode'][:22]:22s} {i['operands'][:90]}")
