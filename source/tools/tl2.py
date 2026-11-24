"""Compact per-engine timeline of one physical core (subgraph) in a time window, from a neuron-explorer json.
usage: tl2.py prof.json [sg00] [t0_us] [t1_us]  -> runs of consecutive instructions with the same source line,
and the DMA bytes per 2 us bucket (all queues of that subgraph)."""
import json, sys, collections
d = json.load(open(sys.argv[1])); sg = sys.argv[2] if len(sys.argv) > 2 else "sg00"
t0 = float(sys.argv[3]) * 1e3 if len(sys.argv) > 3 else 0; t1 = float(sys.argv[4]) * 1e3 if len(sys.argv) > 4 else 1e12
base = min(i["timestamp"] for i in d["instruction"] if i.get("fully_qualified_subgraph") == "/" + sg)
ins = sorted([i for i in d["instruction"] if i.get("fully_qualified_subgraph") == "/" + sg and t0 <= i["timestamp"] - base <= t1], key=lambda i: i["timestamp"])
runs, last = [], {}
for i in ins:
    if i["label"] == "Tensor" and i["opcode"] == "LDWEIGHTS":
        continue
    key = (i["label"], i.get("nki_source_location", "").split("/")[-1], i["opcode"])
    r = last.get(i["label"])
    if r is not None and r[0] == key and i["timestamp"] - r[2] < 1500:
        r[2] = i["timestamp"] + i["duration"]; r[3] += 1
    else:
        r = [key, i["timestamp"], i["timestamp"] + i["duration"], 1]; runs.append(r); last[i["label"]] = r
for k, s, e, n in runs:
    print(f"{(s-base)/1e3:8.2f}..{(e-base)/1e3:8.2f} {k[0][:12]:12s} {k[2][:20]:20s} x{n:<4d} {k[1]}")
b = collections.Counter()
for x in d["dma"]:
    if x["subgraph"] == sg and t0 <= x["timestamp"] - base <= t1:
        b[int((x["timestamp"] - base) // 2000)] += x["transfer_size"]
print("DMA GB/s per 2 us bucket:", " ".join(f"{k*2}:{v/2000:.0f}" for k, v in sorted(b.items())))
