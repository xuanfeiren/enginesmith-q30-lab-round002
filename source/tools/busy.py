"""union busy time per engine (excluding semaphore waits) of one physical core in a time window of a profile json.
usage: busy.py prof.json [sg00] [t0_us] [t1_us]"""
import json, sys, collections
d = json.load(open(sys.argv[1])); sg = "/" + (sys.argv[2] if len(sys.argv) > 2 else "sg00")
t0 = float(sys.argv[3]) * 1e3 if len(sys.argv) > 3 else 0; t1 = float(sys.argv[4]) * 1e3 if len(sys.argv) > 4 else 1e15
base = min(i["timestamp"] for i in d["instruction"])
iv = collections.defaultdict(list)
for i in d["instruction"]:
    s = i["timestamp"] - base
    if i["fully_qualified_subgraph"] == sg and t0 <= s <= t1 and "SEMAPHORE" not in i["opcode"]:
        iv[i["label"]].append((s, s + i["duration"]))
for lab, v in sorted(iv.items()):
    v.sort(); tot = 0; cs, ce = v[0]
    for s, e in v[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    tot += ce - cs
    print(f"{lab:14s} busy {tot/1e3:9.1f} us of {(min(t1, max(e for _, e in v)) - max(t0, min(s for s, _ in v)))/1e3:9.1f} us window")
