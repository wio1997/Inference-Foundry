#!/usr/bin/env python3
"""Summarize Loop020 no-profiler DSpark proposer stages in c12 sample window."""
import csv,glob,json,statistics
from pathlib import Path
root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260921_loop020_draft_stage"
sample=json.loads((base/"run1/decode_c12.json").read_text())
lo=min(r["start"] for r in sample["requests"] if r["error"] is None)*1e9
hi=max(r["end"] for r in sample["requests"] if r["error"] is None)*1e9
labels=["set_inputs","pre_attention_metadata","step0_all_attention_groups",
        "later_step_metadata","pre_model","model_run"]
def stat(v):
    v=sorted(v);return {"n":len(v),"median_ms":round(statistics.median(v),6),
                        "p90_ms":round(v[round((len(v)-1)*.9)],6),"max_ms":round(max(v),6)}
out={"window_monotonic_ns":[lo,hi],"filter":"proposer start within successful c12 client window; only no-prefill calls",
     "scope":"Inclusive host-stage times; model_run may wait on device and is not removable work.","ranks":{}}
for path in sorted(glob.glob(str(base/"raw/*.csv"))):
    rank=path.split("rank")[-1].split("_")[0];rows=[]
    for a in csv.reader(open(path)):
        t=list(map(int,a[:7]));batch,ntok,ninput,prefill,graph=map(int,a[7:12])
        if not lo<=t[0]<=hi or prefill:continue
        rows.append((batch,ntok,ninput,graph,[(t[i+1]-t[i])/1e6 for i in range(6)]))
    if not rows:continue
    data=[x[4] for x in rows]
    out["ranks"][rank]={"count":len(rows),"batch_size_values":sorted({x[0] for x in rows}),
                        "input_token_values":sorted({x[2] for x in rows}),
                        "graph_values":sorted({x[3] for x in rows}),
                        "stages":{labels[i]:stat([v[i] for v in data]) for i in range(6)},
                        "total":stat([sum(v) for v in data])}
path=base/"draft_stage_summary.json";path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({r:{"count":v["count"],"graph":v["graph_values"],"total":v["total"]["median_ms"],
                      **{k:x["median_ms"] for k,x in v["stages"].items()}}
                  for r,v in out["ranks"].items()},indent=2))
