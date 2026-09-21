#!/usr/bin/env python3
"""Summarize Loop019 pure-decode QLI max.item and metadata-op timing."""
import csv,glob,json,statistics
from pathlib import Path
root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260921_loop019_qli_subphase"
sample=json.loads((base/"run3/decode_c12.json").read_text())
lo=min(r["start"] for r in sample["requests"] if r["error"] is None)*1e9
hi=max(r["end"] for r in sample["requests"] if r["error"] is None)*1e9
def stat(v):
    v=sorted(v)
    return {"count":len(v),"median_ms":round(statistics.median(v),6),"p90_ms":round(v[round((len(v)-1)*.9)],6),
            "max_ms":round(max(v),6)}
out={"window_monotonic_ns":[lo,hi],"filter":"QLI uncached decode calls whose start is in successful c12 client window; trace does not include cache-hit calls.","ranks":{}}
for path in sorted(glob.glob(str(base/"raw/*.csv"))):
    rank=path.split("rank")[-1].split("_")[0]
    rows=[]
    for a in csv.reader(open(path)):
        t=list(map(int,a[:4]))
        if lo<=t[0]<=hi:
            rows.append((t,int(a[4]),int(a[5]),int(a[6])))
    if not rows:continue
    out["ranks"][rank]={
        "count":len(rows),"num_reqs_values":sorted({r[1] for r in rows}),
        "max_q_values":sorted({r[2] for r in rows}),"max_k_minmax":[min(r[3] for r in rows),max(r[3] for r in rows)],
        "q_item":stat([(r[0][1]-r[0][0])/1e6 for r in rows]),
        "k_item":stat([(r[0][2]-r[0][1])/1e6 for r in rows]),
        "metadata_op_plus_clones":stat([(r[0][3]-r[0][2])/1e6 for r in rows]),
        "total":stat([(r[0][3]-r[0][0])/1e6 for r in rows])}
path=base/"qli_subphase_summary.json";path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({r:{k:v[k]["median_ms"] for k in ("q_item","k_item","metadata_op_plus_clones","total")} | {"count":v["count"]}
                  for r,v in out["ranks"].items()},indent=2))
