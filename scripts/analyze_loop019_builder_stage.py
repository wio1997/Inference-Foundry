#!/usr/bin/env python3
"""Summarize Loop019 flag-gated DSA-CP builder stage trace inside client request window."""
import csv,glob,json,statistics
from pathlib import Path
root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260921_loop019_builder_stage"
sample=json.loads((base/"run1/decode_c12.json").read_text())
lo=min(r["start"] for r in sample["requests"] if r["error"] is None)*1e9
hi=max(r["end"] for r in sample["requests"] if r["error"] is None)*1e9
def stat(v):
    v=sorted(v)
    return {"n":len(v),"median_ms":round(statistics.median(v),6),
            "p90_ms":round(v[round((len(v)-1)*.9)],6),
            "max_ms":round(max(v),6)}
out={"client_window_monotonic_ns":[lo,hi],"filter":"row start within successful 12-request sample client window; pure decode num_input_tokens<=96","ranks":{}}
for path in sorted(glob.glob(str(base/"raw/*.csv"))):
    rank=path.split("rank")[-1].split("_")[0]
    rows=[]
    for a in csv.reader(open(path)):
        t=list(map(int,a[:4]))
        if not lo<=t[0]<=hi:continue
        first=int(a[4]); nreq=int(a[5]);ntok=int(a[6])
        if ntok>96:continue
        rows.append((first,nreq,ntok,[(t[i+1]-t[i])/1e6 for i in range(3)]))
    groups={}
    for label,flag in (("first",1),("other",0)):
        v=[x[3] for x in rows if x[0]==flag]
        groups[label]={"shared":stat([x[0] for x in v]),"slot":stat([x[1] for x in v]),
                       "request_metadata":stat([x[2] for x in v]),"total":stat([sum(x) for x in v])}
    out["ranks"][rank]={"rows":len(rows),"first_count":sum(x[0] for x in rows),
                        "input_token_values":sorted({x[2] for x in rows}),"groups":groups}
path=base/"stage_summary.json";path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({r:{"steps":v["first_count"],
                      "first_total_ms":v["groups"]["first"]["total"]["median_ms"],
                      "first_req_ms":v["groups"]["first"]["request_metadata"]["median_ms"],
                      "first_shared_ms":v["groups"]["first"]["shared"]["median_ms"],
                      "other_total_ms":v["groups"]["other"]["total"]["median_ms"]}
                  for r,v in out["ranks"].items()},indent=2))
