#!/usr/bin/env python3
"""Summarize Loop019 build_req_metadata subphases in a no-profiler client window."""
import csv,glob,json,statistics
from pathlib import Path
root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260921_loop019_builder_req"
sample=json.loads((base/"run2/decode_c12.json").read_text())
lo=min(r["start"] for r in sample["requests"] if r["error"] is None)*1e9
hi=max(r["end"] for r in sample["requests"] if r["error"] is None)*1e9
labels=["device_local","rope_local","cpu_local","max_lengths","compressor","sas","qli","return_metadata"]
def stat(v):
    v=sorted(v)
    return {"n":len(v),"median_ms":round(statistics.median(v),6),
            "p90_ms":round(v[round((len(v)-1)*.9)],6),
            "max_ms":round(max(v),6)}
out={"client_window_monotonic_ns":[lo,hi],"filter":"row start within successful c12 request window; pure decode num_input_tokens<=96","phases":labels,"ranks":{}}
for path in sorted(glob.glob(str(base/"raw_req/*.csv"))):
    rank=path.split("rank")[-1].split("_")[0]
    rows=[]
    for a in csv.reader(open(path)):
        t=list(map(int,a[:9]))
        if not lo<=t[0]<=hi:continue
        first=int(a[9]);nreq=int(a[10]);ntok=int(a[11]);ratio=int(a[12])
        if ntok>96:continue
        rows.append((first,nreq,ntok,ratio,[(t[i+1]-t[i])/1e6 for i in range(8)]))
    groups={}
    for label,flag in (("first",1),("other",0)):
        v=[x[4] for x in rows if x[0]==flag]
        groups[label]={labels[i]:stat([x[i] for x in v]) for i in range(8)}
        groups[label]["total"]=stat([sum(x) for x in v])
    out["ranks"][rank]={"rows":len(rows),"first_count":sum(x[0] for x in rows),
                        "input_token_values":sorted({x[2] for x in rows}),
                        "first_ratio_values":sorted({x[3] for x in rows if x[0]}),
                        "groups":groups}
path=base/"request_subphases_summary.json";path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({r:{"steps":v["first_count"],"first_ms":{k:val["median_ms"] for k,val in v["groups"]["first"].items()},
                      "other_total_ms":v["groups"]["other"]["total"]["median_ms"]}
                  for r,v in out["ranks"].items()},indent=2))
