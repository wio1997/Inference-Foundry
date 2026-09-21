#!/usr/bin/env python3
"""Estimate per-rank timestamp offsets from common collective end times, audit residual skew."""
import glob,json,statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
root=Path("/data/wio/Inference_Foundry");base=root/"evidence/20260920_decode_c12_profile"
w=json.loads((base/"run4/device_c12_summary.json").read_text())
lo=datetime.fromisoformat(w["profile_start_utc"]).timestamp()*1e6
hi=datetime.fromisoformat(w["request_done_utc"]).timestamp()*1e6
rows=[]
for r in range(8):
    p=glob.glob(str(base/f"torch_raw/*rank{r}_*/ASCEND_PROFILER_OUTPUT/communication.json"))[0]
    d=json.loads(Path(p).read_text())["step"]["collective"]; a=[]
    for name,event in d.items():
        t=event.get("Communication Time Info",{});s=t.get("Start Timestamp(us)")
        if s is None or not lo<=float(s)<=hi:continue
        a.append((name.split("__")[0],float(s),float(t["Elapse Time(ms)"])*1000))
    rows.append(a)
n=len(rows[0]);assert n==48960 and all(len(a)==n for a in rows)
assert all(len({rows[r][i][0] for r in range(8)})==1 for i in range(n))
offset=[0]+[statistics.median(rows[r][i][1]+rows[r][i][2]-rows[0][i][1]-rows[0][i][2] for i in range(n)) for r in range(1,8)]
def pct(v,p):return sorted(v)[round((len(v)-1)*p)]
rawstart=[];rawend=[];correctedstart=[];correctedend=[];last=Counter();first=Counter()
for i in range(n):
    s=[rows[r][i][1] for r in range(8)]
    e=[rows[r][i][1]+rows[r][i][2] for r in range(8)]
    ss=[s[r]-offset[r] for r in range(8)];ee=[e[r]-offset[r] for r in range(8)]
    rawstart.append(max(s)-min(s));rawend.append(max(e)-min(e))
    correctedstart.append(max(ss)-min(ss));correctedend.append(max(ee)-min(ee))
    last[ss.index(max(ss))]+=1;first[ss.index(min(ss))]+=1
def stats(v):return {k:round(f(v)/1000,6) for k,f in {"median":lambda x:pct(x,.5),"p90":lambda x:pct(x,.9),"p99":lambda x:pct(x,.99),"max":max}.items()}
out={"n":n,"method":"Median per-rank collective end-time difference relative to rank0; assumes common completion and stable offsets. This is a sensitivity analysis, not independent clock synchronization.",
"estimated_offset_us_from_rank0":[round(x,3) for x in offset],
"raw_start_spread_ms":stats(rawstart),"raw_end_spread_ms":stats(rawend),
"corrected_start_spread_ms":stats(correctedstart),"corrected_end_spread_ms":stats(correctedend),
"corrected_earliest_start_rank_counts":dict(sorted(first.items())),"corrected_latest_start_rank_counts":dict(sorted(last.items()))}
p=root/"evidence/20260921_decode_comm_audit/collective_clock_sensitivity.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
