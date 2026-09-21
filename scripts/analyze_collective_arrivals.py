#!/usr/bin/env python3
"""Align saved TP8 collective sequence by call index and quantify arrival skew."""
import glob,json,statistics
from collections import Counter
from pathlib import Path
root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260920_decode_c12_profile"
window=json.loads((base/"run4/device_c12_summary.json").read_text())
from datetime import datetime
lo=datetime.fromisoformat(window["profile_start_utc"]).timestamp()*1e6
hi=datetime.fromisoformat(window["request_done_utc"]).timestamp()*1e6
rows=[]
for rank in range(8):
    files=glob.glob(str(base/f"torch_raw/*rank{rank}_*/ASCEND_PROFILER_OUTPUT/communication.json"))
    assert len(files)==1,files
    data=json.loads(Path(files[0]).read_text())["step"]["collective"]
    rankrows=[]
    for name,event in data.items():
        info=event.get("Communication Time Info",{})
        st=info.get("Start Timestamp(us)")
        if st is None or not lo<=float(st)<=hi: continue
        rankrows.append((name.split("__")[0],float(st),float(info["Elapse Time(ms)"])*1000))
    rows.append(rankrows)
assert len(set(map(len,rows)))==1,[len(x) for x in rows]
n=len(rows[0])
assert all(len({rows[r][i][0] for r in range(8)})==1 for i in range(n)),"collective order mismatch"
def pct(v,p):
    a=sorted(v);return a[round((len(a)-1)*p)]
spread=[]; endspread=[]; earliest=[]; last=Counter(); first=Counter(); types={}
for i in range(n):
    st=[rows[r][i][1] for r in range(8)]
    end=[rows[r][i][1]+rows[r][i][2] for r in range(8)]
    ty=rows[0][i][0]
    x=max(st)-min(st);y=max(end)-min(end)
    spread.append(x);endspread.append(y);earliest.append(min(st))
    last[st.index(max(st))]+=1;first[st.index(min(st))]+=1
    types.setdefault(ty,[]).append(x)
res={"window_utc":[window["profile_start_utc"],window["request_done_utc"]],
 "alignment":"Collective type sequence identical at all 8 ranks; rank timestamps used as reported, no clock correction.",
 "count":n,"latest_start_rank_counts":dict(sorted(last.items())),"earliest_start_rank_counts":dict(sorted(first.items())),
 "start_spread_ms":{"median":pct(spread,.5)/1000,"p90":pct(spread,.9)/1000,"p99":pct(spread,.99)/1000,"max":max(spread)/1000},
 "end_spread_ms":{"median":pct(endspread,.5)/1000,"p90":pct(endspread,.9)/1000,"p99":pct(endspread,.99)/1000,"max":max(endspread)/1000},
 "by_type_start_spread_ms":{k:{"count":len(v),"median":pct(v,.5)/1000,"p90":pct(v,.9)/1000,"p99":pct(v,.99)/1000} for k,v in types.items()},
 "sample_first_3":[{"type":rows[0][i][0],"start_us":[rows[r][i][1] for r in range(8)],"duration_us":[rows[r][i][2] for r in range(8)]} for i in range(3)]}
out=root/"evidence/20260921_decode_comm_audit/collective_arrival_skew.json";out.write_text(json.dumps(res,indent=2)+"\n")
print(json.dumps(res,indent=2))
