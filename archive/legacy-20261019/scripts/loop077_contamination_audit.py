#!/usr/bin/env python3
"""Detect stale Run386 client overlap with Run389 service from saved outputs."""
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"evidence/20260927_loop077_bound"
OUT=BASE/"run393"

def source_window(run,kind):
    x=json.loads((BASE/f"run{run}"/f"{kind}.json").read_text())
    req=x["requests"]
    assert len(req)==(48 if kind=="warmup48" else 12)
    assert all(y["output_tokens"]==1024 and y["error"] is None for y in req)
    times=sorted((y["start"],y["end"]) for y in req)
    points=sorted([(a,1) for a,b in times]+[(b,-1) for a,b in times])
    n=maximum=0
    for t,delta in points:
        n+=delta
        maximum=max(maximum,n)
    return {"n":len(req),"start":min(y["start"] for y in req),"end":max(y["end"] for y in req),"max_inflight":maximum,
            "external_tps":x["summary"]["output_tps"]}

result={"status":"Run389_current_c12_timing_INVALID_stale_Run386_client_overlap","windows":{},"server_post_200":None}
for kind in ("warmup48","bench"):
    old=source_window(386,kind)
    new=source_window(389,kind)
    combined=[]
    for run in (386,389):
        req=json.loads((BASE/f"run{run}"/f"{kind}.json").read_text())["requests"]
        combined.extend((y["start"],y["end"]) for y in req)
    points=sorted([(a,1) for a,b in combined]+[(b,-1) for a,b in combined])
    concurrent=maximum=0
    above12=0.0
    previous=None
    for when,delta in points:
        if previous is not None and concurrent>12:
            above12+=when-previous
        concurrent+=delta
        maximum=max(maximum,concurrent)
        previous=when
    result["windows"][kind]={"run386":old,"run389":new,"overlap_seconds":max(0,min(old["end"],new["end"])-max(old["start"],new["start"])),
                              "combined_max_inflight":maximum,"combined_above12_seconds":above12}
    assert maximum==24
    assert result["windows"][kind]["overlap_seconds"]>30
log=(ROOT/"logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP077-RUN389.log").read_text()
result["server_post_200"]=len(re.findall(r"POST /v1/chat/completions HTTP/1.1\" 200 OK",log))
assert result["server_post_200"]==120
result["registered_run389_clients"]=60
result["stale_run386_clients"]=60
result["runtime_eligible_cohorts_in_run389"]=2
result["decision"]="Run389/390 event intervals are valid under an unintended concurrent client load, not frozen c12 Current. Exclude from Scheduling Bound and Product comparison; retain as contamination diagnostic."
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"contamination.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
