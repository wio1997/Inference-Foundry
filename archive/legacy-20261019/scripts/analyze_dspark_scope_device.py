#!/usr/bin/env python3
"""Align TP0 profiled host scopes and device tasks in the saved warm c12 window.

Counts are temporal co-occurrence, not causal ownership or removable cost.
"""
import bisect,csv,glob,json,statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260920_decode_c12_profile"
phase=json.loads((base/"run4/phase_times.json").read_text())
lo=datetime.fromisoformat(phase["profile_started_utc"]).timestamp()*1e6
hi=datetime.fromisoformat(phase["profiled_request_done_utc"]).timestamp()*1e6
trace=glob.glob(str(base/"torch_raw/*rank0*/ASCEND_PROFILER_OUTPUT/trace_view.json"))[0]
kernel=glob.glob(str(base/"torch_raw/*rank0*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))[0]
events=json.loads(Path(trace).read_text())
scopes={}
for name in ("draft_token","prepare input","forward","sample_token"):
    rows=sorted((float(e["ts"]),float(e["ts"])+float(e["dur"])) for e in events
                if e.get("ph")=="X" and e.get("cat")=="cpu_op" and e.get("name")==name
                and lo<=float(e["ts"])<hi)
    scopes[name]=rows
del events
def union(intervals):
    if not intervals:return 0.
    intervals.sort()
    start,end=intervals[0];total=0.
    for x,y in intervals[1:]:
        if x>end:
            total+=end-start;start,end=x,y
        else:end=max(end,y)
    return total+end-start
by={name:[{"all":[],"compute":[],"hcom":[],"count":Counter()} for _ in rows] for name,rows in scopes.items()}
starts={name:[x for x,y in rows] for name,rows in scopes.items()}
with open(kernel,newline="") as f:
    for row in csv.DictReader(f):
        try:s=float(row["Start Time(us)"]);d=float(row["Duration(us)"])
        except (ValueError,KeyError):continue
        if not lo<=s<hi:continue
        typ=row["Type"];category="hcom" if typ.startswith("hcom_") else "compute"
        for name,items in scopes.items():
            idx=bisect.bisect_right(starts[name],s)-1
            if idx<0:continue
            a,b=items[idx]
            if s>=b:continue
            y=min(s+d,b)
            rec=by[name][idx]
            rec["all"].append((s,y));rec[category].append((s,y));rec["count"][typ]+=1
def stat(v):
    if not v:return None
    z=sorted(v)
    return {"median_ms":round(statistics.median(z),6),"p90_ms":round(z[round((len(z)-1)*.9)],6),"sum_ms":round(sum(z),6)}
out={"window_utc":[phase["profile_started_utc"],phase["profiled_request_done_utc"]],
     "scope":"TP0 temporal co-occurrence of device starts inside host scopes. Device tasks can belong to adjacent async work; host-duration minus device-union is not an E2E saving.",
     "groups":{}}
for name,items in scopes.items():
    data=by[name]
    durations=[(y-x)/1000 for x,y in items]
    allbusy=[union(v["all"])/1000 for v in data]
    compute=[union(v["compute"])/1000 for v in data]
    hcom=[union(v["hcom"])/1000 for v in data]
    counts=Counter();[counts.update(v["count"]) for v in data]
    out["groups"][name]={"scope_count":len(items),"host_duration":stat(durations),
                          "coincident_device_all_union":stat(allbusy),
                          "coincident_compute_union":stat(compute),
                          "coincident_hcom_union":stat(hcom),
                          "top_device_task_counts":counts.most_common(12)}
p=root/"evidence/20260921_dspark_audit/host_device_scope_overlap_tp0.json"
p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out["groups"],indent=2))
