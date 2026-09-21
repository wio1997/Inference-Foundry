#!/usr/bin/env python3
"""Summarize saved c12 HCCL profiler categories; timing is profiled, not an E2E bound."""
import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

root=Path("/data/wio/Inference_Foundry")
base=root/"evidence/20260920_decode_c12_profile"
window=json.loads((base/"run4/device_c12_summary.json").read_text())
lo=datetime.fromisoformat(window["profile_start_utc"]).timestamp()*1e6
hi=datetime.fromisoformat(window["request_done_utc"]).timestamp()*1e6
fields=("Elapse Time(ms)","Transit Time(ms)","Wait Time(ms)","Synchronization Time(ms)","Idle Time(ms)")
result={"profile_window_utc":[window["profile_start_utc"],window["request_done_utc"]],
        "scope":"Saved profiler communication.json values; no inference of removable E2E time.",
        "ranks":{}}
for path in sorted(glob.glob(str(base/"torch_raw/*/ASCEND_PROFILER_OUTPUT/communication.json"))):
    rank=path.split("rank")[1].split("_")[0]
    summary=defaultdict(lambda:{"count":0,**{field:0.0 for field in fields}})
    entries=json.loads(Path(path).read_text())["step"]["collective"]
    included=0
    missing_start=0
    for name,event in entries.items():
        info=event.get("Communication Time Info",{})
        start=info.get("Start Timestamp(us)")
        if start is None:
            missing_start+=1
            continue
        if not lo <= float(start) <= hi:
            continue
        included+=1
        typ=name.split("__")[0]
        row=summary[typ]
        row["count"]+=1
        for field in fields:
            row[field]+=float(info.get(field,0) or 0)
    result["ranks"][rank]={"source":str(path),"collective_entries":len(entries),
                           "included_in_window":included,"missing_start":missing_start,
                           "types":{key:{f:round(v,4) if isinstance(v,float) else v for f,v in row.items()}
                                    for key,row in sorted(summary.items())}}
out=root/"evidence/20260921_decode_comm_audit/collective_time_components.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({r:{k:{"count":v["count"],"elapsed_ms":v["Elapse Time(ms)"],
                     "idle_ms":v["Idle Time(ms)"],"wait_ms":v["Wait Time(ms)"]}
                    for k,v in d["types"].items()}
                  for r,d in result["ranks"].items()},indent=2))
