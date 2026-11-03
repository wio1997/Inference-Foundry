#!/usr/bin/env python3
"""Read four fixed captured-model tasks in one frozen EP8 DB; no device work."""
import argparse, hashlib, json, sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument("--task-root",required=True);ap.add_argument("--db",required=True);a=ap.parse_args()
f=Path(a.task_root)/"agents/fusion/f22-layer24-run2-evidence.json"
assert hashlib.sha256(f.read_bytes()).hexdigest()=="0cae38a2a726f1471f651cd664f0dcb4f608734dbb59f8603d1315ce02084dfb"
x=json.loads(f.read_text()); base=x["window_ns"][0]
assert x["rank"]=="dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt"
ev=[e for e in x["events"] if e["task_id"] in ("1261","1264","1267","1268")];assert len(ev)==4
p=Path(a.db).resolve(strict=True); h=hashlib.sha256()
with p.open("rb") as q:
 for chunk in iter(lambda:q.read(1024*1024),b""):h.update(chunk)
assert h.hexdigest()=="67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58"
c=sqlite3.connect(p.as_uri()+"?mode=ro",uri=True);c.row_factory=sqlite3.Row;c.execute("PRAGMA query_only=ON");c.execute("PRAGMA temp_store=MEMORY")
def rows(sql,args=(),cap=30):
 r=[dict(x) for x in c.execute(sql,args).fetchmany(cap+1)]; return {"rows":r[:cap],"truncated":len(r)>cap}
out=[]
for e in ev:
 found=rows("SELECT * FROM TASK WHERE streamId=? AND modelId=? AND taskId=? AND startNs=? AND endNs=?",(int(e["stream"]),int(e["model"]),int(e["task_id"]),base+e["start_rel_ns"],base+e["end_rel_ns"]),2)
 assert not found["truncated"] and len(found["rows"])==1,(e,found)
 t=found["rows"][0];gid=t["globalTaskId"];conn=t["connectionId"]
 info=rows("SELECT * FROM COMPUTE_TASK_INFO WHERE globalTaskId=?",(gid,),2)
 cann=rows("SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?",(conn,))
 sets=rows("SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=? ORDER BY id",(conn,))
 torch=[]
 if not sets["truncated"]:
  for v in sets["rows"]:
   torch.append({"id":v["id"],"matches":rows("SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.sequenceNumber,a.callchainId,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?",(v["id"],))})
 out.append({"task":t,"compute_info":info,"connection_task_count":c.execute("SELECT count(*) FROM TASK WHERE connectionId=?",(conn,)).fetchone()[0],"global_id_task_count":c.execute("SELECT count(*) FROM TASK WHERE globalTaskId=?",(gid,)).fetchone()[0],"cann":cann,"connection_sets":sets,"pytorch":torch})
schemas={n:[dict(r) for r in c.execute("PRAGMA table_info(\""+n+"\")")] for n in ("TASK","COMPUTE_TASK_INFO","CANN_API","PYTORCH_API","CONNECTION_IDS")}
c.close()
print(json.dumps({"scope":"Four exact EP8 layer24 run2 tasks 1261/1264/1267/1268; task identity, schemas, limited connection matches","source_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"db_sha256":h.hexdigest(),"schemas":schemas,"rows":out,"limits":["Connection IDs/global task IDs may be reused across graph replays; not direct per-run or tensor identity proof.","Shape, proximity, names, and task sequence do not prove data dependence or same tensor/storage.","Limited connection lookup and absent fields cannot establish that unavailable data never exists in other files.","No host/device clock subtraction or NPU work."],"no_device_work":True},ensure_ascii=False,indent=2))
