import json,hashlib,sqlite3,csv,sys,collections
from pathlib import Path
J=Path(__file__).parent;R=J.parents[1];G=R.parents[4];sys.path.insert(0,str(G/"runtime"))
from phase_runner import atomic_json,utc
job=json.loads(Path(sys.argv[1]).read_text());data={};refs=[]
for cp in [2160760,2161164]:
 f=next(p for p in (R/"device_trace").glob("PROF*/mindstudio_profiler_output/msprof*.json") if "_0"+str(cp) in str(p));v=json.loads(f.read_text());names={e["pid"]:e["args"]["name"]for e in v if e.get("ph")=="M" and e.get("name")=="process_name"};threads={(e["pid"],e["tid"]):e["args"]["name"]for e in v if e.get("ph")=="M" and e.get("name")=="thread_name"}
 broadcasts=sorted([dict(index=i,event=e,track=names[e["pid"]],thread=threads.get((e["pid"],e["tid"])))for i,e in enumerate(v)if e.get("ph")=="X" and names.get(e.get("pid"))=="Communication" and e["name"].startswith("hcom_broadcast") and e.get("args",{}).get("data_type")=="INT64" and e.get("args",{}).get("rank_size")==2],key=lambda e:float(e["event"]["ts"]))
 sends=sorted([dict(index=i,event=e)for i,e in enumerate(v)if e.get("ph")=="X" and names.get(e.get("pid"))=="Communication" and e["name"].startswith("hcom_send") and e.get("args",{}).get("data_type")=="BFP16" and e.get("args",{}).get("rank_size")==2],key=lambda e:float(e["event"]["ts"]))
 conns={b["event"]["args"]["connection_id"]for b in broadcasts};linked=[dict(index=i,event=e,track=names.get(e.get("pid")),thread=threads.get((e.get("pid"),e.get("tid"))))for i,e in enumerate(v)if e.get("ph")=="X" and e.get("args",{}).get("connection_id") in conns]
 csvp=next(f.parent.glob("op_summary*.csv"));csvrows=[]
 with csvp.open() as file:
  for x in csv.DictReader(file):
   if x["Task Type"]=="COMMUNICATION" and "broadcast"in x["Op Name"]:csvrows.append(x)
 db=next(f.parent.parent.glob("msprof*.db"));conn=sqlite3.connect("file:"+str(db)+"?mode=ro",uri=True);schema=[dict(table=x[0],sql=x[1])for x in conn.execute("select name,sql from sqlite_master where type='table'")];conn.close()
 data[cp]=dict(trace=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size,broadcasts=broadcasts,sends=sends,linked_events=linked,csv_broadcast=csvrows,database=str(db),schema=schema)
 del v
p0=data[2160760];p1=data[2161164];peer={x["event"]["name"]:x for x in p1["broadcasts"]};paired=[]
for b in p0["broadcasts"]:
 e=b["event"];mate=peer[e["name"]];p=mate["event"];assert e["args"]["count"]==p["args"]["count"]
 t=float(e["ts"]);u=float(p["ts"]);end=t+float(e["dur"])
 paired.append(dict(name=e["name"],count=e["args"]["count"],derived_num_requests=e["args"]["count"]//5,pp0_begin_us=t,pp0_end_us=end,pp1_begin_us=u,pp1_end_us=u+float(p["dur"]),receiver_elapsed_us=float(e["dur"]),sender_elapsed_us=float(p["dur"]),peer_arrival_gap_us=u-t,completion_skew_us=end-u-float(p["dur"]),pp0_array_index=b["index"],pp1_array_index=mate["index"]))
atomic_json(J/"dependency_raw.json",dict(at=utc(),ranks=data,paired_broadcasts=paired,scope="Actual raw array indices, same named2-rank INT64 broadcast/count; request-ID causality and AIV resource occupancy unknown"))
ev=[]
for p in [J/"dependency_raw.json"]:ev.append(dict(id=p.name,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),locator="PP0/PP1 matching collective names, all original host/communication/hardware connection IDs and schemas"))
atomic_json(Path(job["result"]["path"]),dict(schema_version=1,job_id=job["job_id"],status="completed",summary="Existing all28 PP payload broadcasts linked across selected PP rank0 peers; CPU only, no Gap verdict.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[dict(kind="fact",text="Same collective names and counts paired; original host/hardware connection witnesses retained",scope=dict(run="GLM-RUN-0244",tp_rank=0,offline=True),evidence_ids=["dependency_raw.json"])],evidence=ev,unknowns=["No device block occupancy/concurrent independent cohort proof; tail counts not matched half-batch baseline","No inferred removable time or E2E gain"],decision_request=None,next_check_at=None))
