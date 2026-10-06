import json,hashlib,subprocess,sys,time
from pathlib import Path
J=Path(__file__).parent;R=J.parents[1];G=R.parents[4];sys.path.insert(0,str(G/"runtime"))
from phase_runner import atomic_json,utc
job=json.loads(Path(sys.argv[1]).read_text())
assert json.loads((R/"jobs/AUDIT-H2-20261006/cleanup.json").read_text())["profiler_stopped"]
raw=json.loads((R/"jobs/AUDIT-H2-20261006/raw_inventory.json").read_text())
assert len(raw["prof_dirs"])==16
for x in raw["files"]:
 p=Path(x["path"]);assert p.stat().st_size==x["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==x["sha256"]
argv=["docker","exec","glm52-single","bash","-c","source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; exec /usr/local/Ascend/ascend-toolkit/latest/bin/msprof --export=on --output="+str(R/"device_trace")]
with (J/"export.stdout").open("xb") as out,(J/"export.stderr").open("xb") as err:
 p=subprocess.run(argv,stdout=out,stderr=err,timeout=900)
atomic_json(J/"export_exit.json",dict(at=utc(),argv=argv,exit_code=p.returncode,scope="CPU offline export only; no NPU generation"))
assert p.returncode==0
for x in raw["files"]:assert hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"],"Original raw mutated"
files=[]
for f in sorted((R/"device_trace").rglob("*")):
 if f.is_file() and f.suffix in [".csv",".json",".db"]:
  files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
atomic_json(J/"derived_inventory.json",dict(at=utc(),files=files,raw_preserved=True))
ev=[]
for f in [J/"export_exit.json",J/"derived_inventory.json",J/"export.stdout",J/"export.stderr"]:
 ev.append(dict(id=f.name,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size,locator="Offline actual export; Sol must validate device event/rank/clock coverage"))
atomic_json(Path(job["result"]["path"]),dict(schema_version=1,job_id=job["job_id"],status="completed",summary="Existing Run244 raw exported offline, original raw hashes preserved; no new load or performance verdict.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[dict(kind="fact",text="Actual export exit0, original raw unchanged",scope=dict(run="GLM-RUN-0244",offline=True),evidence_ids=["export_exit.json","derived_inventory.json"])],evidence=ev,unknowns=["Actual per-rank event completeness, timing, causal Gap and E2E Gain require Sol analysis"],decision_request=None,next_check_at=None))
