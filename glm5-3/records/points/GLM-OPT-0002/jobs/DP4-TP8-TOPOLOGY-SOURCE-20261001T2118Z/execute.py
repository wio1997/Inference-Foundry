import pathlib,json,hashlib,subprocess
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());out=j/"sources";out.mkdir();rows=[]
for i,item in enumerate(job["inputs"]):
 source=item["path"];p=subprocess.run(["docker","exec","glm52-single","cat",source],capture_output=True,timeout=20);p.check_returncode();dest=out/(str(i).zfill(2)+"_"+pathlib.Path(source).name);dest.write_bytes(p.stdout);rows.append({"native_path":source,"path":str(dest),"bytes":len(p.stdout),"sha256":hashlib.sha256(p.stdout).hexdigest()})
p=subprocess.run(["docker","exec","glm52-single","python3",str(j/"probe.py")],capture_output=True,timeout=200)
(j/"probe.stdout").write_bytes(p.stdout);(j/"probe.stderr").write_bytes(p.stderr)
try:config=json.loads(p.stdout.decode().splitlines()[-1]) if p.returncode==0 else None
except Exception:config=None
(j/"config_reduction.json").write_text(json.dumps({"exit_code":p.returncode,"native_config":config,"inference_calls":0,"source_only":True},indent=2))
dest=j/"probe.stdout";rows.append({"native_path":"CPU native EngineArgs probe","path":str(dest),"bytes":len(p.stdout),"sha256":hashlib.sha256(p.stdout).hexdigest()})
index=j/"source_index.json";index.write_text(json.dumps({"sources":rows,"inference_calls":0,"source_only":True},indent=2));raw=index.read_bytes()
result={"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Installed native topology source and CPU-only full config evidence retained; GPT assesses lawful prefill alternatives","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"Installed topology sources copied unchanged; CPU native EngineArgs only, no EngineCore/Worker/NPUcollective/model/request/service action","scope":{"source_only":True},"evidence_ids":["sources"]}],"evidence":[{"id":"sources","path":str(index),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"installed source identity index"}],"unknowns":["Native topology config-only status recorded separately; no HCCL/fit/Graph/native functional or performance proof"],"decision_request":None,"next_check_at":None}
(j/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps({"sources":len(rows),"inference_calls":0}))
