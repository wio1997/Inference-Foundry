import pathlib,json,sys,urllib.request,re,subprocess,hashlib,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json,utc
r=pathlib.Path(__file__).parent;old=r.parent/"GLM-RUN-0028";repo=r.parents[4]
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed" and not same_process(state["owner"])
assert state["completed_stages"]==[] and state["failure_phase"]=="deploy"
phase=json.loads((old/"deploy.phase.json").read_text());assert phase["status"]=="failed" and phase["exit_code"]==1 and not same_process(phase["child"])
assert "AssertionError: (0, 0)" in (old/"deploy.log").read_text()
assert not (old/"cold_summary.json").exists() and not any((old/n).exists() for n in ["control_DP0","control_DP1","capability","pilot","prefix_study"])
for pin in json.loads((old/"controller_spec.json").read_text())["stages"][0]["sources"]:
 assert hashlib.sha256(pathlib.Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"],pin["path"]
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
identities=json.loads((r/"adopted_model_identities.json").read_text())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
 def command(args,body=None):
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  p=subprocess.run(args,input=body,capture_output=True,timeout=60);p.check_returncode();return p.stdout
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=5) as reply:text=reply.read().decode()
 (r/("initial_"+node+".metrics")).write_text(text)
 for key in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total"]:
  vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(v)==0 for v in vals),(node,key,vals)
 for n in ["coupled_dp_worker.py","coupled_dp_metadata.py"]:
  target="/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_dp_run28/"+n
  data=command(["cat",target]);assert data==(repo/"runtime"/n).read_bytes()
 code="""import pathlib,json,hashlib,logging
from vllm.logger import init_logger
from coupled_dp_metadata import make_sources
p=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
sources=make_sources(p.read_text())
a=init_logger('coupled_dp_metadata');b=init_logger('vllm.glm_coupled_dp_metadata')
print(json.dumps({'native_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_guards_passed':True,'external_INFO_enabled':a.isEnabledFor(logging.INFO),'native_INFO_enabled':b.isEnabledFor(logging.INFO),'inference_calls':0,'constructed_model_worker':False}))
"""
 raw=command(["docker","exec","-i","-e","PYTHONPATH=/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_dp_run28","glm52-single","python3","-"],code.encode())
 (r/("metadata_control_"+node+".stdout")).write_bytes(raw)
 receipt=json.loads(raw.decode().splitlines()[-1]);assert receipt["source_guards_passed"]
 pid=identities["DP"+str(rank)]["pid"]
 env=command(["python3","-c","import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"/environ');print(json.dumps([x.decode() for x in p.read_bytes().split(bytes([0])) if x.startswith(b'PYTHONPATH=')]))"])
 (r/("worker_import_path_"+node+".json")).write_bytes(env);assert "coupled_dp_run28" in json.loads(env)[0]
p=subprocess.run(["ss","-ltnp"],capture_output=True,text=True);p.check_returncode();assert not re.search(r":8002\s",p.stdout)
atomic_json(r/"prepared_identity.json",{"at":utc(),"exact_cohort_retained":True,"previous_run_attempts":0,"operator_changes":False,"installation":"source-conditioned inference from explicit Worker resolution + successful init_device/load/Graph/start, not direct object/shape telemetry","scope":"new bounded real E2E, no telemetry completeness prerequisite"})
