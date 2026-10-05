from pathlib import Path
import json,ast,hashlib,subprocess,shlex,urllib.request,re,sys,types
from datetime import datetime,timezone
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0208"
def ref(f):
 raw=f.read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
prev=json.loads((p/"runs/GLM-RUN-0208/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def check(label):
 proof={};counters={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=90);f=j/(label+"_"+key+".stdout");f.write_bytes(z.stdout);z.check_returncode();v=json.loads(z.stdout)
  known={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
  assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(known[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"]);proof[key]=ref(f)
  with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=15)as response:raw=response.read()
  (j/(label+"_"+key+".metrics")).write_bytes(raw);vals={}
  for l in raw.decode().splitlines():
   if l.startswith("vllm:"):
    name=l.split("{")[0].split()[0]
    if name.endswith("_total")or name in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]:vals[name]=vals.get(name,0)+float(l.rsplit(" ",1)[1])
  assert vals["vllm:num_requests_running"]==vals["vllm:num_requests_waiting"]==vals["vllm:kv_cache_usage_perc"]==0;counters[key]=vals
 return proof,counters
before,bc=check("before")
sourcepath="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp200/issue_budget_scheduler_v3.py";args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c","from pathlib import Path;import sys;sys.stdout.write(Path("+repr(sourcepath)+").read_text())"])]
z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();source=z.stdout.decode();assert hashlib.sha256(z.stdout).hexdigest()=="4ac2c4f76bbbf778e3919090d18338cafe0a19061bc903961429ebd3707f531c";(j/"native_reader_source.py").write_bytes(z.stdout)
module=ast.parse(source);cls=next(x for x in module.body if isinstance(x,ast.ClassDef)and x.name=="IssueBudgetMixin");method=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=="_glm_read_controls")
messages=[];ns=dict(json=json,log=types.SimpleNamespace(info=lambda *a:messages.append(a)));exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),"actual_native_reader_AST","exec"),ns)
c=(r/"controls.py").read_text();assert hashlib.sha256(c.encode()).hexdigest()=="8f2ef3e7f4c04a6f29560a03cc2c1b7fa46996b51e4b382532b7da95a8eb2811"
module=ast.parse(c);writer=next(ast.literal_eval(x.value)for x in module.body if isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id=="write"for y in x.targets));compile(writer,"actual208CAS","exec")
fixture=j/"fixture";fixture.mkdir();f=fixture/"policy.json";initial=dict(schema_version=1,cohort_id="GLM-COHORT-0200",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3)
f.write_text(json.dumps(initial)+chr(10));new=dict(initial,prefill_threshold_tokens=4096,serial=4)
def transaction(previous,new):
 z=subprocess.run([sys.executable,"-c",writer],input=json.dumps(dict(path=str(f),previous=previous,new=new)).encode(),capture_output=True,timeout=15)
 return z
z=transaction(initial,new);assert z.returncode==0and json.loads(f.read_text())==new;positive=json.loads(z.stdout)
dummy=types.SimpleNamespace(_glm_default_budget=8192,_glm_native_budget=8192,_glm_cohort="GLM-COHORT-0200",_glm_budget_path=f,_glm_budget_observed=None)
assert ns["_glm_read_controls"](dummy)==(8192,4096,1)and dummy._glm_budget_observed==(8192,4096,1,4,None)
negative=[]
for name,bad in[("stale_previous",dict(new)),("wrong_serial",dict(new,serial=6)),("wrong_cohort",dict(new,cohort_id="OTHER",serial=5)),("wrong_budget",dict(new,budget_tokens=16384,serial=5)),("wrong_threshold",dict(new,prefill_threshold_tokens=8193,serial=5)),("wrong_cadence",dict(new,prefill_cadence=2,serial=5))]:
 previous=initial if name=="stale_previous"else new
 if name=="stale_previous":bad=dict(new,prefill_threshold_tokens=1024,serial=5)
 raw=f.read_bytes();z=transaction(previous,bad);assert z.returncode!=0and f.read_bytes()==raw;negative.append(dict(case=name,rejected=True,file_unchanged=True))
restored=dict(initial,serial=5);z=transaction(new,restored);assert z.returncode==0and json.loads(f.read_text())==restored
# Actual native reader rejects an outside-allocation threshold by safe fallback; no livepolicy writes.
bad=dict(restored,prefill_threshold_tokens=8193);f.write_text(json.dumps(bad));assert ns["_glm_read_controls"](dummy)==(8192,0,1)and dummy._glm_budget_observed[-1]is not None
after,ac=check("after");assert bc==ac
out=dict(at=datetime.now(timezone.utc).isoformat(),valid=True,kind="CPU_actual_native_reader_and208_atomicCAS",controls_sha256=hashlib.sha256(c.encode()).hexdigest(),native_control_reader4096_legal=True,native_reader_source=ref(j/"native_reader_source.py"),source_alias="Actual167 nativeIssueBudgetMixin._glm_read_controls AST/standardlibrary only; no vLLM-Torch-NPU import",positive_negative_atomic_CAS=True,positive=positive,negative=negative,restoration_positive=True,actual_native_reader_invalid_fallback=True,current_native32_same_idle_vllm_counters=True,physical_before=before,physical_after=after,native_private_policy_writes=0,models=0,workers=0,inference=0,NPU_tensors=0,SDKcalls=0,Current=None,limits=["SyntheticCPU policy files only; actual source reader4096 bounds and atomictransaction proof do not certify GPUsteps/nativeconsumption/E2E/QoS","CurrentD1 PP4TP4DCP4 K1/allocated8192/Graph/epoch/source unchanged; next unique208 nativeSELECTED threshold4096 serial4 and full10112E2E required, restored1024serial5 afterwindow","No framework/operator math/guard changes or model lifecycle; existing fullframework204 native32-STORE-SDK function reused"])
f=j/"reduction.json";f.write_text(json.dumps(out,ensure_ascii=False,indent=2)+chr(10))
result=dict(schema_version=1,job_id=j.name,status="completed",summary="CPU actualnative D1 threshold4096 reader +208 atomicCAS positive-negative/restoration VALID/currentnative32sameidle/counters/nolivepolicy-model-inference; GPUQoS unknown",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="native-readerAST/CPUtransaction/actual32/counters")],unknowns=out["limits"],decision_request=None,next_check_at=None);(j/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10));print(json.dumps(dict(valid=True,proof=ref(f))))
