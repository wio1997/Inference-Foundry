from pathlib import Path
import json,sys,subprocess,shlex,hashlib,urllib.request,re,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0067";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
proof=json.loads((old/"reduction_brief.json").read_text());assert proof["functional_acceptance"]and proof["CPU_issue_budget"]==4096 and proof["effective_public_output_tokens"]==768
owners=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text());history=[];results=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}));policy=dict(schema_version=1,cohort_id="GLM-COHORT-0067",budget_tokens=4096,serial=1)
POLICY="/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run67/issue_budget_policy.json"
SOURCE="/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run67/issue_budget_scheduler_v2.py"
SHA="a0081b66089665571f0b854a28321164de283e0bb0227b420092904dc5858585"
check="""import pathlib,json,sys,hashlib
a=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();o=a['owner']
for x in a['targets']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=pathlib.Path('/proc/'+str(x['pid']));b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
env=dict(v.decode().split('=',1)for v in pathlib.Path('/proc/'+str(o['pid'])+'/environ').read_bytes().split(bytes([0]))if b'='in v)
assert env['GLM_ISSUE_BUDGET_COHORT']==a['policy']['cohort_id']and env['GLM_ISSUE_BUDGET_POLICY']==a['path']and env['VLLM_ENABLE_RESPONSES_API_STORE']=='1'
assert hashlib.sha256(pathlib.Path(a['source']).read_bytes()).hexdigest()==a['sha']
assert json.loads(pathlib.Path(a['path']).read_text())==a['policy'];print(json.dumps(dict(API_same=True,NPU16_same=True,source_same=True,policy=a['policy'])))
"""
write=r"""import json,pathlib,sys,tempfile,os,hashlib
a=json.load(sys.stdin);p=pathlib.Path(a['path']);old=p.read_bytes();assert json.loads(old)==a['previous']
new=a['new'];assert new['schema_version']==1 and new['cohort_id']=='GLM-COHORT-0067'and type(new['budget_tokens'])is int and new['budget_tokens']%128==0 and 512<=new['budget_tokens']<=16384 and new['serial']==a['previous']['serial']+1
data=(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n').encode();fd,tmp=tempfile.mkstemp(prefix='.glm68policy-',dir=str(p.parent))
try:
 os.fchmod(fd,0o600)
 with os.fdopen(fd,'wb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==old
 os.replace(tmp,p)
finally:
 if os.path.exists(tmp):os.unlink(tmp)
assert p.read_bytes()==data;print(json.dumps(dict(previous_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256(data).hexdigest(),path=str(p),new=new,mode=oct(p.stat().st_mode&0o777),atomic_replace=True)))
"""
def run(node,code,payload,stem):
 guard();args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(payload).encode(),capture_output=True,timeout=90);(r/(stem+".stdout")).write_bytes(z.stdout);(r/(stem+".stderr")).write_bytes(z.stderr);z.check_returncode();return json.loads(z.stdout)
def verify(label):
 for key,o in owners.items():
  targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]]
  ack=run(o["host"],check,dict(owner=o,targets=targets,policy=policy,path=POLICY,source=SOURCE,sha=SHA),label+"_"+key)
  with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
  (r/(label+"_"+key+".metrics")).write_bytes(b)
  for name in["num_requests_running","num_requests_waiting"]:
   vals=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert vals and all(float(v)==0 for v in vals)
verify("initial")
for budget in[16384,4096,1024]:
 guard();verify("before_update_"+str(budget));next_policy=dict(policy,budget_tokens=budget,serial=policy["serial"]+1);acks={}
 for node in["166","167"]:
  acks[node]=run(node,write,dict(path=POLICY,previous=policy,new=next_policy),"update_"+str(budget)+"_"+node)
  atomic_json(r/"policy_progress.json",dict(at=utc(),previous=policy,new=next_policy,acks=acks,complete=len(acks)==2,no_new_requests_before_both_acks=True))
 history.append(dict(at=utc(),previous=policy,new=next_policy,acks=acks));atomic_json(r/"policy_history.json",history);policy=next_policy;verify("after_update_"+str(budget))
 w=r/("GLM-RUN-0068-budget"+str(budget)+"-serial"+str(policy["serial"]));w.mkdir()
 for name in["canonical_81932.body.json","adopted_model_identities.json","native_member_identities.json"]:(w/name).write_bytes((r/name).read_bytes())
 atomic_json(w/"policy.json",policy)
 with(w/"mixed.stdout").open("wb")as out,(w/"mixed.stderr").open("wb")as err:
  z=subprocess.run([sys.executable,str(r/"mixed_template.py"),str(w)],stdout=out,stderr=err,timeout=660)
 atomic_json(w/"execution.json",dict(at=utc(),exit_code=z.returncode,template_sha256=hashlib.sha256((r/"mixed_template.py").read_bytes()).hexdigest(),policy=policy));z.check_returncode()
 a=json.loads((w/"mixed_summary.json").read_text());assert a["functional_acceptance"]and a["effective_public_output_tokens"]==768
 verify("after_window_"+str(budget));results.append(dict(window=str(w),policy=policy,effective_public_output_tokens=768,elapsed_s=a["elapsed_s"],requests=[{k:v[k]for k in["id","ttft_s","wall_s","usage","max_native_chunk_gap_s","wire"]}for v in a["requests"]]))
 atomic_json(r/"matrix_progress.json",dict(at=utc(),completed=results,no_new_models=True,no_model_signals=True));print(json.dumps(dict(budget=budget,outputs=768,TTFT=[v["ttft_s"]for v in a["requests"]],chunkgap=[v["max_native_chunk_gap_s"]for v in a["requests"]])),flush=True)
out=dict(at=utc(),measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",new_requests=15,effective_public_output_tokens=2304,windows=results,policy_history=history,retained_physical_cohort="D67 API2/NPU32 originalepochs",models_started=0,model_signals=0,GPUallocation16384_unchanged=True,terminal_policy=policy,limits=["Finite3 ordered samecohort directnative cold mixed windows; no formalSLO/capacity/KEEP/fullAPI/state/PD","Sameworkload freshsalts andnativeprefillcold counters/raw needindependentaudit; order/shape-warmup/thermal/MTPtrajectory may differ","No arbitrary hot topology/KV/capture changes; onlyboundedCPUissuebudget policy onalreadyallocated16k buffers","Eachbothpolicyack/idle before newwindow; atomicpernodefiles are not globalatomic live-request adaptive control"])
atomic_json(r/"matrix_summary.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=out);atomic_json(r/"manifest.json",m);(r/"summary.md").write_text("# "+r.name+"\n\n"+json.dumps(dict(new_requests=15,outputs=2304,policy=policy,models=0,signals=0))+"\n\nIndependentraw/nativebudget/counterauditpending, noKEEP.\n");print(json.dumps(out))

