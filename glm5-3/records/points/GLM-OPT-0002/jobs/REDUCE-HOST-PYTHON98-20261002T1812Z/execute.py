from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0098";p=r.parents[1]
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["failure_phase"]=="arrivals"
raw=(r/"gateway.log").read_text();err=(r/"arrivals.log").read_text();events=[]
assert "ModuleNotFoundError: No module named 'uvicorn'"in raw and 'AssertionError'in err
assert not any((r/name).exists()for name in ["native_dynamic.stdout","validation_native400.json","native_tokenized_counts.json","arrival_plan.json","attempts.json","before_state.json"])
owners=json.loads((p/"runs/GLM-RUN-0096/adopted_model_identities.json").read_text());members=json.loads((p/"runs/GLM-RUN-0096/native_member_identities.json").read_text());opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));acks={}
code="""import pathlib,json,sys,re,subprocess,hashlib
a=json.load(sys.stdin);o=a['owner'];boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for v in [dict(pid=o['pid'],identity=o['identity'])]+a['workers']:
 f=pathlib.Path('/proc/'+str(v['pid']));s=(f/'stat').read_text();x=s[s.rfind(')')+2:].split();assert x[0]not in['Z','X']and dict(boot_id=boot,start_ticks=x[19])==v['identity']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
b=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);actual={int(v)for v in re.findall(r'^\\|\\s+\\d+\\s+\\d+\\s+\\|\\s+(\\d+)\\s+\\|\\s+VLLMWorker',b,re.M)};assert actual==set(v['pid']for v in a['workers'])
policy=json.loads((pathlib.Path(o['argv'][1]).parent/'issue_budget_policy.json').read_text());assert policy==a['policy']
print(json.dumps(dict(API_same=True,NPU16_same=True,policy=policy,npu_smi_sha256=hashlib.sha256(b.encode()).hexdigest())))
"""
policy=json.loads((p/"runs/GLM-RUN-0089/issue_budget_policy.json").read_text())
for key,o in owners.items():
 policy=dict(policy);policy["cohort_id"]="GLM-COHORT-0089"if key=="D0"else"GLM-COHORT-0096";policy["serial"]=3 if key=="D0"else 1
 a=dict(owner=o,workers=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]],policy=policy)
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(a).encode(),capture_output=True,timeout=90);(j/(key+".stdout")).write_bytes(z.stdout);(j/(key+".stderr")).write_bytes(z.stderr);z.check_returncode();acks[key]=json.loads(z.stdout)
 with opener.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
 (j/(key+".metrics")).write_bytes(b);vals=[float(x)for x in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M)];assert len(vals)>=2 and all(x==0 for x in vals)
sources=[]
for f in[r/"state.json",r/"arrivals.phase.json",r/"gateway.log",r/"arrivals.log",r/"arrivals.py",r/"controller_spec.json"]:
 b=f.read_bytes();sources.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
def totals(f):
 out={}
 for l in f.read_text().splitlines():
  if not l or l.startswith("#"):continue
  n=l.split("{")[0].split()[0]
  if n.endswith("_total"):out[n]=out.get(n,0)+float(l.rsplit(" ",1)[1])
 return out
for key in owners:
 assert totals(r/("epoch_prepare_"+key+".metrics"))==totals(j/(key+".metrics"))
out=dict(at=utc(),verdict="INVALID",measurement_valid=False,functional_acceptance=False,new_GPU_models=0,new_inference_requests=0,old_GPU_operations=0,effective_public_output_tokens=0,task_CPU_SDK_events=events,actual_failure="HostPython3.9 gatewaystartup missinguvicorn; wrongcontrollerstage argv omittedexistingcontainerdynamicwrapper; beforeanytokenization/inference/modelops. NativeP89D96sameepochs/counters/idle/policyP3D1 preserved",old_P89_D96_API2_NPU32_unchanged_idle=True,current_policies={k:v["policy"]for k,v in acks.items()},ownership_acks=acks,evidence=sources,limits=["Hostruntimeentryfailure beforeinference/modelops; notnativeGPUfit rejection/noPD orperformance evidence; frozen98source notrewritten"])
atomic_json(r/"reduction_brief.json",out);b=(r/"reduction_brief.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="98 INVALID hostPython missinguvicorn/omittedcontainerwrapper beforeinference; P89D96API2NPU32idle/P3D1/counters unchanged",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="audit",path=str(r/"reduction_brief.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="failedCPUphase/SDK0/unchangedownedoldservice")],unknowns=["IndependentEP16 GPUfit/semantics/PD/performance unexecuted"],decision_request=None,next_check_at=None));print(json.dumps(out))
