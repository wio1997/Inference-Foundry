from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0095";p=r.parents[1]
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["failure_phase"]=="deploy"
raw=(r/"CPU_fullCLI_167.stdout").read_text();err=(r/"CPU_fullCLI_167.stderr").read_text()
events=[json.loads(x)for x in raw.splitlines()if x.startswith('{"event":')]
assert len(events)==2 and events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and all(x["returncode"]==0 for x in events)
assert 'cudagraph_capture_sizes==[6,12,24,48]' in err and 'AssertionError' in err
assert not any((r/name).exists()for name in["pre_cleanup_native_member_identities.json","cleanup_167.receipt.json","start_167.receipt.json","startup_model_identities.json","attempts.json"])
owners=json.loads((p/"runs/GLM-RUN-0089/adopted_model_identities.json").read_text());members=json.loads((p/"runs/GLM-RUN-0089/native_member_identities.json").read_text());opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));acks={}
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
 policy=dict(policy);policy["cohort_id"]="GLM-COHORT-0089"if key=="D0"else"GLM-COHORT-0088";policy["serial"]=3 if key=="D0"else 1
 a=dict(owner=o,workers=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]],policy=policy)
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(a).encode(),capture_output=True,timeout=90);(j/(key+".stdout")).write_bytes(z.stdout);(j/(key+".stderr")).write_bytes(z.stderr);z.check_returncode();acks[key]=json.loads(z.stdout)
 with opener.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
 (j/(key+".metrics")).write_bytes(b);vals=[float(x)for x in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M)];assert len(vals)>=2 and all(x==0 for x in vals)
sources=[]
for f in[r/"state.json",r/"deploy.phase.json",r/"CPU_fullCLI_167.stdout",r/"CPU_fullCLI_167.stderr",r/"CPU_fullCLI_167.receipt.json",r/"api_config_probe.py",r/"deploy.py"]:
 b=f.read_bytes();sources.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
out=dict(at=utc(),verdict="INVALID",measurement_valid=False,functional_acceptance=False,new_GPU_models=0,new_inference_requests=0,old_GPU_operations=0,effective_public_output_tokens=0,SDK_CPU_init_finalize=events,actual_failure="WrongCPUcaptureassertion expected6,12,24,48 withnativeSP_TRUE_TP16; nativeconfigremoved6,12,24andretained48; CPUinit_finalize0/beforeanymodelcleanup",old_P89_D88_API2_NPU32_unchanged_idle=True,current_policies={k:v["policy"]for k,v in acks.items()},ownership_acks=acks,evidence=sources,limits=["CPUtestassertionfailure beforeexactcleanup/modelstartup; notnativeGPUfit rejection/noPD orperformance evidence; frozen95source notrewritten"])
atomic_json(r/"reduction_brief.json",out);b=(r/"reduction_brief.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="95 INVALID captureCPUassertion beforemodelcleanup; oldP89D88API2NPU32idle policyP3D1 verified/noGPUoperations orrequests",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="audit",path=str(r/"reduction_brief.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="failedCPUphase/SDK0/unchangedownedoldservice")],unknowns=["IndependentEP16 GPUfit/semantics/PD/performance unexecuted"],decision_request=None,next_check_at=None));print(json.dumps(out))
