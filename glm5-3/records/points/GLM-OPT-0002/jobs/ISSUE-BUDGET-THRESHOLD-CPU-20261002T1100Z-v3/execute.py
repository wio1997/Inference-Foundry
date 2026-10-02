from pathlib import Path
import json,sys,hashlib,subprocess,shlex,ast
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0065";root=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime");source=root/"issue_budget_scheduler_v3.py";assert hashlib.sha256(source.read_bytes()).hexdigest()=="4ac2c4f76bbbf778e3919090d18338cafe0a19061bc903961429ebd3707f531c"
inner=(r/"api_config_probe.py").read_text()
needle="config=EngineArgs.from_cli_args(args).create_engine_config()"
replacement="""config=EngineArgs.from_cli_args(args).create_engine_config()
from issue_budget_scheduler_v3 import BudgetScheduler,IssueBudgetMixin,proof as budget_proof
assert config.scheduler_config.get_scheduler_cls()is BudgetScheduler
baseline_args=EngineArgs.from_cli_args(args);baseline_args.scheduler_cls=None;baseline=baseline_args.create_engine_config()
bp=budget_proof(config);assert bp["native_scheduler_cls"]==budget_proof(baseline)["native_scheduler_cls"],"native base scheduler mode changed"
assert config.scheduler_config.async_scheduling==baseline.scheduler_config.async_scheduling
import os
policy=Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]);test=object.__new__(IssueBudgetMixin);test._glm_native_budget=16384;test._glm_default_budget=4096;test._glm_budget_path=policy;test._glm_cohort=os.environ["GLM_ISSUE_BUDGET_COHORT"];test._glm_budget_observed=None
def select(obj,expected):
 policy.write_text(json.dumps(obj));actual=test._glm_read_controls();assert actual==expected;return dict(input=obj,selected=actual)
base=dict(schema_version=1,cohort_id=test._glm_cohort,budget_tokens=4096,serial=1)
cases=[]
for obj,expected in[
 (base,(4096,0,1)),
 (dict(base,budget_tokens=16384),(16384,0,1)),
 (dict(base,budget_tokens=1024,prefill_threshold_tokens=512,prefill_cadence=2),(1024,512,2)),
 (dict(base,prefill_threshold_tokens=2048,prefill_cadence=1),(4096,2048,1)),
 (dict(base,prefill_threshold_tokens=2048,prefill_cadence=2),(4096,2048,2)),
 (dict(base,prefill_threshold_tokens=8192),(4096,0,1)),
 (dict(base,prefill_threshold_tokens=True),(4096,0,1)),
 (dict(base,prefill_threshold_tokens=129),(4096,0,1)),
 (dict(base,prefill_threshold_tokens=-128),(4096,0,1)),
 (dict(base,prefill_cadence=3),(4096,0,1)),
 (dict(base,prefill_cadence=True),(4096,0,1)),
 (dict(base,budget_tokens=32768),(4096,0,1)),
 (dict(base,cohort_id="foreign"),(4096,0,1)),
 (dict(base,serial=-1),(4096,0,1)),
 ([],(4096,0,1))]:
 cases.append(select(obj,expected))
policy.write_bytes(b"x"*4097);assert test._glm_read_controls()==(4096,0,1);cases.append(dict(input="oversized4097",selected=(4096,0,1)))
policy.unlink();assert test._glm_read_controls()==(4096,0,1);cases.append(dict(input="missing",selected=(4096,0,1)))
assert config.scheduler_config.prefill_schedule_interval==2 and config.scheduler_config.long_prefill_token_threshold==0
bp.update(async_scheduling=config.scheduler_config.async_scheduling,policy_cases=cases,scope="Policy parsing/native base type only; no nativeScheduler instance/KV/model/comm/GPU/performance")
"""
assert inner.count(needle)==1;inner=inner.replace(needle,replacement)
needle='"event":"full_native_CLI_API_config_valid"';inner=inner.replace(needle,needle+',"issue_budget_CPU_proof":bp');ast.parse(inner);(j/"inner.py").write_text(inner)
planned=json.loads((r/"planned_launch.json").read_text());outrows=[]
for x in planned:
 argv=x["argv"]+["--scheduler-cls","issue_budget_scheduler_v3.BudgetScheduler","--prefill-schedule-interval","2"];node=x["node"]
 script="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(root)+":/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run65:$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" GLM_ISSUE_BUDGET_POLICY="+str(j/("policy_"+node+".json"))+" GLM_ISSUE_BUDGET_COHORT=GLM-COHORT-0069 HCCL_BUFFSIZE=4096; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run65/native_acl_lifecycle.py script "+str(j/"inner.py")+" "+shlex.quote(json.dumps(argv))
 if node=="167":
  cmd=["ssh","-o","BatchMode=yes","root@172.16.10.167","mkdir -p "+shlex.quote(str(j))+" "+shlex.quote(str(root))];subprocess.run(cmd,capture_output=True,check=True)
  for f in[j/"inner.py",source]:subprocess.run(["scp","-q",str(f),"root@172.16.10.167:"+str(f)],capture_output=True,check=True)
 args=["docker","exec","glm52-single","bash","-c",script]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=240);(j/(node+".stdout")).write_bytes(z.stdout);(j/(node+".stderr")).write_bytes(z.stderr);z.check_returncode()
 events=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert len(events)==3 and events[0]["returncode"]==events[-1]["returncode"]==0
 outrows.append(dict(node=node,actualevents=events,stdout=dict(path=str(j/(node+".stdout")),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest())))
out=dict(at=utc(),source=dict(path=str(source),bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest()),rows=outrows,models=0,requests=0,signals=0,scope="TwoactualfullnativeCLI/API/SDKinitfinally0/nativebase-mode preserved/policy9boundarycases each; no instantiatednativeScheduler KV/GPU/E2E/capacity")
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="BudgetCPU two fullnativeCLI/API/nativebase/asyncmode preserved +17budget_threshold_cadence_boundarycases each, zeroGPU/model/requests",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualnativeconfig/source/policy-boundaries")],unknowns=["No nativeScheduler-KV instance or GPU/E2E/budget speed/capacity proof"],decision_request=None,next_check_at=None));print(json.dumps(out))

