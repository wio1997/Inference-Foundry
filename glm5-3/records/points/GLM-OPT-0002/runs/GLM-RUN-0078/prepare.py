from pathlib import Path
import json,sys,subprocess,shlex,hashlib,urllib.request,re,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0077";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
proof=json.loads((old/"reduction_brief.json").read_text());assert proof["functional_acceptance"]and proof["effective_public_output_tokens"]==192
owners=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text());history=[];results=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}));policy=dict(schema_version=1,cohort_id="GLM-COHORT-0077",budget_tokens=4096,prefill_threshold_tokens=2048,prefill_cadence=2,serial=1)
POLICY="/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run77/issue_budget_policy.json"
SOURCE="/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run77/issue_budget_scheduler_v3.py"
SHA="4ac2c4f76bbbf778e3919090d18338cafe0a19061bc903961429ebd3707f531c"
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
ports=subprocess.check_output(["ss","-ltnp"],text=True);assert not re.search(r":8002\s",ports)
cpu=json.loads((r.parents[1]/"jobs/RESPONSES-BUDGET-CLI-CPU-20261002T1326Z/reduction.json").read_text())
assert cpu["CPU_contracts"]["passed"]and cpu["CPU_contracts"]["tests_run"]==14 and cpu["native_actual_requests"]==0
atomic_json(r/"prepared_state.json",dict(at=utc(),retained_native_API2_NPU32=True,policy=policy,port8002_free=True,CPU_gatewayV4_14contracts=True,model_operations=0))
print(json.dumps(dict(prepared=True,epoch_same=True,policy=policy)))
