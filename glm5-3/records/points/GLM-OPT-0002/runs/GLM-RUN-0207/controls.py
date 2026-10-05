from pathlib import Path
import json,sys,subprocess,shlex,hashlib,urllib.request,re,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
r=Path(__file__).parent
owners=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());history=[];results=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}));policy=dict(schema_version=1,cohort_id="GLM-COHORT-0204",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
POLICY={node:"/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp204"if node=="166"else"local_pp200")+"/issue_budget_policy.json"for node in["166","167"]}
policies={"166":dict(policy),"167":dict(policy,cohort_id="GLM-COHORT-0200",budget_tokens=8192,prefill_threshold_tokens=1024,serial=1)}
SOURCE={node:"/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp204"if node=="166"else"local_pp200")+("/issue_budget_scheduler_v5.py")for node in["166","167"]}
SHAS={"166":"d03a5e6b470aab4abab3d42244a494be299b032e528fb01c7c74fff2286deb6b","167":"d03a5e6b470aab4abab3d42244a494be299b032e528fb01c7c74fff2286deb6b"}
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
write='import json,pathlib,sys,tempfile,os,hashlib\na=json.load(sys.stdin);p=pathlib.Path(a["path"]);old=p.read_bytes();previous=a["previous"];assert json.loads(old)==previous\nnew=a["new"];assert set(new)==set(previous)=={"schema_version","cohort_id","budget_tokens","prefill_threshold_tokens","prefill_cadence","serial"}\nassert new["schema_version"]==previous["schema_version"]==1 and new["cohort_id"]==previous["cohort_id"]=="GLM-COHORT-0200"\nassert type(new["budget_tokens"])is int and new["budget_tokens"]==previous["budget_tokens"]==8192\nassert type(new["prefill_threshold_tokens"])is int and new["prefill_threshold_tokens"]in[1024,2048] and new["prefill_threshold_tokens"]%128==0\nassert type(new["prefill_cadence"])is int and new["prefill_cadence"]==previous["prefill_cadence"]==1\nassert type(new["serial"])is int and new["serial"]==previous["serial"]+1\ndata=(json.dumps(new,sort_keys=True,separators=(",",":"))+chr(10)).encode();fd,tmp=tempfile.mkstemp(prefix=".glm207policy-",dir=str(p.parent))\ntry:\n os.fchmod(fd,0o600)\n with os.fdopen(fd,"wb")as f:f.write(data);f.flush();os.fsync(f.fileno())\n assert p.read_bytes()==old\n os.replace(tmp,p)\nfinally:\n if os.path.exists(tmp):os.unlink(tmp)\nassert p.read_bytes()==data;print(json.dumps(dict(previous_sha256=hashlib.sha256(old).hexdigest(),new_sha256=hashlib.sha256(data).hexdigest(),path=str(p),new=new,mode=oct(p.stat().st_mode&0o777),atomic_replace=True)))\n'

def run(node,code,payload,stem):
 guard();args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(payload).encode(),capture_output=True,timeout=90);(r/(stem+".stdout")).write_bytes(z.stdout);(r/(stem+".stderr")).write_bytes(z.stderr);z.check_returncode();return json.loads(z.stdout)
def verify(label):
 for key,o in owners.items():
  targets=[v for v in members[key]["owned_targets"]if v["pid"]in members[key]["npu_worker_pids"]]
  ack=run(o["host"],check,dict(owner=o,targets=targets,policy=policies[o["host"]],path=POLICY[o["host"]],source=SOURCE[o["host"]],sha=SHAS[o["host"]]),label+"_"+key)
  with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
  (r/(label+"_"+key+".metrics")).write_bytes(b)
  for name in["num_requests_running","num_requests_waiting"]:
   vals=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert vals and all(float(v)==0 for v in vals)


def update_D1(next_policy,label):
 verify(label+"_before")
 with http.open("http://127.0.0.1:8000/healthcheck",timeout=15)as response:h=json.loads(response.read());assert h["status"]=="ok"and h["request_num"]==0
 with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as response:v=json.loads(response.read());assert all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]for x in v["replicas"])
 o=owners["node1"];assert o["host"]=="167";previous=policies["167"];targets=[v for v in members["node1"]["owned_targets"]if v["pid"]in members["node1"]["npu_worker_pids"]]
 run("167",check,dict(owner=o,targets=targets,policy=previous,path=POLICY["167"],source=SOURCE["167"],sha=SHAS["167"]),label+"_guard")
 ack=run("167",write,dict(path=POLICY["167"],previous=previous,new=next_policy),label+"_write")
 policies["167"]=dict(next_policy);atomic_json(r/"effective_D1_policy.json",policies["167"]);atomic_json(r/"policy_progress.json",dict(at=utc(),previous=previous,new=policies["167"],ack=ack,D1_only=True,complete=True))
 verify(label+"_after")
 atomic_json(r/(label+"_ack.json"),dict(at=utc(),previous=previous,new=policies["167"],ack=ack,both_independent_engines_idle=True,D1_only_atomic_update=True,D0_policy_unchanged=policies["166"],model_operations=0))
