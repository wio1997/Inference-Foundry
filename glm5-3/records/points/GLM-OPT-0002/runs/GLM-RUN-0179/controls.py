from pathlib import Path
import json,sys,subprocess,shlex,hashlib,urllib.request,re,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
r=Path(__file__).parent
owners=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());history=[];results=[];http=urllib.request.build_opener(urllib.request.ProxyHandler({}));policy=dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3)
POLICY={node:"/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp175"if node=="166"else"local_pp168")+"/issue_budget_policy.json"for node in["166","167"]}
policies={"166":dict(policy),"167":dict(policy,budget_tokens=8192,prefill_threshold_tokens=4096,serial=13)}
SOURCE={node:"/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp175"if node=="166"else"local_pp168")+"/issue_budget_scheduler_v3.py"for node in["166","167"]}
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
write=r"""import json,pathlib,sys,tempfile,os,hashlib
a=json.load(sys.stdin);p=pathlib.Path(a['path']);old=p.read_bytes();assert json.loads(old)==a['previous']
new=a['new'];assert new['schema_version']==1 and new['cohort_id']=='GLM-COHORT-0137'and type(new['budget_tokens'])is int and new['budget_tokens']%128==0 and 512<=new['budget_tokens']<=8192 and type(new['prefill_threshold_tokens'])is int and(new['prefill_threshold_tokens']==0 or new['prefill_threshold_tokens']%128==0 and 128<=new['prefill_threshold_tokens']<=new['budget_tokens'])and type(new['prefill_cadence'])is int and new['prefill_cadence']in[1,2]and new['serial']==a['previous']['serial']+1
data=(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n').encode();fd,tmp=tempfile.mkstemp(prefix='.glm178policy-',dir=str(p.parent))
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
  ack=run(o["host"],check,dict(owner=o,targets=targets,policy=policies[o["host"]],path=POLICY[o["host"]],source=SOURCE[o["host"]],sha=SHA),label+"_"+key)
  with http.open("http://172.16.10."+o["host"]+":"+str(9081 if o["host"]=="166"else 9900)+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
  (r/(label+"_"+key+".metrics")).write_bytes(b)
  for name in["num_requests_running","num_requests_waiting"]:
   vals=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert vals and all(float(v)==0 for v in vals)

def update_D1(next_policy,label):
 verify(label+"_before")
 o=owners["node1"];assert o["host"]=="167";previous=policies["167"];targets=[v for v in members["node1"]["owned_targets"]if v["pid"]in members["node1"]["npu_worker_pids"]]
 run("167",check,dict(owner=o,targets=targets,policy=previous,path=POLICY["167"],source=SOURCE["167"],sha=SHA),label+"_guard")
 ack=run("167",write,dict(path=POLICY["167"],previous=previous,new=next_policy),label+"_write")
 policies["167"]=dict(next_policy);atomic_json(r/"effective_D1_policy.json",policies["167"]);atomic_json(r/"policy_progress.json",dict(at=utc(),previous=previous,new=policies["167"],ack=ack,D1_only=True,complete=True))
 verify(label+"_after")
 atomic_json(r/(label+"_ack.json"),dict(at=utc(),previous=previous,new=policies["167"],ack=ack,both_independent_engines_idle=True,D1_only_atomic_update=True,D0_policy_unchanged=policies["166"],model_operations=0))

def update_D0(next_policy,label):
 verify(label+"_before")
 o=owners["node0"];assert o["host"]=="166";previous=policies["166"];targets=[v for v in members["node0"]["owned_targets"]if v["pid"]in members["node0"]["npu_worker_pids"]]
 run("166",check,dict(owner=o,targets=targets,policy=previous,path=POLICY["166"],source=SOURCE["166"],sha=SHA),label+"_guard")
 ack=run("166",write,dict(path=POLICY["166"],previous=previous,new=next_policy),label+"_write")
 policies["166"]=dict(next_policy);atomic_json(r/"effective_D0_policy.json",policies["166"])
 verify(label+"_after")
 atomic_json(r/(label+"_ack.json"),dict(at=utc(),previous=previous,new=policies["166"],ack=ack,both_independent_engines_idle=True,D0_only_atomic_update=True,D1_policy_unchanged=policies["167"],model_operations=0))
