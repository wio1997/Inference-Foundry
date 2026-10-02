from pathlib import Path
import json,sys,subprocess,shlex,time,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parent.parent;r=p/"runs/GLM-RUN-0075";owners=json.loads((r/"startup_model_identities.json").read_text());samples=[]
probe="""import pathlib,json,sys,subprocess,collections,time
o=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def proc(pid):
 try:
  p=pathlib.Path('/proc/'+str(pid));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
  return dict(pid=pid,ppid=int(v[1]),state=v[0],utime=int(v[11]),stime=int(v[12]),start_ticks=v[19],comm=(p/'comm').read_text().strip(),wchan=(p/'wchan').read_text().strip())
 except (FileNotFoundError,PermissionError,ProcessLookupError):return None
a=proc(o['pid']);assert a and a['start_ticks']==o['identity']['start_ticks']and boot==o['identity']['boot_id']
assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
rows={x['pid']:x for d in pathlib.Path('/proc').iterdir()if d.name.isdigit()for x in[proc(int(d.name))]if x}
ids={o['pid']}
while True:
 more=ids|{pid for pid,x in rows.items()if x['ppid']in ids}
 if more==ids:break
 ids=more
out=[]
for pid in sorted(ids):
 x=rows.get(pid)
 if not x:continue
 counts=collections.Counter()
 for t in pathlib.Path('/proc/'+str(pid)+'/task').iterdir():
  try:counts[(t/'wchan').read_text().strip()]+=1
  except (FileNotFoundError,ProcessLookupError):pass
 x['thread_wchan_counts']=dict(counts);out.append(x)
z=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=60)
print(json.dumps(dict(at=time.time(),boot_id=boot,API_same=True,descendants=out,npu_returncode=z.returncode,npu_smi=z.stdout,npu_stderr=z.stderr)))
"""
for k in range(3):
 current={"at":utc(),"state":json.loads((r/"state.json").read_text()),"nodes":{}}
 for key,o in owners.items():
  args=["python3","-c",probe]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(o).encode(),capture_output=True,timeout=100);f=j/(str(k)+"_"+key+".stdout");f.write_bytes(z.stdout);(j/(str(k)+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode()
  d=json.loads(z.stdout);log="/data/tiankuan/wio/glm52-pd/deploy/logs/D_run75_"+str(o["rank"])+".log"
  args=["cat",log]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,capture_output=True,timeout=30);f=j/(str(k)+"_"+key+".native.log");f.write_bytes(z.stdout);z.check_returncode()
  lines=z.stdout.decode(errors="replace").splitlines();x=dict(API_same=d["API_same"],active_descendants=len([v for v in d["descendants"]if v["state"]not in["Z","X"]]),process_states=dict(__import__("collections").Counter(v["state"]for v in d["descendants"])),worker_samples=[v for v in d["descendants"]if v["comm"].startswith("VLLM")],log_bytes=len(z.stdout),log_sha256=hashlib.sha256(z.stdout).hexdigest(),recent_lines=lines[-8:],error_lines=[l for l in lines if any(s in l for s in["Error:","Traceback","OutOfMemory","HCCL timeout","EI000","EZ9999"])][-20:])
  current["nodes"][key]=x
 samples.append(current);atomic_json(j/"samples.json",samples)
 if k<2:time.sleep(30)
out=dict(at=utc(),read_only=True,samples=samples,limits=["Proc CPU/wchan and npu samples do not prove collective progress or root cause; no signals/profiler/GPU requests/native source changes"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run75 readonly3samples API identities/descendantCPUwaits/NPU/native rawlogs; no model operations",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="3samples/originalnative/API/CPU/wchan/NPU")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(samples=len(samples),last={k:dict(active=v["active_descendants"],log_bytes=v["log_bytes"],errors=v["error_lines"])for k,v in samples[-1]["nodes"].items()})))
