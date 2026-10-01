
import json,pathlib,subprocess,shlex,sys,time,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=pathlib.Path(job["inputs"][0]["path"]);owners=json.loads((r/"startup_model_identities.json").read_text())
probe=r"""
import pathlib,json,subprocess,os,hashlib
owner=json.loads(os.environ["GLM_ROOT"]);p=pathlib.Path("/proc")/str(owner["pid"]);s=(p/"stat").read_text()
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
assert {"boot_id":boot,"start_ticks":s[s.rfind(")")+2:].split()[19]}==owner["identity"]
assert [x.decode()for x in (p/"cmdline").read_bytes().split(bytes([0]))if x]==owner["argv"]
rows=subprocess.check_output(["ps","-eo","pid,ppid,etimes,pcpu,rss,state,comm,args"],text=True).splitlines()[1:];ps={}
for l in rows:
 x=l.split(None,7)
 if len(x)==8:ps[int(x[0])]=x
desc={owner["pid"]}
while True:
 more=desc|{p for p,v in ps.items()if int(v[1])in desc}
 if more==desc:break
 desc=more
out=[]
for pid in sorted(desc):
 if pid not in ps or ps[pid][6]!="bishengir-compi":continue
 p=pathlib.Path("/proc")/str(pid);stat=(p/"stat").read_text();f=stat[stat.rfind(")")+2:].split();args=[x.decode()for x in(p/"cmdline").read_bytes().split(bytes([0]))if x]
 ir=p/"root"/args[1].lstrip("/");raw=ir.read_bytes()
 item={"pid":pid,"boot_id":boot,"start_ticks":f[19],"user_ticks":int(f[11]),"system_ticks":int(f[12]),"clock_hz":os.sysconf("SC_CLK_TCK"),"cpu_percent":float(ps[pid][3]),"elapsed_s":int(ps[pid][2]),"rss_KiB":int(ps[pid][4]),"argv":args,"IR":{"container_path":args[1],"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}}
 out.append(item)
print(json.dumps({"root_owner":owner,"compile_processes":out,"count":len(out)}))
"""
samples=[];artifacts=[]
for n in range(2):
 sample={"at":utc(),"controller":json.loads((r/"state.json").read_text()),"nodes":{}}
 for rank,node in [(0,"166"),(1,"167")]:
  args=["env","GLM_ROOT="+json.dumps(owners["Node"+str(rank)]),"python3","-c",probe]
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  p=subprocess.run(args,capture_output=True,text=True,timeout=20);p.check_returncode();data=json.loads(p.stdout)
  path=j/("sample"+str(n)+"_"+node+".json");atomic_json(path,data);sample["nodes"][node]=data
  if n==0 and data["compile_processes"]:
   item=data["compile_processes"][0];path=j/(node+"_kernel.ttadapter.mlir")
   args=["cat","/proc/"+str(item["pid"])+"/root"+item["IR"]["container_path"]]
   if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
   copy=subprocess.run(args,capture_output=True,timeout=20);copy.check_returncode();assert hashlib.sha256(copy.stdout).hexdigest()==item["IR"]["sha256"];path.write_bytes(copy.stdout)
   artifacts.append({"path":str(path),"bytes":len(copy.stdout),"sha256":item["IR"]["sha256"]})
 samples.append(sample)
 if n==0:time.sleep(10)
deltas={}
for node in ["166","167"]:
 a={v["pid"]:v for v in samples[0]["nodes"][node]["compile_processes"]};b={v["pid"]:v for v in samples[1]["nodes"][node]["compile_processes"]}
 deltas[node]=[{"pid":pid,"same_identity":a[pid]["start_ticks"]==v["start_ticks"],"cpu_seconds_delta":((v["user_ticks"]+v["system_ticks"])-(a[pid]["user_ticks"]+a[pid]["system_ticks"]))/v["clock_hz"],"elapsed_s_delta":v["elapsed_s"]-a[pid]["elapsed_s"]}for pid,v in b.items()if pid in a]
out={"valid_read_only_diagnostic":True,"native_requests":0,"native_service_actions":0,"samples":samples,"cpu_deltas":deltas,"IR_artifacts":artifacts,"limits":["Active CPU compiler proves work consumption, not forward progress/finite compile time or device deadlock","IR copied unchanged, no compiler options/operator/native source mutations","No functional/performance/KEEP verdict"]}
p=j/"reduction.json";atomic_json(p,out);raw=p.read_bytes()
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Exact owned native compiler CPU tick/elapsed/IR identity samples saved; no inference or service actions","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[{"id":"reduction","path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"native CPU compiler evidence"}],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"counts":{k:v["count"]for k,v in samples[-1]["nodes"].items()},"IR":artifacts}))
