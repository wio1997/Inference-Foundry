
import pathlib,json,subprocess,shlex,re,time,sys,hashlib,urllib.request,zipfile,io,os
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());run=pathlib.Path(job["inputs"][0]["path"])
owners=json.loads((run/"startup_model_identities.json").read_text())
# Task-private diagnostic binary only, no native/operator/package installations.
private=pathlib.Path("/data/tiankuan/wio/glm52-pd/deploy/private/pyspy_diag");private.mkdir(exist_ok=True,mode=0o700)
binary=private/"py-spy";download=None
try:
 if not binary.exists():
  with urllib.request.urlopen("https://pypi.org/pypi/py-spy/0.4.1/json",timeout=15)as response:package=json.load(response)
  urls=[x for x in package["urls"]if "manylinux" in x["filename"]and "x86_64"in x["filename"]and x["packagetype"]=="bdist_wheel"];assert len(urls)==1
  with urllib.request.urlopen(urls[0]["url"],timeout=30)as response:raw=response.read()
  assert hashlib.sha256(raw).hexdigest()==urls[0]["digests"]["sha256"]
  with zipfile.ZipFile(io.BytesIO(raw))as archive:
   names=[n for n in archive.namelist()if n.endswith("/py-spy")];assert len(names)==1;binary.write_bytes(archive.read(names[0]))
  binary.chmod(0o700);download={"url":urls[0]["url"],"wheel_sha256":hashlib.sha256(raw).hexdigest(),"binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),"version":"0.4.1"}
except Exception as error:download={"unavailable":repr(error)}
def command(node,args,timeout=15):
 if node=="167":args=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"argv":args}
probe=r"""
import pathlib,json,subprocess,os
owner=json.loads(os.environ["GLM_DIAG_OWNER"])
def identity(pid):
 p=pathlib.Path("/proc")/str(pid);s=(p/"stat").read_text()
 return {"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":s[s.rfind(")")+2:].split()[19]}
assert identity(owner["pid"])==owner["identity"]
assert [x.decode()for x in (pathlib.Path("/proc")/str(owner["pid"])/"cmdline").read_bytes().split(bytes([0]))if x]==owner["argv"]
rows=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True).splitlines()[1:];ps={}
for row in rows:
 x=row.split(None,3)
 if len(x)==4:ps[int(x[0])]=(int(x[1]),x[2],x[3])
desc={owner["pid"]}
while True:
 more=desc|{p for p,(parent,_,_)in ps.items()if parent in desc}
 if more==desc:break
 desc=more
assert all(p in desc for p,(_,comm,_)in ps.items()if comm.startswith("VLLM"))
workers=[]
for pid in sorted(desc):
 if pid not in ps:continue
 p=pathlib.Path("/proc")/str(pid)
 if ps[pid][1].startswith("VLLM"):
  tasks=[]
  for t in (p/"task").iterdir():
   try:tasks.append({"tid":t.name,"comm":(t/"comm").read_text().strip(),"wchan":(t/"wchan").read_text().strip()})
   except FileNotFoundError:pass
  workers.append({"pid":pid,"identity":identity(pid),"comm":ps[pid][1],"tasks":tasks})
print(json.dumps({"native_owner_verified":owner,"owned_workers":workers}))
"""
samples=[]
for sample in range(2):
 row={"at":utc(),"controller":json.loads((run/"state.json").read_text()),"nodes":{}}
 for rank,node in [(0,"166"),(1,"167")]:
  val=command(node,["env","GLM_DIAG_OWNER="+json.dumps(owners["Node"+str(rank)]),"python3","-c",probe])
  if val["exit_code"]!=0:raise RuntimeError(val["stderr"])
  data=json.loads(val["stdout"]);p=j/("sample"+str(sample)+"_"+node+"_proc.json");atomic_json(p,data)
  log=command(node,["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/TP_run36_"+str(rank)+".log"])
  (j/("sample"+str(sample)+"_"+node+".log")).write_text(log["stdout"])
  smi=command(node,["docker","exec","glm52-single","npu-smi","info"],timeout=20);(j/("sample"+str(sample)+"_"+node+".npu-smi")).write_text(smi["stdout"])
  row["nodes"][node]={"owned_worker_count":len(data["owned_workers"]),"proc_ref":str(p),"native_log_last_lines":log["stdout"].splitlines()[-5:],"smi_exit":smi["exit_code"]}
  if sample==1 and binary.exists():
   if node=="167":
    command(node,["mkdir","-p",str(private)])
    cp=subprocess.run(["scp","-q",str(binary),"root@172.16.10.167:"+str(binary)],capture_output=True);cp.check_returncode();command(node,["chmod","700",str(binary)])
   help=command(node,[str(binary),"dump","--help"]);(j/(node+"_pyspy_help.txt")).write_text(help["stdout"])
   if "--nonblocking"in help["stdout"]:
    # Only owned workers, no signals/stops/restarts, one representative rank per node.
    workers=[v for v in data["owned_workers"]if v["comm"].startswith("VLLM::Worker")]
    if not workers:workers=[v for v in data["owned_workers"]if "Worker"in v["comm"]]
    if workers:
     dump=command(node,[str(binary),"dump","--pid",str(workers[0]["pid"]),"--nonblocking"],timeout=20)
     (j/(node+"_python_stack.txt")).write_text(dump["stdout"]);(j/(node+"_python_stack.stderr")).write_text(dump["stderr"])
     row["nodes"][node]["pyspy_exit"]=dump["exit_code"];row["nodes"][node]["pyspy_owned_pid"]=workers[0]["pid"]
 samples.append(row)
 if sample==0:time.sleep(10)
artifacts=[]
for p in j.iterdir():
 if p.is_file()and p.suffix in [".log",".json",".txt",".npu-smi"]and p.name not in ["job.json","result.json","reduction.json","bridge.json"]:
  raw=p.read_bytes();artifacts.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
out={"valid_read_only_diagnostic":True,"model_requests":0,"native_service_actions":0,"samples":samples,"pyspy_binary":download,"artifacts":artifacts,"limits":["Samples/proc Python stack do not directly prove device kernel/root cause or permanent deadlock","No inference or automatic configuration verdict"]}
p=j/"reduction.json";atomic_json(p,out);raw=p.read_bytes()
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Two actual owned-rank/proc/log/NPU samples, optional task-private nonblocking Python stacks captured; no native inference/control","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[{"id":"reduction","path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"actual raw diagnostics"}],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"requests":0,"samples":len(samples),"pyspy":download}))
