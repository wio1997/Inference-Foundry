import json,os,pathlib,subprocess,re,time,signal,sys
payload=json.load(sys.stdin);owner=payload["owner"];known={x["pid"]:x for x in payload["members"]};pre=payload["preflight"]
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def proc(pid):
 try:
  b=pathlib.Path("/proc/"+str(pid)+"/stat").read_text();x=b[b.rfind(")")+2:].split();return dict(identity=dict(boot_id=boot,start_ticks=x[19]),state=x[0])
 except FileNotFoundError:return None
def active(pid):
 z=proc(pid);return bool(z and z["identity"]==known[pid]["identity"]and z["state"]not in["Z","X"])
assert owner["pid"]in known
if active(owner["pid"]):assert [v.decode()for v in pathlib.Path("/proc/"+str(owner["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==owner["argv"]
def scope():
 top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True);un=[]
 for l in top.splitlines()[1:]:
  x=l.split(None,4)
  if len(x)!=5:continue
  pid=int(x[0]);native="/bin/vllm serve "in x[4]or"native_acl_lifecycle.py cli serve "in x[4]or x[3].startswith("VLLM")or"bishengir-compile "in x[4]
  if native and x[2][0]not in["Z","X"]and (pid not in known or not active(pid)):un.append(dict(pid=pid,ppid=x[1],state=x[2],comm=x[3],proc=proc(pid)))
 if un:print(json.dumps(dict(event="unattributed_active_native",rows=un,signals=0)),flush=True)
 assert not un,"Unknown active native resource; no signals"
 return top
def npu():
 b=subprocess.check_output(["npu-smi","info"],text=True,timeout=60);ids={int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",b,re.M)};assert ids<=known.keys()and all(active(pid)for pid in ids),"foreignNPU";return b,ids
scope();raw,ids=npu()
if pre:assert all(active(pid)for pid in known)and len(ids)==16
print(json.dumps(dict(event="ownership_preflight",at=time.time(),exact_D65_owner=owner,owned_members=len(known),active_members=sum(active(pid)for pid in known),npu_workers=len(ids),signals=0,npu_smi=raw)),flush=True)
if pre:raise SystemExit(0)
def send(pid,sig):
 if not active(pid):return
 assert pid in known
 print(json.dumps(dict(event="signal_attempt",pid=pid,identity=known[pid]["identity"],signal=sig.name,scope="exact_original_D65")),flush=True)
 try:os.kill(pid,sig);print(json.dumps(dict(event="signal_sent",pid=pid,signal=sig.name)),flush=True)
 except ProcessLookupError:print(json.dumps(dict(event="signal_raced_exit",pid=pid,signal=sig.name)),flush=True)
send(owner["pid"],signal.SIGTERM);end=time.monotonic()+45
while time.monotonic()<end and any(active(pid)for pid in known):time.sleep(1)
for pid in sorted(known,reverse=True):send(pid,signal.SIGKILL)
time.sleep(2);assert not any(active(pid)for pid in known)
scope();raw,ids=npu();assert not ids
print(json.dumps(dict(event="cleanup_complete",all_original_D_inactive=True,npu_workers=0,npu_smi=raw,signals_to_unknown=0,signals_to_inactive=0)),flush=True)
