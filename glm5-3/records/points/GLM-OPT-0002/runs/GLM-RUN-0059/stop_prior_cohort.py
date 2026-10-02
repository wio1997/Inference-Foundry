"""Verify exact P58/D56 ownership, stop only P58, and journal every signal."""
import json,subprocess,pathlib,os,signal,time,re
expected=json.loads(os.environ["GLM_EXPECTED_COHORT"]);history=json.loads(pathlib.Path(os.environ["GLM_PRIOR_SNAPSHOT"]).read_text())
assert len(expected)==2 and {o["role"]for o in expected}=={"P","D"}
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def proc(pid):
 try:
  s=pathlib.Path("/proc/"+str(pid)+"/stat").read_text();x=s[s.rfind(")")+2:].split()
  return dict(identity=dict(boot_id=boot,start_ticks=x[19]),state=x[0],ppid=int(x[1]))
 except FileNotFoundError:return None
hist={x["pid"]:x["identity"]for x in history["owned_P_targets"]}
historical_Z=set()
for pid,i in hist.items():
 z=proc(pid)
 if z and z["identity"]==i:
  assert z["state"]in["Z","X"],"historicalP active"
  historical_Z.add(pid)
def table():
 out={}
 for row in subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True).splitlines()[1:]:
  x=row.split(None,4)
  if len(x)==5:out[int(x[0])]=(int(x[1]),x[2],x[3],x[4])
 return out
ps=table();known={}
for o in expected:
 z=proc(o["pid"]);assert z and z["identity"]==o["identity"]and z["state"]not in["Z","X"],"priorAPI missing/reused"
 actual=[x.decode()for x in pathlib.Path("/proc/"+str(o["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x];assert actual==o["argv"];known[o["role"]]=o["pid"]
roots={pid for pid,(_,state,_,a)in ps.items()if state[0]not in["Z","X"]and("/bin/vllm serve "in a or"native_acl_lifecycle.py cli serve "in a)}
assert roots==set(known.values()),"foreign nativeAPI"
def descendants(root):
 result={root}
 while True:
  more=result|{pid for pid,(parent,_,_,_)in ps.items()if parent in result}
  if more==result:return result
  result=more
P=descendants(known["P"]);D=descendants(known["D"]);assert not(P&D)
for pid,(_,state,comm,a)in ps.items():
 if comm.startswith("VLLM")or"bishengir-compile "in a:
  assert pid in P|D or(state[0]in["Z","X"]and pid in historical_Z),"unattributed nativeworker"
ids={pid:proc(pid)["identity"]for pid in P|D}
for x in history["retained_D_targets"]:
 assert x["pid"]in D and ids[x["pid"]]==x["identity"],"oldD membership changed"
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)
npu={int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",raw,re.M)}
assert npu<=P|D and len(npu&D)==16,"unattributed NPU/Dmissing"
receipt=dict(exact_P_root=known["P"],exact_D_root=known["D"],owned_P_targets=[dict(pid=pid,identity=ids[pid],comm=ps[pid][2])for pid in sorted(P)],retained_D_targets=[dict(pid=pid,identity=ids[pid],comm=ps[pid][2])for pid in sorted(D)],historical_P_zombies=len(historical_Z),npu_P=len(npu&P),npu_D=len(npu&D),npu_smi_raw=raw)
print(json.dumps(receipt),flush=True)
if os.environ.get("GLM_CLEANUP_PREFLIGHT_ONLY")=="1":raise SystemExit(0)
def active(pid):
 z=proc(pid);return bool(z and z["identity"]==ids[pid]and z["state"]not in["Z","X"])
def signal_one(pid,sig):
 z=proc(pid)
 if not z or z["identity"]!=ids[pid]or z["state"]in["Z","X"]:return
 assert pid in P and pid not in D
 print(json.dumps(dict(event="signal_attempt",pid=pid,identity=ids[pid],signal=sig.name,scope="exact_P58")),flush=True)
 try:os.kill(pid,sig);print(json.dumps(dict(event="signal_sent",pid=pid,signal=sig.name)),flush=True)
 except ProcessLookupError:print(json.dumps(dict(event="signal_raced_exit",pid=pid,signal=sig.name)),flush=True)
for pid in D:assert proc(pid)["identity"]==ids[pid]
signal_one(known["P"],signal.SIGTERM)
end=time.monotonic()+45
while time.monotonic()<end and any(active(pid)for pid in P):time.sleep(1)
for pid in sorted(P,reverse=True):signal_one(pid,signal.SIGKILL)
time.sleep(2)
assert not any(active(pid)for pid in P),"priorP still active"
for pid in D:
 z=proc(pid);assert z and z["identity"]==ids[pid]and z["state"]not in["Z","X"],"D changed"
after=table()
for pid,(_,state,comm,a)in after.items():
 native="/bin/vllm serve "in a or"native_acl_lifecycle.py cli serve "in a or comm.startswith("VLLM")or"bishengir-compile "in a
 if native:assert pid in D or(state[0]in["Z","X"]and pid in P|historical_Z),"foreign active native aftercleanup"
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60);npu={int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",raw,re.M)}
assert len(npu)==16 and npu<=D,"NPU oldP not released/foreign"
print(json.dumps(dict(P_inactive=True,P_zombies=sum(bool(proc(pid)and proc(pid)["identity"]==ids[pid]and proc(pid)["state"]in["Z","X"])for pid in P),D_every_identity_retained=True,npu_smi_raw=raw,signals_to_D=0,signals_to_historical_Z=0,signals_to_unrelated=0)),flush=True)

