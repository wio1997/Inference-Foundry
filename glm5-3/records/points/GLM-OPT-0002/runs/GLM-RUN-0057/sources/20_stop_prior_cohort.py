"""Remove only the verified failed P cohort; preserve every D process identity."""
import json,subprocess,pathlib,os,signal,time
expected=json.loads(os.environ["GLM_EXPECTED_COHORT"])
assert len(expected)==2 and {o["role"]for o in expected}=={"P","D"}
def identity(pid):
 try:
  raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text()
  return dict(boot_id=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),start_ticks=raw[raw.rfind(")")+2:].split()[19])
 except FileNotFoundError:return None
def table():
 ps={}
 for row in subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True).splitlines()[1:]:
  x=row.split(None,3)
  if len(x)==4:ps[int(x[0])]=(int(x[1]),x[2],x[3])
 return ps
ps=table();roots={pid for pid,(_,_,a)in ps.items()if"/bin/vllm serve "in a or"native_acl_lifecycle.py cli serve "in a};known={}
for o in expected:
 assert identity(o["pid"])==o["identity"],"prior root absent/reused"
 actual=[x.decode()for x in pathlib.Path("/proc/"+str(o["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]
 assert actual==o["argv"];known[o["role"]]=o["pid"]
assert roots==set(known.values()),"other native root; no signal"
def descendants(root):
 targets={root}
 while True:
  more=targets|{pid for pid,(parent,_,_)in ps.items()if parent in targets}
  if more==targets:return targets
  targets=more
P=descendants(known["P"]);D=descendants(known["D"]);assert not(P&D)
assert all(pid in P|D for pid,(_,comm,a)in ps.items()if comm.startswith("VLLM")or"bishengir-compile "in a),"unattributed nativeworker"
ids={pid:identity(pid)for pid in P|D};assert all(ids.values())
def receipt():
 return dict(exact_P_root=known["P"],exact_D_root=known["D"],owned_P_targets=[dict(pid=pid,identity=ids[pid],comm=ps[pid][1])for pid in sorted(P)],retained_D_targets=[dict(pid=pid,identity=ids[pid],comm=ps[pid][1])for pid in sorted(D)])
print(json.dumps(receipt()),flush=True)
if os.environ.get("GLM_CLEANUP_PREFLIGHT_ONLY")=="1":raise SystemExit(0)
for pid in D:assert identity(pid)==ids[pid],"retainedD identity changed"
assert identity(known["P"])==ids[known["P"]]
os.kill(known["P"],signal.SIGTERM)
end=time.monotonic()+45
while time.monotonic()<end and any(identity(pid)==ids[pid]for pid in P):time.sleep(1)
fallback=[]
for pid in sorted(P,reverse=True):
 if identity(pid)==ids[pid]:
  try:os.kill(pid,signal.SIGKILL);fallback.append(pid)
  except ProcessLookupError:pass
time.sleep(2)
assert all(identity(pid)!=ids[pid]for pid in P),"priorP live"
for pid in D:assert identity(pid)==ids[pid],"retainedD identity changed"
after=table();afterroots={pid for pid,(_,_,a)in after.items()if"/bin/vllm serve "in a or"native_acl_lifecycle.py cli serve "in a}
assert afterroots=={known["D"]},"unexpected native root afterP stop"
assert all(pid in D for pid,(_,comm,a)in after.items()if comm.startswith("VLLM")or"bishengir-compile "in a),"unexpected nativeworker afterP stop"
print(json.dumps(dict(P_removed=True,D_every_identity_retained=True,verified_P_sigkill_fallback=fallback,signals_to_D=0,signals_to_unrelated=0)),flush=True)

