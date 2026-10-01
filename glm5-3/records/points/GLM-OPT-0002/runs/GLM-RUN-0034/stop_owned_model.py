import json,subprocess,pathlib,os,signal,time
expected=json.loads(os.environ["GLM_EXPECTED_OWNER"])
def identity(pid):
 try:
  raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text()
  return {"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":raw[raw.rfind(")")+2:].split()[19]}
 except FileNotFoundError:return None
assert pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()==expected["identity"]["boot_id"]
original={k:expected["identity"][k]for k in ["boot_id","start_ticks"]};live=identity(expected["pid"])
assert live is None or live==original,"saved root PID reused; do not touch"
rows=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True).splitlines()[1:];ps={}
for row in rows:
 p=row.split(None,3)
 if len(p)==4:ps[int(p[0])]=(int(p[1]),p[2],p[3])
roots=[p for p,(_,_,a)in ps.items()if "/bin/vllm serve "in a]
assert roots==([expected["pid"]]if live else []),roots
if live:
 actual=[x.decode()for x in pathlib.Path("/proc/"+str(expected["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x];assert actual==expected["argv"]
targets=({expected["pid"]}if live else set())|{p for p,(_,comm,_)in ps.items()if comm.startswith("VLLM")}
# Original native installation receipts identify orphan worker container PIDs;
# vLLM intentionally strips PYTHONPATH from child environ.
import re
rank=expected["argv"][expected["argv"].index("--data-parallel-rank")+1]
log=pathlib.Path("/data/tiankuan/wio/glm52-pd/deploy/logs/DP_run32_"+rank+".log")
receipt_pids=set()
for line in log.read_text(errors="replace").splitlines():
 if "GLM_DP_METADATA_INSTALLED " in line:
  match=re.search(r"pid=(\d+)",line);assert match
  value=json.loads(line.split("GLM_DP_METADATA_INSTALLED ",1)[1]);assert value["dp_rank"]==int(rank)
  receipt_pids.add(int(match[1]))
descendants={expected["pid"]}if live else set()
while True:
 more=descendants|{p for p,(parent,_,_)in ps.items()if parent in descendants}
 if more==descendants:break
 descendants=more
membership=[]
for p in targets:
 ident=identity(p);assert ident and ident["boot_id"]==expected["identity"]["boot_id"]
 if p in descendants:proof="exact live root ancestry"
 else:
  status=pathlib.Path("/proc/"+str(p)+"/status").read_text()
  nspid=next(line.split()[1:]for line in status.splitlines()if line.startswith("NSpid:"))
  assert int(nspid[-1])in receipt_pids and int(ident["start_ticks"])>=int(expected["identity"]["start_ticks"]),"unverified orphan; no action"
  proof="original task metadata installation receipt + namespace PID + cohort boot/start"
 membership.append({"pid":p,"identity":ident,"proof":proof})
print(json.dumps({"membership_before_any_signal":membership,"native_log":str(log),"native_log_sha256":__import__("hashlib").sha256(log.read_bytes()).hexdigest()}),flush=True)
while True:
 more=targets|{p for p,(parent,_,_)in ps.items()if parent in targets}
 if more==targets:break
 targets=more
ids={p:identity(p)for p in targets}
print(json.dumps({"old_root_alive":bool(live),"owned_targets":[{"pid":p,"identity":ids[p],"comm":ps[p][1]}for p in sorted(targets)]}),flush=True)
for p in targets:
 if ids[p]and identity(p)==ids[p]:
  try:os.kill(p,signal.SIGTERM)
  except ProcessLookupError:pass
end=time.monotonic()+25
while time.monotonic()<end and any(identity(p)==ids[p]for p in targets if ids[p]):time.sleep(1)
killed=[]
for p in sorted(targets,reverse=True):
 if ids[p]and identity(p)==ids[p]:
  try:os.kill(p,signal.SIGKILL);killed.append(p)
  except ProcessLookupError:pass
time.sleep(3)
left=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True)
assert not any("/bin/vllm serve "in line or "VLLM::"in line for line in left.splitlines()),"owned native resources remain; no startup"
print(json.dumps({"exact_failed_task_cohort_removed":True,"sigkill_same_identity_pids":killed}))
