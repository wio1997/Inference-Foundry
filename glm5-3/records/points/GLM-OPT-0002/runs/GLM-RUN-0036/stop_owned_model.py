import json,subprocess,pathlib,os,signal,time
expected=json.loads(os.environ["GLM_EXPECTED_OWNER"])
def identity(pid):
 try:
  raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text()
  return {"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":raw[raw.rfind(")")+2:].split()[19]}
 except FileNotFoundError:return None
assert identity(expected["pid"])=={k:expected["identity"][k] for k in ["boot_id","start_ticks"]}
rows=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True).splitlines()[1:]
ps={}
for row in rows:
 p=row.split(None,3)
 if len(p)==4:ps[int(p[0])]=(int(p[1]),p[2],p[3])
roots=[p for p,(_,_,a) in ps.items() if "/bin/vllm serve " in a]
assert roots==[expected["pid"]],roots
actual=[x.decode() for x in pathlib.Path("/proc/"+str(expected["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x];assert actual==expected["argv"]
args=ps[roots[0]][2];assert "/data/tiankuan/wio/GLM-5.2-w8a8" in args and "--port "+expected["argv"][expected["argv"].index("--port")+1] in args
targets={expected["pid"]}
while True:
 more=targets|{p for p,(parent,_,_) in ps.items() if parent in targets}
 if more==targets:break
 targets=more
assert all(p in targets for p,(_,comm,_)in ps.items()if comm.startswith("VLLM")),"unattributed native worker, no signal"
ids={p:identity(p) for p in targets}
print(json.dumps({"unique_task_model_root":roots,"owned_targets":[{"pid":p,"identity":ids[p],"comm":ps[p][1]} for p in sorted(targets)]}),flush=True)
os.kill(expected["pid"],signal.SIGTERM)
end=time.monotonic()+60
while time.monotonic()<end and any(identity(p)==ids[p] for p in targets if ids[p]):time.sleep(2)
killed=[]
for p in sorted(targets,reverse=True):
 if ids[p] and identity(p)==ids[p]:
  try:os.kill(p,signal.SIGKILL);killed.append(p)
  except ProcessLookupError:pass
time.sleep(3)
left=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True)
assert not any("/bin/vllm serve " in line or "VLLM::" in line for line in left.splitlines()),"task model processes remain; reconcile, no startup"
print(json.dumps({"old_task_processes_removed":True,"verified_sigkill_fallback_pids":killed}))
