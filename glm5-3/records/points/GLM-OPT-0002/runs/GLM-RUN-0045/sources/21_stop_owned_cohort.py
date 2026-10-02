import json,subprocess,pathlib,os,signal,time
expected=json.loads(os.environ["GLM_EXPECTED_COHORT"])
def identity(pid):
 try:
  raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text()
  return {"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":raw[raw.rfind(")")+2:].split()[19]}
 except FileNotFoundError:return None
assert len(expected)==1 and len({x["pid"]for x in expected})==1
rows=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True).splitlines()[1:];ps={}
for row in rows:
 x=row.split(None,3)
 if len(x)==4:ps[int(x[0])]=(int(x[1]),x[2],x[3])
roots=[pid for pid,(_,_,argv)in ps.items()if "/bin/vllm serve "in argv]
assert set(roots)=={x["pid"]for x in expected}
for owner in expected:
 assert identity(owner["pid"])==owner["identity"]
 actual=[x.decode()for x in pathlib.Path("/proc/"+str(owner["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x];assert actual==owner["argv"]
targets=set(roots)
while True:
 more=targets|{pid for pid,(parent,_,_)in ps.items()if parent in targets}
 if more==targets:break
 targets=more
assert all(pid in targets for pid,(_,comm,_)in ps.items()if comm.startswith("VLLM")),"unattributed native worker; no signal"
ids={pid:identity(pid)for pid in targets}
print(json.dumps({"exact_root_cohort_verified":roots,"owned_targets":[{"pid":pid,"identity":ids[pid],"comm":ps[pid][1]}for pid in sorted(targets)]}),flush=True)
if os.environ.get("GLM_CLEANUP_PREFLIGHT_ONLY")=="1":raise SystemExit(0)
for owner in expected:
 assert identity(owner["pid"])==owner["identity"]
 try:os.kill(owner["pid"],signal.SIGTERM)
 except ProcessLookupError:pass
end=time.monotonic()+60
while time.monotonic()<end and any(identity(pid)==ids[pid]for pid in targets if ids[pid]):time.sleep(2)
fallback=[]
for pid in sorted(targets,reverse=True):
 if ids[pid]and identity(pid)==ids[pid]:
  try:os.kill(pid,signal.SIGKILL);fallback.append(pid)
  except ProcessLookupError:pass
time.sleep(3)
rows=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True)
assert not any("/bin/vllm serve "in l or "VLLM::"in l or "bishengir-compile "in l for l in rows.splitlines())
print(json.dumps({"exact_one_root_cohort_removed":True,"verified_sigkill_fallback_pids":fallback}))
