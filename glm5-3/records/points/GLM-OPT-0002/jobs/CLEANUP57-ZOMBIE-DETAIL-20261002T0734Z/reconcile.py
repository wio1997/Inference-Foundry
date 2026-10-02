"""Read-only reconciliation of Run57 stopped P and retained D; never sends signals."""
import json,subprocess,pathlib,os,re
expected=json.loads(os.environ["GLM_EXPECTED_COHORT"])
snap=json.loads(pathlib.Path(os.environ["GLM_PRIOR_SNAPSHOT"]).read_text())
assert len(expected)==2 and {o["role"]for o in expected}=={"P","D"}
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def proc(pid):
 try:
  raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text();x=raw[raw.rfind(")")+2:].split()
  return dict(identity=dict(boot_id=boot,start_ticks=x[19]),state=x[0],ppid=int(x[1]))
 except FileNotFoundError:return None
P={x["pid"]:x for x in snap["owned_P_targets"]};D={x["pid"]:x for x in snap["retained_D_targets"]};assert not(P.keys()&D.keys())
pstates=[];dstates=[]
for pid,x in P.items():
 z=proc(pid)
 if z:assert z["identity"]==x["identity"] and z["state"]in["Z","X"],"priorP active/reused"
 pstates.append(dict(pid=pid,**(z or dict(state="absent"))))
for pid,x in D.items():
 z=proc(pid);assert z and z["identity"]==x["identity"] and z["state"]not in["Z","X"],"retainedD absent/reused/zombie"
 dstates.append(dict(pid=pid,**z))
for o in expected:
 if o["role"]=="D":
  assert o["pid"]==snap["exact_D_root"]
  actual=[x.decode()for x in pathlib.Path("/proc/"+str(o["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x];assert actual==o["argv"]
 else:assert o["pid"]==snap["exact_P_root"]
top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True)
active_roots=set()
for row in top.splitlines()[1:]:
 x=row.split(None,4)
 if len(x)!=5:continue
 pid,ppid,stat,comm,args=int(x[0]),int(x[1]),x[2],x[3],x[4]
 native="/bin/vllm serve "in args or"native_acl_lifecycle.py cli serve "in args
 worker=comm.startswith("VLLM")or"bishengir-compile "in args
 if stat[0]in["Z","X"]:
  if native or worker:assert pid in P or pid in D,"unattributed zombie native process"
 elif native:active_roots.add(pid)
 elif worker:assert pid in D,"unattributed active worker"
assert active_roots=={snap["exact_D_root"]}
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)
npu=[int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",raw,re.M)]
assert len(npu)==16 and len(set(npu))==16 and all(pid in D for pid in npu),"NPU workers not exactD"
print(json.dumps(dict(exact_P_root=snap["exact_P_root"],exact_D_root=snap["exact_D_root"],owned_P_targets=snap["owned_P_targets"],retained_D_targets=snap["retained_D_targets"],P_states=pstates,D_states=dstates,P_absent=sum(x["state"]=="absent"for x in pstates),P_zombies=sum(x["state"]in["Z","X"]for x in pstates),P_active=0,D_every_identity_retained=True,npu_D_worker_pids=npu,npu_smi_raw=raw,signals=0,verification_only=True)),flush=True)
