"""Verify P58 has exited and D56 is unchanged; no signals or process starts."""
import json,pathlib,subprocess,os,re
expected=json.loads(os.environ["GLM_EXPECTED_COHORT"]);history=json.loads(pathlib.Path(os.environ["GLM_PRIOR_SNAPSHOT"]).read_text());registered=json.loads(pathlib.Path(os.environ["GLM_PRIOR_P_REGISTERED"]).read_text())
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def proc(pid):
 try:
  raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text();x=raw[raw.rfind(")")+2:].split()
  return dict(identity=dict(boot_id=boot,start_ticks=x[19]),state=x[0],ppid=int(x[1]))
 except FileNotFoundError:return None
P=next(o for o in expected if o["role"]=="P");D=next(o for o in expected if o["role"]=="D");z=proc(P["pid"])
assert not z or z["identity"]!=P["identity"]or z["state"]in["Z","X"],"oldP API stillactive; no action"
z=proc(D["pid"]);assert z and z["identity"]==D["identity"]and z["state"]not in["Z","X"]
assert [x.decode()for x in pathlib.Path("/proc/"+str(D["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==D["argv"]
targets={x["pid"]:x for x in history["retained_D_targets"]};assert D["pid"]==history["exact_D_root"]
for pid,x in targets.items():
 z=proc(pid);assert z and z["identity"]==x["identity"]and z["state"]not in["Z","X"],"D member changed"
priorP=registered["P"+str(P["rank"])];assert priorP["API_owner"]==P
pstates=[]
for x in priorP["physical_workers"]:
 z=proc(x["pid"]);ident={k:x[k]for k in["boot_id","start_ticks"]}
 assert not z or z["identity"]!=ident or z["state"]in["Z","X"],"oldP registeredworker active"
 pstates.append(dict(pid=x["pid"],expected_identity=ident,current=z))
rawtop=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True)
un=[];zombies=[];roots=set()
for row in rawtop.splitlines()[1:]:
 x=row.split(None,4)
 if len(x)!=5:continue
 pid,ppid,state,comm,args=int(x[0]),int(x[1]),x[2],x[3],x[4]
 root="/bin/vllm serve "in args or"native_acl_lifecycle.py cli serve "in args
 native=root or comm.startswith("VLLM")or"bishengir-compile "in args
 if not native:continue
 if state[0]in["Z","X"]:zombies.append(dict(pid=pid,ppid=ppid,state=state,comm=comm));continue
 if root:roots.add(pid)
 if pid not in targets:un.append(dict(pid=pid,ppid=ppid,state=state,comm=comm,proc=proc(pid)))
if un:print(json.dumps(dict(event="unattributed_active_nativeworker",rows=un,signals=0)),flush=True)
assert not un and roots=={D["pid"]},"unattributed active native resource"
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)
npus={int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",raw,re.M)}
assert len(npus)==16 and npus<=targets.keys(),"NPU not D-only"
print(json.dumps(dict(exact_P_root=P["pid"],exact_D_root=D["pid"],owned_P_targets=[],retained_D_targets=history["retained_D_targets"],P_API_state=proc(P["pid"]),P_registered_worker_states=pstates,P_inactive=True,npu_P=0,npu_D=16,historical_P_zombies=len(zombies),inactive_native_zombies=zombies,npu_smi_raw=raw,signals=0,verification_only=True)),flush=True)
