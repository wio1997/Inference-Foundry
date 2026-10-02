"""Reconcile mixed P61 exit state and stop only verified live P; preserve D56."""
import json,pathlib,subprocess,os,re,signal,time
expected=json.loads(os.environ["GLM_EXPECTED_COHORT"])
history=json.loads(os.environ["GLM_PRIOR_SNAPSHOT_JSON"])
registered=json.loads(os.environ["GLM_PRIOR_P_REGISTERED_JSON"])
boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def proc(pid):
    try:
        raw=pathlib.Path("/proc/"+str(pid)+"/stat").read_text();x=raw[raw.rfind(")")+2:].split()
        return dict(identity=dict(boot_id=boot,start_ticks=x[19]),state=x[0],ppid=int(x[1]))
    except FileNotFoundError:return None
def table():
    rows={}
    for row in subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True).splitlines()[1:]:
        x=row.split(None,4)
        if len(x)==5:rows[int(x[0])]=(int(x[1]),x[2],x[3],x[4])
    return rows
def descendants(rows,root):
    found={root}
    while True:
        more=found|{pid for pid,(parent,_,_,_)in rows.items()if parent in found}
        if more==found:return found
        found=more
P=next(o for o in expected if o["role"]=="P");D=next(o for o in expected if o["role"]=="D")
priorP=registered["P"+str(P["rank"])];assert priorP["API_owner"]==P
Dtargets={x["pid"]:x for x in history["retained_D_targets"]};assert D["pid"]==history["exact_D_root"]
def checkD():
    for pid,x in Dtargets.items():
        z=proc(pid);assert z and z["identity"]==x["identity"]and z["state"]not in["Z","X"],"D member changed"
    assert [x.decode()for x in pathlib.Path("/proc/"+str(D["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==D["argv"]
checkD();ps=table();root=proc(P["pid"])
Palive=bool(root and root["identity"]==P["identity"]and root["state"]not in["Z","X"])
if Palive:
    assert [x.decode()for x in pathlib.Path("/proc/"+str(P["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==P["argv"]
    Ptargets=descendants(ps,P["pid"])
else:Ptargets=set()
assert not(Ptargets&Dtargets.keys())
physical={x["pid"]:{k:x[k]for k in["boot_id","start_ticks"]}for x in priorP["physical_workers"]}
for pid,ident in physical.items():
    z=proc(pid)
    if z and z["identity"]==ident and z["state"]not in["Z","X"]:assert Palive and pid in Ptargets,"registeredP active outside root tree"
Pids={pid:proc(pid)["identity"]for pid in Ptargets}
def native_scope(rows,Pallowed,expectedroots):
    roots=set();un=[];inactive=[]
    for pid,(ppid,state,comm,args)in rows.items():
        isroot="/bin/vllm serve "in args or"native_acl_lifecycle.py cli serve "in args
        native=isroot or comm.startswith("VLLM")or"bishengir-compile "in args
        if not native:continue
        if state[0]in["Z","X"]:inactive.append(dict(pid=pid,ppid=ppid,state=state,comm=comm));continue
        if isroot:roots.add(pid)
        if pid not in Pallowed and pid not in Dtargets:un.append(dict(pid=pid,ppid=ppid,state=state,comm=comm,proc=proc(pid)))
    if un:print(json.dumps(dict(event="unattributed_active_nativeworker",rows=un,signals=0)),flush=True)
    assert not un and roots==expectedroots,"unattributed active native resource"
    return inactive
inactive=native_scope(ps,Ptargets,{D["pid"]}|({P["pid"]}if Palive else set()))
def npu():
    raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)
    ids={int(x)for x in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",raw,re.M)}
    return raw,ids
raw,npus=npu();assert npus<=Ptargets|Dtargets.keys()and len(npus&Dtargets.keys())==16,"NPU foreign/Dmissing"
for pid in npus&Ptargets:assert pid in physical and proc(pid)["identity"]==physical[pid],"P NPU identity outside registered original"
receipt=dict(exact_P_root=P["pid"],exact_D_root=D["pid"],P_API_active=Palive,owned_P_targets=[dict(pid=pid,identity=Pids[pid],comm=ps[pid][2])for pid in sorted(Ptargets)],retained_D_targets=history["retained_D_targets"],P_registered_worker_count=len(physical),historical_P_zombies=len(inactive),inactive_native_zombies=inactive,npu_P=len(npus&Ptargets),npu_D=16,npu_smi_raw=raw,signals=0,preflight_only=True)
print(json.dumps(receipt),flush=True)
if os.environ.get("GLM_CLEANUP_PREFLIGHT_ONLY")=="1":raise SystemExit(0)
def active(pid):
    z=proc(pid);return bool(z and z["identity"]==Pids[pid]and z["state"]not in["Z","X"])
def signal_one(pid,sig):
    if not active(pid):return
    assert pid in Ptargets and pid not in Dtargets
    print(json.dumps(dict(event="signal_attempt",pid=pid,identity=Pids[pid],signal=sig.name,scope="exact_live_P61")),flush=True)
    try:os.kill(pid,sig);print(json.dumps(dict(event="signal_sent",pid=pid,signal=sig.name)),flush=True)
    except ProcessLookupError:print(json.dumps(dict(event="signal_raced_exit",pid=pid,signal=sig.name)),flush=True)
checkD()
if Palive:
    signal_one(P["pid"],signal.SIGTERM)
    end=time.monotonic()+45
    while time.monotonic()<end and any(active(pid)for pid in Ptargets):time.sleep(1)
    for pid in sorted(Ptargets,reverse=True):signal_one(pid,signal.SIGKILL)
    time.sleep(2)
assert not any(active(pid)for pid in Ptargets),"P stillactive"
for pid,ident in physical.items():
    z=proc(pid);assert not z or z["identity"]!=ident or z["state"]in["Z","X"],"P registeredworker active afterstop"
checkD();native_scope(table(),set(),{D["pid"]});raw,npus=npu();assert len(npus)==16 and npus<=Dtargets.keys(),"NPU not D-only afterstop"
print(json.dumps(dict(P_inactive=True,P_was_active=Palive,D_every_identity_retained=True,npu_smi_raw=raw,signals_to_D=0,signals_to_unattributed=0,signals_to_inactive_native=0)),flush=True)
