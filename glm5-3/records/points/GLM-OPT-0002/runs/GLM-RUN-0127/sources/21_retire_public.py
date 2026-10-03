from pathlib import Path
import json,sys,hashlib,subprocess,re,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0125"
state=json.loads((old/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
audit=p/"jobs/AUDIT-RUN125-20261002T2352Z-v2/reduction.json";assert hashlib.sha256(audit.read_bytes()).hexdigest()=="9b8cc93c3ba654b50e46e39cd7f743e50b44edd574b5ee1bddcf4da3ff357767"
proof=json.loads(audit.read_text());assert proof["functional_acceptance"]
service=json.loads((old/"public_service_owner.json").read_text());assert service==proof["public_service"]["container_owner"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"before_public_retire"],capture_output=True,timeout=240);(r/"before_public_retire.stdout").write_bytes(z.stdout);(r/"before_public_retire.stderr").write_bytes(z.stderr);z.check_returncode();guard()
code="""import json,sys,pathlib,os,signal,time
a=json.load(sys.stdin);i=a['identity'];p=pathlib.Path('/proc/'+str(a['pid']))
def same():
 try:
  b=(p/'stat').read_text();v=b[b.rfind(')')+2:].split();return v[0]not in['Z','X']and v[19]==i['start_ticks']and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==i['boot_id']
 except FileNotFoundError:return False
assert same()and[v.decode()for v in(p/'cmdline').read_bytes().split(bytes([0]))if v]==a['argv']
os.kill(a['pid'],signal.SIGTERM)
for _ in range(200):
 if not same():break
 time.sleep(.1)
else:raise RuntimeError('owned oldfrontend notstopped')
print(json.dumps(dict(exact_owned_frontend_stopped=True,container_pid=a['pid'],native_model_signals=0,old_gateway_SDK_finalize='notobserved in legacySIGTERM path')))
"""
z=subprocess.run(["docker","exec","-i","glm52-single","python3","-c",code],input=json.dumps(service).encode(),capture_output=True,timeout=45);(r/"public_retire.stdout").write_bytes(z.stdout);(r/"public_retire.stderr").write_bytes(z.stderr);z.check_returncode()
assert not re.search(r":(?:8000|8002)\s",subprocess.check_output(["ss","-ltnp"],text=True))
fault=json.loads((old/"response_owners.json.fault").read_text());assert not fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
owners=json.loads((old/"response_owners.json").read_text())["owners"];assert owners["resp_glm_run125_base"]["replica"]=="D0"and owners["resp_glm_run125_other"]["replica"]=="D1"
atomic_json(r/"public_retire.json",dict(at=utc(),old_owner=service,old_retained_state_dir=str(old),fault_journal_closed=True,preserved_nativeIDs=["resp_glm_run125_base","resp_glm_run125_other"],new_gateway_requires_CPU_SDKfinal0=True,retirement=json.loads(z.stdout),model_operations=0))
