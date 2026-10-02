from pathlib import Path
import json,sys,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parent.parent;r=p/"runs/GLM-RUN-0075";owners=json.loads((r/"startup_model_identities.json").read_text());inner=(j/"inner.py").read_text();out=dict(at=utc(),state=json.loads((r/"state.json").read_text()),nodes={},model_signals=0,inference_requests=0)
probe="""import pathlib,json,sys,subprocess,re
o=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def proc(pid):
 try:
  p=pathlib.Path('/proc/'+str(pid));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(pid=pid,state=v[0],identity=dict(boot_id=boot,start_ticks=v[19]))
 except (FileNotFoundError,ProcessLookupError):return None
a=proc(o['pid']);same=bool(a and a['identity']==o['identity']and a['state']not in['Z','X'])
if same:assert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']
z=subprocess.run(['docker','top','glm52-single','-eo','pid,ppid,stat,comm,args'],capture_output=True,text=True,timeout=30)
n=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=60)
print(json.dumps(dict(API_same_active=same,API_now=a,docker_top=z.stdout,npu_smi=n.stdout,npu_rc=n.returncode)))
"""
for key,o in owners.items():
 def run(args,stem,input=None,timeout=90):
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=input,capture_output=True,timeout=timeout);(j/(stem+".stdout")).write_bytes(z.stdout);(j/(stem+".stderr")).write_bytes(z.stderr);z.check_returncode();return z.stdout
 live=json.loads(run(["python3","-c",probe],key+".resources",json.dumps(o).encode()))
 detail=json.loads(run(["docker","exec","glm52-single","python3","-c",inner],key+".ascend"))
 assert not live["API_same_active"] and "VLLMWorker"not in live["npu_smi"], "native remains active"
 atomic_json(j/(key+".native_detail.json"),detail)
 top_lines=live["docker_top"].splitlines();active=[l for l in top_lines if any(x in l for x in["VLLM","native_acl_lifecycle.py cli serve"])]
 lines=[dict(file=x["path"],line=y["line"],text=y["text"])for x in detail["files"]for y in x["selected_lines"]]
 out["nodes"][key]=dict(API_same_active=live["API_same_active"],API_now=live["API_now"],task_process_lines=active,ascend_files=[{k:x[k]for k in["path","bytes","sha256"]}for x in detail["files"]],selected_error_lines=[x for x in lines if any(w in x["text"].lower()for w in["timeout","failed","error","exception"])][:40],physical_npu_raw=str(j/(key+".resources.stdout")))
assert out["state"]["status"]=="failed" and not same_process(out["state"]["owner"])
assert all(not v["task_process_lines"]for v in out["nodes"].values())
out["limits"]=["Nativevector timeout is observed startup failure, not proven hardwarelimit/operatorrootcause; driver logs scoped only exact75 loggedPIDs","No model resources signalled/restarted; inactive resource states are expected postnativefailure"]
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run75 terminal resource reconciliation and scopedAscend logs; no inference/signals",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualterminalAPI/NPU/drivererrors scopednativePIDs")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(nodes={k:dict(API_active=v["API_same_active"],ascend_files=len(v["ascend_files"]),native_processes=len(v["task_process_lines"]))for k,v in out["nodes"].items()})))
