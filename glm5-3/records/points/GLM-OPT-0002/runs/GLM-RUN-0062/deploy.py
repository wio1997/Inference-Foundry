import json,sys,subprocess,shlex,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import guard,start_watchdog
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0061"
s=json.loads((old/"state.json").read_text());assert s["status"]=="failed" and not same_process(s["owner"])
a=json.loads((old/"reduction_brief.json").read_text());assert a["inference_attempts"]==3 and a["effective_public_output_tokens"]==64 and not a["functional_acceptance"]
owners=json.loads((old/"adopted_model_identities.json").read_text());reg=json.loads((old/"registered_P_engine_identities.json").read_text());source=(r/"stop_prior_cohort.py").read_text()
def command(node,preflight):
 guard()
 history=json.loads((old/("prior_"+node+".owner_preflight.json")).read_text());pair=[v for v in owners.values()if v["host"]==node]
 env=dict(GLM_EXPECTED_COHORT=json.dumps(pair),GLM_PRIOR_SNAPSHOT_JSON=json.dumps(history),GLM_PRIOR_P_REGISTERED_JSON=json.dumps(reg),GLM_CLEANUP_PREFLIGHT_ONLY="1"if preflight else"0")
 code="import os,json;os.environ.update(json.loads("+repr(json.dumps(env))+"));exec("+repr(source)+")"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=180)
 label=("preflight_"if preflight else"cleanup_")+node
 (r/(label+".stdout")).write_bytes(z.stdout);(r/(label+".stderr")).write_bytes(z.stderr);atomic_json(r/(label+".receipt.json"),dict(at=utc(),argv_kind="exactcurrentP61-registered/D56-physical",exit_code=z.returncode,stdout_bytes=len(z.stdout),stdout_sha256=hashlib.sha256(z.stdout).hexdigest(),stderr_bytes=len(z.stderr)))
 z.check_returncode();rows=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith("{")]
 atomic_json(r/(label+".json"),rows);return rows
for node in["166","167"]:command(node,True)
for node in["166","167"]:command(node,False)
D={k:v for k,v in owners.items()if k.startswith("D")};assert len(D)==2
atomic_json(r/"adopted_model_identities.json",D)
atomic_json(r/"deployment_summary.json",dict(at=utc(),new_model_loads=0,D56_API_epochs_retained=True,registered_P61_only_cleanup=True,D647_physical_members_retained=True,P_stopped=True,signals_to_D=0,signals_to_unknown=0))
print(json.dumps({"native_D_retained":list(D),"new_models":0,"P61_stopped":True}))
