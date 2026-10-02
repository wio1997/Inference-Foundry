from pathlib import Path
import json,subprocess,shlex,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0048";site=Path("/data/tiankuan/wio/glm52-pd/deploy")
def ref(f):
 d=f.read_bytes();return{"path":str(f),"bytes":len(d),"sha256":hashlib.sha256(d).hexdigest()}
def cmd(node,args):
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 return subprocess.check_output(args,timeout=90)
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
pins=json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(Path(x["path"]))["sha256"]==x["sha256"]
owners=json.loads((r/"startup_model_identities.json").read_text());logs={}
for node in ["166","167"]:
 data=cmd(node,["cat",str(site/("logs/P_run48_"+str(0 if node=="166"else 1)+".log"))]);f=j/("native_"+node+".log");f.write_bytes(data);t=data.decode(errors="replace")
 assert "KeyError: 'invalid tool call parser: glm48_contract"in t and"glm47_contract"in t
 assert not any(k in t for k in ["GLM_DP_METADATA_INSTALLED","Loading model weights","Starting to load model","GPU KV cache size:"])
 top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]);(j/(node+".processes")).write_bytes(top)
 assert not any("native_acl_lifecycle.py cli serve "in l or"VLLM::"in l for l in top.decode().splitlines())
 o=next(v for v in owners.values()if v["host"]==node)
 assert cmd(node,["python3","-c","import pathlib;assert not pathlib.Path('/proc/"+str(o["pid"])+"').exists();print('exactpriorrootabsent')"]).strip()==b"exactpriorrootabsent"
 logs[node]={"ref":ref(f),"error_lines":[l for l in t.splitlines()if "KeyError:"in l],"sdk_events":[json.loads(l)for l in t.splitlines()if l.startswith('{"event": "task_acl_')]}
assert not(r/"pilot.phase.json").exists()
events=json.loads((r/"deployment_events.json").read_text());starts=[x for x in events if x["event"]=="native_role_started"];assert len(starts)==2 and all(x["role"]=="P"for x in starts)
out={"at":utc(),"run_id":r.name,"measurement_valid":True,"functional_acceptance":False,"status":"failed","verdict":"INVALID","classification":"GPTconfigurationgenerator renamedregisteredtoolparser withoutchangingpluginregistration","source_pins":len(pins),"controller":s,"native_logs":logs,"prior_owners":owners,"current_native_roots":"absentbothhosts/nounattributedworker","P_API_starts":2,"NPU_workers_started":0,"D_started":False,"inference_attempts":0,"effective_output_tokens":0,"actual_old47_cleanup":"yes exactP0rootanddescendants only; P1absent","auditor_signals":0,"new_models":0,"new_requests":0,"limits":["Noengine/NPUweight/portlayout/dualresidentmemoryverdict","Old47cleanup actuallyexecuted; zeroallactions isfalse","ErroroccursnativeAPI validationbeforeweights; newRunpreservesregisteredglm47_contractsymbol"]}
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",failure_phase="deploy",reduction=ref(j/"reduction.json"),results={k:out[k]for k in ["inference_attempts","effective_output_tokens","NPU_workers_started","D_started"]});atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nINVALID configurationgenerator renamedtoolparser to glm48_contract althoughunchangedpluginregisteredglm47_contract. BothnativeAPIstarted/exitedbeforeengineworkers/weights; Dnotstarted/0inference/outputs. Exactold47P0cleanupdidoccur, P1alreadyabsent. 17frozenpins verified; bothroots/workers absent. Noport/memory/operatorverdict; newRunkeepsregisteredparser andaddsactualfullCLI/APIvalidation.\n")
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Run48 configurationINVALID nativeparsername beforeworker/weights; exactbothrootsgone/17pins/0D/inference; old47cleanupdidoccur","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="nativeparsererror/frozenpins/absence/0worker",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"verdict":"INVALID","requests":0,"workers":0,"reduction":ref(j/"reduction.json")}))

