from pathlib import Path
import json,sys,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0086";original=r/"reduction_brief.json"
assert hashlib.sha256(original.read_bytes()).hexdigest()=="132a6f3b710e3402b69c508e98c1ffda806bfeb493ce1346dab250afe8cf9cba"
out=dict(at=utc(),original_reduction_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),correction="originalnative_selected_policy.observed_serial4_policy retainedfirstmatchingthreshold rowserial2from85; label didnotprove86serial4. Newrawlog rows requireserial4 exactpolicy. Originalsource/raw unchanged.",nodes={},models=0,new_requests=0,signals=0)
expected=dict(cohort="GLM-COHORT-0079",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=2,serial=4,policy_error=None,fallback=False,native_max=16384)
for node,rank in[("166",0),("167",1)]:
 path="/data/tiankuan/wio/glm52-pd/deploy/logs/D_run79_"+str(rank)+".log";args=["cat",path]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();f=j/("native_"+node+".log");f.write_bytes(z.stdout)
 lines=[x for x in z.stdout.decode(errors="replace").splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in x]
 selected=[dict(line=x,row=json.loads(x.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1]))for x in lines]
 exact=[x for x in selected if x["row"]==expected];assert len(exact)==1
 out["nodes"][node]=dict(raw=dict(path=str(f),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest()),all_selected=selected,exact_serial4=exact)
out["serial4_both_actual_native_selected"]=True
atomic_json(r/"native_policy_serial4_correction.json",out);f=r/"native_policy_serial4_correction.json";b=f.read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="86 rawnative bothactualselectedserial4/t1024/c2 verified; original reductionfirstthresholdmatchserial2label clarifiedwithout rewriting history",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="correction",path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="bothrawnative exactserial4selection+old2row")],unknowns=["Native policyserial5 fileackrestoredidle; runtimecache onlyselectsafteranothernative schedule"],decision_request=None,next_check_at=None));print(json.dumps(out))
