from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
code="""from pathlib import Path
import json
roots=[Path('/vllm-workspace/vllm/vllm'),Path('/vllm-workspace/vllm-ascend/vllm_ascend'),Path('/usr/local/python3.12.13/lib/python3.12/site-packages/vllm'),Path('/usr/local/python3.12.13/lib/python3.12/site-packages/vllm_ascend')]
suffixes=['v1/core/sched/scheduler.py','config/scheduler.py','engine/arg_utils.py','v1/engine/core.py','entrypoints/openai/api_server.py']
files=[]
for root in roots:
 for suffix in suffixes:
  f=root/suffix
  if f.is_file():files.append(dict(path=str(f),content=f.read_text()))
print(json.dumps(files))
"""
z=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=60);(j/"source_read.stdout").write_bytes(z.stdout);(j/"source_read.stderr").write_bytes(z.stderr);z.check_returncode();files=json.loads(z.stdout);assert len(files)>=3
rows=[]
for n,x in enumerate(files):
 b=x["content"].encode();f=j/(str(n)+"_"+Path(x["path"]).name);f.write_bytes(b);ls=x["content"].splitlines();keys=["max_num_scheduled_tokens","scheduler_cls","self.scheduler.schedule(","def schedule(","token_budget"]
 hits=[i for i,l in enumerate(ls)if any(k in l for k in keys)];groups=[]
 for i in hits:groups.append(dict(line=i+1,context="\n".join(str(k+1)+":"+ls[k]for k in range(max(0,i-5),min(len(ls),i+12)))))
 rows.append(dict(native_path=x["path"],raw=dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),sections=groups))
out=dict(at=utc(),native_sources=rows,new_requests=0,new_models=0,signals=0,limits=["NativeCPU scheduler source research; no budgetpolicy injected/nativeGPU/performance/capacity proof","Preserve fullnativeAPI/K5/KV/metadata/operator contracts; adaptive CPU budget requires native-source and E2E judgement"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="NativeCPU scheduler budget/customclass/core scheduling source captured without SDK import/model/inference",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativeinstalledsource")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(files=len(files),bytes=len(b))))

