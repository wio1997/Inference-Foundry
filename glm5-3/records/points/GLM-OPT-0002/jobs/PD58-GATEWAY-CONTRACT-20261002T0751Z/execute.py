from pathlib import Path
import json,ast,hashlib,subprocess,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;g=j.parents[3];native="/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
z=subprocess.run(["docker","exec","glm52-single","cat",native],capture_output=True,check=True);(j/"native_proxy.py").write_bytes(z.stdout)
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def routes(tree):
 out=[]
 for f in ast.walk(tree):
  if not isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
  for d in f.decorator_list:
   if isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.args and isinstance(d.args[0],ast.Constant)and isinstance(d.args[0].value,str)and d.func.attr in["get","post","put","delete","api_route","head","patch","options"]:
    out.append(dict(path=d.args[0].value,decorator=d.func.attr,handler=f.name,line=f.lineno))
 return sorted(out,key=lambda x:x["line"])
tree=ast.parse(z.stdout);nr=routes(tree);assert {x["path"]for x in nr}>={"/v1/completions","/v1/chat/completions","/v1/models"}
assert not any(x["path"]=="/{path:path}"or"/responses"in x["path"]for x in nr)
ns={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for name in["build_prefill_request","send_request_to_service","assign_instances","handle_completions_impl"]:(j/(name+".source.py")).write_text(ast.get_source_segment(z.stdout.decode(),ns[name])+"\n")
runtime=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
generic=runtime/"response_affinity_gateway.py";gr=routes(ast.parse(generic.read_text()));assert any(x["path"]=="/{path:path}"for x in gr)
sources=[generic,runtime/"response_affinity.py",runtime/"persistent_coupled_placement.py",runtime/"glm_gateway.py"]
out=dict(at=utc(),installed_proxy=dict(native_path=native,source=ref(j/"native_proxy.py"),routes=nr),existing_response_gateway=dict(source=ref(generic),routes=gr),task_sources=[ref(f)for f in sources],native_requests=0,signals=0,models=0,findings=["Installed PD proxy explicitly routes completions/chat/models/cache reset/instance operations; it lacks generic native endpoint and Responses handlers","Existing task response-affinity gateway has transparent catchall and saved native response-owner affinity; prior Run46 finite state/feature evidence remains conditional to that model epoch/config","Native P helper changes stream false, max/min output1 and KV producer flags; actual returned producer metadata is relayed to D; copied body is not universally unmodified public API contract","PD resource release in proxy scheduler is request bookkeeping; producer native shard completion/ACK is a separate dependency per existing lifecycle proof"],next_research=["After actual58pilot/native audit and independent PD-local token control, integrate validated optional chat/completion PD path with complete D native/stateful catchall","Use actualnative request/token lengths/owner identities and finite measurements to choose PD/local routing; not stringlength or emptyKV heuristic as capacity proof","Retain originalpublic sampling/output/status/headers/disconnect contracts and saved Responses owner; test actualE2E dynamic arrivals/variableloads"],limits=["AST/source-only no new service launch or proxy behavior/capacity proof","Run46 compatibility is finite and different deployment; current PD integration unimplemented","No blind oldproxy replacement or oldtestqueue replay"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="NativePD proxy limited explicit API routes vs task native/stateful catchall; actualsource captured, optionalPD composition candidate; no newrequests/signals/models",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="installednativeproxy AST routes/helper/source and task response-affinity boundary",**ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(ref(j/"reduction.json")))

