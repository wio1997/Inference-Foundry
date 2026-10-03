from pathlib import Path
import ast,json,sys,hashlib,types
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from pp_empty_token_guard import guarded_source
source_path=Path("/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py")
raw=source_path.read_bytes();txt=raw.decode();tree=ast.parse(txt)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="GPUModelRunner")
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_update_states")
source=ast.unparse(method)+"\n";fixed=guarded_source(source)
def replay(s,token_lists,index,computed,existing):
 tree=ast.parse(s);branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="not is_last_rank")
 body=ast.Module(body=branch.body,type_ignores=[]);ast.fix_missing_locations(body)
 state=types.SimpleNamespace(num_tokens=existing,output_token_ids=[])
 data=types.SimpleNamespace(new_token_ids=token_lists)
 namespace=dict(req_data=data,i=index,num_computed_tokens=computed,req_state=state)
 error=None
 try:exec(compile(body,"native_pp_state_branch","exec"),namespace)
 except Exception as e:error=type(e).__name__+":"+str(e)
 return dict(error=error,output_token_ids=state.output_token_ids)
rows=[]
for name,tokens,i,computed,existing in [("mixed_empty",[[],[7]],0,6,5),("single_commit",[[],[7]],1,5,5),("multiple_commit",[[7,8]],0,5,5),("no_gap",[[],[7]],0,5,5)]:
 before=replay(source,tokens,i,computed,existing);after=replay(fixed,tokens,i,computed,existing)
 if name=="mixed_empty":assert before["error"].startswith("IndexError") and after==dict(error=None,output_token_ids=[])
 else:assert before==after and not after["error"]
 rows.append(dict(name=name,before=before,after=after,synthetic_metadata=True))
print(json.dumps(dict(event="CPU_PP_empty_token_guard",native_file_sha256=hashlib.sha256(raw).hexdigest(),native_function_line=method.lineno,fixed_function_sha256=hashlib.sha256(fixed.encode()).hexdigest(),metadata_cases=rows,NPU_worker_init=0,device_tensors=0,inference=0,native_source_edits=0,limits=["Actual native AST branch under synthetic metadata, not full state or GPU correctness; real failed scheduler rows were not captured, so exact runtime cause remains conditional."])),flush=True)
