"""Task-private PP state guard; no model/operator changes."""
import ast,hashlib,inspect,json,textwrap
from pathlib import Path
_EXPECTED_GPU_SOURCE = '993f2c926012241bcb8cc7f568c8e82968e4d81ca125641c831ecf501aac02ac'

def guarded_source(source):
 tree=ast.parse(source)
 targets=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.If) and ast.unparse(n.test)=="num_new_tokens == 1":
   targets.append(n)
 if len(targets)!=1:
  raise RuntimeError("Native PP token update shape changed; refuse patch")
 old=targets[0];old.test=ast.BoolOp(op=ast.And(),values=[old.test,ast.Name(id="new_token_ids",ctx=ast.Load())])
 ast.fix_missing_locations(tree)
 return ast.unparse(tree)+"\n"

def install():
 import vllm.v1.worker.gpu_model_runner as native
 if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=_EXPECTED_GPU_SOURCE:
  raise RuntimeError("Native PP base file changed; refuse patch")
 cls=native.GPUModelRunner
 if getattr(cls,"_glm_pp_empty_token_guard",False):return
 original=cls._update_states
 source=textwrap.dedent(inspect.getsource(original))
 fixed=guarded_source(source)
 namespace=dict(native.__dict__)
 exec(compile(fixed,__file__,"exec"),namespace)
 cls._update_states=namespace["_update_states"]
 cls._glm_pp_empty_token_guard=True
 print("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "+json.dumps(dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),fixed_source_sha256=hashlib.sha256(fixed.encode()).hexdigest(),change="Skip last-token indexing when this request has no committed tokens; preserves all nonempty commits",model_operator_edits=0)),flush=True)
