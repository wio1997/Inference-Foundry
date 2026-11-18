"""Pinned V4.1 NPU frequency indexing only; eight native indexing sites."""
import ast,hashlib,importlib,inspect,json,os
from pathlib import Path
from freqs_compat import gather
_DONE=False
def _guard_cuda_rope(node):
 class Guard(ast.NodeTransformer):
  count=0
  def visit_If(self,n):
   self.generic_visit(n)
   direct_cuda_call=any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id in ('fused_rope_inplace','fused_q_norm_rope') for x in n.body)
   guarded=any(isinstance(x,ast.UnaryOp) and isinstance(x.op,ast.Not) and isinstance(x.operand,ast.Name) and x.operand.id=='_is_npu' for x in ast.walk(n.test))
   if direct_cuda_call and not guarded:
    n.test=ast.BoolOp(op=ast.And(),values=[n.test,ast.UnaryOp(op=ast.Not(),operand=ast.Name(id='_is_npu',ctx=ast.Load()))]);self.count+=1
   return n
 transform=Guard();node=transform.visit(node)
 return node,transform.count
def install(root):
 global _DONE
 if _DONE:return
 import sglang
 package=Path(sglang.__file__).resolve().parent.parent
 for name,h in json.loads((root/'native_freqs_sha256.json').read_text()).items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==h,name
 targets=[('sglang.srt.layers.attention.deepseek_v4_backend','_low_ratio_source_projections',1),('sglang.srt.layers.attention.deepseek_v4_backend','DeepseekV4AttnBackend._low_ratio_write_group',1),('sglang.srt.models.deepseek_v4_dspark','DSparkAttention._compute_q',2),('sglang.srt.models.deepseek_v4_dspark','DSparkAttention.forward',1),('sglang.srt.layers.attention.dsv4.dsv41_sparse','_rope_fq4',1),('sglang.srt.layers.attention.dsv4.v41_indexer.scoring','prefill_requests',1),('sglang.srt.layers.attention.dsv4.v41_indexer.scoring','decode_scores',1)]
 result=[]
 for module,path,expected in targets:
  mod=importlib.import_module(module);obj=mod
  for attr in path.split('.'):obj=getattr(obj,attr)
  fn=inspect.unwrap(obj);node=ast.parse(inspect.getsource(fn).lstrip() if '\n' not in inspect.getsource(fn) else __import__('textwrap').dedent(inspect.getsource(fn))).body[0];node.decorator_list=[]
  class Replace(ast.NodeTransformer):
   count=0
   def visit_Subscript(self,n):
    self.generic_visit(n)
    if (isinstance(n.value,ast.Attribute) and n.value.attr=='freqs_cis') or (isinstance(n.value,ast.Name) and n.value.id=='freqs'):
     self.count+=1;return ast.copy_location(ast.Call(func=ast.Name(id='_ds41_freqs_gather',ctx=ast.Load()),args=[n.value,n.slice],keywords=[]),n)
    return n
  transform=Replace();node=transform.visit(node);assert transform.count==expected,(path,transform.count)
  cuda_guards=0
  if module=='sglang.srt.models.deepseek_v4_dspark':
   node,cuda_guards=_guard_cuda_rope(node);assert cuda_guards==({'DSparkAttention._compute_q':2,'DSparkAttention.forward':1}[path]),(path,cuda_guards)
  unit=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]));ns=dict(fn.__globals__);ns['_ds41_freqs_gather']=gather
  exec(compile(unit,str(root/'freqs_install.py')+'::'+path,'exec'),ns);new=ns[node.name];assert not fn.__code__.co_freevars and not new.__code__.co_freevars,path;fn.__globals__['_ds41_freqs_gather']=gather;fn.__code__=new.__code__;result.append({'module':module,'function':path,'subscript_replacements':transform.count,'native_NPU_RoPE_dispatch_guards':cuda_guards})
 _DONE=True
 (root/f'freqs_install.pid{os.getpid()}.json').write_text(json.dumps({'targets':result,'source_pins_verified':True,'scope':'Factory-scoped V4.1 NPU frequency gather plus CUDA-only RoPE guards; selects existing native NPU branch, preserves fast WO_A, function identities, graphs and original RoPE helper','full_model_PD_accepted':False},indent=2))
