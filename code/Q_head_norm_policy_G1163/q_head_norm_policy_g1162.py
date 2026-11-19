"""Honor configured Q head normalization in the two fixed NPU target paths.
The V4.1 checkpoint disables Q-head RMSNorm; Q-LoRA and KV norm stay unchanged.
"""
from pathlib import Path
import ast,hashlib,inspect,json,os,textwrap
_DONE=False
SOURCE_SHA='6c8254375ef1950ef53202422d1a55935e8721091f3b9bb802b074ffaabc8c6f'

def install(root):
    global _DONE
    if _DONE:return
    import sglang.srt.models.deepseek_v4 as native
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==SOURCE_SHA
    root=Path(root);seen=set()
    def policy(self,q,path):
        key=(id(self),path)
        if key in seen:return
        seen.add(key)
        # One per-path/rank witness is sufficient; do not serialize tensor data.
        rank=__import__('torch').distributed.get_rank()
        name=root/f'Q_head_norm_policy.{path}.rank{rank}.json'
        if not name.exists():name.write_text(json.dumps({'pid':os.getpid(),'rank':rank,'path':path,'is_dsv41':bool(self.is_dsv41),'configured_q_head_norm':bool(self.q_head_norm),'actual_action':'per_head_RMSNorm' if self.q_head_norm else 'skip_per_head_RMSNorm','shape':list(q.shape),'dtype':str(q.dtype),'device':str(q.device),'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Runtimeconfiguredbranch witness; Q-LoRAnorm/KVnorm/RoPE/weights unchanged; quality/performance notaccepted'},indent=2)+'\n')
    changed=[]
    for path in ['_forward_prepare','_forward_prepare_multi_stream_npu']:
        fn=getattr(native.MQALayer,path);node=ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0];node.decorator_list=[]
        class Transform(ast.NodeTransformer):
            count=0
            def visit_Assign(self,n):
                self.generic_visit(n)
                if (len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='q' and isinstance(n.value,ast.Subscript) and isinstance(n.value.value,ast.Call) and isinstance(n.value.value.func,ast.Attribute) and n.value.value.func.attr=='npu_rms_norm' and isinstance(n.value.value.func.value,ast.Name) and n.value.value.func.value.id=='torch_npu'):
                    self.count+=1
                    call=ast.Expr(value=ast.Call(func=ast.Name(id='_ds41_qnorm_policy_g1162',ctx=ast.Load()),args=[ast.Name(id='self',ctx=ast.Load()),ast.Name(id='q',ctx=ast.Load()),ast.Constant(path)],keywords=[]))
                    guard=ast.If(test=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='q_head_norm',ctx=ast.Load()),body=[n],orelse=[])
                    return [ast.copy_location(call,n),ast.copy_location(guard,n)]
                return n
        t=Transform();node=t.visit(node);assert t.count==1,(path,t.count)
        unit=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]));ns=dict(fn.__globals__);ns['_ds41_qnorm_policy_g1162']=policy;exec(compile(unit,str(root/'q_head_norm_policy_g1162.py')+'::'+path,'exec'),ns)
        new=ns[node.name];assert not fn.__code__.co_freevars and not new.__code__.co_freevars
        fn.__globals__['_ds41_qnorm_policy_g1162']=policy;fn.__code__=new.__code__;changed.append({'function':path,'guarded_unconditional_Q_head_norm_calls':1,'function_identity_retained':True})
    _DONE=True
    (root/f'Q_head_norm_policy_install.pid{os.getpid()}.json').write_text(json.dumps({'source_SHA256':SOURCE_SHA,'changes':changed,'scope':'ConfigrespectingNPUtargetQ-headnorm policy, nochangeQ-LoRA/KVnorm/weights/RoPE. Actualbranch/quality/performance gates required.'},indent=2)+'\n')
