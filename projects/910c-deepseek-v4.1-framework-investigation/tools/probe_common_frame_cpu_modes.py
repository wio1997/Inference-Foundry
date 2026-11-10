#!/usr/bin/env python3
"""Audit inference-tensor key guards against the archived CPU prototype."""
import ast,copy,json,os,sys,sysconfig,time
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
def main():
    root=Path(sys.argv[1]);spec=json.loads((root/'spec.json').read_text());(root/'started.json').write_text(json.dumps(dict(status='RUNNING_CPU_MODE_GATE',pid=os.getpid()))+'\n')
    assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0';pure=Path(sysconfig.get_path('purelib'));assert sha256((pure/'torch/__init__.py').read_bytes()).hexdigest()==spec['torch_init_sha256'];sys.path.insert(0,str(pure))
    import torch
    torch.set_num_threads(1)
    def guard():
        assert not any(n=='torch_npu' or n.startswith('torch_npu.') or n=='vllm' or n.startswith('vllm.') for n in sys.modules)
        maps=Path('/proc/self/maps').read_text();assert not any(n in maps for n in ['libtorch_npu','libascendcl','libhccl'])
    guard();p=Path(spec['helper_source_path']);assert sha256(p.read_bytes()).hexdigest()==spec['helper_source_sha256']
    original_tree=ast.parse(p.read_text());cls=next(n for n in original_tree.body if isinstance(n,ast.ClassDef) and n.name=='AscendDSACPMetadataBuilder');methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_local_token_range','_build_local_token_metadata']]
    group=SimpleNamespace(world_size=8,rank_in_group=0);ns={'torch':torch,'get_tp_group':lambda:group,'HAS_TRITON':True}
    module=ast.Module(body=[ast.ClassDef(name='OriginalHelper',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),str(p),'exec'),ns);ns['original']=ns['OriginalHelper']();ns['group']=group
    oldpath=Path(spec['prototype_source_path']);old=oldpath.read_text();assert sha256(oldpath.read_bytes()).hexdigest()==spec['prototype_source_sha256'];tree=ast.parse(old);oldcls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='CPUFrame')
    class Guard(ast.NodeTransformer):
        def visit_Attribute(self,node):
            node=self.generic_visit(node)
            if node.attr=='_version':return ast.copy_location(ast.Call(func=ast.Name(id='version_token',ctx=ast.Load()),args=[node.value],keywords=[]),node)
            return node
    newcls=Guard().visit(copy.deepcopy(oldcls));newcls.name='GuardedCPUFrame'
    ns['version_token']=lambda tensor:None if tensor.is_inference() else tensor._version
    exec(compile(ast.fix_missing_locations(ast.Module(body=[oldcls,newcls],type_ignores=[])),str(oldpath),'exec'),ns)
    def pkt(v):return (str(v.dtype),tuple(v.shape),v.contiguous().numpy().tobytes()) if isinstance(v,torch.Tensor) else v
    rows=[]
    for create_inference,call_inference in [(False,False),(False,True),(True,True)]:
        with torch.inference_mode(create_inference):
            q=torch.tensor([0,6,12,18,24],dtype=torch.int32);l=torch.tensor([32774]*4,dtype=torch.int32)
        with torch.inference_mode(call_inference):
            expected=ns['original']._build_local_token_metadata(4,24,q,l);v1_error=None
            try:ns['CPUFrame']('v1').call(4,24,q,l)
            except RuntimeError as e:v1_error=str(e)
            f=ns['GuardedCPUFrame']('guard');checks=0
            for i in range(26):
                out=f.call(4,24,q,l);assert [pkt(v) for v in out]==[pkt(v) for v in expected];checks+=1
            # Private microcost sensitivity for the guard, not a service delta.
            start=time.perf_counter_ns()
            for epoch in range(30):
                f=ns['GuardedCPUFrame'](epoch)
                for i in range(26):f.call(4,24,q,l)
            us=(time.perf_counter_ns()-start)/1000/30
        rows.append(dict(create_inference=create_inference,call_inference=call_inference,input_is_inference=q.is_inference(),v1_error=v1_error,guarded_byte_comparisons=checks,guarded_us_per_frame_sensitivity=us))
    assert rows[2]['v1_error'] and all(not r['v1_error'] for r in rows[:2]);guard()
    transformed=ast.unparse(newcls);(root/'guarded_cpu_frame.py').write_text(transformed+'\n')
    result=dict(status='PASS_CPU_MODE_KEY_GATE_WITH_LIMITED_V1_COUNTEREXAMPLE',modes=rows,guarded_comparisons=78,bitwise_mismatches=0,prototype_v1_scope='NormalCPUinput version readable; fails for inputs createdunderinference_mode.',guarded_variant='Same frame/prototype, only replace _version reads with None iftensor.is_inference() else _version. Explicit immutable-input lease remainsmandatory; no NumPywrite safety invented.',guarded_source=transformed,guarded_source_sha256=sha256((transformed+'\n').encode()).hexdigest(),torch_version=torch.__version__,no_NPU_libraries=True,new_NPU_work=0,new_profile=0,model_loads=0,service_actions=0,service_requests=0)
    (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],modes=rows,guarded_comparisons=78,new_NPU_work=0)))
if __name__=='__main__':main()
