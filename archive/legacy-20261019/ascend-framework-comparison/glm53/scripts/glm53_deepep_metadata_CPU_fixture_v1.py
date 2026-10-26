"""CPU-only original/overlay property checks; no DeepEP, NPU or MemCache imports."""
import __future__,ast,builtins,hashlib,importlib.util,json,sys,types,typing
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
SOURCE_SHA='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84'
OPT_SHA='daf4825ab2579af4cca617cf2746b8a97d30020e7abde47277bc4abe7e009363'

class TorchCPUProxy:
 def __getattr__(self,name):return getattr(torch,name)
 def histc(self,x,bins,min,max):
  valid=x.reshape(-1);valid=valid[(valid>=0)&(valid<bins)]
  return torch.bincount(valid.to(torch.int64),minlength=bins).to(torch.int32)
class Event:
 pass

def make_module(name,source):
 module=types.ModuleType(name);module.__file__=str(ROOT/'normal_strategy.py');sys.modules[name]=module
 module.__dict__.update({k:v for k,v in vars(typing).items() if not k.startswith('__')});module.__dict__.update(__name__=name,torch=TorchCPUProxy(),EventOverlap=Event,NormalEPCommStrategy=object)
 cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='AlltoAllNormalCommStrategy');cls.decorator_list=[]
 tree=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));exec(compile(tree,module.__file__,'exec',flags=__future__.annotations.compiler_flag),module.__dict__)
 return module
def check_equal(a,b):
 if isinstance(a,torch.Tensor):assert type(b) is type(a) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 elif isinstance(a,Event):assert isinstance(b,Event)
 else:assert type(a) is type(b) and a==b
def main():
 source=(ROOT/'normal_strategy.py').read_bytes();assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
 p=ROOT/'glm53_deepep_metadata_batch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==OPT_SHA
 spec=importlib.util.spec_from_file_location('opt',p);opt=importlib.util.module_from_spec(spec);spec.loader.exec_module(opt)
 old=make_module('fixture_original_normal',source.decode());new=make_module('fixture_modified_normal',source.decode());captured=[]
 def capture_compile(tree,*args,**kwargs):captured.append(tree);return builtins.compile(tree,*args,**kwargs)
 opt.compile=capture_compile;patch=opt.install(new);del opt.compile
 assert len(captured)==1
 transformed=next(x for x in captured[0].body if isinstance(x,ast.FunctionDef) and x.name=='dispatch')
 fresh_count_calls=[x for x in ast.walk(transformed) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='list'];assert len(fresh_count_calls)==1
 assert not any(isinstance(x,ast.Attribute) and x.attr=='cpu' for x in ast.walk(transformed))
 transformed_count_expr=compile(ast.Expression(body=fresh_count_calls[0]),'fixture-count-expression','eval')
 checks=[];real_cpu=torch.Tensor.cpu
 for case in ['balanced3','balanced6','skewed3','sparse0and6','allzero','idle_minus1']:
  inputs=[]
  for rank in range(32):
   n=0 if case=='allzero' or case=='sparse0and6' and rank%2 else 6 if case in ['balanced6','sparse0and6'] else 3
   idx=(torch.arange(n*8).reshape(n,8)+rank*8)%256
   if case=='skewed3':idx=torch.arange(8).expand(n,8).clone()
   if case=='idle_minus1':idx.fill_(-1)
   inputs.append(idx.to(torch.int64))
  proxy=TorchCPUProxy();global_counts=torch.stack([proxy.histc(x,256,0,256) for x in inputs])
  for rank,idx in enumerate(inputs):
   original=old.AlltoAllNormalCommStrategy.__new__(old.AlltoAllNormalCommStrategy);changed=new.AlltoAllNormalCommStrategy.__new__(new.AlltoAllNormalCommStrategy)
   for obj in [original,changed]:
    obj.group=object();obj.group_size=32;obj.rank=rank;obj._alltoall_layout=None;obj._gather_along_first_dim=lambda local,group:global_counts.clone()
   counts=[0]
   def measured_cpu(self,*args,**kwargs):counts[0]+=1;return real_cpu(self,*args,**kwargs)
   try:
    torch.Tensor.cpu=measured_cpu;ref=original.get_dispatch_layout(idx,256);ref_counts=original._alltoall_layout['num_global_tokens_per_local_expert'].sum(axis=0).cpu().numpy().tolist();before=counts[0]
    counts[0]=0;actual=changed.get_dispatch_layout(idx,256);actual_counts=eval(transformed_count_expr,{}, {'self':changed});after=counts[0]
   finally:torch.Tensor.cpu=real_cpu
   assert before==3 and after==1,(case,rank,before,after)
   for a,b in zip(ref,actual):check_equal(a,b)
   for k,a in original._alltoall_layout.items():check_equal(a,changed._alltoall_layout[k])
   assert ref_counts==actual_counts and all(type(a) is type(b) for a,b in zip(ref_counts,actual_counts))
   retained=list(changed._alltoall_layout['_glm53_host_expert_counts']);actual_counts.append(-123);assert changed._alltoall_layout['_glm53_host_expert_counts']==retained
  checks.append({'case':case,'ranks':32,'pass':True,'host_downloads_before':3,'host_downloads_after':1})
 opt.uninstall(new);assert not new.AlltoAllNormalCommStrategy._glm53_metadata_batch_installed
 assert not any(x.startswith(('torch_npu','deep_ep','memcache_hybrid','memfabric_hybrid')) for x in sys.modules)
 out={'project_id':'glm53','pass':True,'model_requests':0,'SDK_init_calls':0,'NPU_devices':False,'torch_version':torch.__version__,'native_NPU_qualification_required':True,'original_source_sha256':SOURCE_SHA,'optimizer_sha256':OPT_SHA,'cases':checks,'exact_count_types_layout_fields_repeat_indices_and_fresh_list_ownership_checked':True,'source_guard_and_rollback_checked':True,'patch':patch}
 (ROOT/'fixture-result.json').write_text(json.dumps(out,indent=2)+'\n');print('GLM53_METADATA_CPU_FIXTURE='+json.dumps(out),flush=True)
if __name__=='__main__':main()
