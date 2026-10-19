from pathlib import Path
from types import SimpleNamespace as S
from dataclasses import dataclass,replace
import importlib.util,json
p=Path(__file__).with_name('moe_cumsum_i32.py');spec=importlib.util.spec_from_file_location('i32',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tensor:
 def __init__(self,values,dtype='i64',device='npu'):self.values=list(values);self.dtype=dtype;self.device=S(type=device);self.shape=(len(values),)
 def is_contiguous(self):return True
 def to(self,dtype):return Tensor(self.values,dtype,self.device.type)
 def cumsum(self,dim,dtype):
  assert dim==0;total=0;out=[]
  for v in self.values:total+=v;out.append(total)
  return Tensor(out,dtype,self.device.type)
@dataclass
class Output:
 group_list:object
 group_list_type:int=1
 hidden_states:object=None
 combine_metadata:object=None
@dataclass
class Input:
 group_list:object
 group_list_type:int
 fusion:bool=True
 dynamic_eplb:bool=False
 activation:object=None
 layer:object=None
native_count=Tensor([3,0,5]+[0]*253);marker=object();output=Output(native_count,1,marker,marker);calls=[]
def dispatch(self,inp):return output
def fused(self,inp,*args,**kw):calls.append(('fused',inp,args,kw));return marker
def gmm1(self,inp,*args,**kw):calls.append(('gmm1',inp,args,kw));return marker
def gmm2(self,inp,*args,**kw):calls.append(('gmm2',inp,args,kw));return marker
candidate,ff,gg1,gg2,stats=m.representations(S(int32='i32',int64='i64'),dispatch,fused,gmm1,gmm2)
obj=S(top_k=8);inp=S(hidden_states=S(shape=(6,6144)));out=candidate(obj,inp)
assert out.group_list_type==0 and out.group_list.dtype=='i64' and out.group_list.values==[3,3,8]+[8]*253
assert out.hidden_states is marker and out.combine_metadata is marker and output.group_list is native_count and output.group_list_type==1
checks=['exact_prefix_output_and_native_dispatch_output_unchanged']
mlp=Input(out.group_list,0,activation=S(name='SILU'),layer=marker);assert ff(obj,mlp,'x',scale=1) is marker
assert calls[-1][1] is mlp and calls[-1][1].group_list is out.group_list;checks.append('fused_GMM1_receives_exact_int64_prefix')
assert gg2(obj,mlp,'x',scale=1) is marker
seen=calls[-1][1];assert seen is not mlp and seen.group_list is native_count and seen.group_list_type==1 and seen.layer is marker
assert mlp.group_list is out.group_list and mlp.group_list_type==0 and calls[-1][2]==('x',) and calls[-1][3]=={'scale':1};checks.append('GMM2_restores_same_original_count_object_without_mutating_caller')
assert stats=={'prefix':1,'fused':1,'restored':1};checks.append('actual_selected_stage_counters')
for method in [ff,gg1,gg2]:
 plain=Input(native_count,1);assert method(obj,plain) is marker and calls[-1][1] is plain
checks.append('native_count_MTP_and_control_passthrough')
plain=Input(Tensor([8]*256),0);assert gg2(obj,plain) is marker and calls[-1][1] is plain;checks.append('unrelated_prefix_is_not_reinterpreted')
for name,bad in [('unfused',replace(mlp,fusion=False)),('eplb',replace(mlp,dynamic_eplb=True)),('interleave',replace(mlp,activation=S(name='SWIGLUOAI_UNINTERLEAVE')))]:
 try:ff(obj,bad);raise RuntimeError('accepted '+name)
 except AssertionError:pass
 checks.append(name+'_rejected_before_fused_operator')
try:gg1(obj,mlp);raise RuntimeError('accepted nonfused')
except AssertionError:pass
checks.append('nonfused_GMM1_rejected')
for name,count,typ in [('wrong_dtype',Tensor([0]*256,'i32'),1),('wrong_shape',Tensor([0]*255),1),('CPU',Tensor([0]*256,device='cpu'),1),('already_prefix',Tensor([0]*256),0)]:
 output=Output(count,typ,marker,marker);assert candidate(obj,inp) is output;checks.append(name+'_native_fallback')
output=Output(Tensor([3072]+[0]*255),1);out=candidate(obj,S(hidden_states=S(shape=(24,6144))));assert out.group_list.values==[3072]*256;checks.append('conservative_3072_route_bound_exact')
print(json.dumps({'completed':True,'passed':len(checks),'checks':checks,'scope':'CPU representation/ownership/consumer-restore guards. Existing graph-owner17 checks reused at unchanged SHA; NPU fragment separately validates exact operations, full model not yet accepted.'}))
