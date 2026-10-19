"""Native utility AST plus ownership/deferred-consumer checks; no NPU proof."""
from pathlib import Path
from types import SimpleNamespace as S
from dataclasses import dataclass,replace
import ast,hashlib,importlib.util,json,random
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe',root/'moe_prefix_gmm2_count.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-moe-count-source/moe_utils.py');raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='2c3dc13f2f7955f0e077df07adf8538929404a8f0af35a0ea9da42d42bcf5b5a'
f=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name=='cumsum_group_list')
class Tensor:
 def __init__(self,values,dtype='i64',device='npu',contiguous=True):self.values=list(values);self.dtype=dtype;self.device=S(type=device);self.shape=(len(values),);self.contiguous=contiguous
 def is_contiguous(self):return self.contiguous
 def __getitem__(self,key):return Tensor(self.values[key] if isinstance(key,slice) else [self.values[key]],self.dtype,self.device.type)
 def unsqueeze(self,dim):assert dim==0;return self
 def cumsum(self,dim):
  assert dim==0;total=0;out=[]
  for v in self.values:total+=v;out.append(total)
  return Tensor(out,self.dtype,self.device.type)
def cat(v,dim):assert dim==0;return Tensor([a for t in v for a in t.values],v[0].dtype,v[0].device.type)
def diff(t):return Tensor([b-a for a,b in zip(t.values,t.values[1:])],t.dtype,t.device.type)
torch=S(Tensor=Tensor,int64='i64',cat=cat,diff=diff);ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns);utility=ns['cumsum_group_list'];checks=[]
rng=random.Random(1919)
for total in [0,48,192,768,3072]:
 for layout in ['head','tail','scatter']:
  values=[0]*256
  for _ in range(total):values[0 if layout=='head' else 255 if layout=='tail' else rng.randrange(256)]+=1
  original=Tensor(values);prefix=original.cumsum(0);recovered=utility(prefix,0,1)
  assert recovered.values==values and prefix.values==original.cumsum(0).values and recovered.dtype=='i64'
checks.append('native_utility_roundtrip_15_sparse_dense_boundary_histograms')
@dataclass(frozen=True)
class Output:
 group_list:object
 group_list_type:int=0
 hidden_states:object=None
 combine_metadata:object=None
@dataclass(frozen=True)
class Input:
 group_list:object
 group_list_type:int
 fusion:bool=True
 dynamic_eplb:bool=False
 activation:object=None
 layer:object=None
marker=object();prefix=Tensor([3,3,8]+[8]*253);output=Output(prefix,0,marker,marker);seen=[];converted=[]
def dispatch(self,inp):return output
def fused(self,inp,*args,**kw):seen.append(('fused',inp,args,kw));return marker
def gmm1(self,inp,*args,**kw):seen.append(('gmm1',inp,args,kw));return marker
def gmm2(self,inp,*args,**kw):seen.append(('gmm2',inp,args,kw));return marker
def convert(g,src,dst):converted.append((g,src,dst));return utility(g,src,dst)
candidate,ff,gg1,gg2,stats=m.representations(torch,dispatch,fused,gmm1,gmm2,convert);obj=S(top_k=8);x=S(hidden_states=S(shape=(6,6144)));out=candidate(obj,x);assert out is output and out.hidden_states is marker and out.combine_metadata is marker and not converted
checks.append('dispatch_preserves_native_prefix_and_other_outputs_without_recovery')
mlp=Input(prefix,0,activation=S(name='SILU'),layer=marker);assert ff(obj,mlp,'a',scale=2) is marker and seen[-1][1] is mlp and not converted
checks.append('original_fused_GMM1_keeps_exact_prefix_without_preceding_recovery')
assert gg2(obj,mlp,'a',scale=2) is marker;actual=seen[-1][1];assert actual is not mlp and actual.group_list_type==1 and actual.group_list.values==[3,0,5]+[0]*253 and actual.layer is marker;assert converted==[(prefix,0,1)] and mlp.group_list is prefix and mlp.group_list_type==0 and prefix.values==[3,3,8]+[8]*253 and seen[-1][2:]==(('a',),{'scale':2})
checks.append('deferred_original_utility_at_GMM2_no_input_or_prefix_mutation')
assert stats=={'prefix':1,'fused':1,'restored':1};checks.append('stage_counters_track_successful_owned_calls')
for method in [ff,gg1,gg2]:
 plain=Input(Tensor([3,0,5]+[0]*253),1);before=len(converted);assert method(obj,plain) is marker and seen[-1][1] is plain and len(converted)==before
checks.append('native_control_and_MTP_count_passthrough')
plain=Input(Tensor([8]*256),0);before=len(converted);assert gg2(obj,plain) is marker and seen[-1][1] is plain and len(converted)==before;checks.append('unrelated_prefix_not_reinterpreted')
for name,bad in [('unfused',replace(mlp,fusion=False)),('eplb',replace(mlp,dynamic_eplb=True)),('interleave',replace(mlp,activation=S(name='SWIGLUOAI_UNINTERLEAVE')))]:
 try:ff(obj,bad);raise RuntimeError('accepted '+name)
 except AssertionError:pass
 checks.append(name+'_rejected')
try:gg1(obj,mlp);raise RuntimeError('accepted nonfused')
except AssertionError:pass
checks.append('nonfused_GMM1_rejected')
for name,prefix,typ in [('count_routing',Tensor([0]*256),1),('dtype',Tensor([0]*256,'i32'),0),('shape',Tensor([0]*255),0),('CPU',Tensor([0]*256,device='cpu'),0),('noncontiguous',Tensor([0]*256,contiguous=False),0)]:
 output=Output(prefix,typ,marker,marker)
 try:candidate(obj,x);raise RuntimeError('accepted '+name)
 except AssertionError:pass
 checks.append('unexpected_'+name+'_fails_closed')
print(json.dumps({'completed':True,'passed':len(checks),'checks':checks,'native_utility_source_sha256':hashlib.sha256(raw).hexdigest(),'helper_sha256':hashlib.sha256((root/'moe_prefix_gmm2_count.py').read_bytes()).hexdigest(),'scope':'CPU exact native utility AST with integer-list tensor stand-in, frozen-dataclass ownership and consumer timing guards. No torch/NPU kernel, compile, TP, full model or performance evidence; original graph-owner17 checks reused only at unchanged source SHA.'}))
