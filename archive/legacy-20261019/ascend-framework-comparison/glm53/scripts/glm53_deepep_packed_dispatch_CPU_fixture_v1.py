"""CPU property checks of actual source dispatch AST and lossless byte transport.
Real NPU routing/collectives are deliberately virtualized; native qualification remains mandatory.
"""
import ast,hashlib,importlib.util,json,struct,sys,types
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
EXPECTED={'normal_strategy.py': '48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84', 'glm53_deepep_metadata_CPU_fixture_v1.py': 'b82d4b25c9931d4cfa364dbbc0a1214d87fc7564b1a8855f6abb4b51180a9133', 'glm53_deepep_metadata_batch_v1.py': 'daf4825ab2579af4cca617cf2746b8a97d30020e7abde47277bc4abe7e009363', 'glm53_deepep_packed_dispatch_v1.py': '336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb'}
for name,h in EXPECTED.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
base=load('CPUbase','glm53_deepep_metadata_CPU_fixture_v1.py');L04=load('L04','glm53_deepep_metadata_batch_v1.py');pack=load('pack','glm53_deepep_packed_dispatch_v1.py')
def same(a,b):
 if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,base.Event):assert isinstance(b,base.Event)
 else:assert type(a)==type(b) and a==b
bits=[0,0x80000000,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00001,0x7fa12345,0x00000001,0x80000001,0x007fffff,0x7f7fffff,0xff7fffff,0x00800000,0xdeadbeef]
def scale_values(n,seed):
 vals=[bits[(i+seed)%len(bits)] for i in range(n)]
 return torch.tensor([v if v<2**31 else v-2**32 for v in vals],dtype=torch.int32).view(torch.float32)
def oracle_wire(q,s):
 # Independent row-by-row Python byte serialization, no candidate helper.
 vals=[]
 for row,b in zip(q.tolist(),s.contiguous().view(torch.int32).tolist()):
  vals.extend(bytes(v&255 for v in row)+struct.pack('<i',b))
 n,h=q.shape
 return torch.tensor(vals,dtype=torch.uint8).reshape(n,h+4).view(torch.int8)
checks=[]
for n in [0,1,2,3,6,17,33]:
 for h in [4,16,128,6144]:
  q=((torch.arange(n*h,dtype=torch.int64)*79+17)%256-128).to(torch.int8).reshape(n,h);s=scale_values(n,3);before=q.clone(),s.clone();packed=pack.pack_payload(torch,q,s);same(packed,oracle_wire(q,s));a,b=pack.unpack_payload(torch,packed,h);same(a,q);same(b,s);same((q,s),before)
  # Lifetimes and nonoverlap: outputs survive deleting transport owner and
  # modifying one output never modifies the other output's logical bytes.
  del packed
  if n:
   keep=b.clone();a.fill_(23);same(b,keep);keep=a.clone();b.view(torch.int32).fill_(0);same(a,keep)
  checks.append({'case':'byte_roundtrip','rows':n,'hidden':h,'pass':True})
for bad_hidden in [1,7]:
 try:pack.pack_payload(torch,torch.zeros((1,bad_hidden),dtype=torch.int8),torch.zeros(1,dtype=torch.float32))
 except AssertionError:pass
 else:raise AssertionError('Unaligned hidden bytes accepted')
source=(ROOT/'normal_strategy.py').read_text();old=base.make_module('L04original',source);new=base.make_module('L09packed',source)
L04.install(old);L04.install(new)
old_layout=new.AlltoAllNormalCommStrategy.get_dispatch_layout;old_combine=new.AlltoAllNormalCommStrategy.combine;old_dispatch=new.AlltoAllNormalCommStrategy.dispatch
captured=[];real_compile=compile
def capture(tree,*args,**kwargs):captured.append(tree);return real_compile(tree,*args,**kwargs)
pack.compile=capture;patch=pack.install(new);del pack.compile
assert new.AlltoAllNormalCommStrategy.get_dispatch_layout is old_layout and new.AlltoAllNormalCommStrategy.combine is old_combine and len(captured)==1
compiled_fn=captured[0].body[0];assert not any(isinstance(n,ast.Attribute) and n.attr in ['cpu','numpy','item'] for n in ast.walk(compiled_fn))
# Source and installation prerequisites fail closed; no fallback that changes data.
try:pack.install(new)
except AssertionError:pass
else:raise AssertionError('Double install accepted')
unqualified=base.make_module('withoutL04',source)
try:pack.install(unqualified)
except AssertionError:pass
else:raise AssertionError('Missing accepted L04 accepted')
class Wait:
 def __init__(self,stats):self.stats=stats
 def wait(self):self.stats['waits']+=1
class Ops:
 def __init__(self,q,s):self.q=q;self.s=s;self.calls=[]
 def npu_moe_init_routing_v2(self,x,idx,**kwargs):
  self.calls.append({'x':x.clone(),'idx':idx.clone(),'kwargs':dict(kwargs)})
  if len(self.calls)==1:return self.q.clone(),torch.arange(self.q.shape[0],dtype=torch.int32),None,self.s.clone()
  order=torch.argsort(idx.reshape(-1),stable=True)
  return x.index_select(0,order),torch.argsort(order).to(torch.int32),None,None
for case in ['balanced3','balanced6','skewed3','sparse0and6','allzero','one_sender','one_row_receiver']:
 for quant in ['int8','bf16']:
  h=6144 if case=='balanced3' else 16
  indices=[]
  for rank in range(32):
   n=0 if case=='allzero' or case=='sparse0and6' and rank%2 or case=='one_sender' and rank!=7 else 6 if case in ['balanced6','sparse0and6'] else 3
   ix=(torch.arange(n*8).reshape(n,8)+rank*8)%256
   if case=='skewed3':ix=torch.arange(8).expand(n,8).clone()
   if case=='one_row_receiver':ix=torch.tensor([[rank*8+j for j in range(8)]],dtype=torch.int64)
   indices.append(ix.to(torch.int64))
  proxy=base.TorchCPUProxy();counts=torch.stack([proxy.histc(x,256,0,256) for x in indices]);qs=[];ss=[]
  for rank,ix in enumerate(indices):
   total=ix.numel();q=((torch.arange(total*h,dtype=torch.int64)*7+rank*31)%256-128).to(torch.int8).reshape(total,h)
   qs.append(q if quant=='int8' else q.to(torch.bfloat16));ss.append(scale_values(total,rank))
  # Reference transport uses row ranges from each sender's expert histogram.
  for rank,ix in enumerate(indices):
   recvq=[];recvs=[]
   for sender in range(32):
    split=counts[sender].reshape(32,8).sum(1).tolist();offset=sum(split[:rank]);size=split[rank];recvq.append(qs[sender][offset:offset+size]);recvs.append(ss[sender][offset:offset+size])
   expected_q=torch.cat(recvq,0);expected_s=torch.cat(recvs,0);outputs=[];stats=[];routing=[]
   for module in [old,new]:
    obj=module.AlltoAllNormalCommStrategy.__new__(module.AlltoAllNormalCommStrategy);obj.group=object();obj.group_size=32;obj.rank=rank;obj._alltoall_layout=None;obj._gather_along_first_dim=lambda local,group:counts.clone();obj.get_dispatch_layout(ix,256)
    stat={'payload_collectives':0,'waits':0};op=Ops(qs[rank],ss[rank]);module.__dict__['torch_npu']=op
    def transport(payload,outs,ins,group,event=None):
     assert event is None and group is obj.group
     assert list(ins)==obj._alltoall_layout['input_splits'] and list(outs)==obj._alltoall_layout['output_splits']
     stat['payload_collectives']+=1
     if payload.dtype==torch.float32:
      same(payload,ss[rank]);recv=expected_s.clone()
     elif quant=='int8' and payload.shape[1]==h+4:
      same(payload,oracle_wire(qs[rank],ss[rank]));recv=oracle_wire(expected_q,expected_s)
     else:
      same(payload,qs[rank]);recv=expected_q.clone()
     return payload,recv,Wait(stat)
    obj._async_all_to_all=transport
    x=torch.zeros((ix.shape[0],h),dtype=torch.bfloat16);weights=torch.ones(ix.shape,dtype=torch.float32)
    out=obj.dispatch(x,None,None,None,None,None,ix,weights,quant_mode=quant)
    outputs.append(out);stats.append(dict(stat));routing.append(op.calls)
   same(outputs[0],outputs[1]);same(routing[0],routing[1])
   assert stats[0]=={'payload_collectives':2 if quant=='int8' else 1,'waits':2 if quant=='int8' else 1}
   assert stats[1]=={'payload_collectives':1,'waits':1}
   if quant=='int8':
    expectedorder=torch.argsort(obj._alltoall_layout['global_tokens_indices'],stable=True)
    same(outputs[1][0][0],expected_q.index_select(0,expectedorder));same(outputs[1][0][1],expected_s.index_select(0,expectedorder))
  checks.append({'case':case,'quant':quant,'ranks':32,'pass':True,'payload_collectives_before':2 if quant=='int8' else 1,'payload_collectives_after':1})
pack.uninstall(new);assert new.AlltoAllNormalCommStrategy.dispatch is old_dispatch
assert not any(n.startswith(('torch_npu','deep_ep','memcache_hybrid','memfabric_hybrid')) for n in sys.modules)
result={'project_id':'glm53','pass':True,'qualification':'CPU_VIRTUAL_ROUTING_AND_COLLECTIVE_DIFFERENTIAL_ONLY_NATIVE_NPU_CORRECTNESS_PENDING','torch_version':torch.__version__,'model_requests':0,'SDK_init_calls':0,'NPU_devices':False,'checks':checks,'cases_32rank':14,'rank_quant_cases':448,'roundtrips':28,'raw_FP32_bits_include_NaNs_signalingNaNs_signedzero_inf_subnormals':True,'layout_and_combine_identity_unchanged':True,'routing_inputs_kwargs_and_outputs_bitwise_equal':True,'patch':patch,'files_sha256':EXPECTED}
(ROOT/'fixture-result.json').write_text(json.dumps(result,indent=2)+'\n');print('GLM53_PACKED_DISPATCH_CPU_FIXTURE='+json.dumps(result),flush=True)
