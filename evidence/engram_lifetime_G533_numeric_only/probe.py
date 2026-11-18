"""Fixed H5120/HC4 Engram eager equivalence and tensor-liveness probe.

No model loading, serving, network transfer, or end-to-end performance claim.
"""
from pathlib import Path
import ast,datetime,hashlib,json,os,traceback
import torch,torch_npu
from safetensors import safe_open

rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank)
os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],
 'scope':'P-eager actual H5120 HC4 fixed-gate equivalence and allocation liveness only; no full model, cross-host PD, graph replay, accuracy suite or E2E gain acceptance.'}

def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2)+'\n');q.replace(p)

try:
 save();mechanism=json.loads((ROOT/'mechanism.json').read_text())
 trees={}
 for name in ['original','candidate']:
  p=ROOT/(name+'_gate.py');t=ast.parse(p.read_text());fn=next(x for x in t.body if isinstance(x,ast.FunctionDef))
  assert hashlib.sha256((ast.unparse(fn)+'\n').encode()).hexdigest()==mechanism[name+'_function_ast_sha256']
  trees[name]=t
 candidate=ast.parse((ROOT/'candidate_gate.py').read_text())
 fn=next(x for x in candidate.body if isinstance(x,ast.FunctionDef));deleted=[x for x in fn.body if isinstance(x,ast.Delete)]
 assert len(deleted)==1 and [v.id for v in deleted[0].targets]==['key','original']
 fn.body=[x for x in fn.body if not isinstance(x,ast.Delete)]
 assert ast.dump(candidate,include_attributes=False)==ast.dump(trees['original'],include_attributes=False),'Only dead-reference deletion is allowed'
 record['arithmetic_AST_identical_except_two_dead_reference_deletions']=True
 snapshots={}
 def observe(name):snapshots[name]=int(torch.npu.memory_allocated())
 functions={}
 for name,t in trees.items():
  fn=next(x for x in t.body if isinstance(x,ast.FunctionDef))
  where=next(i for i,v in enumerate(fn.body) if isinstance(v,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='out' for x in v.targets))
  fn.body.insert(where,ast.parse('observe('+repr(name)+')').body[0]);ast.fix_missing_locations(t)
  source=ast.unparse(t)+'\n';path=ROOT/f'{name}_instrumented_gate.pid{os.getpid()}.py';path.write_text(source)
  context={'torch':torch,'observe':observe};exec(compile(t,str(path),'exec'),context);functions[name]=context['rotated_gate']
 with safe_open('/models/ds41/optional/quarot.safetensors',framework='pt',device='cpu') as f:
  matrix=f.get_tensor('global_rotation')
 assert tuple(matrix.shape)==(5120,5120)
 block=matrix[:32,:32].contiguous()
 assert torch.equal(matrix,torch.block_diag(*[block]*160))
 rotation=block.float().to(f'npu:{rank}');del matrix,block
 configurations=[(1,'normal'),(16,'image_mask'),(96,'kv_float32'),(127,'signed_zero'),(512,'weight_bf16'),(513,'strided'),(8191,'large'),(8193,'large')]
 for case,(n,kind) in enumerate(configurations):
  gen=torch.Generator().manual_seed(53300+100*rank+case)
  hc,dim=4,5120;stride=2 if kind=='strided' else 1
  x=torch.randn(n,hc,dim*stride,generator=gen).bfloat16().to(f'npu:{rank}')[...,::stride]
  kv=torch.randn(n,(hc+1)*dim*stride,generator=gen).bfloat16().to(f'npu:{rank}')[...,::stride]
  if kind=='kv_float32':kv=kv.float()
  q=torch.randn(hc,dim,generator=gen).to(f'npu:{rank}');k=torch.randn(hc,dim,generator=gen).to(f'npu:{rank}')
  if kind=='signed_zero':q.fill_(-0.0);k.fill_(1)
  if kind=='weight_bf16':q=q.bfloat16();k=k.bfloat16()
  ids=torch.arange(n,device=f'npu:{rank}')%7 if kind=='image_mask' else None
  image_id=3 if ids is not None else None
  args=(x,kv,q,k,1e-6,1e-6,rotation,ids,image_id)
  x_initial=x.clone();kv_initial=kv.clone()
  # Warm both implementations before collecting PyTorch allocation counters.
  for name in ['original','candidate']:
   tmp=functions[name](*args);torch.npu.synchronize();del tmp
  measured={};outputs={}
  for name in ['original','candidate']:
   torch.npu.synchronize();start=int(torch.npu.memory_allocated());torch.npu.reset_peak_memory_stats()
   outputs[name]=functions[name](*args);torch.npu.synchronize()
   measured[name]={'start_allocated_bytes':start,'peak_delta_bytes':int(torch.npu.max_memory_allocated())-start,'pre_output_delta_bytes':snapshots[name]-start}
  assert torch.equal(outputs['original'],outputs['candidate']),'BF16 output must remain bitwise identical; no relaxed tolerance'
  assert torch.equal(x,x_initial) and torch.equal(kv,kv_initial),'Inputs must remain unchanged'
  if ids is not None:
   assert torch.equal(outputs['candidate'][ids==image_id],x[ids==image_id])
  element_bytes=n*hc*dim*4
  if kind=='large':
   peak_reduction=measured['original']['peak_delta_bytes']-measured['candidate']['peak_delta_bytes']
   live_reduction=measured['original']['pre_output_delta_bytes']-measured['candidate']['pre_output_delta_bytes']
   assert live_reduction>=1.8*element_bytes,(n,'Expected two fullFP32 dead buffers released',measured)
   assert peak_reduction>=.8*element_bytes,(n,'Expected at least one fullFP32 buffer in peak reduction',measured)
  record['cases'].append({'rows':n,'kind':kind,'HC':hc,'hidden':dim,'input_dtype':str(x.dtype),'KV_dtype':str(kv.dtype),'bitwise_equal':True,'inputs_unchanged':True,'allocation':measured,'numeric_tolerance':'Exact BF16 equality; arithmetic AST unchanged','scope':'Actual NPU eager gate only; memory counter is PyTorch allocation, not total HBM or end-to-end performance'})
  save();del outputs,x_initial,kv_initial,x,kv,q,k,args,ids
  torch.npu.synchronize()
 record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except BaseException as exc:
 record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
