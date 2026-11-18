"""Real-H5120/HC4 row-chunk gate, eager only, no model/request/graph."""
from pathlib import Path
import ast,datetime,hashlib,json,os,traceback
import torch,torch_npu
from safetensors import safe_open
rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False)
R=Path(__file__).resolve().parent;OUT=R/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'scope':'Fixed nativeEngram eager per-row tensor equivalence/peak; no fullmodel, PD,state,graph,qualitysuite orE2Egain claim'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2)+'\n');q.replace(p)
try:
 save();m=json.loads((R/'mechanism.json').read_text());assert hashlib.sha256((R/'reference_gate.py').read_bytes()).hexdigest()==m['source_reference_file_sha256'];assert hashlib.sha256((R/'candidate_gate.py').read_bytes()).hexdigest()==m['candidate_file_sha256']
 from reference_gate import rotated_gate as reference
 from candidate_gate import rotated_gate as candidate,CHUNK_ROWS
 assert CHUNK_ROWS==256
 record.update(architecture=os.uname().machine,torch_version=torch.__version__,torch_npu_version=getattr(torch_npu,'__version__','unknown'),source_adapter_sha256=m['reference_adapter_sha256'])
 rotation_file=Path('/models/ds41/optional/quarot.safetensors');record['quarot_sha256']=hashlib.sha256(rotation_file.read_bytes()).hexdigest()
 with safe_open(str(rotation_file),framework='pt',device='cpu') as f:matrix=f.get_tensor('global_rotation')
 assert tuple(matrix.shape)==(5120,5120);block=matrix[:32,:32].contiguous();assert torch.equal(matrix,torch.block_diag(*[block]*160));rotation=block.float().to(f'npu:{rank}');del matrix,block
 for case,(rows,kind) in enumerate(m['cases']):
  gen=torch.Generator().manual_seed(59700+100*rank+case);hc,dim=4,5120;stride=2 if kind=='strided' else 1;device=f'npu:{rank}'
  x=torch.randn(rows,hc,dim*stride,generator=gen).bfloat16().to(device)[...,::stride]
  kv=torch.randn(rows,(hc+1)*dim*stride,generator=gen).bfloat16().to(device)[...,::stride]
  if kind in ['kv_float32','large_kv_float32']:kv=kv.float()
  q=torch.randn(hc,dim,generator=gen).to(device);k=torch.randn(hc,dim,generator=gen).to(device)
  if kind=='signed_zero':q.fill_(-0.0);k.fill_(1)
  if kind=='weight_bf16':q=q.bfloat16();k=k.bfloat16()
  ids=torch.arange(rows,device=device)%7 if kind=='image_mask' else None;image_id=3 if ids is not None else None
  args=(x,kv,q,k,1e-6,1e-6,rotation,ids,image_id);before=[v.clone() for v in [x,kv,q,k]]
  for fun in [reference,candidate]:tmp=fun(*args);torch.npu.synchronize();del tmp
  outputs={};peaks={}
  for name,fun in [('reference',reference),('candidate',candidate)]:
   torch.npu.synchronize();start=int(torch.npu.memory_allocated());torch.npu.reset_peak_memory_stats();outputs[name]=fun(*args);torch.npu.synchronize();peaks[name]=int(torch.npu.max_memory_allocated())-start
  assert torch.equal(outputs['reference'].view(torch.int16),outputs['candidate'].view(torch.int16)),(rows,kind,'Exact BF16 integer bit pattern gate failed; preserve without tolerance changes')
  assert all(torch.equal(a,b) for a,b in zip(before,[x,kv,q,k])),'Input/weight mutation'
  if ids is not None:assert torch.equal(outputs['candidate'][ids==image_id].view(torch.int16),x[ids==image_id].view(torch.int16))
  if rows in [8191,8193]:assert peaks['candidate']<768*1024**2 and peaks['reference']-peaks['candidate']>1024**3,(rows,kind,peaks)
  record['cases'].append({'rows':rows,'kind':kind,'HC':hc,'hidden':dim,'kv_dtype':str(kv.dtype),'literal_BF16_int16_bits_equal':True,'inputs_and_weights_unchanged':True,'chunk_rows':256,'actual_invocations':max(1,(rows+255)//256),'peak_delta_bytes':peaks,'numeric_tolerance':'Exact BF16 bits; no relaxation','scope':'Eager selectedtensors/PyTorchallocation; no E2E/modelacceptance'});save()
  del args,outputs,before,x,kv,q,k,ids;torch.npu.synchronize()
 record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except BaseException as exc:record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
