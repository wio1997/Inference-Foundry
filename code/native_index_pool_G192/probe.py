from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
torch.set_grad_enabled(False)
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'scope':'Native generic FP4 index pool allocation and readback with NPU RNE writer; not entire NPU model pool, candidate scoring, PD transfer or model performance'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
try:
 save()
 from fp4_index_pool import make_low_ratio_index_pool
 from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
 pool=make_low_ratio_index_pool(65536,64,torch.bfloat16,128,2,f'npu:{rank}',False)
 record.update(bytes_per_token=pool.get_bytes_per_token(),buffer_shapes=[list(x.shape) for x in pool.index_k_with_scale_buffer],native_pool_class=type(pool).__mro__[1].__name__);assert pool.get_bytes_per_token()==68;save()
 gen=torch.Generator().manual_seed(184)
 for rows in [16,96,8192]:
  c={'rows':rows,'phase':'running'};record['cases'].append(c);save()
  slots_cpu=torch.randperm(65535,generator=gen)[:rows].long()+1
  slots=slots_cpu.to(f'npu:{rank}')
  x_cpu=(torch.randn((rows,128),generator=gen)*.2).bfloat16();x=x_cpu.to(f'npu:{rank}')
  def call():
   pool.set_index_fp4(0,slots,x)
   return pool.get_index_k_dequant(0,slots)
  y=call();torch.npu.synchronize();assert torch.equal(y.cpu(),fake_quant_fp4(x_cpu))
  payload,scale=pool.get_index_k_fp4(0,slots);assert payload.shape==(rows,64) and scale.shape==(rows,)
  other_before=sha(pool.index_k_with_scale_buffer[1]);null_before=sha(pool.get_index_k_dequant(0,torch.zeros(1,dtype=torch.int64,device=x.device)))
  graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=call()
  hashes=[]
  for replay in range(3):
   value=(x_cpu.float()+.03125*(replay+1)).bfloat16();x.copy_(value)
   changed_slots=slots_cpu.roll(replay+1);slots.copy_(changed_slots)
   graph.replay();torch.npu.synchronize();assert torch.equal(captured.cpu(),fake_quant_fp4(value));hashes.append(sha(captured))
  assert len(set(hashes))==3 and other_before==sha(pool.index_k_with_scale_buffer[1])
  assert null_before==sha(pool.get_index_k_dequant(0,torch.zeros(1,dtype=torch.int64,device=x.device)))
  c.update(phase='passed',all_BF16_outputs_bitwise_equal_CPU_native=True,other_layer_unchanged=True,null_slot_unchanged=True,graph_captures=1,changed_input_replays=3,hashes=hashes);save()
 record['phase']='passed';save()
except BaseException as exc:record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
