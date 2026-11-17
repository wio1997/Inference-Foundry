from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
torch.set_grad_enabled(False)
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'scope':'Native per16 E4M3 scale and E2M1 FP4 compatibility; exact CPU native FP8 oracle, no fullmodel/PD/performance claim'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
try:
 save()
 import sglang.kernels.ops.quantization.fp4_fake_quant as native
 from fp4_compat import positive_e4m3_scale,fake_quant_compressed_kv
 source_sha=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest();assert source_sha=='72770fc6635768f49d8c7dc53a053829a6a5e5acead41e555d22cbd385a4cdef'
 record['native_source_sha256']=source_sha
 # Every finite positive code, its adjacent midpoint and both immediate FP32 neighbors.
 values=torch.arange(127,dtype=torch.uint8).view(torch.float8_e4m3fn).float()
 mids=(values[:-1]+values[1:])*.5
 inputs=torch.cat([values,mids,torch.nextafter(mids,torch.zeros_like(mids)),torch.nextafter(mids,torch.full_like(mids,float('inf'))),torch.tensor([0.,2.0**-20,449.,1e6])]).clamp(min=2.0**-9,max=448.)
 expected=inputs.to(torch.float8_e4m3fn).float();actual=positive_e4m3_scale(inputs.to(f'npu:{rank}'));torch.npu.synchronize();assert torch.equal(actual.cpu(),expected)
 record['scale_boundary_cases']=inputs.numel();record['all_scale_boundaries_bitwise_equal']=True;save()
 gen=torch.Generator().manual_seed(170)
 for rows in [16,96,8192]:
  c={'rows':rows,'phase':'running'};record['cases'].append(c);save()
  base=torch.randn((rows,512),generator=gen)
  gains=torch.exp2(torch.randint(-15,11,(rows,32,1),generator=gen).float())
  base=(base.reshape(rows,32,16)*gains).reshape(rows,512).bfloat16();base[0]=0
  x=base.to(f'npu:{rank}');out=fake_quant_compressed_kv(x);torch.npu.synchronize();ref=native.fake_quant_compressed_kv(base);assert torch.equal(out.cpu(),ref)
  graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=fake_quant_compressed_kv(x)
  hashes=[]
  for replay in range(3):
   changed=(base.float()+(.03125*(replay+1))).bfloat16();x.copy_(changed);graph.replay();torch.npu.synchronize();expected=native.fake_quant_compressed_kv(changed)
   assert torch.equal(captured.cpu(),expected);hashes.append(sha(captured))
  assert len(set(hashes))==3
  c.update(phase='passed',all_BF16_outputs_bitwise_equal=True,graph_captures=1,changed_input_replays=3,hashes=hashes);save()
 record['phase']='passed';save()
except BaseException as exc:
 record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
