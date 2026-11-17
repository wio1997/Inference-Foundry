from pathlib import Path
import datetime,hashlib,json,os,traceback
import torch,torch_npu
torch.set_grad_enabled(False)
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],
 'scope':'Native fixed-image FP4, RoPE, compressor prerequisite; independent devices, not full checkpoint/PD/model performance'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
save()
try:
 from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4,fake_quant_compressed_kv
 from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail,DeepseekV41Compressor
 import sglang.srt.layers.attention.dsv4.dsv41_sparse as native
 record['native_source_sha256']=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest();save()
 gen=torch.Generator().manual_seed(166)
 def exercise(name,build):
  c={'name':name,'phase':'running'};record['cases'].append(c);save()
  try:
   x,fn,ref=build();y=fn(x);torch.npu.synchronize();expected=ref(x.cpu())
   torch.testing.assert_close(y.cpu(),expected,rtol=.02,atol=.003)
   c['max_abs_error']=float((y.cpu().float()-expected.float()).abs().max())
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=fn(x)
   hashes=[]
   for replay in range(3):
    v=(x.cpu().float()+.03125*(replay+1)).to(x.dtype);x.copy_(v);g.replay();torch.npu.synchronize()
    e=ref(v);torch.testing.assert_close(captured.cpu(),e,rtol=.02,atol=.003);hashes.append(sha(captured))
   assert len(set(hashes))==3
   c.update(phase='passed',graph_captures=1,changed_input_replays=3,hashes=hashes)
  except Exception as exc:c.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
  save()
 for rows in [16,96,8192]:
  for label,fn in [('index_FP4',fake_quant_fp4),('compressed_KV_FP4',fake_quant_compressed_kv)]:
   def build(rows=rows,fn=fn):
    x=(torch.randn((rows,512),generator=gen)*.2).bfloat16().to(f'npu:{rank}');return x,fn,fn
   exercise(f'{label}_{rows}',build)
 def rope_build():
  x=(torch.randn((96,8,512),generator=gen)*.2).bfloat16();theta=torch.randn((96,32),generator=gen);freq=torch.polar(torch.ones_like(theta),theta)
  freq_npu=freq.to(f'npu:{rank}')
  return x.to(f'npu:{rank}'),lambda v:rope_tail(v,freq_npu,64),lambda v:rope_tail(v,freq,64)
 exercise('native_complex_RoPE',rope_build)
 for ratio in [1,2]:
  for rows in [16,96,8192]:
   def build(ratio=ratio,rows=rows):
    cpu=DeepseekV41Compressor(5120,512,ratio,1e-6,fused_compress=False).eval()
    for p in cpu.parameters():
     with torch.no_grad():p.copy_((torch.randn(p.shape,generator=gen)*.01).to(p.dtype))
    import copy
    npu=copy.deepcopy(cpu).to(f'npu:{rank}')
    x=(torch.randn((rows,5120),generator=gen)*.2).bfloat16()
    def call(model,v):
     kv,score=model.project(v)
     if ratio==2:kv=model.pool_pairs(kv.reshape(-1,2,512),score.reshape(-1,2,512))
     return model.finish(kv)
    return x.to(f'npu:{rank}'),lambda v:call(npu,v),lambda v:call(cpu,v)
   exercise(f'native_C{ratio}_compress_{rows}',build)
 record['phase']='passed' if all(c['phase']=='passed' for c in record['cases']) else 'native_capability_gaps';save()
except BaseException as exc:
 record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
