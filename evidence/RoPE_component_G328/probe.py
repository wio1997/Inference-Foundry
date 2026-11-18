from pathlib import Path
import datetime,hashlib,json,os,traceback
import torch,torch_npu
from rope_compat import apply_rotary_mul_inplace
import sglang.srt.layers.attention.dsv4.dsv41_sparse as native
torch.set_grad_enabled(False);rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'native_rope_source_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),'scope':'Native FP32 complex tail rotation with actual BF16 cos/sin cache coefficient contract; component only, not custom-op bit identity, fullmodel, PD, quality or performance'}
def save():
 p=OUT/f'rank{rank}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2));t.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
gen=torch.Generator().manual_seed(328)
def reference(x,c,s):
 y=x.clone();v=x[...,448:].double().reshape(*x.shape[:-1],32,2);cc=c[:,0,0,::2].double().view(-1,1,32);ss=s[:,0,0,::2].double().view(-1,1,32);y[...,448:]=torch.stack((v[...,0]*cc-v[...,1]*ss,v[...,1]*cc+v[...,0]*ss),-1).flatten(-2).to(x.dtype);return y
save()
try:
 for rows in [1,16,96,8192]:
  for inverse in [False,True]:
   case={'rows':rows,'inverse':inverse,'phase':'running'};record['cases'].append(case);save()
   # Strided Q plus K alias mirrors an in-place partial-tail interface.
   cpu=(torch.randn((rows,8,520),generator=gen)*.2).bfloat16();qbase=cpu.to(f'npu:{rank}');q=qbase[...,:512]
   kcpu=(torch.randn((rows,1,512),generator=gen)*.2).bfloat16();k=kcpu.to(f'npu:{rank}')
   theta=torch.randn((rows,32),generator=gen);c=(1.1*theta.cos()).repeat_interleave(2,-1).bfloat16().view(rows,1,1,64);s=(1.1*theta.sin()).repeat_interleave(2,-1).bfloat16().view(rows,1,1,64)
   if inverse:s=-s
   cn=c.to(f'npu:{rank}');sn=s.to(f'npu:{rank}')
   apply_rotary_mul_inplace(q,k,cn,sn,448);torch.npu.synchronize();max_error=0.
   def check(vq,vk,cc,ss):
    global max_error
    for actual,source in [(q,vq),(k,vk)]:
     got=actual.cpu();expected=reference(source,cc,ss);assert torch.equal(got[...,:448],source[...,:448]);torch.testing.assert_close(got[...,448:],expected[...,448:],rtol=.008,atol=.003);max_error=max(max_error,float((got.float()-expected.float()).abs().max()))
   check(cpu[...,:512],kcpu,c,s);assert torch.equal(qbase.cpu()[...,512:],cpu[...,512:])
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):apply_rotary_mul_inplace(q,k,cn,sn,448)
   hashes=[]
   for step in range(3):
    vq=(cpu[...,:512].float()+.03125*(step+1)).bfloat16();vk=(kcpu.float()+.0625*(step+1)).bfloat16();cs=c.roll(step+1,0);ss=s.roll(step+1,0);q.copy_(vq);k.copy_(vk);cn.copy_(cs);sn.copy_(ss);g.replay();torch.npu.synchronize();check(vq,vk,cs,ss);hashes.append(sha(q))
   assert len(set(hashes))==3 and torch.equal(qbase.cpu()[...,512:],cpu[...,512:]);case.update(phase='passed',graphs=1,changed_replays=3,max_abs_error=max_error,changed_hashes=hashes);save()
 record['phase']='passed';save()
except BaseException as e:record.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
