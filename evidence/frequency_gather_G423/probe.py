"""Actual NPU frequency bit-copy and original native RoPE/FP4 chain; no model load."""
from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False);torch.npu.set_device(rank)
R=Path(__file__).resolve().parent;O=R/'evidence';O.mkdir(exist_ok=True);rec={'rank':rank,'device':torch.npu.get_device_name(rank),'phase':'initializing','cases':[],'scope':'FP32-view gather of complex64 frequencies, exact bit copy and changed input NPUgraphs, plus unchanged native RoPE/FP4 chain CPU comparison. No wholemodel, EP, PD, quality or performance acceptance.'}
def save():
 p=O/f'rank{rank}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(rec,indent=2));t.replace(p)
def bits(x):return torch.view_as_real(x).contiguous().view(torch.int32).cpu() if x.is_complex() else x.contiguous().view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32).cpu()
def sha(x):return hashlib.sha256(bits(x).numpy().tobytes()).hexdigest()
save()
try:
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 from freqs_install import install
 install(R)
 from freqs_compat import gather
 import sglang.srt.layers.attention.dsv4.dsv41_sparse as sp
 from fp4_compat import fake_quant_compressed_kv
 old_quant=sp.fake_quant_compressed_kv;sp.fake_quant_compressed_kv=lambda x:fake_quant_compressed_kv(x) if x.device.type=='npu' else old_quant(x)
 rec['source_pins_verified']=True;save()
 gen=torch.Generator().manual_seed(415);raw=torch.randn((263296,64,2),generator=gen);raw[0,0]=torch.tensor([0.,-0.]);nb=raw.to(f'npu:{rank}')
 for strided in [False,True]:
  f=torch.view_as_complex(raw)[:,::2] if strided else torch.view_as_complex(raw)[:,:32]
  fn=torch.view_as_complex(nb)[:,::2] if strided else torch.view_as_complex(nb)[:,:32]
  for rows in [0,1,16,96,8192]:
   for ndim in [1,2]:
    ids=(torch.arange(rows*(2 if ndim==2 else 1))*129+131071)%263296
    if ndim==2:ids=ids.reshape(rows,2)
    ids=ids.long();idx=ids.to(f'npu:{rank}');c={'kind':'gather','rows':rows,'index_dims':ndim,'strided':strided,'phase':'running'};rec['cases'].append(c);save()
    def invoke():return gather(fn,idx)
    out=invoke();torch.npu.synchronize();assert torch.equal(bits(out),bits(f[ids]))
    if rows==0:c.update(phase='passed',graphs=0,changed_replays=0,bitwise_exact=True);save();continue
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=invoke()
    hashes=[]
    for step in range(3):
     ii=(ids+7*(step+1))%263296;idx.copy_(ii);g.replay();torch.npu.synchronize();assert torch.equal(bits(captured),bits(f[ii]));hashes.append(sha(captured))
    assert len(set(hashes))==3;c.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,bitwise_exact=True);save()
 # Original source function includes the NPU compatibility gather; CPU branch
 # remains the original direct index and original CPU quantizer.
 angle=torch.arange(263296).float().view(-1,1)*torch.arange(1,33).float().view(1,-1)*.0001
 freq=torch.polar(torch.ones_like(angle),angle);nf=torch.view_as_complex(torch.view_as_real(freq).to(f'npu:{rank}'))
 for rows in [1,16,96,8192]:
  for width,compressed in [(128,False),(512,True)]:
   x=(torch.randn((rows,width),generator=gen)*.2).bfloat16();xn=x.to(f'npu:{rank}');ids=(torch.arange(rows)*257+1)%263296;idx=ids.to(f'npu:{rank}');c={'kind':'native_RoPE_FP4_chain','rows':rows,'width':width,'compressed':compressed,'phase':'running'};rec['cases'].append(c);save()
   def invoke_chain():return sp._rope_fq4(xn,nf,64,compressed_kv=compressed,positions=idx)
   def check(y,xx,ii):
    expected=sp._rope_fq4(xx,freq,64,compressed_kv=compressed,positions=ii);yy=y.cpu();torch.testing.assert_close(yy,expected,rtol=.02,atol=.003);return float((yy.float()-expected.float()).abs().max())
   errors=[check(invoke_chain(),x,ids)];g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=invoke_chain()
   hashes=[]
   for step in range(3):
    xx=(x.float()+.03125*(step+1)).bfloat16();ii=(ids+17*(step+1))%263296;xn.copy_(xx);idx.copy_(ii);g.replay();torch.npu.synchronize();errors.append(check(captured,xx,ii));hashes.append(sha(captured))
   assert len(set(hashes))==3;c.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,max_abs_error=max(errors),oracle='CPU original native RoPE/FP4 function; compressed scale bridge independently checked earlier, no independent wholemodel quality claim');save()
 rec['phase']='passed';save()
except BaseException:
 rec['phase']='failed';rec['traceback']=traceback.format_exc();save();raise
