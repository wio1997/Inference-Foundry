"""Actual fixed-image fused sqrtsoftplus384/top6 gate: FP64 CPU oracle + NPU graphs."""
from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False);torch.npu.set_device(rank)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
r={'rank':rank,'pid':os.getpid(),'device':torch.npu.get_device_name(rank),'phase':'initializing','cases':[],'scope':'Actual fixed-image native Triton fused sqrtsoftplus gate; 384 experts/top6, FP32 logits/bias, no group/sharedslots, FP64 CPU oracle, changed-input NPUgraphs. No wholemodel, checkpoint, PD orperformance acceptance.'}
def save():
 p=OUT/f'rank{rank}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def sha(vals):return hashlib.sha256(b''.join(v.contiguous().cpu().numpy().tobytes() for v in vals)).hexdigest()
def ref(x,b):
 a=torch.where(x.double()>20,x.double(),torch.log1p(torch.exp(x.double()))).sqrt();ids=torch.argsort(a+b.double(),dim=-1,descending=True,stable=True)[:,:6];w=a.gather(1,ids);return w/w.sum(-1,keepdim=True),ids.int()
save()
try:
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 import sglang
 package=Path(sglang.__file__).resolve().parent.parent
 r['source_SHA256']=json.loads((ROOT/'native_source_sha256.json').read_text())
 for n,h in r['source_SHA256'].items():assert hashlib.sha256((package/n).read_bytes()).hexdigest()==h,n
 from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
 r['native_source_pins_verified']=True;save()
 for rows in [0,1,16,96,8192]:
  for mode in ['structured','extreme','ties']:
   e=torch.arange(384).float();row=torch.arange(rows).float().view(-1,1)
   x=((e.view(1,-1)+row*7)%384)/32-6
   bias=(e%11)*.001
   if mode=='extreme':x=x*(80/6)
   elif mode=='ties':x=torch.zeros((rows,384));bias=torch.zeros(384)
   xn=x.to(f'npu:{rank}');bn=bias.to(f'npu:{rank}')
   case={'rows':rows,'mode':mode,'phase':'running'};r['cases'].append(case);save()
   def invoke():return moe_fused_gate(xn,bn,6,scoring_func='sqrtsoftplus',renormalize=True,routed_scaling_factor=1.5,apply_routed_scaling_factor_on_output=False,sqrtsoftplus_log1p=True)
   errors=[]
   def check(vals,xx,bb):
    w,ids=(v.cpu() for v in vals);ew,ei=ref(xx,bb);assert torch.equal(ids,ei),(mode,rows,'expert selection differs');assert torch.isfinite(w).all();torch.testing.assert_close(w.double(),ew,rtol=5e-5,atol=1e-6);errors.append(float((w.double()-ew).abs().max()) if w.numel() else 0.)
   out=invoke();torch.npu.synchronize();check(out,x,bias)
   if rows==0:case.update(phase='passed',graphs=0,changed_replays=0,max_abs_error=0);save();continue
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=invoke()
   hashes=[]
   for step in range(3):
    xx=torch.roll(x,17*(step+1),dims=1)+.125*(step+1);bb=torch.roll(bias,17*(step+1));bb=bb+(((e.int()//6)==step+2).float()*.5);xn.copy_(xx);bn.copy_(bb);g.replay();torch.npu.synchronize();check(captured,xx,bb);hashes.append(sha(captured))
   assert len(set(hashes))==3
   case.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,max_abs_error=max(errors),FP64_exact_expert_IDs=True);save()
 r['phase']='passed';save()
except BaseException:
 r['phase']='failed';r['traceback']=traceback.format_exc();save();raise
