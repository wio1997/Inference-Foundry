"""Actual fixed-image fused sqrtsoftplus384/top6 gate: FP64 CPU oracle + NPU graphs."""
from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False);torch.npu.set_device(rank)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
r={'rank':rank,'pid':os.getpid(),'device':torch.npu.get_device_name(rank),'phase':'initializing','cases':[],'scope':'Fixed-image native fused sqrtsoftplus router with Ascend log1p/exp/sqrt lowering; target384/top6 and draft128/top3, shared0/1, outputscaleoff/on, FP32 logits/bias vs FP64 oracle and changed NPUgraphs. No wholemodel/checkpoint/PD/performance acceptance.'}
def save():
 p=OUT/f'rank{rank}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def sha(vals):return hashlib.sha256(b''.join(v.contiguous().cpu().numpy().tobytes() for v in vals)).hexdigest()
def ref(x,b,k,shared,apply):
 a=torch.where(x.double()>20,x.double(),torch.log1p(torch.exp(x.double()))).sqrt();ids=torch.argsort(a.float()+b.float(),dim=-1,descending=True,stable=True)[:,:k];w=a.gather(1,ids);w=w/w.sum(-1,keepdim=True)
 if apply:w=w*1.5
 if shared:
  w=torch.cat([w,torch.full((x.shape[0],shared),1. if apply else 1/1.5,dtype=torch.float64)],-1);ids=torch.cat([ids,torch.arange(x.shape[1],x.shape[1]+shared).expand(x.shape[0],-1)],-1)
 return w,ids.int()
save()
try:
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 import sglang
 package=Path(sglang.__file__).resolve().parent.parent
 r['source_SHA256']=json.loads((ROOT/'native_source_sha256.json').read_text())
 for n,h in r['source_SHA256'].items():assert hashlib.sha256((package/n).read_bytes()).hexdigest()==h,n
 from route_compat import get_gate
 moe_fused_gate=get_gate(ROOT)
 r['native_source_pins_verified']=True;save()
 for n,k,shared,apply in [(n,k,shared,apply) for n,k in [(384,6),(128,3)] for shared in [0,1] for apply in [False,True]]:
  for rows in [0,1,16,96,8192]:
   for mode in ['structured','extreme','ties']:
    e=torch.arange(n).float();row=torch.arange(rows).float().view(-1,1)
    x=((e.view(1,-1)+row*7)%n)/32-6
    bias=(e%11)*.001
    if mode=='extreme':x=x*(80/6)
    elif mode=='ties':x=torch.zeros((rows,n));bias=torch.zeros(n)
    xn=x.to(f'npu:{rank}');bn=bias.to(f'npu:{rank}')
    case={'experts':n,'routed_topk':k,'shared_slots':shared,'apply_scale':apply,'rows':rows,'mode':mode,'phase':'running'};r['cases'].append(case);save()
    def invoke():return moe_fused_gate(xn,bn,k+shared,num_fused_shared_experts=shared,scoring_func='sqrtsoftplus',renormalize=True,routed_scaling_factor=1.5,apply_routed_scaling_factor_on_output=apply,sqrtsoftplus_log1p=True)
    errors=[]
    def check(vals,xx,bb):
     w,ids=(v.cpu() for v in vals);ew,ei=ref(xx,bb,k,shared,apply);assert torch.equal(ids,ei),(mode,rows,'expert selection differs');assert torch.isfinite(w).all();torch.testing.assert_close(w.double(),ew,rtol=5e-5,atol=1e-6);errors.append(float((w.double()-ew).abs().max()) if w.numel() else 0.)
    out=invoke();torch.npu.synchronize();check(out,x,bias)
    if rows==0:case.update(phase='passed',graphs=0,changed_replays=0,max_abs_error=0);save();continue
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=invoke()
    hashes=[]
    for step in range(3):
     xx=torch.roll(x,17*(step+1),dims=1)+.125*(step+1);bb=torch.roll(bias,17*(step+1));bb=bb+(((e.int()//6)==step+2).float()*.5);xn.copy_(xx);bn.copy_(bb);g.replay();torch.npu.synchronize();check(captured,xx,bb);hashes.append(sha(captured))
    assert len(set(hashes))==3
    case.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,max_abs_error=max(errors),CPU_FP32_exact_expert_IDs=True);save()
 r['phase']='passed';save()
except BaseException:
 r['phase']='failed';r['traceback']=traceback.format_exc();save();raise
