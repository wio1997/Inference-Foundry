"""Actual fixed-image mHC mix/combine/post interfaces; no model weights or PD."""
from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.set_grad_enabled(False);torch.npu.set_device(rank)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'scope':'Real native mHC mix_stats/combine/post with H=5120/HC4, fixed image, CPUFP64 oracle and changed-input graphs. H5120 post uses existing native Triton split-hidden port; originalCANNdimfailure retained. No full model/checkpoint or PD/performance acceptance.'}
def save():
 p=OUT/f'rank{rank}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2));t.replace(p)
def sha(values):return hashlib.sha256(b''.join(x.contiguous().view(torch.uint8).cpu().numpy().tobytes() for x in values)).hexdigest()
def stats_ref(x,w,sc,ba):
 f=x.double().flatten(1);m=torch.nn.functional.linear(f,w.double())*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6);pre=torch.sigmoid(m[:,:4]*sc[0].double()+ba[:4].double())+1e-6;post=2*torch.sigmoid(m[:,4:8]*sc[1].double()+ba[4:8].double());v=m[:,8:].reshape(-1,4,4)*sc[2].double()+ba[8:].double().reshape(4,4);v=v.softmax(-1)+1e-6;v=v/(v.sum(-2,keepdim=True)+1e-6)
 for _ in range(19):v=v/(v.sum(-1,keepdim=True)+1e-6);v=v/(v.sum(-2,keepdim=True)+1e-6)
 return tuple(y.float() for y in (pre,post,v))
save()
try:
 from sglang.srt.hardware_backend.npu.utils import init_npu_backend
 init_npu_backend()
 import sglang.srt.models.deepseek_v4_mhc as model
 import sglang.kernels.ops.layernorm.mhc as native
 from mhc_install import install
 install(ROOT)
 from post_install import install as install_post
 install_post(ROOT)
 record['sources']={k:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for k,m in [('model_mhc',model),('kernel_mhc',native)]};record['native_post_schema']=str(torch.ops.custom.npu_hc_post._schemas);save()
 cfg=model.HcConfig(mult=4,sinkhorn_iters=20,eps=1e-6,rms_eps=1e-6,hidden=5120,pre_from_prev=False,cp_prefill=False)
 gen=torch.Generator().manual_seed(361);sc=torch.tensor([.4,.6,.8]);ba=torch.randn(24,generator=gen)*.1;sn=sc.to(f'npu:{rank}');bn=ba.to(f'npu:{rank}')
 for rows in [1,16,96,8192]:
  x=(torch.randn((rows,4,5120),generator=gen)*.2).bfloat16();y=(torch.randn((rows,5120),generator=gen)*.2).bfloat16();pre=torch.sigmoid(torch.randn((rows,4),generator=gen));post=2*pre;comb=torch.softmax(torch.randn((rows,4,4),generator=gen),-1)
  xn=x.to(f'npu:{rank}');yn=y.to(f'npu:{rank}');pn=pre.to(f'npu:{rank}');pon=post.to(f'npu:{rank}');cn=comb.to(f'npu:{rank}')
  for kind in ['combine','post']:
   c={'kind':kind,'rows':rows,'phase':'running'};record['cases'].append(c);save()
   # Each interface starts from its declared CPU inputs, not the previous
   # interface's final replay mutation. G361 original fixture failure retained.
   xn.copy_(x);yn.copy_(y);pn.copy_(pre);pon.copy_(post);cn.copy_(comb)
   def invoke():return (native.mhc_combine(xn.flatten(1),pn,4,torch.bfloat16),) if kind=='combine' else (model.post(cfg,yn,xn,pon,cn),)
   def expected(vx,vy,vp,vpo,vc):
    if kind=='combine':return ((vx.double()*vp.double().unsqueeze(-1)).sum(1).bfloat16(),)
    return ((vy.double().unsqueeze(1)*vpo.double().unsqueeze(-1)+torch.einsum('tji,tjh->tih',vc.double(),vx.double())).bfloat16(),)
   error=[]
   def check(out,args):
    for o,e in zip(out,expected(*args)):
     o=o.cpu();assert torch.isfinite(o).all();torch.testing.assert_close(o,e,rtol=.008,atol=.003);error.append(float((o.float()-e.float()).abs().max()))
   out=invoke();torch.npu.synchronize();check(out,(x,y,pre,post,comb));g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=invoke()
   hashes=[]
   for step in range(3):
    vx=(x.float()+.03125*(step+1)).bfloat16();vy=(y.float()+.0625*(step+1)).bfloat16();vp=pre+.007*(step+1);vpo=post+.009*(step+1);vc=comb+.003*(step+1);xn.copy_(vx);yn.copy_(vy);pn.copy_(vp);pon.copy_(vpo);cn.copy_(vc);g.replay();torch.npu.synchronize();check(captured,(vx,vy,vp,vpo,vc));hashes.append(sha(captured))
   assert len(set(hashes))==3;c.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,max_abs_error=max(error));save()
  if rows<=96:
   c={'kind':'actual_model_mix_stats','rows':rows,'phase':'running'};record['cases'].append(c);save();weight=(torch.randn((24,20480),generator=gen)*.005).float();wn=weight.to(f'npu:{rank}');hc=model.HcSubLayer(cfg,wn,sn,bn,None)
   def invoke_stats():return tuple(model._mix_stats_impl(hc,xn))
   def check_stats(vals,xx,ss,bb):
    err=[]
    for v,e in zip(vals,stats_ref(xx,weight,ss,bb)):
     v=v.cpu();torch.testing.assert_close(v,e,rtol=1e-4,atol=1e-5);err.append(float((v-e).abs().max()))
    return max(err)
   # Reset the residual after prior interface graph replays.
   xn.copy_(x);errors=[check_stats(invoke_stats(),x,sc,ba)];g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=invoke_stats()
   hashes=[]
   for step in range(3):
    vx=(x.float()+.03125*(step+1)).bfloat16();ss=sc+.007*(step+1);bb=ba+.023*(step+1);xn.copy_(vx);sn.copy_(ss);bn.copy_(bb);g.replay();torch.npu.synchronize();errors.append(check_stats(captured,vx,ss,bb));hashes.append(sha(captured))
   assert len(set(hashes))==3;c.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,max_abs_error=max(errors));save();sn.copy_(sc);bn.copy_(ba)
 record['phase']='passed';save()
except BaseException as e:record.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
