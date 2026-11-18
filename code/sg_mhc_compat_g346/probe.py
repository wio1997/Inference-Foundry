from pathlib import Path
import hashlib,json,os,traceback
import torch,torch_npu
from mhc_compat import split_sinkhorn
import sglang.kernels.ops.layernorm.mhc as native
torch.set_grad_enabled(False);rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True);record={'rank':rank,'pid':os.getpid(),'phase':'initializing','cases':[],'native_mhc_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),'scope':'Reuse native FP32hc4/20iter split/Sinkhorn; component, not fullmodel/quality/PD/performance'}
def save():
 p=OUT/f'rank{rank}.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2));t.replace(p)
def sha(values):return hashlib.sha256(b''.join(x.contiguous().view(torch.uint8).cpu().numpy().tobytes() for x in values)).hexdigest()
def ref(x,scale,base,eps):
 x=x.double();scale=scale.double();base=base.double();pre=torch.sigmoid(x[...,:4]*scale[0]+base[:4])+eps;post=2*torch.sigmoid(x[...,4:8]*scale[1]+base[4:8]);v=x[...,8:].reshape(*x.shape[:2],4,4)*scale[2]+base[8:].reshape(4,4);v=v.softmax(-1)+eps;v=v/(v.sum(-2,keepdim=True)+eps)
 for _ in range(19):v=v/(v.sum(-1,keepdim=True)+eps);v=v/(v.sum(-2,keepdim=True)+eps)
 return pre.float(),post.float(),v.float()
gen=torch.Generator().manual_seed(346);save()
try:
 for rows in [0,1,16,96,8192]:
  for magnitude in [1.,40.]:
   c={'rows':rows,'magnitude':magnitude,'phase':'running'};record['cases'].append(c);save()
   b,s=(2,rows//2) if rows>=16 else (rows,1);full=(torch.randn((b,s,48),generator=gen)*magnitude).float();source=full[...,::2];scale=torch.tensor([.4,.6,.8]);base=torch.randn(24,generator=gen)*.1;xn=full.to(f'npu:{rank}')[...,::2];sn=scale.to(f'npu:{rank}');bn=base.to(f'npu:{rank}');got=split_sinkhorn(xn,sn,bn);torch.npu.synchronize();errors=[]
   def check(vals,x,sc,ba):
    expected=ref(x,sc,ba,1e-6)
    for v,e in zip(vals,expected):
     v=v.cpu();assert v.shape==e.shape;assert torch.isfinite(v).all();torch.testing.assert_close(v,e,rtol=3e-5,atol=3e-6);errors.append(float((v-e).abs().max()) if v.numel() else 0.)
   check(got,source,scale,base)
   if rows==0:
    c.update(phase='passed',graphs=0,changed_replays=0,hashes=[],max_abs_error=0.,scope='Idle zero-token eager shape gate; no empty standalone graph capture required');save();continue
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=split_sinkhorn(xn,sn,bn)
   hashes=[]
   for step in range(3):
    x=source+.03125*(step+1);sc=scale+.007*(step+1);ba=base+.023*(step+1);xn.copy_(x);sn.copy_(sc);bn.copy_(ba);g.replay();torch.npu.synchronize();check(captured,x,sc,ba);hashes.append(sha(captured))
   assert not rows or len(set(hashes))==3;c.update(phase='passed',graphs=1,changed_replays=3,hashes=hashes,max_abs_error=max(errors));save()
 record['phase']='passed';save()
except BaseException as e:record.update(phase='failed',error=repr(e),traceback=traceback.format_exc());save();raise
