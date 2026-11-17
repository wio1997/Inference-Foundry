"""Actual NPU kernel vs independent FP64 CPU oracle; no complete model/PD acceptance."""
from pathlib import Path
import datetime,hashlib,json,math,os,traceback
import torch,torch_npu
rank=int(os.environ['LOCAL_RANK']);os.sched_setaffinity(0,{168+rank});torch.set_num_threads(1);torch.npu.set_device(rank)
OUT=Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
record={'rank':rank,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'phase':'initializing','cases':[],'scope':'Sparse attention kernel vs independent FP64 oracle with causal candidate rows, finite sink and changed-input graph; not compressor/indexer selection, full model, PD or performance acceptance'}
def save():
 p=OUT/f'rank{rank}.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(record,indent=2));q.replace(p)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def oracle(q,ori,cmp,oi,ci,sink):
 outputs=[]
 for row in range(q.shape[0]):
  kv=torch.cat([ori[oi[row][oi[row]>=0].long()],cmp[ci[row][ci[row]>=0].long()]],0).double()
  logits=q[row].double()@kv.T/math.sqrt(512)
  full=torch.cat([logits,sink.double()[:,None]],1)
  weights=torch.softmax(full,dim=1)[:,:kv.shape[0]]
  outputs.append(weights@kv)
 return torch.stack(outputs).float()
def check(actual,expected):
 actual=actual.float().cpu();error=(actual-expected).abs()
 torch.testing.assert_close(actual,expected,rtol=.02,atol=.003)
 return {'max_abs_error':float(error.max()),'rms_error':float(error.square().mean().sqrt()),'all_finite':bool(torch.isfinite(actual).all())}
try:
 save()
 from lowratio_sparse_attention import sparse_shared_kv
 record.update(device=torch.npu.get_device_name(rank));save()
 generator=torch.Generator().manual_seed(114)
 ori_cpu=(torch.randn((2048,512),generator=generator)*.2).bfloat16()
 cmp_cpu=(torch.randn((2048,512),generator=generator)*.2).bfloat16()
 ori=ori_cpu.to(f'npu:{rank}');cmp=cmp_cpu.to(f'npu:{rank}')
 # Multiple requests and odd/even pair, window/page/tail boundaries; full512 selection at777.
 cases=[('decode',[127,777]),('verify',[126,127,128,129,130,131]*2),('prefill',[126,127,128,129,254,255,256,511,512,777]*2)]
 for ratio in [1,2]:
  for mode,positions in cases:
   t=len(positions);oi_cpu=torch.full((t,128),-1,dtype=torch.int32);ci_cpu=torch.full((t,512),-1,dtype=torch.int32)
   for row,pos in enumerate(positions):
    request=row%2;offset=request*1024
    legal=torch.arange(max(0,pos-127),pos+1,dtype=torch.int32)+offset;oi_cpu[row,:len(legal)]=legal
    n=(pos+1)//ratio;candidates=torch.arange(n,dtype=torch.int32)
    order=torch.randperm(n,generator=generator)[:512];chosen=candidates[order]+offset;ci_cpu[row,:len(chosen)]=chosen
    assert all(int(x)-offset<=pos for x in oi_cpu[row] if x>=0)
    assert all((int(x)-offset+1)*ratio<=pos+1 for x in ci_cpu[row] if x>=0)
   q_cpu=(torch.randn((t,8,512),generator=generator)*.2).bfloat16();sink_cpu=torch.linspace(-2.,3.,8)
   q=q_cpu.to(f'npu:{rank}');oi=oi_cpu.to(q.device);ci=ci_cpu.to(q.device);sink=sink_cpu.to(q.device)
   c={'ratio':ratio,'mode':mode,'rows':t,'positions':positions,'phase':'compile_and_warmup'};record['cases'].append(c);save()
   for _ in range(2):out=sparse_shared_kv(q,ori,cmp,oi,ci,sink);torch.npu.synchronize()
   expected=oracle(q_cpu,ori_cpu,cmp_cpu,oi_cpu,ci_cpu,sink_cpu);c['eager']=check(out,expected);c['phase']='graph';save()
   graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph,pool=torch.npu.graph_pool_handle(),auto_dispatch_capture=True):captured=sparse_shared_kv(q,ori,cmp,oi,ci,sink)
   hashes=[];replays=[]
   for replay in range(3):
    changed_q=(torch.randn(q_cpu.shape,generator=generator)*(.15+.1*replay)).bfloat16()
    changed_ori=(ori_cpu.float()+.03125*(replay+1)).bfloat16();changed_cmp=(cmp_cpu.float()-.015625*(replay+1)).bfloat16()
    changed_sink=sink_cpu+.25*(replay+1)
    # Change selected candidate IDs as well: drop a legal candidate, preserving mask padding semantics.
    changed_ci=ci_cpu.clone();changed_ci[:,replay]=-1
    q.copy_(changed_q);ori.copy_(changed_ori);cmp.copy_(changed_cmp);sink.copy_(changed_sink);ci.copy_(changed_ci)
    graph.replay();torch.npu.synchronize();expected=oracle(changed_q,changed_ori,changed_cmp,oi_cpu,changed_ci,changed_sink)
    replays.append(check(captured,expected));hashes.append(sha(captured))
   assert len(set(hashes))==3
   c.update(phase='passed',graph_captures=1,changed_input_replays=3,replay_errors=replays,hashes=hashes);save()
   del graph,q,oi,ci,sink,out,captured
   # Next case uses its original fixture after the changed-input replays.
   ori.copy_(ori_cpu);cmp.copy_(cmp_cpu);torch.npu.synchronize()
 record['phase']='passed';save()
except BaseException as exc:
 record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
