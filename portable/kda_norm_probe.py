"""Four-die reference-shaped projection and chunk-state validation."""
import datetime,hashlib,json,os,pathlib,platform,sys,time
import torch,torch_npu
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
import deepinfer.model_executor.models.glm5_next.kda as kda
import deepinfer.model_executor.models.glm5_next.chunk_prefill as pref
from vllm_ascend.ops.kda import run_chunk_kda
rank,die=map(int,sys.argv[1:3]);assert 0<=rank<4 and die==12+rank
root=pathlib.Path(os.environ['GLM_EXPERIMENT_RUN_DIR']);out=root/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==str(die) and torch_npu.npu.device_count()==1
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);torch.ones(1,device='npu');torch_npu.npu.synchronize();(out/'ready').write_text(str(os.getpid()))
for _ in range(1800):
 if (out/'go').exists():break
 time.sleep(.1)
else:raise RuntimeError('host PID/device mapping not verified')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
w=pathlib.Path('/opt/tiankuan/yxy/models/GLM-5.3-Flash-w8a8');ref=root.parent/'round406-ref-batch-a1'/f'rank{rank}'
report={'rank':rank,'die':die,'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'probe_sha256':sha(__file__),'kda_sha256':sha(kda.__file__),'chunk_prefill_sha256':sha(pref.__file__),'index_sha256':sha(w/'quant_model_weights.safetensors.index.json'),'config_sha256':sha(w/'config.json'),'status':'running'}
def save():
 p=out/'result.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');q.replace(p)
def copy_state(s):return tuple(t.detach().clone() for t in s.conv),s.recurrent.detach().clone(),s.tokens_seen
def equal_state(s,c):return s.tokens_seen==c[2] and torch.equal(s.recurrent,c[1]) and all(torch.equal(a,b) for a,b in zip(s.conv,c[0],strict=True))
save()
try:
 assert report['index_sha256']=='dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4' and report['config_sha256']=='33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943'
 info=json.loads((ref/'result.json').read_text());cap=ref/'boundary-capture.pt';batch=ref/'projection-input-batch.pt';assert info['status']=='passed' and sha(cap)==info['capture_sha256'] and sha(batch)==info['batch_input_sha256'];reference=torch.load(cap,map_location='cpu',weights_only=True);x=torch.load(batch,map_location='cpu',weights_only=True);assert x.shape==(41,4096) and torch.equal(x[:1],reference['0:norm_attn']);x=x.to('npu')
 with torch.inference_mode():
  m=kda.Glm5NextKDA.from_checkpoint(ModelSlimCheckpoint(w,max_read_bytes=2*1024**3),0,'npu',backend='ascendc_chunk_prefill');h=m.num_heads;dim=m.head_dim;lo=rank*(h//4);hi=(rank+1)*(h//4);checks={};seen=[];metrics={}
  m.reference_sharded_projection=True
  calls=[];orig_chunk=pref.chunk_prefill_sharded
  def wrapped_chunk(q,k,v,g,b,state,alog,bias,*,shards,lower_bound):
   result=orig_chunk(q,k,v,g,b,state,alog,bias,shards=shards,lower_bound=lower_bound)
   calls.append({'operands':[z.detach().cpu().clone() for z in (q,k,v,g,b)],'core':result[0].detach().cpu().clone(),'state_before':state.detach().cpu().clone()})
   return result
  pref.chunk_prefill_sharded=wrapped_chunk
  # The previous round established equal raw first-eight operands. Capture
  # them once from the real 41-token invocation and test normalization alone.
  initial=m.initial_state();y,state=m(x,initial);torch_npu.npu.synchronize();assert len(calls)==1
  from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
  import vllm.third_party.flash_linear_attention.ops.l2norm as normmod
  report['l2norm_sha256']=sha(normmod.__file__)
  assert report['l2norm_sha256']=='520f4513ef4fc97e6b88dc07877cca8e29da0d34e44673fe989e3af002d86381'
  def stats(a,b):
   assert a.shape==b.shape and a.dtype==b.dtype
   d=a.float()-b.float();return {'bitwise_equal':torch.equal(a,b),'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item(),'different_elements':int(torch.count_nonzero(d).item())}
  metrics={}
  for name,t in zip(('q','k'),calls[0]['operands'][:2],strict=True):
   a=t.to('npu');width=h//4;small=[];large=[]
   for shard in range(4):
    part=a[:,shard*width:(shard+1)*width].contiguous();small.append(l2norm_fwd(part[:8].contiguous().unsqueeze(0)));large.append(l2norm_fwd(part.unsqueeze(0))[:,:8])
   small=torch.cat(small,dim=2);large=torch.cat(large,dim=2);torch_npu.npu.synchronize();metrics[name]={'first_eight':stats(small.cpu(),large.cpu()),'position0':stats(small[:,:1].cpu(),large[:,:1].cpu())}
  checks={'finite':bool(torch.isfinite(y).all().item()),'tokens_seen':state.tokens_seen==41,'zero_state':bool(torch.count_nonzero(calls[0]['state_before']).item()==0)}
  report.update(checks=checks,normalization_metrics=metrics,reference_capture_sha256=sha(cap),batch_input_sha256=sha(batch),scope='Real recall17 layer0 fixed operands: installed per-shard Q/K l2 normalization at lengths 8 vs 41, one die; no end-to-end accuracy claim.',status='passed' if all(checks.values()) else 'failed',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert all(checks.values())

except BaseException as e:report.update(status='failed',failure=repr(e));save();raise
finally:torch_npu.npu.synchronize()
