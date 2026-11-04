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
w=pathlib.Path(os.environ['GLM_MODEL_DIR']);ref=pathlib.Path(os.environ['GLM_REFERENCE_DIR'])/f'rank{rank}'
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
  # Use one retained actual 41-token projection input; evaluate exposed
  # low-level intermediates with the same operands truncated to eight tokens.
  initial=m.initial_state();y,state=m(x,initial);torch_npu.npu.synchronize();assert len(calls)==1
  from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
  from fla_npu.ops.ascendc import chunk_kda_fwd,ASCENDC_CTYPES_OPS
  import inspect
  binding=inspect.getsourcefile(ASCENDC_CTYPES_OPS['npu_chunk_kda_fwd']);report['binding_sha256']=sha(binding)
  assert report['binding_sha256']=='2aae5a0d7131f78f26a285f2b2cce2cbae3e2001fb20cf7e8fd373daf9e4cd8a'
  def stats(a,b):
   assert a.shape==b.shape and a.dtype==b.dtype
   d=a.float()-b.float();return {'bitwise_equal':torch.equal(a,b),'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item(),'different_elements':int(torch.count_nonzero(d).item())}
  operands=[z.to('npu') for z in calls[0]['operands']];q,k,v,g,b=operands;width=h//4;results={name:[] for name in ('baseline','gate16','gate24')}
  for mode in results:
   changed=g.clone()
   if mode!='baseline':
    pos=16 if mode=='gate16' else 24
    changed[pos]=changed[pos]-4.0
   assert torch.equal(changed[:8],g[:8])
   for shard in range(4):
    sl=slice(shard*width,(shard+1)*width);n=32
    qn=l2norm_fwd(q[:n,sl].unsqueeze(0).contiguous());kn=l2norm_fwd(k[:n,sl].unsqueeze(0).contiguous())
    outputs=chunk_kda_fwd(qn,kn,v[:n,sl].unsqueeze(0).contiguous(),changed[:n,sl].unsqueeze(0).contiguous(),b[:n,sl].float().sigmoid().unsqueeze(0).contiguous(),dim**-0.5,64,layout='BSND',initial_state=initial.recurrent[sl].unsqueeze(0).clone().contiguous(),output_final_state=True,cu_seqlens=(0,n),chunk_indices=(0,0),safe_gate=True,lower_bound=m.lower_bound,use_gate_in_kernel=True,A_log=m.a_log.reshape(-1)[sl].float().contiguous(),dt_bias=m.dt_bias.reshape(h,dim)[sl].reshape(-1).float().contiguous(),disable_recompute=False,return_intermediate_states=False,state_v_first=True)
    torch_npu.npu.synchronize();results[mode].append([None if z is None else z.detach().cpu().clone() for z in outputs[:5]])
  intervention_metrics={}
  for name,index in (('core',0),('aqk',3),('akk',4)):
   axis=2 if index==0 else 1; values={mode:torch.cat([z[index] for z in rows],dim=axis) for mode,rows in results.items()}
   values={mode:(z[:,:8] if index==0 else torch.tril(z[:,:,:8,:8])) for mode,z in values.items()}
   assert all(torch.isfinite(z).all() for z in values.values()),name
   intervention_metrics[name]={mode:stats(values['baseline'],values[mode]) for mode in ('gate16','gate24')}
  checks={'finite':bool(torch.isfinite(y).all().item()),'tokens_seen':state.tokens_seen==41,'zero_state':bool(torch.count_nonzero(calls[0]['state_before']).item()==0)}
  report.update(checks=checks,intervention_metrics=intervention_metrics,reference_capture_sha256=sha(cap),batch_input_sha256=sha(batch),scope='Real recall17 layer0 fixed32 operands; perturb future raw gate at position16 or control position24; first8 held fixed; single die.',status='passed' if all(checks.values()) else 'failed',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert all(checks.values())

except BaseException as e:report.update(status='failed',failure=repr(e));save();raise
finally:torch_npu.npu.synchronize()
