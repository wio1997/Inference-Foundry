"""Bounded two- or four-rank nonzero-state rollback probe. Operator owns allocation and go barrier."""
import datetime,hashlib,importlib.metadata,json,os,pathlib,platform,sys,time,resource
from datetime import timedelta
resource.setrlimit(resource.RLIMIT_AS,(256*1024**3,256*1024**3))
import torch,torch_npu,torch.distributed as dist
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.native_model import load_native_text_runner
from deepinfer.model_executor.models.glm5_next.experimental_fp32_expanded_dsa import make_experimental_fp32_expanded_dsa_operations
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.state_archive import serialize_snapshot,deserialize_snapshot
from deepinfer.model_executor.models.glm5_next.rank_text_install import install_rank_text_seams
rank=int(sys.argv[1]);die=int(sys.argv[2]);port=int(sys.argv[3]);mode=sys.argv[4];world_size=int(os.environ['WORLD_SIZE']);assert mode == 'bounded' and world_size in (2,4) and die>=0 and 0<=rank<world_size;injected_rank=min(2,world_size-1)
root=pathlib.Path(os.environ['GLM_EXPERIMENT_RUN_DIR']);out=root/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==str(die) and torch_npu.npu.device_count()==1
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);torch.ones(1,device='npu');torch_npu.npu.synchronize();(out/'ready').write_text(str(os.getpid()))
for _ in range(1800):
 if (out/'go').exists():break
 time.sleep(.1)
else:raise RuntimeError('host PID/device mapping not verified')
model=pathlib.Path(os.environ['GLM_MODEL_DIR']);source=pathlib.Path(os.environ['GLM_SOURCE_DIR'])/'deepinfer/model_executor/models/glm5_next';lib=pathlib.Path(os.environ['GLM_ASCEND_LIBRARY']);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report={'rank':rank,'die':die,'mode':mode,'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'weight_index_sha256':sha(model/'quant_model_weights.safetensors.index.json'),'config_sha256':sha(model/'config.json'),'template_sha256':sha(model/'chat_template.jinja'),'install_sha256':sha(source/'rank_text_install.py'),'probe_sha256':sha(__file__),'library_sha256':sha(lib),'versions':{n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')},'status':'running','phase':'setup','scope':'45-layer distributed nonzero-state rollback; dense gate/up and experts sharded; attention/down/router/embedding/head replicated; experimental short-context DSA; not independent quality or production EP','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 p=out/'result.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');q.replace(p)
def state_eq(a,b,config):
 checks={}
 for i,s in enumerate(config.layers):
  x,y=a.layers[i],b.layers[i]
  if s.attention=='linear_attention':same=x.tokens_seen==y.tokens_seen and torch.equal(x.recurrent,y.recurrent) and all(torch.equal(p,q) for p,q in zip(x.conv,y.conv,strict=True))
  else:same=x.tokens_seen==y.tokens_seen and all(torch.equal(getattr(x,k),getattr(y,k)) for k in ('latent','compressed','tail','tail_positions'))
  checks[str(i)]=bool(same)
 return checks
save()
try:
 assert report['weight_index_sha256']=='dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4' and report['config_sha256']=='33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943'
 if mode=='bounded':dist.init_process_group(backend='hccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world_size,timeout=timedelta(seconds=300))
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(root/'hf-cache')
 report.update(phase='assembly',scope='45-layer layer-major dense KDA chunks; singleton rank mask, chunk-to-recurrent handoff, state rollback; no quality claim');save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=32)
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=2,max_context_tokens=32,logical_block_size=128,hc_backend='ascend_installed',kda_backend='ascendc')
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  report['installed']=install_rank_text_seams(runner,reader,rank=rank,world_size=world_size,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement();selected=[]
  for layer in runner.stack.layers:
   if layer.attention_kind=='linear_attention' and layer.mlp_kind=='dense':
    layer.attention.backend='ascendc_chunk_prefill';layer.native_chunk_prefill=True;selected.append(layer.layer_id)
  assert selected==[0,1,2];selected_layers=tuple(selected)
  import deepinfer.model_executor.models.glm5_next.chunk_prefill as chunkmod
  chunk_calls=[];chunk_original=chunkmod.chunk_prefill_sharded
  def observed_chunk(*a,**kw):chunk_calls.append(len(a[0]));return chunk_original(*a,**kw)
  chunkmod.chunk_prefill_sharded=observed_chunk
  runner.enable_rank_batch_commit(max_tokens=2,layer_major_dense_kda=True);report['rank_commit_agreement_enabled']=True;report['kda_backend']='ascendc'
  tokens=[154822,154824,154826,25062]
  def step(q,token,pos):
   z=q.step(torch.tensor([token],dtype=torch.int64),expected_tokens_seen=pos);torch_npu.npu.synchronize();assert torch.isfinite(z.logits).all();return z.logits.detach().cpu().clone()
  report['phase']='clean';save()
  q=runner.open_request(MixedRequestState.capture('clean',config,runner.stack.layout,states))
  try:
   clean=[step(q,t,i) for i,t in enumerate(tokens[:2])];zclean=q.step(torch.tensor(tokens[2:],dtype=torch.int64),expected_tokens_seen=2);torch_npu.npu.synchronize();clean.extend([zclean.logits[i:i+1].cpu().clone() for i in range(2)]);clean_state=q.snapshot();report['clean_chunk_calls']=list(chunk_calls);assert chunk_calls==[2,2,2];chunk_calls.clear()
  finally:q.close()
  q=runner.open_request(MixedRequestState.capture('batch',config,runner.stack.layout,states))
  try:
   prefix=[step(q,t,i) for i,t in enumerate(tokens[:2])];before=q.snapshot()
   report['prefix_logits_equal']=[torch.equal(x,y) for x,y in zip(clean[:2],prefix,strict=True)]
   report['nonzero_kda']=any(bool(torch.count_nonzero(before.layers[i].recurrent).item()) for i,spec in enumerate(config.layers) if spec.attention=='linear_attention')
   assert before.tokens_seen==2 and all(report['prefix_logits_equal']) and report['nonzero_kda']
   report['phase']='singleton-mask-mismatch';save()
   selected=runner.stack.layers[0]
   if rank==injected_rank:selected.attention.chunk_prefill_single=True
   try:
    try:q.step(torch.tensor([tokens[2]],dtype=torch.int64),expected_tokens_seen=2)
    except ValueError as exc:assert str(exc)=='rank batch preflight failed or metadata differs'
    else:raise AssertionError('mismatched singleton mask unexpectedly entered stack')
   finally:selected.attention.chunk_prefill_single=False
   single_mismatch=q.snapshot();report['single_mask_mismatch_coordinated']=True;report['single_mask_state_equal']=state_eq(before,single_mismatch,config)
   assert single_mismatch.tokens_seen==2 and all(report['single_mask_state_equal'].values())
   report['phase']='layer-mask-mismatch';save()
   changed=runner.stack.layers[0]
   if rank==injected_rank:changed.native_chunk_prefill=False
   try:
    try:q.step(torch.tensor(tokens[2:],dtype=torch.int64),expected_tokens_seen=2)
    except ValueError as exc:assert str(exc)=='rank batch preflight failed or metadata differs'
    else:raise AssertionError('mismatched layer mask unexpectedly entered stack')
   finally:changed.native_chunk_prefill=True
   masked=q.snapshot();report['layer_mask_mismatch_coordinated']=True;report['layer_mask_state_equal']=state_eq(before,masked,config)
   assert masked.tokens_seen==2 and all(report['layer_mask_state_equal'].values())
   report['phase']='batch-metadata-mismatch';save()
   try:q.step(torch.tensor(tokens[2:],dtype=torch.int64),expected_tokens_seen=1 if rank==injected_rank else 2)
   except ValueError as exc:assert str(exc)=='rank batch preflight failed or metadata differs'
   else:raise AssertionError('stale rank position unexpectedly committed')
   stale=q.snapshot();report['stale_preflight_coordinated']=True;report['stale_state_equal']=state_eq(before,stale,config)
   assert stale.tokens_seen==2 and all(report['stale_state_equal'].values())
   bad_input=[tokens[2],tokens[3]] if rank==injected_rank else torch.tensor(tokens[2:],dtype=torch.int64)
   try:q.step(bad_input,expected_tokens_seen=2)
   except ValueError as exc:assert str(exc)=='rank batch preflight failed or metadata differs'
   else:raise AssertionError('malformed rank token input unexpectedly committed')
   malformed=q.snapshot();report['malformed_preflight_coordinated']=True;report['malformed_state_equal']=state_eq(before,malformed,config)
   assert malformed.tokens_seen==2 and all(report['malformed_state_equal'].values())
   report['phase']='batch-input-mismatch';save()
   bad=[tokens[2],25063 if rank==injected_rank else tokens[3]]
   try:q.step(torch.tensor(bad,dtype=torch.int64),expected_tokens_seen=2)
   except ValueError as exc:assert str(exc)=='rank batch token inputs differ'
   else:raise AssertionError('unequal rank tokens unexpectedly committed')
   mismatch=q.snapshot();report['rank_input_mismatch_coordinated']=True;report['rank_input_mismatch_state_equal']=state_eq(before,mismatch,config)
   assert mismatch.tokens_seen==2 and all(report['rank_input_mismatch_state_equal'].values())
   report['phase']='second-head-failure';save();original_head=runner.head.forward;calls=[0]
   def injected_head(hidden):
    calls[0]+=1
    if rank==injected_rank and calls[0]==2:raise RuntimeError('injected second head failure')
    return original_head(hidden)
   runner.head.forward=injected_head
   try:
    try:q.step(torch.tensor(tokens[2:],dtype=torch.int64),expected_tokens_seen=2)
    except RuntimeError as exc:
     assert str(exc)=='rank batch head failed'
     assert (isinstance(exc.__cause__,RuntimeError) and str(exc.__cause__)=='injected second head failure')==(rank==injected_rank)
    else:raise AssertionError('head failure unexpectedly committed')
   finally:runner.head.forward=original_head
   after=q.snapshot();report['failure_chunk_calls']=list(chunk_calls);assert chunk_calls==[2,2,2];chunk_calls.clear();report['failure_coordinated']=True;report['failure_state_equal']=state_eq(before,after,config)
   assert after.tokens_seen==2 and all(report['failure_state_equal'].values())
   report['phase']='batch-retry';save();z=q.step(torch.tensor(tokens[2:],dtype=torch.int64),expected_tokens_seen=2);torch_npu.npu.synchronize();retry_state=q.snapshot()
   report['retry_chunk_calls']=list(chunk_calls);assert chunk_calls==[2,2,2];report['retry_logits_equal']=[torch.equal(clean[i+2],z.logits[i:i+1].cpu()) for i in range(2)]
   report['retry_state_equal']=state_eq(clean_state,retry_state,config);report['request_owners_remaining']=1
   assert all(report['retry_logits_equal']) and all(report['retry_state_equal'].values()) and retry_state.tokens_seen==4
   chunk_calls.clear()
   for i in selected_layers:runner.stack.layers[i].attention.chunk_prefill_single=True
   try:
    singleton=q.step(torch.tensor([tokens[3]],dtype=torch.int64),expected_tokens_seen=4);torch_npu.npu.synchronize();report['singleton_chunk_calls']=list(chunk_calls);assert chunk_calls==[1,1,1] and singleton.token_stop==5
   finally:
    for i in selected_layers:runner.stack.layers[i].attention.chunk_prefill_single=False
   chunk_calls.clear();decode=q.step(torch.tensor([tokens[3]],dtype=torch.int64),expected_tokens_seen=5);torch_npu.npu.synchronize();report['decode_chunk_calls']=list(chunk_calls);assert not chunk_calls and decode.token_stop==6
  finally:q.close()
  report['request_owners_remaining']=len(runner._requests);assert report['request_owners_remaining']==0
  if hasattr(torch_npu.npu,'max_memory_allocated'):report['peak_npu_allocated_bytes']=int(torch_npu.npu.max_memory_allocated())
  if mode=='bounded':dist.barrier();torch_npu.npu.synchronize()
 report['status']='passed';report['phase']='complete'
except BaseException as e:report.update(status='failed',failure=repr(e));raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('RESULT',json.dumps(report),flush=True)
