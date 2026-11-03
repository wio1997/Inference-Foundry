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
 report.update(phase='assembly',scope='45-layer distributed nonzero-state selected-rank head failure rollback and retry; no quality claim');save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=32)
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=1,max_context_tokens=32,logical_block_size=128,hc_backend='ascend_installed',kda_backend='ascendc')
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  report['installed']=install_rank_text_seams(runner,reader,rank=rank,world_size=world_size,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement();report['rank_commit_agreement_enabled']=True;report['kda_backend']='ascendc'
  tokens=[154822,154824,154826,25062]
  def step(q,token,pos):
   z=q.step(torch.tensor([token],dtype=torch.int64),expected_tokens_seen=pos);torch_npu.npu.synchronize();assert torch.isfinite(z.logits).all();return z.logits.detach().cpu().clone()
  report['phase']='clean';save()
  q=runner.open_request(MixedRequestState.capture('clean',config,runner.stack.layout,states))
  try:
   clean=[];position_states={}
   for i,t in enumerate(tokens):
    clean.append(step(q,t,i))
    if i in (1,2):position_states[i+1]=q.snapshot()
   clean_state=q.snapshot()
  finally:q.close()
  position=2 if rank==injected_rank else 3
  q=runner.open_request(MixedRequestState.capture('position-test',config,runner.stack.layout,position_states[position].layers))
  try:
   report['phase']='position_mismatch';save();position_before=q.snapshot()
   try:q.step(torch.tensor([tokens[3]],dtype=torch.int64),expected_tokens_seen=position)
   except ValueError as exc:assert str(exc)=='rank commit preflight failed'
   else:raise AssertionError('locally valid but rank-inconsistent position unexpectedly committed')
   position_after=q.snapshot();report['position_mismatch_coordinated']=True;report['position_before']=position;report['position_after']=position_after.tokens_seen;report['position_mismatch_state_equal']=state_eq(position_before,position_after,config)
   assert position_after.tokens_seen==position and len(report['position_mismatch_state_equal'])==45 and all(report['position_mismatch_state_equal'].values())
  finally:q.close()
  q=runner.open_request(MixedRequestState.capture('embedding-fix-test',config,runner.stack.layout,states))
  try:
   report['phase']='invalid_token_preflight';save();before_embedding=q.snapshot();original_embed=runner.boundary.embed
   def forbidden_embed(ids):raise AssertionError('embedding entered for invalid token dtype')
   runner.boundary.embed=forbidden_embed
   try:
    try:q.step(torch.tensor([tokens[0]],dtype=torch.float32 if rank==injected_rank else torch.int64),expected_tokens_seen=0)
    except ValueError as exc:assert str(exc)=='rank commit preflight failed'
    else:raise AssertionError('invalid rank token dtype unexpectedly committed')
   finally:runner.boundary.embed=original_embed
   invalid_after=q.snapshot();report['invalid_dtype_coordinated']=True;report['invalid_dtype_state_equal']=state_eq(before_embedding,invalid_after,config)
   assert invalid_after.tokens_seen==0 and all(report['invalid_dtype_state_equal'].values())
   report['phase']='embedding_failure';save()
   def injected_embed(ids):
    if rank==injected_rank:raise RuntimeError('injected embedding reader failure')
    return original_embed(ids)
   runner.boundary.embed=injected_embed
   try:
    try:q.step(torch.tensor([tokens[0]],dtype=torch.int64),expected_tokens_seen=0)
    except RuntimeError as exc:
     assert str(exc)=='rank embedding preparation failed'
     assert (isinstance(exc.__cause__,RuntimeError) and str(exc.__cause__)=='injected embedding reader failure')==(rank==injected_rank)
    else:raise AssertionError('rank embedding failure unexpectedly committed')
   finally:runner.boundary.embed=original_embed
   embedding_after=q.snapshot();report['embedding_failure_coordinated']=True;report['embedding_failure_state_equal']=state_eq(before_embedding,embedding_after,config)
   assert embedding_after.tokens_seen==0 and all(report['embedding_failure_state_equal'].values())
  finally:q.close()
  q=runner.open_request(MixedRequestState.capture('failure-retry-other' if rank==injected_rank else 'failure-retry',config,runner.stack.layout,states))
  try:
   report['phase']='identity_mismatch';save();identity_before=q.snapshot()
   try:q.step(torch.tensor([tokens[0]],dtype=torch.int64),expected_tokens_seen=0)
   except ValueError as exc:assert str(exc)=='rank commit preflight failed'
   else:raise AssertionError('rank-specific request identity unexpectedly committed')
   identity_after=q.snapshot();report['identity_mismatch_coordinated']=True;report['identity_mismatch_tokens_seen']=identity_after.tokens_seen;report['identity_mismatch_state_equal']=state_eq(identity_before,identity_after,config)
   assert identity_after.tokens_seen==0 and len(report['identity_mismatch_state_equal'])==45 and all(report['identity_mismatch_state_equal'].values())
  finally:q.close()
  q=runner.open_request(MixedRequestState.capture('failure-retry',config,runner.stack.layout,states))
  try:
   report['phase']='prefix';save();prefix=[step(q,t,i) for i,t in enumerate(tokens[:3])];before=q.snapshot()
   report['prefix_logits_equal']=[torch.equal(x,y) for x,y in zip(clean[:3],prefix,strict=True)]
   report['nonzero_kda']=any(bool(torch.count_nonzero(before.layers[i].recurrent).item()) for i,spec in enumerate(config.layers) if spec.attention=='linear_attention')
   assert before.tokens_seen==3 and all(report['prefix_logits_equal']) and report['nonzero_kda']
   report['phase']='legacy_stale_preflight';save()
   original_batch_limit=runner._rank_batch_limit;runner._rank_batch_limit=0
   try:
    try:q.step(torch.tensor([tokens[3]],dtype=torch.int64),expected_tokens_seen=2 if rank==injected_rank else 3)
    except ValueError as exc:
     assert str(exc)=='rank commit preflight failed'
     assert (isinstance(exc.__cause__,ValueError) and str(exc.__cause__)=='stale or invalid expected request token position')==(rank==injected_rank)
    else:raise AssertionError('rank-specific stale position unexpectedly entered stack')
   finally:runner._rank_batch_limit=original_batch_limit
   stale=q.snapshot();report['legacy_stale_coordinated']=True;report['legacy_stale_tokens_seen']=stale.tokens_seen;report['legacy_stale_state_equal']=state_eq(before,stale,config)
   assert stale.tokens_seen==3 and len(report['legacy_stale_state_equal'])==45 and all(report['legacy_stale_state_equal'].values())
   del stale
   report['phase']='inject';save();original_head=runner.head.forward
   def injected_head(hidden):
    if rank==injected_rank:raise RuntimeError('injected nonzero-state selected-rank head failure')
    return original_head(hidden)
   runner.head.forward=injected_head
   try:
    try:q.step(torch.tensor([tokens[3]],dtype=torch.int64),expected_tokens_seen=3)
    except RuntimeError as exc:
     assert str(exc)=='rank head/commit preparation failed'
     assert (isinstance(exc.__cause__,RuntimeError) and str(exc.__cause__)=='injected nonzero-state selected-rank head failure')==(rank==injected_rank)
    else:raise AssertionError('rank head failure unexpectedly committed')
   finally:runner.head.forward=original_head
   after=q.snapshot();report['failure_coordinated']=True;report['failure_tokens_seen']=after.tokens_seen;report['failure_state_equal']=state_eq(before,after,config)
   assert after.tokens_seen==3 and len(report['failure_state_equal'])==45 and all(report['failure_state_equal'].values())
   del after
   report['phase']='rank_input_mismatch';save();original_embed=runner.boundary.embed
   def forbidden_mismatch_embed(ids):raise AssertionError('embedding entered for rank token mismatch')
   runner.boundary.embed=forbidden_mismatch_embed
   try:
    try:q.step(torch.tensor([25063 if rank==injected_rank else tokens[3]],dtype=torch.int64),expected_tokens_seen=3)
    except ValueError as exc:assert str(exc)=='rank commit preflight failed',repr(exc)
    else:raise AssertionError('rank input mismatch unexpectedly committed')
   finally:runner.boundary.embed=original_embed
   report['rank_input_mismatch_before_embedding']=True
   mismatch=q.snapshot();report['rank_input_mismatch_coordinated']=True;report['rank_input_mismatch_tokens_seen']=mismatch.tokens_seen;report['rank_input_mismatch_state_equal']=state_eq(before,mismatch,config)
   assert mismatch.tokens_seen==3 and len(report['rank_input_mismatch_state_equal'])==45 and all(report['rank_input_mismatch_state_equal'].values())
   del before,mismatch
   report['phase']='retry';save();retry=step(q,tokens[3],3);retry_state=q.snapshot()
  finally:q.close()
  report['tokens']=tokens;report['retry_logits_equal']=torch.equal(clean[3],retry);report['retry_state_equal']=state_eq(clean_state,retry_state,config);report['request_owners_remaining']=len(runner._requests)
  assert report['retry_logits_equal'] and len(report['retry_state_equal'])==45 and all(report['retry_state_equal'].values()) and report['request_owners_remaining']==0
  if hasattr(torch_npu.npu,'max_memory_allocated'):report['peak_npu_allocated_bytes']=int(torch_npu.npu.max_memory_allocated())
  if mode=='bounded':dist.barrier();torch_npu.npu.synchronize()
 report['status']='passed';report['phase']='complete'
except BaseException as e:report.update(status='failed',failure=repr(e));raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('RESULT',json.dumps(report),flush=True)
