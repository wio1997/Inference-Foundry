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
 report.update(phase='assembly',scope='45-layer nonzero state, rank2 40-token chunk bound mismatch, coordinated rejection and clean retry; no quality claim');save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=64)
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=40,max_context_tokens=64,logical_block_size=128,hc_backend='ascend_installed',kda_backend='ascendc')
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  report['installed']=install_rank_text_seams(runner,reader,rank=rank,world_size=world_size,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8,max_dense_input_tokens=40)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement();selected=[]
  for layer in runner.stack.layers:
   if layer.attention_kind=='linear_attention' and layer.mlp_kind=='dense':
    layer.attention.backend='ascendc_chunk_prefill';layer.attention.reference_sharded_projection=True;layer.native_chunk_prefill=True;layer.native_chunk_max_tokens=40;selected.append(layer.layer_id)
  assert selected==[0,1,2]
  report['selected_layers']=selected
  import deepinfer.model_executor.models.glm5_next.chunk_prefill as chunkmod
  chunk_calls=[];chunk_original=chunkmod.chunk_prefill_sharded
  def observed_chunk(*a,**kw):chunk_calls.append(len(a[0]));return chunk_original(*a,**kw)
  chunkmod.chunk_prefill_sharded=observed_chunk
  runner.enable_rank_batch_commit(max_tokens=40,layer_major_dense_kda=True);report['rank_commit_agreement_enabled']=True;report['kda_backend']='ascendc'
  ids=[154822,154824,154826,25062,287,29905,371,25,12035,154827,5050,6234,1372,374,220,99419,13,154828,154841,154842,40,686,6099,220,99419,13,154827,3838,572,847,6234,1372,30,17815,448,1172,279,1372,13,154828,154841]
  def open_prefixed(name):
   q=runner.open_request(MixedRequestState.capture(name,config,runner.stack.layout,states));q.step(torch.tensor(ids[:1],dtype=torch.int64),expected_tokens_seen=0);return q
  clean=open_prefixed('clean40')
  try:
   report['phase']='clean40';save();z=clean.step(torch.tensor(ids[1:],dtype=torch.int64),expected_tokens_seen=1);torch_npu.npu.synchronize();clean_logits=z.logits.detach().cpu().clone();clean_state=clean.snapshot();assert clean_state.tokens_seen==41
  finally:clean.close()
  q=open_prefixed('retry40')
  try:
   before=q.snapshot();report['phase']='mismatched_chunk_bound40';save();layer=runner.stack.layers[selected[0]]
   if rank==injected_rank:layer.native_chunk_max_tokens=8
   try:
    try:q.step(torch.tensor(ids[1:],dtype=torch.int64),expected_tokens_seen=1)
    except ValueError as exc:
     report['failure_coordinated']=str(exc)=='rank batch preflight failed or metadata differs' and ((isinstance(exc.__cause__,ValueError) and str(exc.__cause__)=='invalid or exceeded KDA chunk bound')==(rank==injected_rank))
    else:report['failure_coordinated']=False
   finally:layer.native_chunk_max_tokens=40
   after=q.snapshot();report['rollback_state_equal']=state_eq(before,after,config);report['rollback_tokens_seen']=after.tokens_seen;report['phase']='retry40';save();retry=q.step(torch.tensor(ids[1:],dtype=torch.int64),expected_tokens_seen=1);torch_npu.npu.synchronize();final=q.snapshot();report['retry_logits_equal']=bool(torch.equal(clean_logits,retry.logits.cpu()));report['retry_state_equal']=state_eq(clean_state,final,config)
   report['checks']={'failure_coordinated':report['failure_coordinated'],'rollback_45_states':len(report['rollback_state_equal'])==45 and all(report['rollback_state_equal'].values()),'rollback_prefix_unchanged':after.tokens_seen==1,'retry_logits_equal':report['retry_logits_equal'],'retry_45_states':len(report['retry_state_equal'])==45 and all(report['retry_state_equal'].values()),'retry_tokens_seen':final.tokens_seen==41};assert all(report['checks'].values())
  finally:q.close()
  report['request_owners_remaining']=len(runner._requests);assert report['request_owners_remaining']==0
 report.update(status='passed',phase='complete');save()
except BaseException as e:report.update(status='failed',failure=repr(e));save();raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();save()
