"""Portable bounded distributed worker; non-four-rank paths are not hardware-validated.

The operator must arrange and verify host PID-to-die mapping before writing the go files.
"""
import datetime,hashlib,importlib.metadata,json,os,pathlib,platform,sys,time,resource
from datetime import timedelta
from contextlib import ExitStack
from deepinfer.model_executor.models.glm5_next.rank_cache_scope import optional_rank_cache_session
from deepinfer.model_executor.models.glm5_next.rank_expert_adapter import ReplicatedRankExperts
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
rank=int(sys.argv[1]);die=int(sys.argv[2]);port=int(sys.argv[3]);world_size=int(os.environ['WORLD_SIZE']);assert world_size in (2,4,8,16) and 0<=rank<world_size and die>=0
root=pathlib.Path(os.environ['GLM_EXPERIMENT_RUN_DIR']);out=root/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==str(die) and torch_npu.npu.device_count()==1
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);torch.ones(1,device='npu');torch_npu.npu.synchronize();(out/'ready').write_text(str(os.getpid()))
for _ in range(1800):
 if (out/'go').exists():break
 time.sleep(.1)
else:raise RuntimeError('host PID/device mapping not verified')
model=pathlib.Path(os.environ['GLM_MODEL_DIR']);source=pathlib.Path(os.environ['GLM_SOURCE_DIR'])/'deepinfer/model_executor/models/glm5_next';lib=pathlib.Path(os.environ['GLM_ASCEND_LIBRARY']);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report={'rank':rank,'die':die,'world_size':world_size,'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'weight_index_sha256':sha(model/'quant_model_weights.safetensors.index.json'),'config_sha256':sha(model/'config.json'),'template_sha256':sha(model/'chat_template.jinja'),'install_sha256':sha(source/'rank_text_install.py'),'probe_sha256':sha(__file__),'library_sha256':sha(lib),'versions':{n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')},'status':'running','phase':'setup','scope':'distributed 45-layer one recall chat; replicated attention/router/embedding/head, sharded dense gate/up and selected experts; same-runtime baseline, not independent quality or production EP','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 p=out/'result.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');q.replace(p)
def archive_state_equal(a,b,config):
 checks={}
 for i,spec in enumerate(config.layers):
  x,y=a.layers[i],b.layers[i]
  if spec.attention=='linear_attention':same=x.tokens_seen==y.tokens_seen and torch.equal(x.recurrent,y.recurrent) and all(torch.equal(u,v) for u,v in zip(x.conv,y.conv,strict=True))
  else:same=x.tokens_seen==y.tokens_seen and all(torch.equal(getattr(x,k),getattr(y,k)) for k in ('latent','compressed','tail','tail_positions'))
  checks[str(i)]=bool(same)
 return checks
save()
try:
 assert report['weight_index_sha256']=='dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4' and report['config_sha256']=='33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943' and report['template_sha256']=='34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891'
 dist.init_process_group(backend='hccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world_size,timeout=timedelta(seconds=300))
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(root/'hf-cache')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 request=json.loads((root/'request.json').read_text());layer_major=request.get('experimental_layer_major_kda',False);sparse_kda=request.get('experimental_sparse_kda_layer4',False);prefill40=request.get('experimental_kda_prefill40',False);assert type(prefill40) is bool and (not prefill40 or layer_major), 'prefill40 requires layer-major KDA';assert type(sparse_kda) is bool and (not sparse_kda or layer_major), 'sparse KDA layer4 requires layer-major';assert type(layer_major) is bool and (not layer_major or world_size==4), 'experimental layer-major KDA requires four ranks';messages=request['messages'];ids=request['prompt_ids'];cap=request['max_new_tokens'];context=request['context_tokens'];expected=request['expected_final'];cache_bytes=request['cache_mib']*1024**2;kda_backend=request['kda_backend'];assert kda_backend in ('ascendc_installed_decode','ascendc')
 encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,reasoning_effort='low');actual_ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
 assert ids==actual_ids and len(ids)+cap<=context<=128, 'request tokenizer or token bound mismatch'
 report.update(messages=messages,expected_final=expected,context_bound=context,cache_mib=request['cache_mib'],kda_backend=kda_backend)
 stops=sorted(json.loads((model/'generation_config.json').read_text())['eos_token_id']);assert stops==[154820,154827,154829]
 report.update(prompt_ids=ids,stop_ids=stops,tokenizer_eos_id=tok.eos_token_id,generation_cap=cap,phase='assembly',prompt_argmax=[]);save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=context)
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=(40 if prefill40 else 8) if layer_major else 1,max_context_tokens=context,logical_block_size=128,hc_backend='ascend_installed',kda_backend=kda_backend)
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  report['expert_lease_cache_mib']=request['cache_mib'] if cache_bytes>0 else 128
  report['installed']=install_rank_text_seams(runner,reader,rank=rank,world_size=world_size,cache_bytes=report['expert_lease_cache_mib']*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8,max_dense_input_tokens=40 if prefill40 else 8)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement()
  selected=[]
  if layer_major:
   if kda_backend!='ascendc':raise ValueError('experimental layer-major KDA requires ascendc backend')
   for layer in runner.stack.layers:
    if layer.attention_kind=='linear_attention' and layer.mlp_kind=='dense':
     layer.attention.backend='ascendc_chunk_prefill';layer.attention.reference_sharded_projection=True;layer.native_chunk_prefill=True;layer.native_chunk_max_tokens=40 if prefill40 else 8;selected.append(layer.layer_id)
   assert selected==[0,1,2]
   if sparse_kda:
    layer=runner.stack.layers[4]
    if layer.attention_kind!='linear_attention' or layer.mlp_kind!='sparse':raise ValueError('layer4 must be sparse-MLP KDA')
    layer.attention.backend='ascendc_chunk_prefill';layer.attention.reference_sharded_projection=True;layer.native_chunk_prefill=True;layer.native_chunk_max_tokens=40 if prefill40 else 8;selected.append(4)
   runner.enable_rank_batch_commit(max_tokens=40 if prefill40 else 8,layer_major_dense_kda=True)
  report.update(rank_commit_agreement_enabled=True,experimental_layer_major_kda=layer_major,experimental_sparse_kda_layer4=sparse_kda,experimental_kda_prefill40=prefill40,chunk_layers=selected,prefill_groups=[])
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  scope=ExitStack();session=scope.enter_context(optional_rank_cache_session(reader,enabled=cache_bytes>0,rank=rank,world_size=world_size,cache_bytes=cache_bytes,max_assembled_bytes=16*1024**2));adapters=[layer.mlp.experts for layer in runner.stack.layers if isinstance(getattr(layer.mlp,'experts',None),ReplicatedRankExperts)]
  if session is not None:
   for adapter in adapters:adapter.cache_session=session
  q=runner.open_request(MixedRequestState.capture(request['run_id'],config,runner.stack.layout,states))
  def step(token,pos):
   z=q.step(torch.tensor([token],dtype=torch.int64),expected_tokens_seen=pos);torch_npu.npu.synchronize();assert torch.isfinite(z.logits).all();return z.logits
  try:
   report['phase']='prompt';save()
   if layer_major:
    for pos in range(0,len(ids),40 if prefill40 else 8):
     part=ids[pos:pos+(40 if prefill40 else 8)]
     if len(part)==1:
      for i in selected:runner.stack.layers[i].attention.chunk_prefill_single=True
     try:z=q.step(torch.tensor(part,dtype=torch.int64),expected_tokens_seen=pos)
     finally:
      for i in selected:runner.stack.layers[i].attention.chunk_prefill_single=False
     torch_npu.npu.synchronize();assert z.token_stop==pos+len(part) and torch.isfinite(z.logits).all()
     report['prefill_groups'].append(len(part));report['prompt_argmax'].extend(z.logits.argmax(dim=-1).cpu().tolist());logits=z.logits[-1:];save()
   else:
    for i,token in enumerate(ids):
     logits=step(token,i);report['prompt_argmax'].append(int(logits.argmax(-1).item()));save()
   generated=[];report['phase']='generation';report['rank_agreement']=True;save();pos=len(ids)
   for i in range(cap):
    token=int(logits.argmax(-1).item())
    local=torch.tensor([token],dtype=torch.int64,device='npu');gathered=[torch.empty_like(local) for _ in range(world_size)];dist.all_gather(gathered,local)
    values=[int(v.item()) for v in gathered];report['rank_agreement'] = report['rank_agreement'] and all(v==token for v in values)
    generated.append(token);report['generated_ids']=generated;report['generation_tokens_done']=len(generated);save()
    if not report['rank_agreement']:raise RuntimeError(('rank token disagreement',values))
    if token in stops or i==cap-1:break
    logits=step(token,pos);pos+=1
    if request['restore_after_first_token'] and i==0:
     # Every live rank must finish archive preparation before any rank
     # enters restore verification. A local failure must not strand peers.
     prepare_status=torch.zeros((1,),dtype=torch.int32,device='npu');failure=None;snapshot=None;blob=None
     try:
      snapshot=q.snapshot();blob=serialize_snapshot(snapshot,config,runner.stack.layout,max_tokens=context,max_archive_bytes=512*1024**2)
      report['archive_sha256']=hashlib.sha256(blob).hexdigest();report['archive_bytes']=len(blob);report['archive_tokens_seen']=pos;report['archive_schema_version']=torch.load(__import__('io').BytesIO(blob),map_location='cpu',weights_only=True)['version'];assert report['archive_schema_version']==2
      q.close()
     except Exception as exc:failure=exc
     prepare_status.fill_(int(failure is not None));dist.all_reduce(prepare_status,op=dist.ReduceOp.MAX)
     if int(prepare_status.item()):raise RuntimeError('rank archive preparation failed') from failure
     failure=None;restored=None
     try:
      restored=deserialize_snapshot(blob,config,runner.stack.layout,expected_request_id=request['run_id'],expected_tokens_seen=pos,device='npu',max_tokens=context,max_archive_bytes=512*1024**2)
      report['archive_state_equal']=archive_state_equal(snapshot,restored,config)
      if snapshot.request_id!=restored.request_id or snapshot.tokens_seen!=restored.tokens_seen or len(report['archive_state_equal'])!=45 or not all(report['archive_state_equal'].values()):raise ValueError('restored archive state mismatch')
     except Exception as exc:failure=exc
     restore_status=torch.tensor([int(failure is not None)],dtype=torch.int32,device='npu');dist.all_reduce(restore_status,op=dist.ReduceOp.MAX)
     if int(restore_status.item()):raise RuntimeError('rank archive restore verification failed') from failure
     report['archive_state_rank_agreement']=True;q=runner.open_request(restored)
     report['restore_after_first_token_completed']=True;save()
   report['stop_set_reached']=generated[-1] in stops;report['stop_token_id']=generated[-1] if report['stop_set_reached'] else None;report['stop_marker']=tok.decode([generated[-1]],skip_special_tokens=False) if report['stop_set_reached'] else None;report['tokenizer_eos_reached']=generated[-1]==tok.eos_token_id;report['cap_reached']=len(generated)==cap and not report['stop_set_reached'];report['generated_text']=tok.decode(generated,skip_special_tokens=False)
   body=generated[:-1] if report['stop_set_reached'] else generated
   text=tok.decode(body,skip_special_tokens=False);report['reason_final_boundary_present']='</think>' in text
   report['reasoning_text'],report['final_text']=(text.rsplit('</think>',1) if report['reason_final_boundary_present'] else (text,None))
   report['final_answer_exact']=report['final_text'] is not None and report['final_text'].strip()==expected if expected is not None else None
   report['semantic_pass']=report['rank_agreement'] and report['stop_set_reached'] and not report['cap_reached'] and report['final_text'] is not None and bool(report['final_text'].strip()) and (expected is None or report['final_answer_exact']) and (not request['restore_after_first_token'] or (report.get('restore_after_first_token_completed',False) and report.get('archive_state_rank_agreement',False) and len(report.get('archive_state_equal',{}))==45 and all(report['archive_state_equal'].values())))
   save();assert report['semantic_pass'], 'recall did not produce naturally terminated exact final answer'

  finally:
   q.close()
   for adapter in adapters:adapter.cache_session=None
   if session is not None:report['cache_before_close']={k:v for k,v in session.stats().items() if k!='keys_lru'}
   scope.close()
   if session is not None:report['cache_after_close']={k:v for k,v in session.stats().items() if k!='keys_lru'}
  if hasattr(torch_npu.npu,'max_memory_allocated'):report['peak_npu_allocated_bytes']=int(torch_npu.npu.max_memory_allocated())
  dist.barrier();torch_npu.npu.synchronize()
 report['status']='passed';report['phase']='complete'
except BaseException as e:report.update(status='failed',failure=repr(e));raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('RESULT',json.dumps(report),flush=True)
