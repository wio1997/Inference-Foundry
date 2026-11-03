"""Fixed real chat regression on four replicated-input cooperating ranks."""
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
from deepinfer.model_executor.models.glm5_next.rank_cache_scope import optional_rank_cache_session
from deepinfer.model_executor.models.glm5_next.rank_expert_adapter import ReplicatedRankExperts
rank=int(sys.argv[1]);die=int(sys.argv[2]);port=int(sys.argv[3]);assert die>=0 and 0<=rank<4
root=pathlib.Path(os.environ['GLM_EXPERIMENT_RUN_DIR']);out=root/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==str(die) and torch_npu.npu.device_count()==1
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);torch.ones(1,device='npu');torch_npu.npu.synchronize();(out/'ready').write_text(str(os.getpid()))
for _ in range(1800):
 if (out/'go').exists():break
 time.sleep(.1)
else:raise RuntimeError('host PID/device mapping not verified')
model=pathlib.Path(os.environ['GLM_MODEL_DIR']);source=pathlib.Path(os.environ['GLM_SOURCE_DIR'])/'deepinfer/model_executor/models/glm5_next';lib=pathlib.Path(os.environ['GLM_ASCEND_LIBRARY']);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report={'rank':rank,'die':die,'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'weight_index_sha256':sha(model/'quant_model_weights.safetensors.index.json'),'config_sha256':sha(model/'config.json'),'template_sha256':sha(model/'chat_template.jinja'),'install_sha256':sha(source/'rank_text_install.py'),'probe_sha256':sha(__file__),'library_sha256':sha(lib),'versions':{n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')},'status':'running','phase':'setup','scope':'four-rank 45-layer experimental layer-major chunk prefill unrestored versus archive-v2 restored reusable-session fixed chat; replicated attention/router/embedding/head, sharded dense gate/up and selected experts; same-runtime baseline, not independent quality or production EP','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
 p=out/'result.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');q.replace(p)
save()
try:
 assert report['weight_index_sha256']=='dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4' and report['config_sha256']=='33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943' and report['template_sha256']=='34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891'
 dist.init_process_group(backend='hccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,timeout=timedelta(seconds=300))
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(root/'hf-cache')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 messages=[{'role': 'user', 'content': 'My secret number is 17.'}, {'role': 'assistant', 'content': 'I will remember 17.'}, {'role': 'user', 'content': 'What was my secret number? Reply with only the number.'}]
 encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,reasoning_effort='low');ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
 baseline_ids=[154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 5050, 6234, 1372, 374, 220, 99419, 13, 154828, 154841, 154842, 40, 686, 6099, 220, 99419, 13, 154827, 3838, 572, 847, 6234, 1372, 30, 17815, 448, 1172, 279, 1372, 13, 154828, 154841]
 baseline_argmax=[154822, 565, 220, 48064, 25, 3969, 25, 576, 198, 785, 3405, 374, 374, 220, 18, 13, 4615, 154842, 1474, 40, 3003, 6099, 429, 99419, 438, 3555, 785, 374, 847, 6234, 1372, 30, 220, 448, 1101, 279, 1372, 13, 154842, 154842, 1474]
 baseline_generated=[1474, 4588, 311, 19056, 279, 1372, 220, 99419, 13, 154842, 99419, 154827]
 assert ids==baseline_ids
 stops=sorted(json.loads((model/'generation_config.json').read_text())['eos_token_id']);assert stops==[154820,154827,154829]
 report.update(prompt_ids=ids,stop_ids=stops,tokenizer_eos_id=tok.eos_token_id,generation_cap=16,phase='assembly',prompt_argmax=[]);save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=64)
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=40,max_context_tokens=64,logical_block_size=128,hc_backend='ascend_installed',kda_backend='ascendc')
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  report['installed']=install_rank_text_seams(runner,reader,rank=rank,world_size=4,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8,max_dense_input_tokens=40)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement();selected=[]
  for layer in runner.stack.layers:
   if layer.attention_kind=='linear_attention' and (layer.mlp_kind=='dense' or layer.layer_id==4):
    layer.attention.backend='ascendc_chunk_prefill';layer.attention.reference_sharded_projection=True;layer.native_chunk_prefill=True;layer.native_chunk_max_tokens=40;selected.append(layer.layer_id)
  assert selected==[0,1,2,4]
  runner.enable_rank_batch_commit(max_tokens=40,layer_major_dense_kda=True)
  sparse_rows=[];sparse_mlp=runner.stack.layers[4].mlp;old_sparse_forward=sparse_mlp.forward
  def observed_sparse(x):sparse_rows.append(len(x));return old_sparse_forward(x)
  sparse_mlp.forward=observed_sparse
  report['rank_commit_agreement_enabled']=True;report['chunk_layers']=selected
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  initial=MixedRequestState.capture('round416-cache',config,runner.stack.layout,states)
  adapters=[layer.mlp.experts for layer in runner.stack.layers if isinstance(getattr(layer.mlp,'experts',None),ReplicatedRankExperts)]
  assert len(adapters)==42 and all(a.cache_session is None for a in adapters)
  def compare_state(a,b):
   result={}
   for j,spec in enumerate(config.layers):
    x,y=a.layers[j],b.layers[j]
    if spec.attention=='linear_attention':same=x.tokens_seen==y.tokens_seen and torch.equal(x.recurrent,y.recurrent) and all(torch.equal(u,w) for u,w in zip(x.conv,y.conv,strict=True))
    else:same=x.tokens_seen==y.tokens_seen and all(torch.equal(getattr(x,k),getattr(y,k)) for k in ('latent','compressed','tail','tail_positions'))
    result[str(j)]=bool(same)
   return result
  def run_path(label):
   report['phase']=label;report['prompt_tokens_done']=0;save();q=runner.open_request(initial);values=[];argmax=[];generated=[]
   def step(token,pos):
    z=q.step(torch.tensor([token],dtype=torch.int64),expected_tokens_seen=pos);torch_npu.npu.synchronize();assert torch.isfinite(z.logits).all();return z.logits.detach().cpu().clone()
   try:
    groups=[]
    for pos in range(0,len(ids),40):
     part=ids[pos:pos+40]
     if len(part)==1:
      for i in selected:runner.stack.layers[i].attention.chunk_prefill_single=True
     try:z=q.step(torch.tensor(part,dtype=torch.int64),expected_tokens_seen=pos)
     finally:
      for i in selected:runner.stack.layers[i].attention.chunk_prefill_single=False
     torch_npu.npu.synchronize();assert z.token_stop==pos+len(part) and torch.isfinite(z.logits).all();logits=z.logits.detach().cpu().clone();values.append(logits);argmax.extend(logits.argmax(dim=-1).tolist());groups.append(len(part));report['prompt_tokens_done']=pos+len(part);save()
    report[label+'_prefill_groups']=groups;logits=logits[-1:]
    pos=len(ids)
    for i in range(16):
     token=int(logits.argmax(-1).item());generated.append(token)
     if token in stops or i==15:break
     logits=step(token,pos);values.append(logits);pos+=1
     if label=='reusable_session' and i==0:
      before=q.snapshot();blob=serialize_snapshot(before,config,runner.stack.layout,max_tokens=64,max_archive_bytes=512*1024**2);q.close()
      restored=deserialize_snapshot(blob,config,runner.stack.layout,expected_request_id=before.request_id,expected_tokens_seen=pos,device='npu',max_tokens=64,max_archive_bytes=512*1024**2)
      report['archive_state_equal']=compare_state(before,restored);report['archive_tokens_seen']=pos;report['archive_sha256']=hashlib.sha256(blob).hexdigest();assert len(report['archive_state_equal'])==45 and all(report['archive_state_equal'].values());q=runner.open_request(restored);save()
    final_state=q.snapshot()
   finally:q.close()
   return values,argmax,generated,final_state
  direct,first_argmax,first_gen,first_state=run_path('default_per_call')
  assert first_gen[-1] in stops
  with optional_rank_cache_session(reader,enabled=True,rank=rank,world_size=4,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2) as session:
   try:
    for adapter in adapters:adapter.cache_session=session
    cached,second_argmax,second_gen,second_state=run_path('reusable_session')
    report['logit_equal']=[torch.equal(x,y) for x,y in zip(direct,cached,strict=True)];report['generated_equal']=first_gen==second_gen;report['prompt_argmax_equal']=first_argmax==second_argmax
    state_equal={}
    for j,layer_spec in enumerate(config.layers):
     x,y=first_state.layers[j],second_state.layers[j]
     if layer_spec.attention=='linear_attention':same=x.tokens_seen==y.tokens_seen and torch.equal(x.recurrent,y.recurrent) and all(torch.equal(u,w) for u,w in zip(x.conv,y.conv,strict=True))
     else:same=x.tokens_seen==y.tokens_seen and all(torch.equal(getattr(x,k),getattr(y,k)) for k in ('latent','compressed','tail','tail_positions'))
     state_equal[str(j)]=bool(same)
    report['state_equal']=state_equal;assert len(state_equal)==45 and all(state_equal.values())
    report['cache_before_close']={k:v for k,v in session.stats().items() if k!='keys_lru'};assert all(report['logit_equal']) and report['generated_equal'] and report['prompt_argmax_equal'] and report['cache_before_close']['used_bytes']<=128*1024**2 and not report['cache_before_close']['active_lease']
   finally:
    for adapter in adapters:adapter.cache_session=None
    torch_npu.npu.synchronize()
  report['cache_after_close']={k:v for k,v in session.stats().items() if k!='keys_lru'};assert report['cache_after_close']['closed'] and report['cache_after_close']['used_bytes']==0 and not report['cache_after_close']['active_lease']
  generated=second_gen;report['generated_ids']=generated;report['stop_set_reached']=generated[-1] in stops;report['stop_token_id']=generated[-1] if report['stop_set_reached'] else None;report['stop_marker']=tok.decode([generated[-1]],skip_special_tokens=False) if report['stop_set_reached'] else None;report['tokenizer_eos_reached']=generated[-1]==tok.eos_token_id;report['cap_reached']=len(generated)==16 and not report['stop_set_reached'];report['generated_text']=tok.decode(generated,skip_special_tokens=False)
  report['semantic_pass']=report['generated_equal'] and report['prompt_argmax_equal'] and report['stop_token_id']==154827 and report['stop_marker']=='<|user|>' and tok.decode(generated[:-1],skip_special_tokens=False).rsplit('</think>',1)[-1].strip()=='17' and not report['tokenizer_eos_reached'] and not report['cap_reached'];assert report['semantic_pass']
  if hasattr(torch_npu.npu,'max_memory_allocated'):report['peak_npu_allocated_bytes']=int(torch_npu.npu.max_memory_allocated())
  dist.barrier();torch_npu.npu.synchronize()
 report['sparse_mlp_call_sizes']=sparse_rows;report['checks']={'sparse_moe_tokenwise':len(sparse_rows)>=80 and all(n==1 for n in sparse_rows),'full_logits':bool(report['logit_equal']) and all(report['logit_equal']),'generated':report['generated_equal'],'prompt_argmax':report['prompt_argmax_equal'],'semantic':report['semantic_pass'],'state_equal':len(report['state_equal'])==45 and all(report['state_equal'].values()),'archive_restore_all45':report['archive_tokens_seen']==42 and len(report['archive_state_equal'])==45 and all(report['archive_state_equal'].values()),'cache_released':report['cache_after_close']['closed'] and report['cache_after_close']['used_bytes']==0 and not report['cache_after_close']['active_lease']};report['status']='passed' if all(report['checks'].values()) else 'failed';report['phase']='complete'
except BaseException as e:report.update(status='failed',failure=repr(e));raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('RESULT',json.dumps(report),flush=True)
