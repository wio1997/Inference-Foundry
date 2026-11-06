"""Bounded two-owner four-rank greedy CLI worker. Operator owns allocation and go barrier."""
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
rank=int(sys.argv[1]);die=int(sys.argv[2]);port=int(sys.argv[3]);mode=sys.argv[4];world_size=int(os.environ['WORLD_SIZE']);assert mode == 'bounded' and world_size==4 and die>=0 and 0<=rank<world_size;injected_rank=min(2,world_size-1)
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
 request=json.loads((root/'request.json').read_text());assert request['mode']=='two_owner_cli' and request['max_requests']==2 and request['kda_backend']=='ascendc' and 1<=request['context_tokens']<=64 and 1<=request['max_new_tokens']<=16 and len(request['cases'])==2
 report.update(phase='assembly',scope='45-layer all-KDA two-owner bounded greedy CLI; serialized steps; no quality or throughput claim');save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=request['context_tokens'])
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=40,max_context_tokens=request['context_tokens'],logical_block_size=128,hc_backend='ascend_installed',kda_backend='ascendc',max_requests=2)
  report['request_owner_limit']=runner.max_requests;assert runner.max_requests==2
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  report['installed']=install_rank_text_seams(runner,reader,rank=rank,world_size=world_size,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8,max_dense_input_tokens=40)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement();selected=[]
  for layer in runner.stack.layers:
   if layer.attention_kind=='linear_attention':
    layer.attention.backend='ascendc_chunk_prefill';layer.attention.reference_sharded_projection=True;layer.native_chunk_prefill=True;layer.native_chunk_max_tokens=40;selected.append(layer.layer_id)
  assert selected[:3]==[0,1,2] and 4 in selected and len(selected)>4
  assert selected==[i for i,s in enumerate(config.layers) if s.attention=='linear_attention']
  report['selected_layers']=selected
  import deepinfer.model_executor.models.glm5_next.chunk_prefill as chunkmod
  chunk_calls=[];chunk_original=chunkmod.chunk_prefill_sharded
  def observed_chunk(*a,**kw):chunk_calls.append(len(a[0]));return chunk_original(*a,**kw)
  chunkmod.chunk_prefill_sharded=observed_chunk
  runner.enable_rank_batch_commit(max_tokens=40,layer_major_dense_kda=True);report['rank_commit_agreement_enabled']=True;report['kda_backend']='ascendc'
  sparse_rows=[];sparse_mlp=runner.stack.layers[4].mlp;old_sparse_forward=sparse_mlp.forward
  def observed_sparse(x):sparse_rows.append(len(x));return old_sparse_forward(x)
  sparse_mlp.forward=observed_sparse
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
  stops=sorted(json.loads((model/'generation_config.json').read_text())['eos_token_id']);assert stops==[154820,154827,154829]
  cases=request['cases']
  assert len({case['request_id'] for case in cases})==2
  for case in cases:
   encoded=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,reasoning_effort='low');actual=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
   assert actual==case['prompt_ids'] and len(actual)+request['max_new_tokens']<=request['context_tokens']
  qa=runner.open_request(MixedRequestState.capture(cases[0]['request_id'],config,runner.stack.layout,states));qb=None
  try:
   qb=runner.open_request(MixedRequestState.capture(cases[1]['request_id'],config,runner.stack.layout,states));report['concurrent_owners']=len(runner._requests);assert report['concurrent_owners']==2
   requests=[qa,qb];logits=[None,None];positions=[0,0];generated=[[],[]];prompt_argmax=[[],[]];prefill_groups=[[],[]];done=[False,False];rank_agreement=True
   for idx in range(2):
    report['phase']='prefill_'+str(idx);save();ids=cases[idx]['prompt_ids']
    for start in range(0,len(ids),40):
     part=ids[start:start+40]
     if len(part)==1:
      for i in selected:runner.stack.layers[i].attention.chunk_prefill_single=True
     try:z=requests[idx].step(torch.tensor(part,dtype=torch.int64),expected_tokens_seen=start)
     finally:
      for i in selected:runner.stack.layers[i].attention.chunk_prefill_single=False
     torch_npu.npu.synchronize();assert z.token_stop==start+len(part) and bool(torch.isfinite(z.logits).all());prompt_argmax[idx].extend(z.logits.argmax(dim=-1).cpu().tolist());prefill_groups[idx].append(len(part));logits[idx]=z.logits[-1:]
    positions[idx]=len(ids)
   report['phase']='interleaved_generation';save()
   for turn in range(request['max_new_tokens']):
    for idx in range(2):
     if done[idx]:continue
     token=int(logits[idx].argmax(-1).item());local=torch.tensor([token],dtype=torch.int64,device='npu');gathered=[torch.empty_like(local) for _ in range(world_size)];dist.all_gather(gathered,local);rank_agreement=rank_agreement and all(int(v.item())==token for v in gathered);assert rank_agreement
     generated[idx].append(token)
     if token in stops:done[idx]=True;continue
     if len(generated[idx])>=request['max_new_tokens']:done[idx]=True;continue
     z=requests[idx].step(torch.tensor([token],dtype=torch.int64),expected_tokens_seen=positions[idx]);torch_npu.npu.synchronize();assert bool(torch.isfinite(z.logits).all());logits[idx]=z.logits;positions[idx]+=1
    if all(done):break
   report['generated_ids']=generated;report['prompt_argmax']=prompt_argmax;report['prefill_groups']=prefill_groups;report['rank_agreement']=rank_agreement;report['stop_ids']=stops;report['tokenizer_eos_id']=tokenizer.eos_token_id
   report['final_texts']=[];report['stop_set_reached']=[];report['cap_reached']=[]
   for idx in range(2):
    stopped=bool(generated[idx]) and generated[idx][-1] in stops;report['stop_set_reached'].append(stopped);report['cap_reached'].append(len(generated[idx])>=request['max_new_tokens'] and not stopped);body=generated[idx][:-1] if stopped else generated[idx];text=tokenizer.decode(body,skip_special_tokens=False);report['final_texts'].append(text.rsplit('</think>',1)[1].strip() if '</think>' in text else None)
   report['tokenizer_eos_reached']=[bool(g) and g[-1]==tokenizer.eos_token_id for g in generated];report['stop_markers']=[tokenizer.decode([g[-1]],skip_special_tokens=False) if g and g[-1] in stops else None for g in generated];report['checks']={'two_live_owners':report['concurrent_owners']==2,'prompt_ids_verified':True,'rank_agreement':rank_agreement,'both_stop_without_cap':all(report['stop_set_reached']) and not any(report['cap_reached']),'final_answers_valid':all(report['final_texts'][i] is not None and bool(report['final_texts'][i]) and (cases[i]['expected_final'] is None or report['final_texts'][i]==cases[i]['expected_final']) for i in range(2))};assert all(report['checks'].values())
  finally:
   if qb is not None:qb.close()
   qa.close()
  report['request_owners_remaining']=len(runner._requests);assert report['request_owners_remaining']==0
 report.update(status='passed',phase='complete');save()
except BaseException as e:report.update(status='failed',failure=repr(e));save();raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();save()
