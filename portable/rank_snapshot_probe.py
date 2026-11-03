"""Current-source four-rank 45-layer archive-v2 snapshot and continuation restore."""
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
rank=int(sys.argv[1]);die=int(sys.argv[2]);port=int(sys.argv[3]);mode='four';assert mode in ('one','four') and die>=0 and 0<=rank<4
root=pathlib.Path(os.environ['GLM_EXPERIMENT_RUN_DIR']);out=root/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==str(die) and torch_npu.npu.device_count()==1
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);torch.ones(1,device='npu');torch_npu.npu.synchronize();(out/'ready').write_text(str(os.getpid()))
for _ in range(1800):
 if (out/'go').exists():break
 time.sleep(.1)
else:raise RuntimeError('host PID/device mapping not verified')
model=pathlib.Path(os.environ['GLM_MODEL_DIR']);source=pathlib.Path(os.environ['GLM_SOURCE_DIR'])/'deepinfer/model_executor/models/glm5_next';lib=pathlib.Path(os.environ['GLM_ASCEND_LIBRARY']);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report={'rank':rank,'die':die,'mode':mode,'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'weight_index_sha256':sha(model/'quant_model_weights.safetensors.index.json'),'config_sha256':sha(model/'config.json'),'template_sha256':sha(model/'chat_template.jinja'),'install_sha256':sha(source/'rank_text_install.py'),'probe_sha256':sha(__file__),'library_sha256':sha(lib),'versions':{n:importlib.metadata.version(n) for n in ('torch','torch-npu','flash-linear-attention-npu-a3')},'status':'running','phase':'setup','scope':'45-layer four-rank 8+5 prompt, archive-v2 prefix snapshot and <=4 greedy tokens; dense gate/up and experts sharded; attention/down/router/embedding/head replicated; experimental short-context DSA; not independent quality or production EP','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
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
 if mode=='four':dist.init_process_group(backend='hccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,timeout=timedelta(seconds=300))
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(root/'hf-cache')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 messages=[{'role':'user','content':'What is 8 plus 5? Reply only with the number.'}];encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,reasoning_effort='low');full_ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded;ids=full_ids;assert ids==[154822,154824,154826,25062,287,29905,371,25,12035,154827,3838,374,220,23,5519,220,20,30,17815,1172,448,279,1372,13,154828,154841]
 stops=sorted(json.loads((model/'generation_config.json').read_text())['eos_token_id']);assert stops==[154820,154827,154829]
 report.update(full_prompt_ids=full_ids,prefix_ids=ids,prefix_is_truncated=len(ids)<len(full_ids),stop_ids=stops,tokenizer_eos_id=tok.eos_token_id,generation_cap=4,phase='assembly');save()
 torch.ops.load_library(str(lib));config=Glm5NextConfig.from_checkpoint(model);reader=ModelSlimCheckpoint(model,max_read_bytes=2*1024**3)
 with torch.inference_mode():
  ops=make_experimental_fp32_expanded_dsa_operations(config,max_total_tokens=32)
  runner=load_native_text_runner(reader,'npu',dsa_operations=ops,expert_policy='selected_per_token',max_expert_checkpoint_bytes=512*1024**2,max_expert_resident_bytes=512*1024**2,vocab_start=0,vocab_stop=config.vocab_size,max_head_rows=config.vocab_size,max_tokens=1,max_context_tokens=32,logical_block_size=128,hc_backend='ascend_installed',kda_backend='ascendc')
  for i,layer in enumerate(runner.stack.layers):
   if config.layers[i].attention=='deepseek_sparse_attention':layer.attention.operation=ops[i].bind(layer.attention.mla,layer.attention.indexer)
  states={i:(layer.attention.initial_state() if config.layers[i].attention=='linear_attention' else state_from_aligned_prefix(torch.empty((0,512),dtype=torch.bfloat16,device='npu'),torch.empty((0,128),dtype=torch.bfloat16,device='npu'),runner.stack.layout)) for i,layer in enumerate(runner.stack.layers)}
  owner='round415-patched-snapshot';report['archive_owner']=owner;report['archive_layout']={'latent_dim':runner.stack.layout.latent_dim,'index_dim':runner.stack.layout.index_dim,'pool_size':runner.stack.layout.pool_size,'topk_tokens':runner.stack.layout.topk_tokens,'logical_block_size':runner.stack.layout.logical_block_size,'lookahead':runner.stack.layout.lookahead};initial=MixedRequestState.capture(owner,config,runner.stack.layout,states)
  def step(q,token,pos):
   z=q.step(torch.tensor([token],dtype=torch.int64),expected_tokens_seen=pos);torch_npu.npu.synchronize();assert torch.isfinite(z.logits).all();return z.logits.detach().cpu().clone()
  def gen(q,last):
   logits=[];generated=[];pos=len(ids)
   for j in range(4):
    token=int(last.argmax(-1).item());generated.append(token)
    if mode=='four':
     value=torch.tensor([token],dtype=torch.int32,device='npu');copies=[torch.empty_like(value) for _ in range(4)];dist.all_gather(copies,value);assert all(int(t.item())==token for t in copies),'rank generation mismatch'
    if token in stops or j==3:break
    last=step(q,token,pos);logits.append(last);pos+=1
   return logits,generated,q.snapshot()
  def run_path(label,archive=False):
   report['phase']=label;save();q=runner.open_request(initial)
   try:
    pre=[]
    for i,t in enumerate(ids):
     pre.append(step(q,t,i));report['prompt_tokens_done']=i+1;save()
    snap=q.snapshot();blob=serialize_snapshot(snap,config,runner.stack.layout,max_tokens=32,max_archive_bytes=512*1024**2) if archive else None
    rest,genids,final=gen(q,pre[-1])
   finally:q.close()
   return pre,rest,genids,final,blob
  report['installed']=install_rank_text_seams(runner,reader,rank=rank if mode=='four' else 0,world_size=4 if mode=='four' else 1,cache_bytes=128*1024**2,max_assembled_bytes=16*1024**2,max_intermediate_rows=12288,max_packed_rows=8)
  assert len(report['installed']['dense_layers'])==3 and len(report['installed']['sparse_layers'])==42
  runner.enable_rank_commit_agreement()
  report['rank_commit_agreement_enabled']=True
  c,d,generated,final_b,blob=run_path('ranked',archive=True)
  report['generated_ids']=generated;report['baseline_generated_ids']=[154842,99366,154827];report['baseline_tokens_equal']=generated==report['baseline_generated_ids']
  report['archive_sha256']=hashlib.sha256(blob).hexdigest();report['archive_bytes']=len(blob);report['archive_schema_version']=torch.load(__import__('io').BytesIO(blob),map_location='cpu',weights_only=True)['version'];assert report['archive_schema_version']==2
  restored=deserialize_snapshot(blob,config,runner.stack.layout,expected_request_id=owner,expected_tokens_seen=len(ids),device='npu',max_tokens=32,max_archive_bytes=512*1024**2)
  report['phase']='restore';save();q=runner.open_request(restored)
  try:e,replay,final_c=gen(q,c[-1])
  finally:q.close()
  report['restore_logits_equal']=[torch.equal(x,y) for x,y in zip(d,e,strict=True)];report['restore_generated_equal']=generated==replay;report['restore_state_equal']=state_eq(final_b,final_c,config)
  report['prompt_argmax']=[int(x.argmax(-1).item()) for x in c];report['generated_text']=tok.decode(generated,skip_special_tokens=False);report['stop_set_reached']=generated[-1] in stops;report['stop_token_id']=generated[-1] if report['stop_set_reached'] else None;report['stop_marker']=tok.decode([generated[-1]],skip_special_tokens=False) if report['stop_set_reached'] else None;report['tokenizer_eos_reached']=generated[-1]==tok.eos_token_id;report['cap_reached']=len(generated)==4 and not report['stop_set_reached']
  report['owners_closed']=len(runner._requests)==0
  assert report['baseline_tokens_equal'] and all(report['restore_logits_equal']) and report['restore_generated_equal'] and all(report['restore_state_equal'].values()) and report['owners_closed']
  if hasattr(torch_npu.npu,'max_memory_allocated'):report['peak_npu_allocated_bytes']=int(torch_npu.npu.max_memory_allocated())
  if mode=='four':dist.barrier();torch_npu.npu.synchronize()
 report['checks']={'baseline_tokens':report['baseline_tokens_equal'],'restore_logits':all(report['restore_logits_equal']),'restore_generated':report['restore_generated_equal'],'restore_states':all(report['restore_state_equal'].values()),'owners_closed':report['owners_closed']};report['status']='passed' if all(report['checks'].values()) else 'failed';report['phase']='complete'
except BaseException as e:report.update(status='failed',failure=repr(e));raise
finally:
 if dist.is_initialized():dist.destroy_process_group()
 report['destroyed']=not dist.is_initialized();report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('RESULT',json.dumps(report),flush=True)
