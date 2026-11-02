"""Bounded four-rank installed reference; operator owns allocation and go barrier."""
from __future__ import annotations
import ast,datetime,hashlib,importlib.metadata,json,os,pathlib,platform,resource,sys,time,types
resource.setrlimit(resource.RLIMIT_AS,(192*1024**3,192*1024**3))
import torch,torch_npu
rank=int(sys.argv[1]);die=int(sys.argv[2]);assert die>=0 and 0<=rank<4
root=pathlib.Path(os.environ['GLM_EXPERIMENT_RUN_DIR']);out=root/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==str(die) and torch_npu.npu.device_count()==1
torch.set_num_threads(2);torch_npu.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False);torch.ones(1,device='npu');torch_npu.npu.synchronize();(out/'ready').write_text(str(os.getpid()))
for _ in range(1800):
 if (out/'go').exists():break
 time.sleep(.1)
else:raise RuntimeError('host PID/device mapping not verified')
W=pathlib.Path(os.environ['GLM_MODEL_DIR']);upstream=pathlib.Path(os.environ['GLM_REFERENCE_ROOT']);P=upstream/'quantization/methods/w8a8/w8a8_dynamic.py';request=json.loads((root/'request.json').read_text());ids=request['prompt_ids'];cap=request['max_new_tokens'];context=request['context_tokens'];expected=request['expected_final'];assert 1<=cap<=64 and len(ids)+cap<=context<=128;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report={'rank':rank,'die':die,'architecture':platform.machine(),'device':torch_npu.npu.get_device_name(0),'index_sha256':sha(W/'quant_model_weights.safetensors.index.json'),'config_sha256':sha(W/'config.json'),'upstream_quant_sha256':sha(P),'probe_sha256':sha(__file__),'versions':{n:importlib.metadata.version(n) for n in ('torch','torch-npu')},'status':'running','scope':'actual installed vLLM-Ascend offline four-rank external launcher; current checkpoint and eager prefetch; compare bounded complete greedy output with retained DeepInfer baseline'}
def save():
 p=out/'result.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(report,indent=2)+'\n');q.replace(p)
save()
try:
 assert report['index_sha256']=='dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4' and report['config_sha256']=='33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943'
 model_source=upstream/'models/glm5next/model.py';assert sha(model_source)=='2a9381fa5897578bf4922b4b2bef99635dbc58236cd0f31fe5f93664b7252bbe'
 offload_source=upstream/'model_executor/offloader/prefetch.py';assert sha(offload_source)=='b5a641d00c4e8cc9b825f35100394f831a30b24ac574d727249edb08723b9de7'
 os.environ['VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY']='1';os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['LOCAL_RANK']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 cache=str(root/'cache')
 for key in ('HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):os.environ[key]=cache+'/'+key.lower()
 for key in ('HF_HOME','XDG_CACHE_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):pathlib.Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
 report.update(phase='import',prompt_ids=ids,baseline_first_token=None,expected_final=expected,generation_cap=cap,source_note='four-way TP, external launcher, actual upstream model; explicit prefetch backend, host pinning disabled, group=2, count=1, step=1, w13/w2 weights');save()
 from vllm import LLM,SamplingParams
 from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
 original_load_model=NPUModelRunner.load_model
 def synchronized_load_model(self,*args,**kwargs):
  value=original_load_model(self,*args,**kwargs)
  sync=root/'post-load-rendezvous';sync.mkdir(exist_ok=True)
  (sync/f'rank{rank}').write_text(str(os.getpid()))
  deadline=time.monotonic()+600
  while not all((sync/f'rank{i}').is_file() for i in range(4)):
   if time.monotonic()>=deadline:raise TimeoutError('post-load rendezvous timed out')
   time.sleep(0.5)
  return value
 NPUModelRunner.load_model=synchronized_load_model
 report.update(phase='load');save()
 llm=LLM(model=str(W),tensor_parallel_size=4,distributed_executor_backend='external_launcher',enforce_eager=True,offload_backend='prefetch',offload_group_size=2,offload_num_in_group=1,offload_prefetch_step=1,offload_params={'w13_weight','w2_weight'},max_model_len=context,max_num_seqs=1,gpu_memory_utilization=0.85,trust_remote_code=False,disable_log_stats=True)
 report.update(phase='generate');save()
 outputs=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=cap,logprobs=5,stop_token_ids=[154820,154827,154829],include_stop_str_in_output=True),use_tqdm=False)
 assert len(outputs)==1 and len(outputs[0].outputs)==1
 completion=outputs[0].outputs[0]
 generated=list(completion.token_ids);assert 0<len(generated)<=cap
 logprobs=completion.logprobs
 tops=[[{'token_id':int(k),'logprob':float(v.logprob)} for k,v in step.items()] for step in logprobs] if logprobs else []
 stop_reason=completion.stop_reason; finish_reason=completion.finish_reason
 text=completion.text;reason,final=(text.rsplit('</think>',1) if '</think>' in text else (text,None));report.update(reasoning_text=reason,final_text=final,reason_final_boundary_present=final is not None,final_answer_exact=final is not None and final.strip()==expected if expected is not None else None,stop_token_id=stop_reason if isinstance(stop_reason,int) else None,tokenizer_eos_reached=stop_reason==154820)
 checks={'expected_final':expected is None or (final is not None and final.strip()==expected),'natural_stop':finish_reason=='stop' and stop_reason in (154820,154827,154829),'bounded_output':0<len(generated)<=cap,'finite_top_logprobs':len(tops)==len(generated) and all(__import__('math').isfinite(v['logprob']) for step in tops for v in step)}
 report.update(generated_ids=generated,generated_text=completion.text,top_logprobs=tops,finish_reason=finish_reason,stop_reason=stop_reason,tokenizer_eos_id=154820,stop_ids=[154820,154827,154829],cap_reached=finish_reason=='length',baseline_ids=[],baseline_comparison='compare separately with the matched adaptation run',checks=checks,peak_npu_allocated_bytes=torch_npu.npu.max_memory_allocated(),peak_npu_reserved_bytes=torch_npu.npu.max_memory_reserved(),status='passed' if all(checks.values()) else 'failed',phase='complete',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert all(checks.values())
except BaseException as e:report.update(status='failed',failure=repr(e),memory_status=[x for x in pathlib.Path('/proc/self/status').read_text().splitlines() if x.startswith(('VmPeak:', 'VmSize:', 'VmRSS:'))],address_space_limit=resource.getrlimit(resource.RLIMIT_AS));save();raise
finally:torch_npu.npu.synchronize();print('RESULT',json.dumps(report),flush=True)
