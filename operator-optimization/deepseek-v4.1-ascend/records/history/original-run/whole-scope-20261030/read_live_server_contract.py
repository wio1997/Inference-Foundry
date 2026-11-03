#!/usr/bin/env python3
"""Read documented JSON routes from the owned live service; keep whitelisted fields.

No inference requests, target imports, debug mutation or process injection.
"""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

CODE=r'''
import datetime,json,urllib.request,urllib.error
o=urllib.request.build_opener(urllib.request.ProxyHandler({}));base='http://127.0.0.1:8172'
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'owned service read-only advertised route and JSON config; no worker-object identity claim','no_inference_or_npu_work':True}
def read(path):
 with o.open(base+path,timeout=10) as r:return r.status,json.load(r)
try:
 status,schema=read('/openapi.json');paths=schema.get('paths',{})
 out['openapi_status']=status;out['advertised_server_info']='/server_info' in paths
 if '/server_info' not in paths:
  out['status']='SERVER_INFO_NOT_ADVERTISED'
 else:
  out['server_info_parameters']=[{'name':x.get('name'),'in':x.get('in'),'schema':x.get('schema')} for x in paths['/server_info'].get('get',{}).get('parameters',[])]
  status,data=read('/server_info?config_format=json');out['http_status']=status
  out['response_keys']=list(data) if isinstance(data,dict) else None
  configs=data.get('vllm_config',data) if isinstance(data,dict) else None
  if isinstance(configs,str):
   try:configs=json.loads(configs)
   except json.JSONDecodeError:configs=None
  if not isinstance(configs,dict):out['status']='JSON_CONFIG_NOT_STRUCTURED_NO_REPR_SAVED'
  else:
   whitelist={
    'model_config':['model','dtype','quantization','runner','model_impl','max_model_len','served_model_name','enable_prompt_embeds','is_multimodal_model','skip_mm_profiling'],
    'parallel_config':['tensor_parallel_size','data_parallel_size','pipeline_parallel_size','prefill_context_parallel_size','decode_context_parallel_size','enable_expert_parallel','enable_batch_sharded_sampling','disable_custom_all_reduce'],
    'speculative_config':['method','num_speculative_tokens','enforce_eager','rejection_sample_method','draft_sample_method','parallel_drafting','enable_adaptive_verification','use_local_argmax_reduction'],
    'cache_config':['block_size','cache_dtype','enable_prefix_caching','prefix_caching_hash_algo','gpu_memory_utilization'],
    'scheduler_config':['max_num_batched_tokens','max_num_seqs','async_scheduling','enable_chunked_prefill','disable_chunked_mm_input','use_v2_model_runner'],
    'compilation_config':['mode','cudagraph_mode','cudagraph_capture_sizes','backend','custom_ops'],
    'engram_config':['cpu_offload','dp_shared_memory'],
    'additional_config':['enable_fused_mc2','enable_dsa_cp','enable_flashcomm1','enable_shared_expert_dp','rejection_sampler_config','lm_head_tensor_parallel_size','enable_lm_head_tensor_parallel']}
   out['config_section_names']=list(configs);out['config']={}
   for section,keys in whitelist.items():
    value=configs.get(section)
    if isinstance(value,dict):out['config'][section]={k:value[k] for k in keys if k in value}
    else:out['config'][section]={'structured':False,'is_null':value is None}
   out['status']='WHITELISTED_LIVE_CONFIG_READ'
except (urllib.error.URLError,ValueError,KeyError,TypeError) as e:
 out['status']='READ_FAILED';out['error_type']=type(e).__name__
 if isinstance(e,urllib.error.HTTPError):out['http_status']=e.code
print(json.dumps(out,indent=2))
'''

payload=base64.b64encode(CODE.encode()).decode()
code="import base64;exec(base64.b64decode('"+payload+"'))"
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-20261028','python3','-c',code])
root=Path(__file__).resolve().parent
with (root/'live-server-contract.json').open('w') as out,(root/'live-server-read.stderr').open('w') as err:
    p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=40,check=False)
print('returncode',p.returncode)
sys.exit(p.returncode)
