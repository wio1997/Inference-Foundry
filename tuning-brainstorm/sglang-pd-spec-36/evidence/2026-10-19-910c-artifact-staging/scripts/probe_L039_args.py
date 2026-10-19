"""Actual SGLang argument/model-config resolution in the910C image, metadata only."""
import hashlib,importlib.metadata,importlib.util,json,pathlib,subprocess,sys,traceback
v={'status':'STARTING','roles':[],'model_weights_mounted':False,'model_weights_loaded':False,'PD_MTP_serving_executed':False,'performance':'NOT_MEASURED','source_files':{}}
try:
 root=pathlib.Path('/sgl-workspace/sglang')
 for name in ['python/sglang/srt/server_args.py','python/sglang/srt/configs/model_config.py','python/sglang/srt/arg_groups/pd_disaggregation_hook.py','python/sglang/srt/arg_groups/speculative_hook.py','python/sglang/srt/hardware_backend/npu/utils.py']:
  data=(root/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();p=subprocess.run(['git','-C',str(root),'rev-parse','HEAD:'+name],capture_output=True,text=True,timeout=10);assert p.returncode==0 and blob==p.stdout.strip(),name
  v['source_files'][name]={'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':blob,'matches_HEAD':True}
 v['image_source_commit']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert v['image_source_commit']=='ce0a8fea5f4245839d174b2b8c55692be1810a7f'
 v['packages']={}
 for name in ['torch','torch-npu','sglang','transformers','torchair','sgl-kernel-npu','sglang-router','sgl-model-gateway']:
  try:v['packages'][name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:v['packages'][name]=None
 v['module_specs']={}
 for name in ['llm_datadist','custom_ops','sgl_kernel_npu','mooncake','sglang_router','sgl_model_gateway']:
  try:s=importlib.util.find_spec(name);v['module_specs'][name]=None if s is None else s.origin
  except Exception as e:v['module_specs'][name]=type(e).__name__+': '+str(e)
 from sglang.srt.server_args import prepare_server_args
 from sglang.srt.arg_groups.overrides import resolved_view
 from sglang.srt.configs.model_config import ModelConfig
 for role in ['prefill','decode']:
  row={'role':role}
  try:
   argv=['--model-path','/model','--device','npu','--tp-size','16','--ep-size','16','--dp-size','1','--nnodes','1','--node-rank','0','--quantization','modelslim','--dtype','bfloat16','--kv-cache-dtype','bfloat16','--speculative-algorithm','NEXTN','--speculative-num-steps','3','--speculative-eagle-topk','1','--speculative-num-draft-tokens','4','--disaggregation-mode',role,'--disaggregation-transfer-backend','ascend','--context-length','33792','--chunked-prefill-size','4096','--max-running-requests','3','--mem-fraction-static','0.90','--host','0.0.0.0','--port','39044' if role=='prefill' else '39046','--disaggregation-bootstrap-port','39098']
   row['argv']=argv;args=prepare_server_args(argv);cfg=resolved_view(args)
   names=['model_path','device','tp_size','ep_size','dp_size','pp_size','dcp_size','attn_cp_size','enable_dp_attention','quantization','dtype','kv_cache_dtype','speculative_algorithm','speculative_num_steps','speculative_eagle_topk','speculative_num_draft_tokens','speculative_draft_model_path','speculative_draft_model_quantization','disaggregation_mode','disaggregation_transfer_backend','context_length','chunked_prefill_size','max_running_requests','mem_fraction_static','attention_backend','prefill_attention_backend','decode_attention_backend','moe_runner_backend','moe_a2a_backend','page_size','sampling_backend','cuda_graph_config','disable_overlap_schedule','disable_radix_cache','disaggregation_decode_extra_slots']
   row['resolved']={n:getattr(cfg,n) for n in names if hasattr(cfg,n)}
   assert cfg.disaggregation_mode==role and cfg.device=='npu' and cfg.speculative_algorithm is not None and cfg.speculative_num_steps==3 and cfg.speculative_eagle_topk==1 and cfg.speculative_num_draft_tokens==4
   row['models']={}
   for draft in [False,True]:
    model=ModelConfig.from_server_args(args,is_draft_model=draft);hf=model.hf_config
    row['models']['draft' if draft else 'target']={'architecture':hf.architectures,'num_hidden_layers':hf.num_hidden_layers,'dtype':str(model.dtype),'quantization':getattr(model,'quantization',None),'context_len':getattr(model,'context_len',None),'is_draft_model':getattr(model,'is_draft_model',None),'num_nextn_predict_layers':getattr(hf,'num_nextn_predict_layers',None),'vocab_size':getattr(hf,'vocab_size',None)}
   row['status']='PASS'
  except BaseException as e:row['status']='UNAVAILABLE';row['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
  v['roles'].append(row)
 v['status']='PASS' if all(x['status']=='PASS' for x in v['roles']) else 'UNAVAILABLE'
except BaseException as e:v['status']='UNAVAILABLE';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print('L039_RESULT='+json.dumps(v,default=str));sys.exit(0 if v['status']=='PASS' else 1)
