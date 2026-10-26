import pathlib,json,os,hashlib,inspect,importlib
R=pathlib.Path('/run-evidence');node=int(os.environ['SGLANG_HOST_IP'].rsplit('.',1)[1]);assert node in (46,47)
role='prefill' if node==46 else 'decode'
import torch,torch_npu
from sglang.srt.server_args import prepare_server_args
from sglang.srt.arg_groups.overrides import resolved_view,mamba_extra_buffer_of
native_bindings={}
for rel,sha in json.loads((R/'native-source-pins.json').read_text()).items():
 p=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt')/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;native_bindings[rel]=sha
args=prepare_server_args(json.loads((R/f'argv{node}.json').read_text()));args.resolve_once();args.check_server_args();v=resolved_view(args)
fields=['served_model_name','context_length','max_running_requests','max_total_tokens','chunked_prefill_size','max_prefill_tokens','tp_size','dp_size','ep_size','kv_cache_dtype','dtype','mamba_ssm_dtype','mamba_radix_cache_strategy','mamba_max_states_per_path','max_mamba_cache_size','mamba_track_interval','disable_radix_cache','disable_cuda_graph','disable_overlap_schedule','disaggregation_mode','disaggregation_transfer_backend','num_reserved_decode_tokens']
resolved={k:getattr(v,k) for k in fields};assert v.tp_size==16 and v.dp_size==v.ep_size==1 and v.context_length==133120 and v.max_running_requests==32 and v.max_total_tokens==2580480
assert v.chunked_prefill_size==v.max_prefill_tokens==4096 and v.num_reserved_decode_tokens==640
assert v.disable_radix_cache is (node==47) and mamba_extra_buffer_of(v) is (node==46)
assert v.mamba_radix_cache_strategy=='extra_buffer' and v.mamba_track_interval==128 and v.max_mamba_cache_size==256 and v.mamba_max_states_per_path==1
assert v.disable_cuda_graph and v.disable_overlap_schedule and v.disaggregation_mode==role and v.disaggregation_transfer_backend=='ascend' and v.quantization=='modelslim' and v.dtype==v.kv_cache_dtype=='bfloat16' and v.mamba_ssm_dtype=='float32'
assert getattr(v,'speculative_algorithm',None) in (None,'','NONE') and getattr(v,'speculative_num_steps',None) in (None,0)
assert os.environ['FLASH_PD_COMPRESSION_ABI_AUDIT']==os.environ['FLASH_PD_SCORE_ABI_AUDIT']==os.environ['FLASH_MHC_FIXED_ROWS']==os.environ['FLASH_PD_STATE_AUDIT']==os.environ['FLASH_PD_PREFILL_COHORT_GATE']=='0'
assert os.environ['FLASH_KDA_PREFIX_ONCE_AUDIT']=='0' and os.environ['FLASH_PD_PERF_MODE']=='1'
assert not (R/'request-context-enabled').exists() and os.environ['FLASH_PD_STARTUP_DTYPE_META']=='1'
assert os.environ['FLASH_SDMA_VERIFY_BYTES']=='0' and os.environ['FLASH_PD_LAYOUT_ONCE']=='0'
assert os.environ['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma' and os.environ['ASCEND_MF_STORE_URL'].startswith('tcp://')
from sglang.srt.environ import envs
wait_budgets={'bootstrap':envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get(),'waiting':envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()}
assert wait_budgets=={'bootstrap':10800,'waiting':10800}
manifest=json.loads((R/'combined-manifest.json').read_text());assert manifest['status']=='R247_CANDIDATE_DELTA_NOT_ORIGINAL_46' and len(manifest['files'])==47
imports={}
for row in manifest['files']:
 name=row['target'].removesuffix('.py').replace('/','.');m=importlib.import_module(name);p=pathlib.Path(inspect.getfile(m));assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];imports[name]={'path':str(p),'sha256':row['sha256']}
ops=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt/hardware_backend/npu/ops')
score=(ops/'flash_kpool_score.py').read_text();caller=(ops/'flash_kpool_indexer_npu.py').read_text();query_parallel=(ops/'flash_indexer_query_parallel.py').read_text()
assert 'do_not_specialize=["requests", "pools_count", "blocks_count", "TABLE_SR"]' in score
assert 'do_not_specialize=["token_offset", "rows", "max_pool_seq_len", "num_reqs", "num_cache_blocks", "table_stride_r"]' in score
assert 'TABLE_SR: tl.constexpr' not in score and 'table_stride_r: tl.constexpr' not in score
assert 'query_parallel_indexer(' in caller and 'group=parallel.attn_tp_group' in caller
assert 'rank=parallel.attn_tp_rank' in caller and 'world_size=parallel.attn_tp_size' in caller
assert 'prefill=forward_batch.forward_mode.is_extend()' in caller
assert 'all_gather' in query_parallel and 'torch.int32' in query_parallel and 'glm5_next_lightning_indexer_triton' in query_parallel
runtime=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt/runtime_context.py').read_text()
assert all(token in runtime for token in ('attn_tp_group','attn_tp_rank','attn_tp_size'))
model=pathlib.Path(v.model_path)
model_metadata={}
for filename,sha in {'config.json':'33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943',
                     'quant_model_description.json':'9aae1f3c7657115b7e24294bf6d668f6ea062e2c8fe9eccad4a3862284fe665f',
                     'tokenizer_config.json':'98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc',
                     'tokenizer.json':'19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'}.items():
 assert hashlib.sha256((model/filename).read_bytes()).hexdigest()==sha,filename;model_metadata[filename]=sha
from sglang.srt.models.glm5_next import _FLASH_MHC_FIXED_ROWS
from sglang.kernels.ops.layernorm import mhc
assert _FLASH_MHC_FIXED_ROWS is False and not mhc._use_tilelang_mhc_pre() and not mhc._use_tilelang_mhc_post()
assert hashlib.sha256(pathlib.Path(inspect.getfile(mhc.hc_pre)).read_bytes()).hexdigest()=='3cb4b6787b854e2bc9c2163cbec5c7f9ee2783aeea5d6efb5d22e56812c008d5'
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from types import SimpleNamespace
req=SimpleNamespace(origin_input_ids=[0]*80000,output_ids=[],sampling_params=SimpleNamespace(max_new_tokens=600));fake=SimpleNamespace(num_reserved_decode_tokens=640,_rebootstrap_prefill_len=DecodePreallocQueue._rebootstrap_prefill_len);fill=DecodePreallocQueue._pre_alloc_fill_len(req);need=DecodePreallocQueue._required_admission_tokens(fake,req,fill,0,0);assert ((fill+127)//128)*128+640<=80640 and 32*need<=v.max_total_tokens
torch.npu.set_device(0);tiny=torch.empty(1,device='npu');tiny.record_stream(torch.npu.current_stream());torch.npu.synchronize();del tiny
(R/'preflight.json').write_text(json.dumps({'project_id':'glm53-flash','unit':'r247','passed':True,'node':node,'resolved':resolved,'imports':imports,'native_source_bindings':native_bindings,'native_PD_wait_budgets_seconds':wait_budgets,'native_C32_admission_need_each':need,'C32_required_tokens':32*need,'native_mhc_selected':True,'native_mamba_extra_buffer':mamba_extra_buffer_of(v),'role_cache_policy':'Prefillradixextra_buffer' if node==46 else 'Decodeforcednativechunkcache','record_stream_API_supported':True,'native_sdma':{'protocol':os.environ['ASCEND_MF_TRANSFER_PROTOCOL'],'store_url':os.environ['ASCEND_MF_STORE_URL'],'transfer_backend':v.disaggregation_transfer_backend,'byte_verify':os.environ['FLASH_SDMA_VERIFY_BYTES']},'model_metadata_sha256':model_metadata,'candidate_delta':True,'original_46_overlay_unchanged':False,'model_requests':0},indent=2)+'\n');print('R247_B_INDEXER_OPTIMIZED_NATIVE_PREFLIGHT_PASS')
