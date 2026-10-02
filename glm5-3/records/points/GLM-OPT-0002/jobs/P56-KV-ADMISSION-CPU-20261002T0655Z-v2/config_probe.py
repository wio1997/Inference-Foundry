import sys,json,dataclasses
from pathlib import Path
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.openai.api_server import validate_api_server_args
from vllm.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.engine.arg_utils import EngineArgs
argv=json.loads(sys.argv[1]);parser=FlexibleArgumentParser();sub=parser.add_subparsers(dest="subparser");cmd=ServeSubcommand();cmd.subparser_init(sub);args=parser.parse_args(argv[3:]);cmd.validate(args)
if args.model_tag is not None:args.model=args.model_tag
if args.tool_parser_plugin:ToolParserManager.import_tool_parser(args.tool_parser_plugin)
if args.reasoning_parser_plugin:ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)
validate_api_server_args(args)
config=EngineArgs.from_cli_args(args).create_engine_config()
assert config.parallel_config.tensor_parallel_size==16 and config.parallel_config.decode_context_parallel_size==16 and config.parallel_config.data_parallel_size==2
assert args.tool_call_parser=="glm47_contract"
capture_geometry=None
if config.kv_transfer_config.kv_role=="kv_consumer":
 import copy
 cc=config.compilation_config
 assert config.speculative_config.num_speculative_tokens==5
 assert cc.cudagraph_capture_sizes==[6]and cc.max_cudagraph_capture_size==6
 negative_cc=copy.deepcopy(cc);negative_cc.cudagraph_capture_sizes=[1,2,4];negative_cc.max_cudagraph_capture_size=4
 try:negative_cc.adjust_cudagraph_sizes_for_spec_decode(6,16)
 except ValueError as err:
  assert "No valid cudagraph sizes after rounding to multiple of 6"in str(err)
 else:raise AssertionError("NativeK5Graphnegativegeometrynotrejected")
 cc.adjust_cudagraph_sizes_for_spec_decode(6,16)
 assert cc.cudagraph_capture_sizes==[6]and cc.max_cudagraph_capture_size==6
 capture_geometry={"native_adjust_method":True,"sizes":cc.cudagraph_capture_sizes,"max":cc.max_cudagraph_capture_size,"uniform_decode_query_len":6,"negative_run51_rejected":True}

# CPU import and exact source validation before removing the prior task cohort.
from vllm.config import set_current_vllm_config
with set_current_vllm_config(config):
 from vllm_ascend.utils import register_ascend_customop
 register_ascend_customop(config)
 from sfa_workspace_worker import SFAWorkspaceWorker
 from coupled_atomic_mq_worker import CoupledAtomicMQWorker
 from sfa_workspace_guard import proof
 assert issubclass(SFAWorkspaceWorker,CoupledAtomicMQWorker)
 workspace_proof=proof()
 import os,hashlib
 from vllm_ascend.utils import get_hccl_config_for_pg_options
 assert os.environ.get("HCCL_BUFFSIZE")=="512"
 pg_policy={name:get_hccl_config_for_pg_options(name)for name in["mc2","tp","ep","dcp","dp"]}
 assert pg_policy["mc2"]is None and all(pg_policy[name]["hccl_buffer_size"]==200 for name in["tp","ep","dcp"])
 HCCL_buffer_policy=dict(actual_env_MB=512,native_pg_options=pg_policy,native_utils_sha256=hashlib.sha256(Path("/vllm-workspace/vllm-ascend/vllm_ascend/utils.py").read_bytes()).hexdigest(),SDK_communicators_created=0)
# Reconstruct exact native observed cache specs with only whitelisted scalar/dtype values.
import ast,copy,torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs,KVCacheGroupSpec,KVCacheConfig,KVQuantMode
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec,AscendMLAAttentionSpec
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups,get_kv_cache_capacity,generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
register_all_kvcache_specs(config)
from vllm.v1.request import Request
from vllm import SamplingParams
meta=json.loads(Path(sys.argv[2]).read_text())
constructors={"AscendSFAIndexerCacheSpec":AscendSFAIndexerCacheSpec,"AscendMLAAttentionSpec":AscendMLAAttentionSpec}
attrs={"torch.bfloat16":torch.bfloat16,"torch.int8":torch.int8,"KVQuantMode.NONE":KVQuantMode.NONE}
specs={}
for entry,indexes in meta["kv_group2layeridx"].values():
 text=entry["kv_cache_spec"]["repr"].replace("<KVQuantMode.NONE: 0>","KVQuantMode.NONE");expr=ast.parse(text,mode="eval").body
 assert isinstance(expr,ast.Call)and isinstance(expr.func,ast.Name)and expr.func.id==entry["kv_cache_spec_type"]and expr.func.id in constructors and not expr.args
 kwargs={}
 for kw in expr.keywords:
  assert kw.arg is not None
  if isinstance(kw.value,ast.Constant):value=kw.value.value
  elif isinstance(kw.value,ast.Attribute):
   name=ast.unparse(kw.value);assert name in attrs;value=attrs[name]
  else:raise AssertionError("Unapproved native metadata value")
  kwargs[kw.arg]=value
 spec=constructors[expr.func.id](**kwargs)
 assert entry["kv_cache_group_id"]==0
 for layer in entry["layer_names"]:assert layer not in specs;specs[layer]=spec
uniform=UniformTypeKVCacheSpecs.from_specs(specs);assert uniform is not None and uniform.block_size==128
group=KVCacheGroupSpec(layer_names=list(specs),kv_cache_spec=uniform)
plans={}
for mb in [966367641,1073741824]:
 kv=get_kv_cache_config_from_groups(config,[group],mb);plans[str(mb)]=dict(num_blocks=kv.num_blocks,page_size_bytes=uniform.page_size_bytes,capacity=get_kv_cache_capacity(config,kv))
assert plans["966367641"]["num_blocks"]==41 and plans["1073741824"]["num_blocks"]>41
rows=[]
for nblocks,ntokens in[(41,81920),(41,81921),(41,81931),(41,81932),(42,81931),(42,81932),(plans["1073741824"]["num_blocks"],81932)]:
 kv=generate_scheduler_kv_cache_config([KVCacheConfig(num_blocks=nblocks,kv_cache_tensors=[],kv_cache_groups=[group])])
 manager=KVCacheManager(kv,max_model_len=81933,scheduler_block_size=2048,hash_block_size=2048,max_in_flight_tokens=config.max_in_flight_tokens,enable_caching=config.cache_config.enable_prefix_caching,use_eagle=False,dcp_world_size=16,pcp_world_size=1,watermark=config.scheduler_config.watermark)
 request=Request(request_id="fixture-"+str(nblocks)+"-"+str(ntokens),prompt_token_ids=[1]*ntokens,sampling_params=SamplingParams(max_tokens=1,temperature=0),pooling_params=None)
 free=manager.block_pool.get_num_free_blocks()
 need=manager.coordinator.get_num_blocks_to_allocate(request_id=request.request_id,num_tokens=ntokens,new_computed_blocks=manager.empty_kv_cache_blocks.blocks,num_encoder_tokens=0,total_computed_tokens=0,num_local_computed_tokens=0,num_tokens_main_model=ntokens,apply_admission_cap=True)
 allocated=manager.allocate_slots(request,1024,num_lookahead_tokens=1,full_sequence_must_fit=config.scheduler_config.scheduler_reserve_full_isl)
 row=dict(num_blocks=nblocks,null_reserved=1,free_before=free,prompt_tokens=ntokens,required_fullsequence_blocks=need,native_admitted=allocated is not None,full_sequence_must_fit=config.scheduler_config.scheduler_reserve_full_isl,effective_block_size=manager.coordinator.block_size)
 assert row["effective_block_size"]==2048 and free==nblocks-1 and row["native_admitted"]==(need<=free)
 rows.append(row)
assert rows[0]["native_admitted"]and all(not x["native_admitted"]for x in rows[1:4])and all(x["native_admitted"]for x in rows[4:])
print("ADMISSION_RECEIPT "+json.dumps(dict(metadata_engine_id=meta["engine_id"],layer_specs=len(specs),cache_page_bytes=uniform.page_size_bytes,plans=plans,rows=rows,config_max_in_flight_tokens=config.max_in_flight_tokens,watermark=config.scheduler_config.watermark,NPU_workers=0,models=0,requests=0,limits=["Actualnative metadata specs/manager/planner onCPU with synthetic promptID list, not productiontokenizer/scheduler livecapture","Matches observedemptyP pool41/40free andcapacitywait; noGPUpeak/dualfit/newE2E/capacity claim","Nativeguards/operators unchanged; candidate P1GiB planonly"])))

try:
 args.tool_call_parser="glm48_contract";validate_api_server_args(args)
except KeyError:negative=True
else:raise AssertionError("nativeAPIfailedtorejectinvalidregisteredparser")
print(json.dumps({"event":"full_native_CLI_API_config_valid","tool_parser":"glm47_contract","invalid_glm48_contract_rejected":negative,"kv_role":config.kv_transfer_config.kv_role,"kv_port":config.kv_transfer_config.kv_port,"max_model_len":config.model_config.max_model_len,"TP":16,"DCP":16,"DP":2,"speculative_tokens":config.speculative_config.num_speculative_tokens,"capture_geometry":capture_geometry,"unused_SFA_workspace_proof":workspace_proof,"HCCL_buffer_policy":HCCL_buffer_policy,"NPU_workers_started":0,"models":0,"requests":0}))

