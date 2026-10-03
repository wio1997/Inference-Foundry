from pathlib import Path
import sys,json,os,hashlib,traceback
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.openai.api_server import validate_api_server_args
from vllm.tool_parsers import ToolParserManager
from vllm.engine.arg_utils import EngineArgs
case=sys.argv[1];r=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0168")
launch=json.loads((r/"candidate_plan.json").read_text());argv=list(launch["argv"])
for k,v in launch["environment"].items():os.environ[k]=str(v)
sys.path[:0]=[str(r),"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137"]
def setarg(k,v):argv[argv.index(k)+1]=str(v)
pcp,dcp,partition,nnodes=1,4,"22,20,20,16",1
setarg("--tensor-parallel-size",4);setarg("--pipeline-parallel-size",4);setarg("--decode-context-parallel-size",4);setarg("--master-port",29960);os.environ["VLLM_PP_LAYER_PARTITION"]=partition
evidence={"case":case,"argv":argv,"PP_partition":partition,"SDK_communicators_created":0,"workers":0,"model_instances":0,"inference":0,"NPU_tensor_allocations":0}
try:
 parser=FlexibleArgumentParser();sub=parser.add_subparsers(dest="subparser");cmd=ServeSubcommand();cmd.subparser_init(sub);args=parser.parse_args(argv[3:]);cmd.validate(args)
 if args.model_tag is not None:args.model=args.model_tag
 if args.tool_parser_plugin:ToolParserManager.import_tool_parser(args.tool_parser_plugin)
 validate_api_server_args(args)
 config=EngineArgs.from_cli_args(args).create_engine_config()
 from issue_budget_scheduler_v3 import proof
 from vllm.config import set_current_vllm_config
 with set_current_vllm_config(config):
  from vllm_ascend.ascend_config import get_ascend_config
  from vllm_ascend.utils import register_ascend_customop,enable_dsa_cp,enable_dsa_cp_with_o_proj_tp
  register_ascend_customop(config)
  from vllm_ascend.attention.context_parallel.sfa_cp import resolve_sfa_metadata_builder,resolve_sfa_impl
  cls=resolve_sfa_metadata_builder();impl=resolve_sfa_impl(config)
  ac=get_ascend_config()
  evidence.update(ascend_metadata=cls.__module__+"."+cls.__name__,ascend_impl=impl.__module__+"."+impl.__name__,DSACP=bool(enable_dsa_cp()),o_proj_tp=bool(enable_dsa_cp_with_o_proj_tp()),SP=config.parallel_config.use_sequence_parallel_moe,shared_expert_overlap=ac.multistream_overlap_shared_expert)
 pc=config.parallel_config
 evidence.update(config_accepted=True,TP=pc.tensor_parallel_size,PP=pc.pipeline_parallel_size,PCP=pc.prefill_context_parallel_size,DCP=pc.decode_context_parallel_size,DP=pc.data_parallel_size,world=pc.world_size,world_across_dp=pc.world_size_across_dp,nnodes=pc.nnodes,local_world_size=pc.world_size//pc.nnodes,K=config.speculative_config.num_speculative_tokens,KV_bytes=config.cache_config.kv_cache_memory_bytes,scheduler=proof(config))
 assert evidence["TP"]==4and evidence["PP"]==4and evidence["PCP"]==pcp and evidence["DCP"]==dcp and evidence["DP"]==1and evidence["world"]==16*pcp and evidence["local_world_size"]==16
 hf=json.loads(Path("/data/tiankuan/wio/GLM-5.2-w8a8/config.json").read_text())
 parts=list(map(int,partition.split(",")));boundaries=[sum(parts[:i])for i in range(1,len(parts))]
 evidence["hf_boundary_types"]={str(i):hf["indexer_types"][i]for i in boundaries}
 assert all(x=="full"for x in evidence["hf_boundary_types"].values())and sum(parts)==hf["num_hidden_layers"]
 from vllm.distributed.utils import get_pp_indices
 evidence["native_PP_indices"]=[get_pp_indices(hf["num_hidden_layers"],i,4)for i in range(4)]
 from vllm_ascend.attention.sfa_v1 import AscendSFABackend
 from vllm_ascend.utils import refresh_block_size
 evidence.update(native_cache_block_size=config.cache_config.block_size,effective_DCPblock_size=config.cache_config.block_size*pc.decode_context_parallel_size,native_speculative_method=config.speculative_config.method,native_use_eagle=config.speculative_config.use_eagle(),max_concurrent_batches=config.max_concurrent_batches,max_in_flight_tokens=config.max_in_flight_tokens,backend_supported_block_sizes=AscendSFABackend.get_supported_kernel_block_sizes())
 config.cache_config.block_size=64;refresh_block_size(config);evidence["explicit64_coerced_to_nativeblock"]=config.cache_config.block_size;assert config.cache_config.block_size==128
 evidence["native_config_only"]=True

except Exception as err:
 evidence.update(config_accepted=False,error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc())
 if True:raise
print(json.dumps({"event":"CPU_geometry_config",**evidence}))
