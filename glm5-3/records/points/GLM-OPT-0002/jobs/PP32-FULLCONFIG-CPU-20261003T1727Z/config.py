from pathlib import Path
import sys,json,os,hashlib,traceback
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.openai.api_server import validate_api_server_args
from vllm.tool_parsers import ToolParserManager
from vllm.engine.arg_utils import EngineArgs
case="PP4_K1_static";r=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0175")
launch=json.loads((r/"candidate_plan.json").read_text());argv=list(launch["argv"])
for k,v in launch["environment"].items():os.environ[k]=str(v)
sys.path[:0]=[str(r),"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137"]
def setarg(k,v):argv[argv.index(k)+1]=str(v)
setarg("--scheduler-cls","issue_budget_scheduler_v5.BudgetScheduler")
pcp,dcp,partition,nnodes=1,8,"22,20,20,16",2
spec=json.loads(argv[argv.index("--speculative-config")+1]);spec["num_speculative_tokens"]=3;setarg("--speculative-config",json.dumps(spec));setarg("--tensor-parallel-size",8);setarg("--pipeline-parallel-size",4);setarg("--decode-context-parallel-size",8);setarg("--master-port",29960);os.environ["VLLM_PP_LAYER_PARTITION"]=partition
setarg("--nnodes",2);setarg("--master-port",29962)
evidence={"case":case,"argv":argv,"PP_partition":partition,"SDK_communicators_created":0,"workers":0,"model_instances":0,"inference":0,"NPU_tensor_allocations":0}
results=[]
for node in [0,1]:
 setarg("--node-rank",node)
 if node==1:argv.append("--headless")
 evidence=dict(evidence,case="PP32_node"+str(node),argv=list(argv),actual_node_rank=node)
 try:
 
  parser=FlexibleArgumentParser();sub=parser.add_subparsers(dest="subparser");cmd=ServeSubcommand();cmd.subparser_init(sub);args=parser.parse_args(argv[3:]);cmd.validate(args)
  if args.model_tag is not None:args.model=args.model_tag
  if args.tool_parser_plugin:ToolParserManager.import_tool_parser(args.tool_parser_plugin)
  validate_api_server_args(args)
  config=EngineArgs.from_cli_args(args).create_engine_config()
  from issue_budget_scheduler_v5 import proof
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
  evidence.update(config_accepted=True,TP=pc.tensor_parallel_size,PP=pc.pipeline_parallel_size,PCP=pc.prefill_context_parallel_size,DCP=pc.decode_context_parallel_size,DP=pc.data_parallel_size,world=pc.world_size,world_across_dp=pc.world_size_across_dp,nnodes=pc.nnodes,local_world_size=pc.world_size//pc.nnodes,K=config.speculative_config.num_speculative_tokens,KV_bytes=config.cache_config.kv_cache_memory_bytes,use_v2_model_runner=config.use_v2_model_runner)
 
  assert (pc.tensor_parallel_size,pc.pipeline_parallel_size,pc.decode_context_parallel_size,pc.data_parallel_size,pc.nnodes,pc.world_size)==(8,4,8,1,2,32)
  assert pc.world_size//pc.nnodes==16and config.speculative_config.num_speculative_tokens==3and config.kv_transfer_config is None
  from issue_budget_scheduler_v5 import proof
  evidence["scheduler"]=proof(config)
  from vllm.distributed.utils import get_pp_indices
  evidence["PP_indices"]=[get_pp_indices(config.model_config.hf_config.num_hidden_layers,k,4)for k in range(4)]
  evidence["native_max_concurrent_batches"]=config.max_concurrent_batches
  evidence["effective_DCPblock_size"]=config.cache_config.block_size*8
  evidence["admission_maxseq"]=config.scheduler_config.max_num_seqs
  evidence["headless"]=node==1
  results.append(evidence)
 except Exception:
  traceback.print_exc();raise
print(json.dumps({"event":"CPU_geometry_config","config_accepted":True,"TP":8,"PP":4,"DCP":8,"K":3,"workers":0,"NPU_tensor_allocations":0,"model_instances":0,"inference":0,"cases":results}))
