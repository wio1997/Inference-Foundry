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
pcp,dcp,partition,nnodes=1,4,"22,20,20,16",1
spec=json.loads(argv[argv.index("--speculative-config")+1]);spec["num_speculative_tokens"]=1;setarg("--speculative-config",json.dumps(spec));setarg("--tensor-parallel-size",4);setarg("--pipeline-parallel-size",4);setarg("--decode-context-parallel-size",4);setarg("--master-port",29960);os.environ["VLLM_PP_LAYER_PARTITION"]=partition
evidence={"case":case,"argv":argv,"PP_partition":partition,"SDK_communicators_created":0,"workers":0,"model_instances":0,"inference":0,"NPU_tensor_allocations":0}
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

 import inspect
 from vllm.v1.core.sched.scheduler import Scheduler
 from vllm.v1.core.sched.async_scheduler import AsyncScheduler
 def source(fn):
  path=Path(inspect.getsourcefile(fn));raw=path.read_bytes();name=path.name
  out=j/("native_"+name);out.write_bytes(raw)
  return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),snapshot=str(out),qualname=getattr(fn,"__qualname__",repr(fn)))
 j=Path(__file__).parent
 evidence["actual_scheduler_class"]=source(Scheduler)
 evidence["actual_schedule_method"]=source(Scheduler.schedule)
 evidence["actual_async_class"]=source(AsyncScheduler)
 evidence["actual_async_schedule"]=source(AsyncScheduler.schedule)
 evidence["actual_method_source"]=inspect.getsource(Scheduler.schedule)
 print(json.dumps({"event":"CPU_geometry_config",**evidence}))
except Exception:
 traceback.print_exc();raise
