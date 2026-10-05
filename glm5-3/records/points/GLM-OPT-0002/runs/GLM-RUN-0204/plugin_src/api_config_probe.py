import sys,json,os
from pathlib import Path
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.openai.api_server import validate_api_server_args
from vllm.tool_parsers import ToolParserManager
from vllm.engine.arg_utils import EngineArgs
from issue_budget_scheduler_v5 import BudgetScheduler,proof
from vllm.config import set_current_vllm_config
argv=json.loads(sys.argv[1]);parser=FlexibleArgumentParser();sub=parser.add_subparsers(dest="subparser");cmd=ServeSubcommand();cmd.subparser_init(sub);a=parser.parse_args(argv[3:]);cmd.validate(a)
if a.model_tag is not None:a.model=a.model_tag
ToolParserManager.import_tool_parser(a.tool_parser_plugin);validate_api_server_args(a)
c=EngineArgs.from_cli_args(a).create_engine_config();pc=c.parallel_config
assert (pc.tensor_parallel_size,pc.pipeline_parallel_size,pc.decode_context_parallel_size,pc.data_parallel_size,pc.nnodes,pc.world_size)==(8,2,8,1,1,16)
assert pc.world_size//pc.nnodes==16and bool(a.headless)==(pc.node_rank==1)
assert c.scheduler_config.get_scheduler_cls()is BudgetScheduler and c.scheduler_config.max_num_batched_tokens==8192
assert c.kv_transfer_config is None and not pc.enable_expert_parallel and c.speculative_config.num_speculative_tokens==int(os.environ['GLM_STATIC_K'])
assert c.cache_config.kv_cache_memory_bytes==3221225472 and c.scheduler_config.max_num_seqs==8
with set_current_vllm_config(c):
 from vllm_ascend.utils import register_ascend_customop,enable_dsa_cp,enable_dsa_cp_with_o_proj_tp
 register_ascend_customop(c)
 from vllm_ascend.ascend_config import get_ascend_config
 from vllm_ascend.attention.context_parallel.sfa_cp import resolve_sfa_metadata_builder,resolve_sfa_impl
 assert not enable_dsa_cp()and not enable_dsa_cp_with_o_proj_tp()and not pc.use_sequence_parallel_moe and not get_ascend_config().multistream_overlap_shared_expert
 assert resolve_sfa_metadata_builder().__name__=="AscendSFADCPMetadataBuilder"and resolve_sfa_impl(c).__name__=="AscendSFADCPImpl"
 from vllm.distributed.utils import get_pp_indices
 assert [get_pp_indices(78,k,2)for k in range(2)]==[(0,42),(42,78)]
 assert all(c.model_config.hf_text_config.indexer_types[k]=="full"for k in[42])
 from atomic_mq_worker import AtomicMQWorker
 from pp_empty_token_guard import guarded_source
 from vllm.v1.executor.multiproc_executor import MultiprocExecutor
 obj=object.__new__(MultiprocExecutor);obj.parallel_config=pc;parallel=obj._get_parallel_sizes()
 assert obj.world_size==16and obj.local_world_size==16
 proof_v5=proof(c)
policy=json.loads(Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]).read_text());assert policy==dict(schema_version=1,cohort_id="GLM-COHORT-0204",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
print(json.dumps(dict(event="fullCLI_PP32_exactplan",TP=8,PP=2,DCP=8,K=c.speculative_config.num_speculative_tokens,nnodes=1,world=16,local_world=16,node_rank=pc.node_rank,headless=a.headless,parallel_sizes=parallel,scheduler=proof_v5,workers=0,models=0,inference=0,NPU_tensors=0,SDKcommunicators=0)))

# CPU key dispatch with the complete native VllmConfig, not a minimal namespace.
import copy,inspect,hashlib
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
assert sorted(c.compilation_config.cudagraph_capture_sizes)==[4,8,12,16,20,24,28,32]
assert c.compilation_config.max_cudagraph_capture_size==32
source=Path(inspect.getfile(CudagraphDispatcher));assert hashlib.sha256(source.read_bytes()).hexdigest()=="51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63"
allrows=[]
for label,captures in [("resident202",[4,8,16,32]),("candidate204",[4,8,12,16,20,24,28,32])]:
 cfg=copy.copy(c);cfg.compilation_config=copy.deepcopy(c.compilation_config);cfg.compilation_config.cudagraph_capture_sizes=captures;cfg.compilation_config.max_cudagraph_capture_size=32
 dispatcher=CudagraphDispatcher(cfg);dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY,4)
 rows=[]
 for batch in range(1,9):
  tokens=4*batch;mode,descriptor=dispatcher.dispatch(tokens,uniform_decode=True)
  expected=tokens if label=="candidate204"else next(x for x in captures if x>=tokens)
  assert mode==CUDAGraphMode.FULL and descriptor.num_tokens==expected and descriptor.num_reqs==expected//4 and descriptor.uniform
  rows.append(dict(native_decode_batch=batch,input_tokens=tokens,mode=mode.name,padded_tokens=descriptor.num_tokens,padded_requests=descriptor.num_reqs,uniform=descriptor.uniform))
 for tokens in [4,8,12,20,32]:
  mode,descriptor=dispatcher.dispatch(tokens,uniform_decode=False);assert mode==CUDAGraphMode.NONE and descriptor.num_tokens==tokens
 allrows.append(dict(label=label,captures=captures,dispatches=rows,nonuniform_preserved_NONE=True))
# Actual native caller sources are copied only; source/callback context is not GPU-step proof.
import vllm.v1.worker.gpu_model_runner as base_runner
import vllm_ascend.worker.worker as native_worker
modules=[inspect.getmodule(CudagraphDispatcher),base_runner,native_worker]
for name,cls in vars(native_worker).items():
 if isinstance(cls,type)and "ModelRunner"in name:
  module=inspect.getmodule(cls)
  if module is not None and getattr(module,"__file__",None)and module not in modules:modules.append(module)
outdir=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/DENSE-GRAPH-PP2-FULLCLI-CPU-20261005T0625Z")/"graph_source_current";outdir.mkdir();sources=[]
for index,module in enumerate(modules):
 src=Path(module.__file__);raw=src.read_bytes();text=raw.decode();snapshot=outdir/(str(index)+"_"+src.name);snapshot.write_bytes(raw);lines=text.splitlines();hits=[]
 for num,line in enumerate(lines):
  if any(key in line for key in ["cudagraph_dispatcher.dispatch","uniform_decode","use_v2_model_runner","VLLM_USE_V2_MODEL_RUNNER"]):
   hits.append(dict(line=num+1,context="\\n".join(lines[max(0,num-3):min(len(lines),num+4)])))
 sources.append(dict(module=module.__name__,path=str(src),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),snapshot=str(snapshot),hits=hits[:35]))
print(json.dumps(dict(event="native_dense_Graph_CPU",full_native_config=True,K=3,TP=8,PP=2,DCP=8,Graph_max32=True,native_dispatch=allrows,source=sources,new_workers=0,new_models=0,inference=0,NPU_tensors=0,SDKcommunicators=0,limits=["CPU dispatch and caller sources do not certify actual native GPU batch occupancy, capture fit, speedup or hardware bound"])))
