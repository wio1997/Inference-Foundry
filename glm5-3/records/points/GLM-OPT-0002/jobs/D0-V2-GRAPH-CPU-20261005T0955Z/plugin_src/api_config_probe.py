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
policy=json.loads(Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]).read_text());assert policy==dict(schema_version=1,cohort_id="GLM-COHORT-0214",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
print(json.dumps(dict(event="fullCLI_PP32_exactplan",TP=8,PP=2,DCP=8,K=c.speculative_config.num_speculative_tokens,nnodes=1,world=16,local_world=16,node_rank=pc.node_rank,headless=a.headless,parallel_sizes=parallel,scheduler=proof_v5,workers=0,models=0,inference=0,NPU_tensors=0,SDKcommunicators=0)))

assert c.use_v2_model_runner and c.max_concurrent_batches==3
assert c._get_v2_model_runner_unsupported_features()==[]
assert c.speculative_config.method=="mtp"
from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support
support=resolve_spec_pp_support(c);assert support is not None and not support.bypass_upstream_pp_guard and support.architectures is None
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager,_is_compatible
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes
from vllm.config import CUDAGraphMode
# Pure CPU native candidate method on uninitialized shell. No __init__, graph pool, PP group, NPU or captured flag.
obj=object.__new__(CudaGraphManager);obj.vllm_config=c;obj.compilation_config=c.compilation_config;obj.cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY;obj.max_num_reqs=8;obj.decode_query_len=3;obj.lora_capture_cases=[0];obj._candidates={};obj._capture_descs={}
CudaGraphManager._init_candidates(obj)
sizes=collect_sorted_captured_token_sizes(obj._capture_descs);assert sizes==[3,6,9,12,15,18,21,24]
rows=[]
for b in range(1,9):
 tok=3*b;candidate=obj._candidates[(tok,0)];matches=[v for v in candidate if _is_compatible(v,b,tok,3,0)]
 assert len(matches)==1 and matches[0].cg_mode==CUDAGraphMode.FULL and matches[0].num_tokens==tok and matches[0].num_reqs==b
 rows.append(dict(batch=b,query_width=3,tokens=tok,num_reqs=matches[0].num_reqs,mode=matches[0].cg_mode.name))
for b,t in [(1,2),(2,5),(3,8)]:
 assert all(not _is_compatible(x,b,t,None,0)for x in obj._candidates[(t,0)])
print(json.dumps(dict(event="V2_native_CPU_candidates",rows=rows,capture_descriptor_sizes=sizes,actual_native_methods=True,NPU_tensors=0,graphs_captured=0,real_dispatch_unknown=True)))
import inspect,hashlib
sources=[]
for module in ["vllm.config.vllm","vllm.v1.worker.gpu.cudagraph_utils","vllm_ascend.worker.v2.model_runner","vllm_ascend.worker.v2.pp_utils","vllm_ascend.worker.v2.spec_decode.mtp.speculator","vllm_ascend.worker.v2.spec_decode.autoregressive.speculator","vllm_ascend.worker.v2.aclgraph_utils","vllm_ascend.worker.v2.attn_utils","vllm.v1.worker.gpu.model_runner","vllm.v1.worker.gpu.pp_utils","vllm.v1.worker.gpu.spec_decode.mtp.speculator","vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"]:
 import importlib
 m=importlib.import_module(module);src=Path(inspect.getfile(m));raw=src.read_bytes();snapshot=Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_v2_graph_cpu_216')/"v2_source_current"/(module.replace(".","_")+".py");snapshot.parent.mkdir(exist_ok=True);snapshot.write_bytes(raw)
 sources.append(dict(module=module,path=str(src),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),snapshot=str(snapshot)))
features=[dict(module=x["module"],**dict(workers=0,models=0,inference=0,NPU_tensors=0))for x in sources]
print(json.dumps(dict(event="native_dense_Graph_CPU",full_native_config=True,V2=True,K=2,TP=8,PP=2,DCP=8,Graph_max24=True,requested_capture_sizes=c.compilation_config.cudagraph_capture_sizes,effective_capture_sizes=sizes,native_dispatch=rows,source=sources,max_concurrent_batches=c.max_concurrent_batches,spec_method=c.speculative_config.method,PP_support=dict(guard_bypass=False,needs_aux_hidden_states=support.needs_aux_hidden_states,architectures=None),mtp_class=AscendMTPSpeculator.__name__,unsupported_native_config=[],thinking_token_budget_native_warning=True,request_functional_equivalence_unknown=True,models=0,inference=0,NPU_tensors=0,limits=["CPUconfig only; runner notinstantiated; noV2Graphdispatch/capture/model/inference proof","NativeV2 warns thinking_token_budget unsupported; all native feature behavior must bepreserved before productioncandidate","PP2nativeV1concurrency2 vsV2CPU3 isconfiguredcapacity notGPU overlap evidence"])))
