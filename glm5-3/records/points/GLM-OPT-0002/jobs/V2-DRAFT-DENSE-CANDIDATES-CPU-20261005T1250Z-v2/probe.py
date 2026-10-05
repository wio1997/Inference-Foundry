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
policy=json.loads(Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]).read_text());assert policy==dict(schema_version=1,cohort_id="GLM-COHORT-0222",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
print(json.dumps(dict(event="fullCLI_PP32_exactplan",TP=8,PP=2,DCP=8,K=c.speculative_config.num_speculative_tokens,nnodes=1,world=16,local_world=16,node_rank=pc.node_rank,headless=a.headless,parallel_sizes=parallel,scheduler=proof_v5,workers=0,models=0,inference=0,NPU_tensors=0,SDKcommunicators=0)))

assert c.use_v2_model_runner and c.max_concurrent_batches==3
assert c._get_v2_model_runner_unsupported_features()==[]
assert c.speculative_config.method=="mtp"
from vllm.config import replace,CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager,_is_compatible
from vllm_ascend.worker.v2.aclgraph_utils import collect_sorted_captured_token_sizes
# A separate native fullCLI validates proposed values; copying the populated
# config preserves Ascend runtime extension oot_compiler (replace(comp) cannot).
import copy
dense_argv=list(argv);dense_argv[dense_argv.index("--compilation-config")+1]=json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=list(range(1,9)),max_cudagraph_capture_size=8))
dense_args=parser.parse_args(dense_argv[3:]);cmd.validate(dense_args)
if dense_args.model_tag is not None:dense_args.model=dense_args.model_tag
validate_api_server_args(dense_args)
validated=EngineArgs.from_cli_args(dense_args).create_engine_config()
assert validated._get_v2_model_runner_unsupported_features()==[]
assert validated.compilation_config.cudagraph_capture_sizes==list(range(1,9))
dense_comp=copy.copy(c.compilation_config);dense_comp.cudagraph_capture_sizes=list(range(1,9));dense_comp.max_cudagraph_capture_size=8
dense=replace(c,compilation_config=dense_comp)
assert hasattr(dense.compilation_config,"oot_compiler") and dense.compilation_config.oot_compiler is c.compilation_config.oot_compiler

assert c.compilation_config.cudagraph_capture_sizes==[3,6,9,12,15,18,21,24]
assert not c.speculative_config.enforce_eager
results=[]
for label,conf in [("old_target_3n_reused_for_draft",c),("draft_dense_1_to_8",dense)]:
 obj=object.__new__(CudaGraphManager);obj.vllm_config=conf;obj.compilation_config=conf.compilation_config;obj.cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY;obj.max_num_reqs=8;obj.decode_query_len=1;obj.lora_capture_cases=[0];obj._candidates={};obj._capture_descs={}
 CudaGraphManager._init_candidates(obj)
 sizes=collect_sorted_captured_token_sizes(obj._capture_descs);rows=[]
 for b in range(1,9):
  matches=[v for v in obj._candidates[(b,0)] if v.cg_mode==CUDAGraphMode.FULL and _is_compatible(v,b,b,1,0)]
  assert len(matches)==1
  v=matches[0];rows.append(dict(active_reqs=b,graph_tokens=v.num_tokens,graph_reqs=v.num_reqs,padding=v.num_tokens-b))
  if label=="draft_dense_1_to_8":assert v.num_tokens==v.num_reqs==b
 results.append(dict(label=label,sizes=sizes,rows=rows))
assert results[0]["rows"][0]["graph_tokens"]==results[0]["rows"][0]["graph_reqs"]==3
assert results[1]["sizes"]==list(range(1,9))
out=dict(CPU_contract_VALID=True,native_fullCLI_guard_passed=True,actual_native_candidate_methods=True,dispatch=results,target3n_config_unchanged=True,proposed_draft_decode_dense_only=True,static_K=2,TP=8,PP=2,DCP=8,NPU_tensors=0,modelstarts=0,inference=0,runtime_activation=False,limits=["CPU actualcandidate descriptors only, no nativeGPU/capturedgraph/replay/function/performance proof","Old decode candidate active1 padded to3; densecandidate everyactive1..8 no padding; actual221draft runtimeoperands unobserved","Proposal changes draft enforce_eager false plus private decodecapturedsizes1..8; target and draftprefillGraph3n preserved, nativeguards/math/operators unchanged"])
Path(__file__).parent.joinpath("probe.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(dict(CPU_VALID=True,original_batch1_pad2=True,dense_no_padding=True)))
