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
assert (pc.tensor_parallel_size,pc.pipeline_parallel_size,pc.decode_context_parallel_size,pc.data_parallel_size,pc.nnodes,pc.world_size)==(4,4,4,1,1,16)
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
 assert [get_pp_indices(78,k,4)for k in range(4)]==[(0,22),(22,42),(42,62),(62,78)]
 assert all(c.model_config.hf_text_config.indexer_types[k]=="full"for k in[22,42,62])
 from atomic_mq_worker import AtomicMQWorker
 from pp_empty_token_guard import guarded_source
 from vllm.v1.executor.multiproc_executor import MultiprocExecutor
 obj=object.__new__(MultiprocExecutor);obj.parallel_config=pc;parallel=obj._get_parallel_sizes()
 assert obj.world_size==16and obj.local_world_size==16
 proof_v5=proof(c)
policy=json.loads(Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]).read_text());assert policy==dict(schema_version=1,cohort_id="GLM-COHORT-0199",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
print(json.dumps(dict(event="fullCLI_PP32_exactplan",TP=4,PP=4,DCP=4,K=c.speculative_config.num_speculative_tokens,nnodes=1,world=16,local_world=16,node_rank=pc.node_rank,headless=a.headless,parallel_sizes=parallel,scheduler=proof_v5,workers=0,models=0,inference=0,NPU_tensors=0,SDKcommunicators=0)))
