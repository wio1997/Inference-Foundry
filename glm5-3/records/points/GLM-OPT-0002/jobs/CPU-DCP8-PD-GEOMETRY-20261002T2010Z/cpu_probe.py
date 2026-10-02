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
from issue_budget_scheduler_v3 import BudgetScheduler,proof as budget_proof
assert config.scheduler_config.get_scheduler_cls()is BudgetScheduler
baseline_args=EngineArgs.from_cli_args(args);baseline_args.scheduler_cls=None;baseline=baseline_args.create_engine_config()
bp=budget_proof(config);assert bp["native_scheduler_cls"]==budget_proof(baseline)["native_scheduler_cls"]and config.scheduler_config.async_scheduling==baseline.scheduler_config.async_scheduling
import os
policy=json.loads(Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]).read_text());assert policy==dict(schema_version=1,cohort_id="GLM-COHORT-0104",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)and os.environ["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0104"
assert config.scheduler_config.prefill_schedule_interval==2 and config.scheduler_config.long_prefill_token_threshold==0
bp.update(issue_budget_policy=policy,async_scheduling=config.scheduler_config.async_scheduling)

assert config.parallel_config.tensor_parallel_size==16 and config.parallel_config.decode_context_parallel_size==8 and config.parallel_config.data_parallel_size==1
assert args.tool_call_parser=="glm47_contract"
capture_geometry=None
kv=config.kv_transfer_config
assert kv.kv_connector=="MooncakeConnectorV1"and kv.kv_role in["kv_producer","kv_consumer"]
assert kv.kv_connector_extra_config==dict(use_ascend_direct=True,prefill=dict(dp_size=1,tp_size=16),decode=dict(dp_size=1,tp_size=16))
if True:
 import copy
 cc=config.compilation_config
 assert config.speculative_config.num_speculative_tokens==3
 assert cc.cudagraph_capture_sizes==[4,8,16,32]and cc.max_cudagraph_capture_size==32
 negative_cc=copy.deepcopy(cc);negative_cc.cudagraph_capture_sizes=[1,2];negative_cc.max_cudagraph_capture_size=2
 try:negative_cc.adjust_cudagraph_sizes_for_spec_decode(4,16)
 except ValueError as err:
  assert "No valid cudagraph sizes after rounding to multiple of "in str(err)
 else:raise AssertionError("NativeK5Graphnegativegeometrynotrejected")
 cc.adjust_cudagraph_sizes_for_spec_decode(4,16)
 assert cc.cudagraph_capture_sizes==[4,8,16,32]and cc.max_cudagraph_capture_size==32
 capture_geometry={"native_adjust_method":True,"sizes":cc.cudagraph_capture_sizes,"max":cc.max_cudagraph_capture_size,"uniform_decode_query_len":4,"negative_run51_rejected":True}

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
 from vllm_ascend.utils import enable_dsa_cp,enable_dsa_cp_with_o_proj_tp
 from vllm_ascend.ascend_config import get_ascend_config
 from vllm_ascend.attention.context_parallel.sfa_cp import resolve_sfa_metadata_builder,resolve_sfa_impl
 from sfa_workspace_guard import eligible
 cls=resolve_sfa_metadata_builder();implementation=resolve_sfa_impl(config)
 assert not enable_dsa_cp_with_o_proj_tp()
 from vllm_ascend.quantization.methods.w8a8.w8a8_static import AscendW8A8LinearMethod
 assert AscendW8A8LinearMethod.supports_tp_weight_switch
 import inspect
 quant_src=inspect.getsource(AscendW8A8LinearMethod)
 assert all(v in quant_src for v in ["aclnn_input_scale","aclnn_input_scale_reciprocal","aclnn_input_offset"])
 assert not enable_dsa_cp()and not get_ascend_config().enable_dsa_cp and not config.parallel_config.use_sequence_parallel_moe
 assert not get_ascend_config().multistream_overlap_shared_expert
 assert json.loads(argv[argv.index("--additional-config")+1])["multistream_overlap_shared_expert"]is False
 assert cls.__name__=="AscendSFADCPMetadataBuilder"and implementation.__name__=="AscendSFADCPImpl"
 assert json.loads(argv[argv.index("--additional-config")+1])["enable_flashcomm1"]is False
 assert args.worker_cls=="atomic_mq_worker.AtomicMQWorker"
 assert not eligible(object.__new__(cls),config)
 workspace_proof.update(optional_guard_installed=False,native_inherited_workspace_allocation_retained=True)
 DSA_proof=dict(kv_connector=kv.kv_connector,o_proj_tp_enabled=bool(enable_dsa_cp_with_o_proj_tp()),W8A8_TP_weight_switch_supported=True,enabled=False,SP_MoE=False,shared_expert_overlap=False,metadata_builder=cls.__module__+"."+cls.__name__,implementation=implementation.__module__+"."+implementation.__name__,old_zero_workspace_guard_eligible=False,optional_guard_installed=False)
 import os,hashlib
 from vllm_ascend.utils import get_hccl_config_for_pg_options
 expected_HCCL="768"
 assert os.environ.get("HCCL_BUFFSIZE")==expected_HCCL
 expected_batch=4096
 assert config.scheduler_config.max_num_seqs==8 and config.cache_config.kv_cache_memory_bytes==6442450944
 assert config.scheduler_config.max_num_batched_tokens==expected_batch
 pg_policy={name:get_hccl_config_for_pg_options(name)for name in["mc2","tp","ep","dcp","dp"]}
 assert pg_policy["mc2"]is None and all(pg_policy[name]["hccl_buffer_size"]==200 for name in["tp","ep","dcp"])
 HCCL_buffer_policy=dict(actual_env_MB=int(os.environ["HCCL_BUFFSIZE"]),batch_tokens=config.scheduler_config.max_num_batched_tokens,native_pg_options=pg_policy,native_utils_sha256=hashlib.sha256(Path("/vllm-workspace/vllm-ascend/vllm_ascend/utils.py").read_bytes()).hexdigest(),SDK_communicators_created=0)
assert os.environ.get("PROFILING_MODE")=="dynamic"
# Native dynamic depth is explicitly incompatible with DCP16; confirm using the same native EngineArgs API, without any workers.
negative_args=EngineArgs.from_cli_args(args);negative_args.speculative_config=dict(json.loads(argv[argv.index("--speculative-config")+1]),num_speculative_tokens_per_batch_size=[[1,8,3]])
try:negative_args.create_engine_config()
except ValueError as err:assert "Dynamic speculative decoding and decode context parallelism is not supported"in str(err)
else:raise AssertionError("NativeDCPdynamicdepthguardnotrejected")
try:
 args.tool_call_parser="glm48_contract";validate_api_server_args(args)
except KeyError:negative=True
else:raise AssertionError("nativeAPIfailedtorejectinvalidregisteredparser")
print(json.dumps({"event":"full_native_CLI_API_config_valid","issue_budget_CPU_proof":bp,"tool_parser":"glm47_contract","invalid_glm48_contract_rejected":negative,"kv_role":kv.kv_role,"kv_port":kv.kv_port,"max_model_len":config.model_config.max_model_len,"TP":16,"DCP":8,"DP":1,"speculative_tokens":config.speculative_config.num_speculative_tokens,"capture_geometry":capture_geometry,"unused_SFA_workspace_proof":workspace_proof,"DSA_CP_native_proof":DSA_proof,"HCCL_buffer_policy":HCCL_buffer_policy,"NPU_workers_started":0,"models":0,"requests":0}))


import ast,hashlib,types
f=Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py');tree=ast.parse(f.read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='_get_local_remote_cp_params')
ns={'ReqMeta':types.SimpleNamespace};exec(compile(ast.Module(body=[node],type_ignores=[]),str(f),'exec'),ns);func=ns[node.name];rows=[]
for rank in range(8):
 obj=types.SimpleNamespace(block_size=128,dcp_rank=rank,dcp_size=8,pcp_rank=0,pcp_size=1);meta=types.SimpleNamespace(remote_block_size=128,remote_pcp_size=1,remote_dcp_size=16)
 result=func(obj,meta);assert result==(128,rank,8,16,1);rows.append(dict(rank=rank,result=result))
obj=types.SimpleNamespace(block_size=512,dcp_rank=0,dcp_size=8,pcp_rank=0,pcp_size=1)
try:func(obj,meta)
except AssertionError:negative=True
else:raise AssertionError('native incompatibleblockratio not rejected')
print(json.dumps(dict(event='native_CP_geometry_CPU_valid',source_path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),method_lines=[node.lineno,node.end_lineno],rows=rows,native_incompatible_blockratio_rejected=negative,scope='Extracted original puremethod; fullPDtransfer/indexer/commit semantics pending actualE2E')))
