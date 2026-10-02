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
from issue_budget_scheduler_v3 import BudgetScheduler,IssueBudgetMixin,proof as budget_proof
assert config.scheduler_config.get_scheduler_cls()is BudgetScheduler
baseline_args=EngineArgs.from_cli_args(args);baseline_args.scheduler_cls=None;baseline=baseline_args.create_engine_config()
bp=budget_proof(config);assert bp["native_scheduler_cls"]==budget_proof(baseline)["native_scheduler_cls"],"native base scheduler mode changed"
assert config.scheduler_config.async_scheduling==baseline.scheduler_config.async_scheduling
import os
policy=Path(os.environ["GLM_ISSUE_BUDGET_POLICY"]);test=object.__new__(IssueBudgetMixin);test._glm_native_budget=16384;test._glm_default_budget=4096;test._glm_budget_path=policy;test._glm_cohort=os.environ["GLM_ISSUE_BUDGET_COHORT"];test._glm_budget_observed=None
def select(obj,expected):
 policy.write_text(json.dumps(obj));actual=test._glm_read_controls();assert actual==expected;return dict(input=obj,selected=actual)
base=dict(schema_version=1,cohort_id=test._glm_cohort,budget_tokens=4096,serial=1)
cases=[]
for obj,expected in[
 (base,(4096,0,1)),
 (dict(base,budget_tokens=16384),(16384,0,1)),
 (dict(base,budget_tokens=1024,prefill_threshold_tokens=512,prefill_cadence=2),(1024,512,2)),
 (dict(base,prefill_threshold_tokens=2048,prefill_cadence=1),(4096,2048,1)),
 (dict(base,prefill_threshold_tokens=2048,prefill_cadence=2),(4096,2048,2)),
 (dict(base,prefill_threshold_tokens=8192),(4096,0,1)),
 (dict(base,prefill_threshold_tokens=True),(4096,0,1)),
 (dict(base,prefill_threshold_tokens=129),(4096,0,1)),
 (dict(base,prefill_threshold_tokens=-128),(4096,0,1)),
 (dict(base,prefill_cadence=3),(4096,0,1)),
 (dict(base,prefill_cadence=True),(4096,0,1)),
 (dict(base,budget_tokens=32768),(4096,0,1)),
 (dict(base,cohort_id="foreign"),(4096,0,1)),
 (dict(base,serial=-1),(4096,0,1)),
 ([],(4096,0,1))]:
 cases.append(select(obj,expected))
policy.write_bytes(b"x"*4097);assert test._glm_read_controls()==(4096,0,1);cases.append(dict(input="oversized4097",selected=(4096,0,1)))
policy.unlink();assert test._glm_read_controls()==(4096,0,1);cases.append(dict(input="missing",selected=(4096,0,1)))
assert config.scheduler_config.prefill_schedule_interval==2 and config.scheduler_config.long_prefill_token_threshold==0
bp.update(async_scheduling=config.scheduler_config.async_scheduling,policy_cases=cases,scope="Policy parsing/native base type only; no nativeScheduler instance/KV/model/comm/GPU/performance")

assert config.parallel_config.tensor_parallel_size==16 and config.parallel_config.decode_context_parallel_size==16 and config.parallel_config.data_parallel_size==2
assert args.tool_call_parser=="glm47_contract"
capture_geometry=None
if config.kv_transfer_config.kv_role=="kv_consumer":
 import copy
 cc=config.compilation_config
 assert config.speculative_config.num_speculative_tokens==5
 assert cc.cudagraph_capture_sizes==[6,12,24,48]and cc.max_cudagraph_capture_size==48
 negative_cc=copy.deepcopy(cc);negative_cc.cudagraph_capture_sizes=[1,2,4];negative_cc.max_cudagraph_capture_size=4
 try:negative_cc.adjust_cudagraph_sizes_for_spec_decode(6,16)
 except ValueError as err:
  assert "No valid cudagraph sizes after rounding to multiple of 6"in str(err)
 else:raise AssertionError("NativeK5Graphnegativegeometrynotrejected")
 cc.adjust_cudagraph_sizes_for_spec_decode(6,16)
 assert cc.cudagraph_capture_sizes==[6,12,24,48]and cc.max_cudagraph_capture_size==48
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
 expected_HCCL="4096"
 assert os.environ.get("HCCL_BUFFSIZE")==expected_HCCL
 expected_batch=16384
 assert config.scheduler_config.max_num_seqs==8 and config.cache_config.kv_cache_memory_bytes==13743895347
 assert config.scheduler_config.max_num_batched_tokens==expected_batch
 pg_policy={name:get_hccl_config_for_pg_options(name)for name in["mc2","tp","ep","dcp","dp"]}
 assert pg_policy["mc2"]is None and all(pg_policy[name]["hccl_buffer_size"]==200 for name in["tp","ep","dcp"])
 HCCL_buffer_policy=dict(actual_env_MB=int(os.environ["HCCL_BUFFSIZE"]),batch_tokens=config.scheduler_config.max_num_batched_tokens,native_pg_options=pg_policy,native_utils_sha256=hashlib.sha256(Path("/vllm-workspace/vllm-ascend/vllm_ascend/utils.py").read_bytes()).hexdigest(),SDK_communicators_created=0)
try:
 args.tool_call_parser="glm48_contract";validate_api_server_args(args)
except KeyError:negative=True
else:raise AssertionError("nativeAPIfailedtorejectinvalidregisteredparser")
print(json.dumps({"event":"full_native_CLI_API_config_valid","issue_budget_CPU_proof":bp,"tool_parser":"glm47_contract","invalid_glm48_contract_rejected":negative,"kv_role":config.kv_transfer_config.kv_role,"kv_port":config.kv_transfer_config.kv_port,"max_model_len":config.model_config.max_model_len,"TP":16,"DCP":16,"DP":2,"speculative_tokens":config.speculative_config.num_speculative_tokens,"capture_geometry":capture_geometry,"unused_SFA_workspace_proof":workspace_proof,"HCCL_buffer_policy":HCCL_buffer_policy,"NPU_workers_started":0,"models":0,"requests":0}))

