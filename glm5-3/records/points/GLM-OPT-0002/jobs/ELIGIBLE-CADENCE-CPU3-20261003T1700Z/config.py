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
 from vllm_ascend.platform import _validate_draft_decode_context_parallel_config
 import copy
 assert config.speculative_config.num_speculative_tokens==1 and config.speculative_config.method=="mtp"
 cc=copy.deepcopy(config.compilation_config);cc.adjust_cudagraph_sizes_for_spec_decode(2,8)
 evidence["K1_graph_geometry"]=dict(uniform_decode_query_len=2,sizes=cc.cudagraph_capture_sizes,max=cc.max_cudagraph_capture_size)
 negative=copy.copy(config);negative.speculative_config=copy.copy(config.speculative_config)
 negative.speculative_config.num_speculative_tokens_per_batch_size=[(1,8,1)]
 try:_validate_draft_decode_context_parallel_config(negative)
 except ValueError as error:
  assert "Dynamic speculative decoding and decode context parallelism is not supported"in str(error)
  evidence["dynamic_K_DCP4_rejected_native"]=str(error)
 else:raise AssertionError("NativeDCP4 dynamicK guard not enforced")
 evidence["native_config_only"]=True

except Exception as err:
 evidence.update(config_accepted=False,error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc())
 if True:raise

from types import SimpleNamespace as NS
from issue_budget_scheduler_v5 import EligibleCadenceMixin,checked_local_config,decode_candidate_has_native_tokens,checked_native_eligibility_source
checks=[]
for p,v in [("data_parallel_size",2),("enable_expert_parallel",True)]:
 bad=NS(parallel_config=NS(data_parallel_size=1,enable_expert_parallel=False,tensor_parallel_size=4,pipeline_parallel_size=4,decode_context_parallel_size=4),kv_transfer_config=None);setattr(bad.parallel_config,p,v)
 try:checked_local_config(bad)
 except RuntimeError:checks.append(p+"_rejected")
 else:raise AssertionError("unsafe local cadence geometry accepted")
bad=NS(parallel_config=NS(data_parallel_size=1,enable_expert_parallel=False),kv_transfer_config=object())
try:checked_local_config(bad)
except RuntimeError:checks.append("KV_transfer_rejected")
else:raise AssertionError("KV transfer accepted")
# Explicit simulation only: MRO/delegation, flags, bounded evidence and config restoration.
class FakeNativeBase:
 def schedule(self,flag=False):
  self.calls.append(dict(flag=flag,budget=self.max_num_scheduled_tokens,threshold=self.scheduler_config.long_prefill_token_threshold,step=self.current_step))
  self.current_step+=1
  if self.fail:raise ValueError("synthetic delegated exception")
  defer=flag and not self.prefill_capacity_bound and any(not r.is_prefill_chunk for r in self.running)
  tokens={}
  for r in self.running:
   if self.current_step<r.next_decode_eligible_step:continue
   if r.num_output_placeholders>0 and r.num_computed_tokens+2-r.num_output_placeholders>=r.num_prompt_tokens+r.max_tokens:continue
   if min(r.num_tokens_with_spec+r.num_output_placeholders-r.num_computed_tokens,self.max_model_len-r.num_computed_tokens-self.num_sampled_tokens_per_step)<=0:continue
   if defer and r.is_prefill_chunk:continue
   tokens[r.request_id]=1
  out=NS(total_num_scheduled_tokens=sum(tokens.values()),num_scheduled_tokens=tokens)
  self.outputs.append(out);return out
class FakeLocal(EligibleCadenceMixin,FakeNativeBase):
 def _glm_read_controls(self):
  self._glm_budget_observed=(1024,512,self.cadence,7,None);return 1024,512,self.cadence
f=object.__new__(FakeLocal);f.current_step=0;f.max_model_len=144384;f.num_sampled_tokens_per_step=1;f.need_mamba_block_aligned_split=False;f._glm_local_cadence_bypass_proofs=0;f._glm_local_cadence_serial=None;f._glm_local_cadence_proofs=0;f.cadence=2;f.running=[NS(request_id="prefill",is_prefill_chunk=True,has_encoder_inputs=False,next_decode_eligible_step=0,num_tokens_with_spec=100,num_output_placeholders=0,num_computed_tokens=10,num_prompt_tokens=100,max_tokens=128),NS(request_id="decode",is_prefill_chunk=False,has_encoder_inputs=False,next_decode_eligible_step=0,num_tokens_with_spec=11,num_output_placeholders=0,num_computed_tokens=10,num_prompt_tokens=1,max_tokens=128)];f.prefill_capacity_bound=False;f.max_num_scheduled_tokens=8192;f.scheduler_config=NS(long_prefill_token_threshold=0);original=f.scheduler_config;f.calls=[];f.outputs=[];f.fail=False
a=f.schedule(False);b=f.schedule(False)
assert a is f.outputs[0]and b is f.outputs[1]and f.calls[0]["flag"]is False and f.calls[1]["flag"]is True and a.num_scheduled_tokens=={"prefill":1,"decode":1}and b.num_scheduled_tokens=={"decode":1}
assert f.max_num_scheduled_tokens==8192and f.scheduler_config is original and original.long_prefill_token_threshold==0and all(x["budget"]==1024and x["threshold"]==512 for x in f.calls)
checks.append("localc2_delegates_authoritative_sequence_and_exact_output")
f.current_step=1;f.prefill_capacity_bound=True;a=f.schedule(False);assert a.num_scheduled_tokens=={"prefill":1,"decode":1};checks.append("native_saturation_guard_retained")
f.prefill_capacity_bound=False;f.running=[NS(request_id="prefill",is_prefill_chunk=True,has_encoder_inputs=False,next_decode_eligible_step=0,num_tokens_with_spec=100,num_output_placeholders=0,num_computed_tokens=10,num_prompt_tokens=100,max_tokens=128)];f.current_step=1;a=f.schedule(False);assert a.num_scheduled_tokens=={"prefill":1};checks.append("prefillonly_progress_retained")
f.running=[NS(request_id="prefill",is_prefill_chunk=True,has_encoder_inputs=False,next_decode_eligible_step=0,num_tokens_with_spec=100,num_output_placeholders=0,num_computed_tokens=10,num_prompt_tokens=100,max_tokens=128),NS(request_id="decode",is_prefill_chunk=False,has_encoder_inputs=False,next_decode_eligible_step=0,num_tokens_with_spec=11,num_output_placeholders=0,num_computed_tokens=10,num_prompt_tokens=1,max_tokens=128)];f.cadence=1;f.current_step=1;a=f.schedule(True);assert f.calls[-1]["flag"]is False and a.num_scheduled_tokens=={"prefill":1,"decode":1};checks.append("c1_flag_ignored_existing_policy_retained")
f.cadence=2;f.current_step=1;decode=f.running[1];decode.next_decode_eligible_step=3
a=f.schedule(False);assert f.calls[-1]["flag"]is False and a.num_scheduled_tokens=={"prefill":1};checks.append("PP_ineligible_decode_conserves_prefill_slot")
f.current_step=1;decode.next_decode_eligible_step=2
a=f.schedule(False);assert f.calls[-1]["flag"]is True and a.num_scheduled_tokens=={"decode":1};checks.append("native_preloop_step_increment_boundary")
f.current_step=1;decode.next_decode_eligible_step=0;decode.num_tokens_with_spec=10
a=f.schedule(False);assert f.calls[-1]["flag"]is False and a.num_scheduled_tokens=={"prefill":1};checks.append("no_residual_decode_tokens_conserves_prefill")
decode.num_tokens_with_spec=11;decode.num_output_placeholders=2;decode.max_tokens=2
assert not decode_candidate_has_native_tokens(f,decode);checks.append("terminal_inflight_placeholders_not_ready")
decode.num_output_placeholders=0;decode.max_tokens=128;f.max_model_len=11
assert not decode_candidate_has_native_tokens(f,decode);checks.append("model_length_cap_not_ready")
f.max_model_len=144384;decode.has_encoder_inputs=True
assert not decode_candidate_has_native_tokens(f,decode);checks.append("encoder_needs_native_fallback")
decode.has_encoder_inputs=False;f.need_mamba_block_aligned_split=True
assert not decode_candidate_has_native_tokens(f,decode);checks.append("Mamba_needs_native_fallback")
f.need_mamba_block_aligned_split=False
assert decode_candidate_has_native_tokens(f,decode);checks.append("positive_residual_decode_ready")
assert checked_native_eligibility_source()=={"BalanceScheduler.schedule":"0006b5b694d887f664d9062587d5463b76c84726497914be41ecc8ec62458e95","Scheduler.schedule":"c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a","SchedulerInterface.schedule":"be6c008664096c5660a8879de72b315bbe479bf1d688be9b25891063ec53a367"};checks.append("full_native_scheduler_source_guard")
f.fail=True
try:f.schedule(False)
except ValueError:pass
else:raise AssertionError("delegated exception swallowed")
assert f.max_num_scheduled_tokens==8192and f.scheduler_config is original;checks.append("native_exception_restores_original_config")
import ast
core=Path("/vllm-workspace/vllm/vllm/v1/engine/core.py");module=ast.parse(core.read_text())
base=next(x for x in module.body if isinstance(x,ast.ClassDef)and x.name=="EngineCore")
method=next(x for x in base.body if isinstance(x,ast.FunctionDef)and x.name=="_should_throttle_prefills")
assert len(method.body)==2and isinstance(method.body[1],ast.Return)and method.body[1].value.value is False
evidence["native_DP1_cadence_inapplicable"]={"core_sha256":hashlib.sha256(core.read_bytes()).hexdigest(),"method_AST_sha256":hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest(),"base_return":False,"override_only_DPEngineCore":True}
evidence.update(synthetic_local_cadence_checks=checks,simulation_only_scheduling_contracts=True,actual_local_throttle_NPU_unverified=True,eligibility_checks_native_preloop_step=True,source_v5_sha256=hashlib.sha256(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/issue_budget_scheduler_v5.py").read_bytes()).hexdigest())
print(json.dumps({"event":"CPU_geometry_config",**evidence}))
