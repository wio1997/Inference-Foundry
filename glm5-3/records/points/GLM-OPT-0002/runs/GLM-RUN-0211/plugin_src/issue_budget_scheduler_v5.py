"""One authoritative scheduler cadence for independent native DP=1 engines.

The native DP=1 EngineCore never raises its DP balancing throttle flag. This
CPU control supplies the flag from the native scheduler call sequence, then
delegates all request/KV/speculation/PP tensor decisions to native schedule().
It is unsupported for coupled DP, EP, or native KV-transfer engines.
"""
import copy,json,hashlib,inspect
from pathlib import Path
from issue_budget_scheduler_v3 import IssueBudgetMixin,default_scheduler_class
from vllm.logger import init_logger
log=init_logger("vllm.glm_local_cadence")

def checked_local_config(config):
    p=config.parallel_config
    if p.data_parallel_size!=1 or p.enable_expert_parallel or config.kv_transfer_config is not None:
        raise RuntimeError("local scheduler cadence requires independent DP1/noEP/noKV")
    return dict(DP=p.data_parallel_size,TP=p.tensor_parallel_size,
                PP=p.pipeline_parallel_size,DCP=p.decode_context_parallel_size,
                no_EP=True,no_KV_transfer=True,authority="single native scheduler",
                sequence_is_not_physical_GPU_steps=True)

def checked_native_eligibility_source():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    expected={"BalanceScheduler.schedule":"0006b5b694d887f664d9062587d5463b76c84726497914be41ecc8ec62458e95",
              "Scheduler.schedule":"c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a",
              "SchedulerInterface.schedule":"be6c008664096c5660a8879de72b315bbe479bf1d688be9b25891063ec53a367"}
    actual={}
    for base in AsyncScheduler.__mro__:
        method=base.__dict__.get("schedule")
        if method is None:continue
        path=Path(inspect.getsourcefile(method));actual[method.__qualname__]=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:
        raise RuntimeError("native eligibility source changed; review control predicate")
    return actual

def decode_candidate_has_native_tokens(scheduler,request):
    """Conservative read-only native RUNNING-loop eligibility, not a KV promise."""
    if request.is_prefill_chunk or request.has_encoder_inputs or scheduler.need_mamba_block_aligned_split:
        return False
    if request.num_output_placeholders>0 and request.num_computed_tokens+2-request.num_output_placeholders>=request.num_prompt_tokens+request.max_tokens:
        return False
    # Native Scheduler.schedule increments current_step once BEFORE the RUNNING loop.
    if scheduler.current_step+1<request.next_decode_eligible_step:
        return False
    n=request.num_tokens_with_spec+request.num_output_placeholders-request.num_computed_tokens
    return min(n,scheduler.max_model_len-request.num_computed_tokens-scheduler.num_sampled_tokens_per_step)>0

class EligibleCadenceMixin(IssueBudgetMixin):
    def __init__(self,*args,**kwargs):
        config=kwargs.get("vllm_config")or(args[0]if args else None)
        if config is None:raise RuntimeError("native VllmConfig missing")
        geometry=checked_local_config(config)
        checked_native_eligibility_source()
        super().__init__(*args,**kwargs)
        self._glm_local_cadence_proofs=0
        self._glm_local_cadence_serial=None
        self._glm_local_cadence_bypass_proofs=0
        log.info("GLM_LOCAL_PREFILL_CADENCE_INSTALLED %s",json.dumps(
            dict(geometry=geometry,native_scheduler=default_scheduler_class(config).__name__,
                 native_DP1_throttle_always_false=True,native_math_changes=0,
                 eligibility_conservation=True,
                 source_control="issue_budget_scheduler_v5")))

    def schedule(self,throttle_prefills=False):
        budget,threshold,cadence=self._glm_read_controls()
        serial=self._glm_budget_observed[3]
        if serial!=self._glm_local_cadence_serial:
            self._glm_local_cadence_serial=serial;self._glm_local_cadence_proofs=0;self._glm_local_cadence_bypass_proofs=0
        # Scheduler.current_step is incremented by the delegated native call.
        # All PP/TP/DCP ranks consume that one SchedulerOutput; no rank-local clock.
        call=self.current_step
        cadence_slot=cadence==2 and call%2!=0
        prefill_ids={r.request_id for r in self.running if r.is_prefill_chunk}
        has_decode=any(not r.is_prefill_chunk for r in self.running)
        can_defer=not self.prefill_capacity_bound
        ready_decode_ids={r.request_id for r in self.running if decode_candidate_has_native_tokens(self,r)}
        local_throttle=cadence_slot and can_defer and bool(ready_decode_ids)
        native_throttle=bool(throttle_prefills)or local_throttle if cadence==2 else False
        if cadence_slot and prefill_ids and has_decode and can_defer and not ready_decode_ids and self._glm_local_cadence_bypass_proofs<4:
            log.info("GLM_LOCAL_PREFILL_CADENCE_BYPASSED %s",json.dumps(dict(
                serial=serial,prefill_cadence=cadence,scheduler_call=call,
                reason="no_native_decode_candidate_tokens",prefill_requests_before=len(prefill_ids),
                has_native_decode=True,local_throttle=False,
                sequence_is_not_physical_GPU_steps=True)))
            self._glm_local_cadence_bypass_proofs+=1
        previous_budget=self.max_num_scheduled_tokens
        previous_config=self.scheduler_config
        step_config=copy.copy(previous_config)
        step_config.long_prefill_token_threshold=threshold
        self.scheduler_config=step_config;self.max_num_scheduled_tokens=budget
        try:
            output=super(IssueBudgetMixin,self).schedule(native_throttle)
            if output.total_num_scheduled_tokens>budget:
                raise RuntimeError("native scheduled output exceeded issue budget")
            if local_throttle and prefill_ids and has_decode and can_defer and self._glm_local_cadence_proofs<4:
                prefill_tokens=sum(output.num_scheduled_tokens.get(k,0)for k in prefill_ids)
                if prefill_tokens:
                    raise RuntimeError("native scheduler did not honor local prefill defer")
                log.info("GLM_LOCAL_PREFILL_CADENCE_APPLIED %s",json.dumps(
                    dict(serial=serial,prefill_cadence=cadence,scheduler_call=call,
                         incoming_native_DP_flag=bool(throttle_prefills),local_throttle=True,
                         prefill_requests_before=len(prefill_ids),has_native_decode=True,
                         ready_decode_candidates_before=len(ready_decode_ids),
                         scheduled_ready_decode_tokens=sum(output.num_scheduled_tokens.get(k,0)for k in ready_decode_ids),
                         native_capacity_bound=False,scheduled_prefill_tokens=prefill_tokens,
                         scheduled_total_tokens=output.total_num_scheduled_tokens,
                         sequence_is_not_physical_GPU_steps=True)))
                self._glm_local_cadence_proofs+=1
            return output
        finally:
            self.max_num_scheduled_tokens=previous_budget
            self.scheduler_config=previous_config

class BudgetScheduler:
    def __new__(cls,*args,**kwargs):
        config=kwargs.get("vllm_config")or(args[0]if args else None)
        if config is None:raise RuntimeError("native VllmConfig missing")
        checked_local_config(config)
        base=default_scheduler_class(config)
        concrete=type("GLMEligibleCadence"+base.__name__,(EligibleCadenceMixin,base),{})
        return concrete(*args,**kwargs)

def proof(config):
    geometry=checked_local_config(config);cls=default_scheduler_class(config);source=checked_native_eligibility_source()
    return dict(native_eligibility_sources=source,geometry=geometry,native_scheduler_cls=cls.__module__+"."+cls.__name__,
                allocated_max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
                native_prefill_interval=config.scheduler_config.prefill_schedule_interval,
                native_threshold=config.scheduler_config.long_prefill_token_threshold,
                native_math_changes=0,SDK_communicators_created=0,model_requests=0,
                effective_standalone_cadence_requires_actual_APPLIED_marker=True,
                useful_decode_requires_positive_scheduled_ready_decode_tokens=True)
