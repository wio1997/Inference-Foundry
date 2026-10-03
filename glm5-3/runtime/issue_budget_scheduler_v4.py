"""One authoritative scheduler cadence for independent native DP=1 engines.

The native DP=1 EngineCore never raises its DP balancing throttle flag. This
CPU control supplies the flag from the native scheduler call sequence, then
delegates all request/KV/speculation/PP tensor decisions to native schedule().
It is unsupported for coupled DP, EP, or native KV-transfer engines.
"""
import copy,json
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

class LocalCadenceMixin(IssueBudgetMixin):
    def __init__(self,*args,**kwargs):
        config=kwargs.get("vllm_config")or(args[0]if args else None)
        if config is None:raise RuntimeError("native VllmConfig missing")
        geometry=checked_local_config(config)
        super().__init__(*args,**kwargs)
        self._glm_local_cadence_proofs=0
        self._glm_local_cadence_serial=None
        log.info("GLM_LOCAL_PREFILL_CADENCE_INSTALLED %s",json.dumps(
            dict(geometry=geometry,native_scheduler=default_scheduler_class(config).__name__,
                 native_DP1_throttle_always_false=True,native_math_changes=0,
                 source_control="issue_budget_scheduler_v4")))

    def schedule(self,throttle_prefills=False):
        budget,threshold,cadence=self._glm_read_controls()
        serial=self._glm_budget_observed[3]
        if serial!=self._glm_local_cadence_serial:
            self._glm_local_cadence_serial=serial;self._glm_local_cadence_proofs=0
        # Scheduler.current_step is incremented by the delegated native call.
        # All PP/TP/DCP ranks consume that one SchedulerOutput; no rank-local clock.
        call=self.current_step
        local_throttle=cadence==2 and call%2!=0
        native_throttle=bool(throttle_prefills)or local_throttle if cadence==2 else False
        prefill_ids={r.request_id for r in self.running if r.is_prefill_chunk}
        has_decode=any(not r.is_prefill_chunk for r in self.running)
        can_defer=not self.prefill_capacity_bound
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
        concrete=type("GLMLocalCadence"+base.__name__,(LocalCadenceMixin,base),{})
        return concrete(*args,**kwargs)

def proof(config):
    geometry=checked_local_config(config);cls=default_scheduler_class(config)
    return dict(geometry=geometry,native_scheduler_cls=cls.__module__+"."+cls.__name__,
                allocated_max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
                native_prefill_interval=config.scheduler_config.prefill_schedule_interval,
                native_threshold=config.scheduler_config.long_prefill_token_threshold,
                native_math_changes=0,SDK_communicators_created=0,model_requests=0,
                effective_standalone_cadence_requires_actual_APPLIED_marker=True)
