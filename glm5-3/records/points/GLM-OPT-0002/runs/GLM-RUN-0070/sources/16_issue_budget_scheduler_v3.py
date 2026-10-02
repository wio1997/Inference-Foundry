"""Task CPU controls delegate request state, KV, sampling and operators to native vLLM."""
import copy,hashlib,json,os
from pathlib import Path
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.logger import init_logger
NATIVE_PATH=Path("/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py")
NATIVE_SHA="c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a"
if hashlib.sha256(NATIVE_PATH.read_bytes()).hexdigest()!=NATIVE_SHA:
    raise RuntimeError("native CPU scheduler source changed")
NATIVE_CORE_PATH=Path("/vllm-workspace/vllm/vllm/v1/engine/core.py")
NATIVE_CORE_SHA="6fdd067f54e5d42c57ff413e292685f5bdf6f343498d85c9262567f1eb746916"
if hashlib.sha256(NATIVE_CORE_PATH.read_bytes()).hexdigest()!=NATIVE_CORE_SHA:
    raise RuntimeError("native globally synchronized cadence source changed")
log=init_logger("vllm.glm_issue_budget")
def default_scheduler_class(config):
    plain=copy.copy(config.scheduler_config)
    plain.scheduler_cls=None
    cls=plain.get_scheduler_cls()
    if not issubclass(cls,Scheduler):
        raise RuntimeError("unsupported native scheduler family")
    return cls
class IssueBudgetMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._glm_native_budget=self.max_num_scheduled_tokens
        self._glm_budget_path=Path(os.environ["GLM_ISSUE_BUDGET_POLICY"])
        self._glm_cohort=os.environ["GLM_ISSUE_BUDGET_COHORT"]
        self._glm_default_budget=int(os.environ.get("GLM_ISSUE_BUDGET_DEFAULT","4096"))
        if self.scheduler_config.prefill_schedule_interval!=2:
            raise RuntimeError("native globally synchronized prefill cadence must be two")
        if self.scheduler_config.long_prefill_token_threshold!=0:
            raise RuntimeError("initial native threshold must be zero")
        if not 512<=self._glm_default_budget<=self._glm_native_budget or self._glm_default_budget%128:
            raise RuntimeError("issue budget exceeds native allocation or alignment")
        self._glm_budget_observed=None
        log.info("GLM_ISSUE_BUDGET_INSTALLED %s",json.dumps(dict(cohort=self._glm_cohort,native_base=type(self).__mro__[2].__name__,native_source_sha256=NATIVE_SHA,native_core_source_sha256=NATIVE_CORE_SHA,allocated_scheduler_max=self._glm_native_budget,default_budget=self._glm_default_budget,default_prefill_threshold=0,default_prefill_cadence=1,native_global_prefill_interval=2,policy_path=str(self._glm_budget_path),native_math_changes=0)))
    def _glm_read_controls(self):
        budget=self._glm_default_budget;threshold=0;cadence=1;serial=None;error=None
        try:
            with self._glm_budget_path.open("rb")as stream:
                raw=stream.read(4097)
            if len(raw)>4096:
                raise ValueError("policy exceeds bounded read")
            p=json.loads(raw)
            if not isinstance(p,dict)or type(p.get("schema_version"))is not int or p.get("schema_version")!=1 or p.get("cohort_id")!=self._glm_cohort:
                raise ValueError("policy schema or cohort mismatch")
            b=p.get("budget_tokens");t=p.get("prefill_threshold_tokens",0);c=p.get("prefill_cadence",1);serial=p.get("serial")
            if type(b)is not int or b%128 or not 512<=b<=self._glm_native_budget:
                raise ValueError("policy budget outside native bound/alignment")
            if type(t)is not int or (t!=0 and(t%128 or not 128<=t<=b)):
                raise ValueError("native perrequest prefill threshold outside issue budget/alignment")
            if type(c)is not int or c not in[1,2]:
                raise ValueError("unsupported native globally synchronized cadence")
            if type(serial)is not int or serial<0:
                raise ValueError("policy serial invalid")
            budget,threshold,cadence=b,t,c
        except (OSError,ValueError,TypeError)as e:
            error=type(e).__name__+":"+str(e)
        state=(budget,threshold,cadence,serial,error)
        if state!=self._glm_budget_observed:
            log.info("GLM_ISSUE_BUDGET_SELECTED %s",json.dumps(dict(cohort=self._glm_cohort,budget_tokens=budget,prefill_threshold_tokens=threshold,prefill_cadence=cadence,serial=serial,policy_error=error,fallback=error is not None,native_max=self._glm_native_budget)))
            self._glm_budget_observed=state
        return budget,threshold,cadence
    def schedule(self,throttle_prefills=False):
        budget,threshold,cadence=self._glm_read_controls()
        previous_budget=self.max_num_scheduled_tokens
        previous_config=self.scheduler_config
        # Isolate temporary CPU policy from EngineCore and worker config objects.
        step_config=copy.copy(previous_config)
        step_config.long_prefill_token_threshold=threshold
        self.scheduler_config=step_config
        self.max_num_scheduled_tokens=budget
        try:
            # EngineCore computes the native globally synchronized cadence flag.
            # Cadence one ignores that flag; cadence two delegates it unchanged.
            native_throttle=bool(throttle_prefills)if cadence==2 else False
            output=super().schedule(native_throttle)
            if output.total_num_scheduled_tokens>budget:
                raise RuntimeError("native scheduled output exceeded issue budget")
            return output
        finally:
            self.max_num_scheduled_tokens=previous_budget
            self.scheduler_config=previous_config
class BudgetScheduler:
    def __new__(cls,*args,**kwargs):
        config=kwargs.get("vllm_config")or(args[0]if args else None)
        if config is None:
            raise RuntimeError("native VllmConfig missing")
        base=default_scheduler_class(config)
        concrete=type("GLMIssueBudget"+base.__name__,(IssueBudgetMixin,base),{})
        return concrete(*args,**kwargs)
def proof(config):
    cls=default_scheduler_class(config)
    return dict(native_scheduler_cls=cls.__module__+"."+cls.__name__,native_source_sha256=NATIVE_SHA,allocated_max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,native_max_num_scheduled_tokens=config.scheduler_config.max_num_scheduled_tokens,native_global_prefill_interval=config.scheduler_config.prefill_schedule_interval,native_long_prefill_threshold=config.scheduler_config.long_prefill_token_threshold,operator_changes=0,SDK_communicators_created=0,model_requests=0)

