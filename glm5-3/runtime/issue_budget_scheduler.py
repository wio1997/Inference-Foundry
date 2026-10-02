"""Task-owned CPU issue budget; native scheduling, KV and operators are delegated."""
import copy,hashlib,json,os
from pathlib import Path
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.logger import init_logger
NATIVE_PATH=Path("/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py")
NATIVE_SHA="c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a"
if hashlib.sha256(NATIVE_PATH.read_bytes()).hexdigest()!=NATIVE_SHA:
    raise RuntimeError("native CPU scheduler source changed")
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
        if not 512<=self._glm_default_budget<=self._glm_native_budget:
            raise RuntimeError("issue budget exceeds native allocation or minimum")
        self._glm_budget_observed=None
        log.info("GLM_ISSUE_BUDGET_INSTALLED %s",json.dumps(dict(cohort=self._glm_cohort,native_base=type(self).__mro__[2].__name__,native_source_sha256=NATIVE_SHA,allocated_scheduler_max=self._glm_native_budget,default_budget=self._glm_default_budget,policy_path=str(self._glm_budget_path),native_math_changes=0)))
    def _glm_read_budget(self):
        value=self._glm_default_budget;serial=None;error=None
        try:
            # Bounded local read, no network or coordinator barrier in schedule().
            if self._glm_budget_path.stat().st_size>4096:
                raise ValueError("policy exceeds bounded read")
            p=json.loads(self._glm_budget_path.read_text())
            if not isinstance(p,dict)or type(p.get("schema_version"))is not int or p.get("schema_version")!=1 or p.get("cohort_id")!=self._glm_cohort:
                raise ValueError("policy schema or cohort mismatch")
            n=p.get("budget_tokens");serial=p.get("serial")
            if type(n)is not int or n%128 or not 512<=n<=self._glm_native_budget:
                raise ValueError("policy budget outside native bound/alignment")
            if type(serial)is not int or serial<0:
                raise ValueError("policy serial invalid")
            value=n
        except (OSError,ValueError,TypeError)as e:
            error=type(e).__name__+":"+str(e)
        state=(value,serial,error)
        if state!=self._glm_budget_observed:
            log.info("GLM_ISSUE_BUDGET_SELECTED %s",json.dumps(dict(cohort=self._glm_cohort,budget_tokens=value,serial=serial,policy_error=error,fallback=error is not None,native_max=self._glm_native_budget)))
            self._glm_budget_observed=state
        return value
    def schedule(self,*args,**kwargs):
        previous=self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens=self._glm_read_budget()
        try:
            output=super().schedule(*args,**kwargs)
            # The original schedule() has already checked all its native constraints.
            if output.total_num_scheduled_tokens>self.max_num_scheduled_tokens:
                raise RuntimeError("native scheduled output exceeded issue budget")
            return output
        finally:
            self.max_num_scheduled_tokens=previous
class BudgetScheduler:
    """Native factory preserves the selected synchronous or async scheduler."""
    def __new__(cls,*args,**kwargs):
        config=kwargs.get("vllm_config")or(args[0]if args else None)
        if config is None:
            raise RuntimeError("native VllmConfig missing")
        base=default_scheduler_class(config)
        concrete=type("GLMIssueBudget"+base.__name__,(IssueBudgetMixin,base),{})
        return concrete(*args,**kwargs)
def proof(config):
    cls=default_scheduler_class(config)
    return dict(native_scheduler_cls=cls.__module__+"."+cls.__name__,native_source_sha256=NATIVE_SHA,allocated_max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,native_max_num_scheduled_tokens=config.scheduler_config.max_num_scheduled_tokens,operator_changes=0,SDK_communicators_created=0,model_requests=0)
