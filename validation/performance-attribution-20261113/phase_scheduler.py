# SPDX-License-Identifier: Apache-2.0
"""通过原生scheduler-cls入口注入固定函数；两臂仅策略布尔值不同。"""
from pathlib import Path
import inspect
import hashlib
import os
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.config.scheduler import SchedulerConfig
from phase_policy import PhasePolicy,CORE_SHA256,transformed_method

_namespace=dict(Scheduler.schedule.__globals__)
_source=Path(inspect.getsourcefile(Scheduler)).read_text()
_pins={AsyncScheduler:'e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508',
       SchedulerConfig:'822179da62be4bca0619c8590082ae09123f5c4415f9cc2037be673748239edb'}
for _cls,_expected in _pins.items():
    if hashlib.sha256(Path(inspect.getsourcefile(_cls)).read_bytes()).hexdigest()!=_expected:
        raise ValueError('fixed async/config source identity changed')
exec(compile(transformed_method(_source),'<fixed-sdk-phase-policy>','exec'),_namespace)


class PhaseControlledAsyncScheduler(AsyncScheduler):
    schedule=_namespace['schedule']

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        value=os.environ['ATTRIBUTION_PHASE_POLICY']
        if value not in ('0','1'):raise ValueError('explicit0/1 policy flag required')
        if (self.connector is not None or self.ec_connector is not None or self.use_pp
                or self.use_v2_model_runner or self.num_spec_tokens!=5):
            raise ValueError('fixed V1 async MTP5 text-only runtime required')
        if (self.parallel_config.tensor_parallel_size!=8 or self.parallel_config.data_parallel_size!=1
                or self.max_num_running_reqs!=8 or self.max_model_len!=2048
                or self.scheduler_config.max_num_batched_tokens!=1792):
            raise ValueError('fixed8die/context/budget differs')
        root=Path(os.environ['ATTRIBUTION_PHASE_STATS_DIR'])
        if not root.is_absolute() or not root.is_dir():raise ValueError('existing absolute evidence directory required')
        identity={'scheduler_class':type(self).__module__+'.'+type(self).__name__,
                  'entry_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'method_sha256':hashlib.sha256(transformed_method(_source).encode()).hexdigest(),
                  'async_scheduler_sha256':_pins[AsyncScheduler],'scheduler_config_sha256':_pins[SchedulerConfig],
                  'async_scheduling':self.scheduler_config.async_scheduling,
                  'max_batched_tokens':self.scheduler_config.max_num_batched_tokens,
                  'max_scheduled_tokens':self.max_num_scheduled_tokens,'max_sequences':self.max_num_running_reqs}
        if not identity['async_scheduling']:raise ValueError('async scheduling was disabled')
        self._attribution_phase_policy=PhasePolicy(value=='1',CORE_SHA256,root/f'phase-policy-{os.getpid()}.json',runtime_identity=identity)

    def shutdown(self):
        try:result=super().shutdown()
        except BaseException as exc:
            self._attribution_phase_policy.close(type(exc).__name__)
            raise
        self._attribution_phase_policy.close()
        return result
