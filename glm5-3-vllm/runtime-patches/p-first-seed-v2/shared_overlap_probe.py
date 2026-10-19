"""Same-instance diagnostic of the existing shared-expert overlap branch.

No tensor/kernel implementation is replaced. Only the native executor's
existing multistream_overlap boolean is selected between completed batches.
The SP-only special entry is excluded: its initialization selects another op.
"""
from collections import Counter
import functools
import json
import os
from pathlib import Path
import time


def install(ns):
    from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts
    Runner=ns['NPUModelRunner']
    root=Path('/out/divergence-probe');root.mkdir(exist_ok=True)
    state={'executors':None,'active':None,'records':0,'current_enabled':None}

    def emit(**row):
        row.update(pid=os.getpid(),monotonic_ns=time.monotonic_ns())
        with (root/('overlap-policy-%d.jsonl'%os.getpid())).open('a') as out:
            out.write(json.dumps(row,sort_keys=True)+'\n')

    def executors(self):
        if state['executors'] is not None:return state['executors']
        found={};counts={}
        for label,model in [('target',self.model),('drafter',getattr(self.drafter,'model',None))]:
            assert model is not None and hasattr(model,'named_modules'),label
            count=0
            for name,module in model.named_modules():
                shared=getattr(module,'ascend_shared_experts',None)
                if shared is None:continue
                assert isinstance(shared,AscendSharedExperts)
                assert shared.multistream_overlap is True
                assert shared.parallel_mode().name!='SEQUENCE_PARALLEL_ONLY', 'SP changes the custom-op entry at initialization; refuse this runtime toggle'
                found[id(shared)]=shared;count+=1
            assert count>0,'No native shared executors found in '+label
            counts[label]=count
        state['executors']=list(found.values())
        emit(kind='shared_overlap_inventory',tp_rank=self.tp_rank,counts=counts,
             unique_executors=len(found),native_initial_flag=True,SP_only_excluded=True)
        return state['executors']

    forward_original=AscendSharedExperts.forward
    @functools.wraps(forward_original)
    def shared_forward(self,*args,**kwargs):
        frame=state['active']
        if frame is not None:
            key=frame['phase']+('_on' if self.multistream_overlap else '_off')
            frame['calls'][key]+=1
        return forward_original(self,*args,**kwargs)
    AscendSharedExperts.forward=shared_forward

    model_original=Runner._model_forward
    @functools.wraps(model_original)
    def model_forward(self,*args,**kwargs):
        if (root/'overlap-probe-disabled').exists():
            if state['executors'] is not None and state['current_enabled'] is not True:
                for executor in state['executors']:executor.multistream_overlap=True
                state['current_enabled']=True;ns['_glm53_overlap_current_enabled']=True
            return model_original(self,*args,**kwargs)
        ids=list(self.input_batch.req_ids)
        if not ids or not all(rid in self.requests for rid in ids):
            return model_original(self,*args,**kwargs)
        # Mode is changed by the host only after all requests in a cohort join.
        # Native Async waits remain unchanged and no extra device wait is added.
        enabled=not (root/'shared-overlap-disabled').exists()
        try:shared=executors(self)
        except Exception as exc:
            ns['_glm53_overlap_current_enabled']=None
            if state['records']<200:
                state['records']+=1
                emit(kind='shared_overlap_refused',tp_rank=self.tp_rank,error=repr(exc),requested_enabled=enabled)
            return model_original(self,*args,**kwargs)
        if state['current_enabled'] is not enabled:
            for executor in shared:executor.multistream_overlap=enabled
            state['current_enabled']=enabled
        ns['_glm53_overlap_current_enabled']=enabled
        frame={'ids':ids,'rank':self.tp_rank,'enabled':enabled,'calls':Counter(),'phase':'target'}
        state['active']=frame if ids and all(rid in self.requests for rid in ids) else None
        try:return model_original(self,*args,**kwargs)
        finally:frame['phase']='after_target'
    Runner._model_forward=model_forward

    sample_original=Runner.sample_tokens
    @functools.wraps(sample_original)
    def sample(self,*args,**kwargs):
        try:return sample_original(self,*args,**kwargs)
        finally:
            frame=state['active'];state['active']=None
            if frame is not None and state['records']<200:
                state['records']+=1
                emit(kind='shared_overlap_actual_calls',tp_rank=frame['rank'],
                     request_ids=frame['ids'],enabled=frame['enabled'],calls=dict(frame['calls']),
                     scope='Native executor forward calls only; no proof that stream ordering is the cause of any output difference')
    Runner.sample_tokens=sample
