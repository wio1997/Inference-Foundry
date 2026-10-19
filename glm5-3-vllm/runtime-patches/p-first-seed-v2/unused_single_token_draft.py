"""Diagnostic selection of the existing native skip branch for D max1 batches."""
import functools
import json
import os
from pathlib import Path
import time


def install(ns):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT')=='0', 'P exported proposals must never be skipped'
    Runner=ns['NPUModelRunner'];root=Path('/out/divergence-probe');root.mkdir(exist_ok=True)
    state={'decision':None,'records':0}

    def emit(**row):
        if state['records']>=400 or (root/'unused-draft-trace-disabled').exists():return
        state['records']+=1;row.update(pid=os.getpid(),monotonic_ns=time.monotonic_ns())
        with (root/('unused-draft-%d.jsonl'%os.getpid())).open('a') as out:
            out.write(json.dumps(row,sort_keys=True)+'\n')

    fits_original=Runner._input_fits_in_drafter
    @functools.wraps(fits_original)
    def fits(self,common_attn_metadata):
        native=fits_original(self,common_attn_metadata)
        ids=list(self.input_batch.req_ids)
        params=self.parallel_config
        role=getattr(self.vllm_config.kv_transfer_config,'kv_role',None)
        real=bool(ids and all(rid in self.requests and self.requests[rid].sampling_params is not None for rid in ids))
        eligible=bool(real and role=='kv_consumer' and self.use_async_scheduling
                      and params.tensor_parallel_size==16 and params.data_parallel_size==1
                      and params.pipeline_parallel_size==1 and self.num_spec_tokens==5
                      and all(rid in self.requests and self.requests[rid].sampling_params is not None
                              and self.requests[rid].sampling_params.max_tokens==1 for rid in ids))
        requested=(root/'skip-unused-single-token-draft').exists()
        selected=bool(native and eligible and requested)
        state['decision']={'tp_rank':self.tp_rank,'request_ids':ids,'native_fits':bool(native),
                           'eligible_all_max1_D_TP16_DP1_PP1':eligible,
                           'requested_skip':requested,'selected_skip':selected,'role':str(role),
                           'max_tokens':[getattr(self.requests[rid].sampling_params,'max_tokens',None) for rid in ids if rid in self.requests]}
        if real:emit(kind='unused_draft_decision',**state['decision'])
        else:state['decision']=None
        return False if selected else native
    Runner._input_fits_in_drafter=fits

    skip_original=Runner._skip_drafting
    @functools.wraps(skip_original)
    def skip(self,*args,**kwargs):
        result=skip_original(self,*args,**kwargs)
        decision=state['decision']
        if decision is not None:emit(kind='native_skip_drafting_executed',**decision)
        return result
    Runner._skip_drafting=skip
