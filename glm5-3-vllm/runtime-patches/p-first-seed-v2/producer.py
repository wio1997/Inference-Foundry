"""Export only requested, already-computed MTP1 proposals via native completion.

The pinned D2H is queued on the actual MTP writer stream BEFORE the original
Async constructor snapshots that stream. FIFO protects the source from reuse;
the original copy-stream dependency/event covers host readiness. No new waits.
"""
import functools
import json
import os
from pathlib import Path
import time


def install(ns):
    import torch
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    Runner, Async = ns['NPUModelRunner'], ns['AsyncGPUModelRunnerOutput']
    root=Path('/out/p-first-draft');root.mkdir(exist_ok=True)
    state={'requested':{},'frame':None}
    def emit(**values):
        values.update(monotonic_ns=time.monotonic_ns(),pid=os.getpid())
        with (root/('worker-%d.jsonl'%os.getpid())).open('a') as f:f.write(json.dumps(values,sort_keys=True)+'\n')
    def stream():return int(torch.npu.current_stream().npu_stream)
    execute_original=Runner.execute_model
    @functools.wraps(execute_original)
    def execute(self,scheduler_output,*args,**kwargs):
        for rid in scheduler_output.finished_req_ids:state['requested'].pop(rid,None)
        for item in scheduler_output.scheduled_new_reqs:
            state['requested'][item.req_id]=item.export_p_first_draft is True
        return execute_original(self,scheduler_output,*args,**kwargs)
    Runner.execute_model=execute

    sample_original=Runner.sample_tokens
    @functools.wraps(sample_original)
    def sample(self,*args,**kwargs):
        frame=None
        if self.is_kv_producer and self.execute_model_state is not None and self.speculative_config.num_speculative_tokens==1:
            ids=list(self.input_batch.req_ids)
            wanted={rid for rid in ids if state['requested'].get(rid) and
                    self.requests[rid].sampling_params.max_tokens==1 and
                    len(self.requests[rid].prompt_token_ids or []) in (256,257)}
            if wanted:
                frame={'runner':self,'req_ids':ids,'wanted':wanted,'mtp_calls':0,
                    'lengths':{rid:len(self.requests[rid].prompt_token_ids) for rid in wanted}}
        state['frame']=frame
        try:return sample_original(self,*args,**kwargs)
        finally:state['frame']=None
    Runner.sample_tokens=sample

    draft_original=AscendSpecDecodeBaseProposer._run_merged_draft
    @functools.wraps(draft_original)
    def draft(self,*args,**kwargs):
        frame=state['frame']
        if frame is not None:
            frame['mtp_calls']+=1;frame['writer_stream']=stream()
        result=draft_original(self,*args,**kwargs)
        if frame is not None:
            frame['writer_return_stream']=stream();frame['writer_return_ns']=time.monotonic_ns()
        return result
    AscendSpecDecodeBaseProposer._run_merged_draft=draft

    init_original=Async.__init__
    @functools.wraps(init_original)
    def init(self,*args,**kwargs):
        frame=state['frame'];self._glm_export=None
        if frame is not None:
            runner=frame['runner'];tensor=runner._draft_token_ids
            valid=(frame['mtp_calls']==1 and frame.get('writer_stream')==frame.get('writer_return_stream')==stream()
                   and torch.cuda.Event is torch.npu.Event and isinstance(tensor,torch.Tensor)
                   and tensor.ndim==2 and tuple(tensor.shape)==(len(frame['req_ids']),1)
                   and tensor.dtype in (torch.int32,torch.int64) and tensor.is_contiguous()
                   and runner.parallel_config.tensor_parallel_size==16 and runner.dcp_size==1
                   and runner.parallel_config.prefill_context_parallel_size==1)
            if valid:
                # Queue on the same writer stream. The original constructor
                # below then waits on this stream and records its native event.
                cpu=torch.empty(tensor.shape,dtype=tensor.dtype,device='cpu',pin_memory=True)
                cpu.copy_(tensor,non_blocking=True)
                self._glm_export={'cpu':cpu,'source_keepalive':tensor,'meta':frame,
                    'constructor_stream':stream(),'copy_queued_ns':time.monotonic_ns(),
                    'cpu_pinned':bool(cpu.is_pinned())}
            else:emit(kind='P_draft_refused',tp_rank=runner.tp_rank,reason='writer/topology/tensor contract not established')
        init_original(self,*args,**kwargs)
    Async.__init__=init

    get_original=Async.get_output
    @functools.wraps(get_original)
    def get(self):
        result=get_original(self)  # Preserve the existing event synchronization.
        export=getattr(self,'_glm_export',None)
        if export is None:return result
        frame=export['meta'];ready=bool(self.async_copy_ready_event.query())
        if not ready:
            emit(kind='P_draft_refused',tp_rank=frame['runner'].tp_rank,reason='native event not ready')
            return result
        values=export['cpu'].tolist();by_id=dict(zip(frame['req_ids'],values));spec=[];published=[]
        for i,rid in enumerate(result.req_ids):
            tokens=by_id.get(rid,[])
            if (rid in frame['wanted'] and len(tokens)==1 and type(tokens[0]) is int and tokens[0]>=0
                and len(result.sampled_token_ids[i])==1):
                spec.append(tokens);published.append({'req_id':rid,'draft':tokens,'first':result.sampled_token_ids[i][0],'n':frame['lengths'][rid]})
            else:spec.append([])
        result.spec_token_ids=spec if published else None
        emit(kind='P_draft_export',tp_rank=frame['runner'].tp_rank,requests=published,
             native_event_ready=ready,writer_stream=frame['writer_stream'],constructor_stream=export['constructor_stream'],
             writer_return_ns=frame['writer_return_ns'],copy_queued_ns=export['copy_queued_ns'],
             cpu_pinned=export['cpu_pinned'],scope='Unverified MTP1 proposal, not output history; new tiny D2H requires SDMA profiling')
        return result
    Async.get_output=get
