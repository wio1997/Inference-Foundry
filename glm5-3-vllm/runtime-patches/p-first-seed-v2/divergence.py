"""Capture already-available target tensors at two shared logical histories.

Copies are queued on the target-return stream before the native sampler can
mutate/reuse them. The original Async output wait is preserved. A diagnostic
copy-event only checks readiness and never adds a synchronization/wait.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def install(ns):
    import torch
    Runner,Async=ns['NPUModelRunner'],ns['AsyncGPUModelRunnerOutput']
    root=Path('/out/divergence-probe');root.mkdir(exist_ok=True)
    allowed=set(json.loads(os.environ.get('GLM53_DIVERGENCE_HISTORY_SHA256','[]')))
    assert len(allowed)==2 and all(len(x)==64 for x in allowed)
    limit=8 if os.environ.get('GLM53_SHARED_OVERLAP_DIAGNOSTIC')=='1' else 6
    state={'frame':None,'inflight':{},'counts':{},'sequence':0}
    def emit(**data):
        data.update(pid=os.getpid(),monotonic_ns=time.monotonic_ns())
        with (root/('worker-%d.jsonl'%os.getpid())).open('a') as f:f.write(json.dumps(data,sort_keys=True)+'\n')
    def stream():return int(torch.npu.current_stream().npu_stream)
    def raw(tensor):return tensor.contiguous().view(torch.uint8).numpy().tobytes()
    def save(name,tensor):
        data=raw(tensor);path=root/(name+'.bin')
        with path.open('xb') as f:f.write(data)
        return {'path':path.name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
    def finish(frame):
        if frame.get('dumped'):return True
        if not frame['copy_event'].query():return False
        # Source buffers are no longer needed once the diagnostic SDMA copies
        # completed; retain only the independent CPU snapshots for file output.
        frame['source_keepalive']=[];cpus=frame['cpu'];rank=frame['rank'];prefix='r%d-f%d'%(rank,frame['sequence'])
        inputs={k:save(prefix+'-'+k,cpus[k]) for k in ('input_ids','positions')}
        for rid,meta in frame['wanted'].items():
            index=frame['req_ids'].index(rid);key=prefix+'-'+hashlib.sha256(rid.encode()).hexdigest()[:16];hidden=cpus['sample_hidden'][index];logits=cpus['logits'][index]
            hinfo=save(key+'-hidden',hidden);ldata=raw(logits);linfo={'sha256':hashlib.sha256(ldata).hexdigest(),'bytes':len(ldata),'shape':list(logits.shape),'dtype':str(logits.dtype)}
            # Local logits shards are cheap to retain on every rank; a gathered
            # full-vocab vector is retained only on rank0, with hashes elsewhere.
            if logits.numel()<=32768 or rank==0:linfo.update(save(key+'-logits',logits))
            vals,idx=torch.topk(logits.float(),min(16,logits.numel()))
            emit(kind='target_tensor_snapshot',schema='glm53-first-divergence/v1',diagnostic_mode=frame['mode'],actual_shared_overlap_enabled=frame['actual_shared_overlap'],tp_rank=rank,request_id=rid,history_sha256=meta['sha256'],prompt_ids=meta['prompt_ids'],computed_before=meta['computed'],scheduled_tokens=meta['scheduled'],batch_index=index,batch_req_ids=frame['req_ids'],query_start_loc=frame['query_start_loc'],input_snapshots=inputs,hidden=hinfo,logits=linfo,logits_top16_local=[{'index':int(i),'value':float(v)} for i,v in zip(idx.tolist(),vals.tolist())],global_vocab_size=frame['global_vocab_size'],lmhead=frame['lmhead'],writer_stream=frame['writer_stream'],constructor_stream=frame.get('constructor_stream'),same_constructor_stream=frame.get('constructor_stream')==frame['writer_stream'],diagnostic_copy_event_ready=True,native_event_ready=frame.get('native_event_ready'),native_output=frame.get('native_output'),scope='Target-return tensors copied before native sampling; CPU snapshots, not proof of full intermediate/KV equality or a performance run')
        frame['dumped']=True;state['inflight'].pop(frame['sequence'],None);return True
    def drain():
        for frame in list(state['inflight'].values()):
            if frame.get('native_output') is not None:
                try:finish(frame)
                except Exception as exc:emit(kind='probe_dump_error',tp_rank=frame['rank'],error=repr(exc))

    sample_original=Runner.sample_tokens
    @functools.wraps(sample_original)
    def sample(self,*args,**kwargs):
        drain();frame=None
        try:
            values=self.execute_model_state
            if not (root/'disabled').exists() and values is not None and self.use_async_scheduling and values[2] is None:
                schedule=values[0];ids=list(self.input_batch.req_ids);wanted={}
                mode=('shared-off' if (root/'shared-overlap-disabled').exists() else 'shared-on') if os.environ.get('GLM53_SHARED_OVERLAP_DIAGNOSTIC')=='1' else 'native'
                for i,rid in enumerate(ids):
                    request=self.requests[rid];prompt=request.prompt_token_ids or [];computed=int(self.input_batch.num_computed_tokens_cpu[i]);scheduled=schedule.num_scheduled_tokens.get(rid,0)
                    if request.sampling_params.max_tokens!=1 or computed!=0 or scheduled!=len(prompt):continue
                    digest=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()
                    reserved=sum(meta['sha256']==digest for meta in wanted.values())
                    if digest in allowed and state['counts'].get((digest,mode),0)+reserved<limit:wanted[rid]={'sha256':digest,'prompt_ids':list(prompt),'computed':computed,'scheduled':scheduled}
                if wanted:
                    # _preprocess passes the padded token count to the actual
                    # forward. Retain that whole buffer, then compare each
                    # logical query using query_start_loc, not padded tails.
                    tensors={'logits':values[1],'sample_hidden':values[5],'positions':values[8],'input_ids':self.input_ids.gpu[:values[8].numel()]}
                    assert torch.cuda.Event is torch.npu.Event and all(isinstance(t,torch.Tensor) and t.device.type in ('npu','privateuseone') and t.is_contiguous() for t in tensors.values())
                    assert tensors['logits'].ndim==2 and tensors['sample_hidden'].ndim==2 and tensors['logits'].shape[0]==tensors['sample_hidden'].shape[0]==len(ids)
                    assert tensors['positions'].ndim==1 and tensors['positions'].numel()==tensors['input_ids'].numel()
                    assert self.parallel_config.tensor_parallel_size==16 and self.dcp_size==1
                    state['sequence']+=1
                    model=getattr(self,'model',None);head=getattr(model,'lm_head',None)
                    frame={'sequence':state['sequence'],'mode':mode,'actual_shared_overlap':ns.get('_glm53_overlap_current_enabled'),'rank':self.tp_rank,'req_ids':ids,'wanted':wanted,'query_start_loc':[int(x) for x in self.query_start_loc.np[:len(ids)+1]],'writer_stream':stream(),'cpu':{},'source_keepalive':list(tensors.values()),'global_vocab_size':self.input_batch.vocab_size,'lmhead':{'class':None if head is None else type(head).__name__,'vocab_start_index':getattr(head,'vocab_start_index',None),'num_embeddings_per_partition':getattr(head,'num_embeddings_per_partition',None)},'copy_event':torch.npu.Event()}
                    state['inflight'][frame['sequence']]=frame
                    try:
                        for name,tensor in tensors.items():
                            cpu=torch.empty(tensor.shape,dtype=tensor.dtype,device='cpu',pin_memory=True);frame['cpu'][name]=cpu;cpu.copy_(tensor,non_blocking=True)
                    finally:frame['copy_event'].record()
                    for meta in wanted.values():
                        key=(meta['sha256'],mode);state['counts'][key]=state['counts'].get(key,0)+1
        except Exception as exc:
            emit(kind='probe_capture_refused',tp_rank=getattr(self,'tp_rank',None),error=repr(exc));frame=None
        state['frame']=frame
        try:return sample_original(self,*args,**kwargs)
        finally:state['frame']=None
    Runner.sample_tokens=sample

    init_original=Async.__init__
    @functools.wraps(init_original)
    def init(self,*args,**kwargs):
        frame=state['frame'];self._glm_divergence_frame=frame
        if frame is not None:frame['constructor_stream']=stream()
        init_original(self,*args,**kwargs)
    Async.__init__=init

    get_original=Async.get_output
    @functools.wraps(get_original)
    def get(self):
        result=get_original(self)  # Original native event wait remains intact.
        frame=getattr(self,'_glm_divergence_frame',None)
        if frame is not None and not frame.get('dumped'):
            try:
                frame['native_event_ready']=bool(self.async_copy_ready_event.query());frame['native_output']={'req_ids':list(result.req_ids),'sampled_token_ids':[list(x) for x in result.sampled_token_ids]}
                finish(frame)
            except Exception as exc:emit(kind='probe_dump_error',tp_rank=frame['rank'],error=repr(exc))
        return result
    Async.get_output=get
