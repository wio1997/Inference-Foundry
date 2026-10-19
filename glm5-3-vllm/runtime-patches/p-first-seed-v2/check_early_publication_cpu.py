"""Exercise real native fences/schema and the new retirement hooks on CPU."""
from pathlib import Path
from types import SimpleNamespace as S, ModuleType
from collections import deque
from dataclasses import dataclass, field, replace
import ast
import argparse
import importlib.util
import json
import os
import pickle
import sys
import tempfile

os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='0'
root=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location(name,root/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
retire=load('early_sample_retirement');publication=load('early_sample_publication')
p=argparse.ArgumentParser();p.add_argument('--build-root',type=Path,required=True);args=p.parse_args()
build=args.build_root
schemas=ModuleType('_glm53_cpu_schemas');schemas.__dict__.update(dataclass=dataclass,field=field)
sys.modules[schemas.__name__]=schemas
for filename,name in [('output.py','SchedulerOutput'),('model_outputs.py','ModelRunnerOutput')]:
    tree=ast.parse((build/filename).read_text());node=next(z for z in tree.body if isinstance(z,ast.ClassDef) and z.name==name)
    exec('from __future__ import annotations\n'+ast.unparse(node),schemas.__dict__)
Output=schemas.ModelRunnerOutput
native=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-source/vllm/vllm/v1/core/sched/scheduler.py')
node=next(z for z in ast.parse(native.read_text()).body if isinstance(z,ast.ClassDef) and z.name=='Scheduler')
names=['_request_blocks_can_be_freed','_free_request_blocks','_free_cow_retained_blocks','_drain_deferred_frees']
methods=[z for z in node.body if isinstance(z,ast.FunctionDef) and z.name in names]
assert len(methods)==4
fence_class=ast.ClassDef(name='NativeFences',bases=[],keywords=[],body=methods,decorator_list=[])
ns={};exec('from __future__ import annotations\n'+ast.unparse(fence_class),ns);NativeFences=ns['NativeFences']
checks=[]
def rejection(fn,label):
    try:fn();raise RuntimeError('accepted '+label)
    except AssertionError:checks.append(label)
def worker_output(seq,retired,early=True):
    return Output(req_ids=['r'],req_id_to_index={'r':0},glm53_gpu_step_seq=seq,
                  glm53_gpu_retired_through=retired,glm53_gpu_early_sample=early,glm53_gpu_retire_rank_count=1)
outputs=[worker_output(2,2,False) for _ in range(15)]+[worker_output(2,1)]
assert retire.aggregate_metadata(outputs)==(2,1,True)
checks.append('last_of16_worker_prevents_premature_retirement')
rejection(lambda:retire.aggregate_metadata(outputs[:15]),'missing_worker_rejected')
outputs[-1].glm53_gpu_step_seq=3
rejection(lambda:retire.aggregate_metadata(outputs),'mismatched_worker_step_rejected')
outputs[-1].glm53_gpu_step_seq=2
outputs[-1].glm53_gpu_retire_rank_count=16
rejection(lambda:retire.aggregate_metadata(outputs),'already_aggregated_worker_rejected')
out=worker_output(2,1)
assert pickle.loads(pickle.dumps(out)).glm53_gpu_retired_through==1
assert 'glm53_gpu_step_seq' in schemas.SchedulerOutput.__dataclass_fields__
checks.append('actual_declared_ModelRunnerOutput_IPC_fields_roundtrip')

class Aggregator:
    def aggregate(self, outputs, output_rank=0):return outputs[output_rank]
retire.install_aggregator(dict(KVOutputAggregator=Aggregator))
agg=Aggregator();assert agg.aggregate([None]*16) is None
checks.append('deferred_execute_all16_None_is_passthrough_not_retirement')
rejection(lambda:agg.aggregate([None]*15),'deferred_execute_missing_rank_rejected')
outputs=[worker_output(2,2,False) for _ in range(15)]+[worker_output(2,1)]
first=outputs[0];merged=agg.aggregate(outputs)
assert merged is first and merged.glm53_gpu_retired_through==1 and merged.glm53_gpu_retire_rank_count==16
checks.append('aggregator_keeps_native_output_and_minimum_retirement')

class Scheduler(NativeFences):
    def __init__(self):
        self.vllm_config=S(kv_transfer_config=S(is_kv_consumer=True,is_kv_producer=False),
                          max_concurrent_batches=2,parallel_config=S(tensor_parallel_size=16,pipeline_parallel_size=1))
        self.defer_block_free=True;self.sched_step_seq=0;self.processed_step_seq=0
        self.deferred_frees=deque();self.released=[];self.freed=[];self.next_tokens=6;self.copies=[]
        self.requests={}
        self.kv_cache_manager=S(free=lambda request:self.freed.append(request),
            pop_blocks_for_free=lambda request:request.blocks,
            block_pool=S(free_blocks=lambda blocks:self.released.extend(list(blocks))))
    def schedule(self):
        self.sched_step_seq+=int(self.next_tokens>0)
        return S(total_num_scheduled_tokens=self.next_tokens,kv_cache_block_copies=self.copies,
                 new_block_ids_to_zero=[],glm53_gpu_step_seq=None,glm53_gpu_step_has_work=False)
    def has_requests(self):return False
    def finish_requests(self, ids, status):
        request=self.requests.pop(ids[0]);request.status=status
        self._free_request_blocks(request);return [request]

with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);retire.install_scheduler(dict(Scheduler=Scheduler,RequestStatus=S(FINISHED_ABORTED='abort')),d)
    s=Scheduler();one=s.schedule();two=s.schedule();request=S(last_sched_seq=2,blocks=[11,12])
    s._free_request_blocks(request);assert not s.released and s.has_requests()
    def ack(sched,seq):
        o=worker_output(sched.glm53_gpu_step_seq,seq);o.glm53_gpu_retire_rank_count=16
        s._glm53_apply_retirement_ack(sched,o)
    ack(one,0);assert not s.released
    ack(two,1);assert not s.released
    s.next_tokens=0;empty=s.schedule();assert not empty.glm53_gpu_step_has_work
    ack(empty,2);assert set(s.released)=={11,12} and not s.has_requests()
    checks+=['abort_before_early_output_keeps_native_KV_fence','next_early_step_retires_only_prior_step',
             'final_empty_drain_releases_KV_and_quiesces']
    s=Scheduler();s._free_cow_retained_blocks([21,22],1);s.next_tokens=0;s.copies=[(1,2)]
    zero=s.schedule();assert zero.glm53_gpu_step_seq==1 and zero.glm53_gpu_step_has_work
    assert not s.released and s.has_requests();ack(zero,1)
    assert set(s.released)=={21,22} and not s.has_requests()
    checks.append('actual_native_CoW_fence_retires_on_zero_token_copy_step')
    o=worker_output(1,0);o.glm53_gpu_retire_rank_count=16
    rejection(lambda:s._glm53_apply_retirement_ack(zero,o),'backwards_retirement_rejected')
    s=Scheduler();one=s.schedule();needle='earlypubabort-'+'a'*32;rid='cmpl-'+needle+'-0'
    request=S(last_sched_seq=1,blocks=[31,32],status='running');s.requests[rid]=request
    (d/'abort-test-once.json').write_text(json.dumps(dict(client_request_id=needle)))
    o=worker_output(1,0);o.req_ids=[rid];o.glm53_gpu_retire_rank_count=16
    s._glm53_apply_retirement_ack(one,o)
    receipt=json.loads((d/('abort-test-receipt-'+needle+'.json')).read_text())
    assert receipt['native_finish_requests_called'] and not s.released and not(d/'abort-test-once.json').exists()
    s.next_tokens=0;empty=s.schedule();ack(empty,1);assert set(s.released)=={31,32}
    checks.append('nonce_scoped_native_abort_injection_keeps_KV_until_real_drain')

current=[]
class Stream:
    def __init__(self):self.events=[]
    def wait_stream(self,other):pass
class StreamContext:
    def __init__(self,stream):self.stream=stream
    def __enter__(self):current.append(self.stream)
    def __exit__(self,*args):current.pop()
default=Stream();current.append(default)
class Event:
    def __init__(self,blocking=True):self.ready=False
    def record(self):self.stream=current[-1];self.stream.events.append(self);self.index=len(self.stream.events)-1
    def query(self):return self.ready
    def synchronize(self):
        for e in self.stream.events[:self.index+1]:e.ready=True
class Tensor:
    def __init__(self,values,device='npu'):
        self.values=[list(v) for v in values];self.device=S(type=device);self.dtype='i64'
        self.ndim=2;self.shape=(len(values),len(values[0]))
    def is_contiguous(self):return True
class Async:
    def __init__(self,model_runner_output,sampled_token_ids,async_output_copy_stream,**kwargs):
        self._model_runner_output=model_runner_output;self._sampled_token_ids=sampled_token_ids
        self.sampled_token_ids_cpu=Tensor(sampled_token_ids.values,'cpu');self.async_copy_ready_event=Event()
        with StreamContext(async_output_copy_stream):self.async_copy_ready_event.record()
    def get_output(self):
        self.async_copy_ready_event.synchronize();out=self._model_runner_output
        out.sampled_token_ids=self.sampled_token_ids_cpu.values;del self._sampled_token_ids;return out
empty_native=Output(req_ids=[],req_id_to_index={})
class Runner:
    def __init__(self):
        self.is_kv_consumer=True;self.is_kv_producer=False;self.use_async_scheduling=True
        self.vllm_config=S(max_concurrent_batches=2);self.async_output_copy_stream=Stream()
        self.parallel_config=S(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,
                               decode_context_parallel_size=1,prefill_context_parallel_size=1)
        self.model_config=S(max_model_len=8192,use_mla=True,is_hybrid=False)
        self.input_batch=S(req_ids=['r'],vocab_size=100);self.requests={'r':S(sampling_params=S(max_tokens=64))}
        self.num_spec_tokens=5;self.dynamic_eplb=False;self.tp_rank=0;self.execute_model_state=None
        self.sampler=S(sampled_token_ids=Tensor([[1,2,3,-1,-1,-1]]),logprobs_tensors=None);self.mutate=False
    def execute_model(self,scheduled):
        if scheduled.total_num_scheduled_tokens:self.execute_model_state=object();return None
        return empty_native
    def _bookkeeping_sync(self,scheduled,sampler):return (None,None,None,[],{},['r'],{'r':0},[])
    def sample_tokens(self,grammar):
        self._bookkeeping_sync(None,self.sampler)
        if self.mutate:self.sampler.sampled_token_ids.values[0][0]=99
        out=Async(model_runner_output=Output(req_ids=['r'],req_id_to_index={'r':0}),
                  sampled_token_ids=self.sampler.sampled_token_ids,logprobs_tensors=None,invalid_req_indices=[],
                  async_output_copy_stream=self.async_output_copy_stream,vocab_size=100,num_nans=None)
        self.input_batch.native_event=out.async_copy_ready_event;self.execute_model_state=None;return out
torch=S(int32='i32',int64='i64',equal=lambda a,b:a.values==b.values,
        npu=S(Event=Event,current_stream=lambda:current[-1],stream=StreamContext))
def scheduled(seq,tokens=6,writes=False):
    return S(glm53_gpu_step_seq=seq,glm53_gpu_step_has_work=bool(tokens or writes),
             total_num_scheduled_tokens=tokens,kv_cache_block_copies=[(1,2)] if writes else [],new_block_ids_to_zero=[])
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);publication.install(dict(torch=torch,NPUModelRunner=Runner,AsyncGPUModelRunnerOutput=Async,
                                       ModelRunnerOutput=Output,enable_sp=lambda:False),d)
    (d/'enable').touch();r=Runner();r.execute_model(scheduled(1));a=r.sample_tokens(None)
    event=a.async_copy_ready_event;out=a.get_output()
    assert out.glm53_gpu_retired_through==0 and out.glm53_gpu_early_sample
    assert not event.query() and r.input_batch.native_event is event and hasattr(a,'_sampled_token_ids')
    checks+=['early_publication_does_not_wait_current_native_late','InputBatch_keeps_original_late_event',
             'native_GPU_source_retained_after_early_publication']
    r.execute_model(scheduled(2));b=r.sample_tokens(None);out=b.get_output()
    assert out.glm53_gpu_retired_through==1 and event.query() and not b.async_copy_ready_event.query()
    checks.append('copy_stream_FIFO_early2_covers_native_late1_only')
    drained=r.execute_model(scheduled(2,0));assert drained.glm53_gpu_retired_through==2
    assert drained is not empty_native and empty_native.glm53_gpu_step_seq is None
    assert not r._glm53_publication_state['retained']
    checks.append('final_drain_waits_original_event_and_preserves_EMPTY_singleton')
    (d/'enable').unlink();r.execute_model(scheduled(3));out=r.sample_tokens(None).get_output()
    assert out.glm53_gpu_retired_through==3 and not out.glm53_gpu_early_sample
    checks.append('native_control_waits_and_retires_current_step')
    zero=r.execute_model(scheduled(4,0,True));assert zero.glm53_gpu_retired_through==4
    checks.append('zero_token_KV_write_gets_real_copy_stream_retirement')
    (d/'enable').touch()
    r2=Runner();r2.execute_model(scheduled(1));a2=r2.sample_tokens(None);a2.get_output()
    r2.execute_model(scheduled(2));b2=r2.sample_tokens(None)
    assert not a2.async_copy_ready_event.query() and not b2.async_copy_ready_event.query()
    r2.execute_model(scheduled(3))
    assert a2.async_copy_ready_event.query() and not b2.async_copy_ready_event.query()
    assert len(r2._glm53_publication_state['retained'])==1
    c2=r2.sample_tokens(None);assert len(r2._glm53_publication_state['retained'])==2
    assert b2.get_output().glm53_gpu_retired_through==1
    assert c2.get_output().glm53_gpu_retired_through==2
    r2.execute_model(scheduled(3,0))
    checks.append('third_GPU_step_waits_oldest_native_late_before_input_reuse')
    (d/'enable').touch();r.mutate=True;r.execute_model(scheduled(5));r.sample_tokens(None).get_output()
    rejection(lambda:r.execute_model(scheduled(5,0)),'late_comparison_detects_changed_source_before_retirement_ACK')
print(json.dumps(dict(completed=True,passed=len(checks),checks=checks,
    scope='CPU actual pinned native KV/CoW fences and declared IPC schema, plus new hook ownership/FIFO/drain behavior. No NPU scheduling, full-model correctness, cancellation or service gain proof.')))
