#!/usr/bin/env python3
"""Execute pinned original Python guards/ownership with opaque tags only.

No Torch/NumPy/model/native imports or numerical equivalence claims. Finite
counterexamples are separate from native bit validation.
"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
from contextlib import nullcontext
import ast, copy, json

R=Path('audit/execution-search')
PINS={s['runtime_path']:s for n in ['target-output-dependencies.json','target-output-aux-dependencies.json'] for s in json.loads((R/n).read_text())['sources']}
USED=[]
def original(path,owner,name,extra):
    pin=PINS['/vllm-workspace/'+path];p=Path(pin['local_path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==pin['runtime_sha256']
    t=ast.parse(raw);body=t.body if owner is None else next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==owner).body
    f=copy.deepcopy(next(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name));f.decorator_list=[]
    used=dict(path=pin['runtime_path'],sha256=pin['runtime_sha256'],owner=owner,method=name,first=f.lineno,last=f.end_lineno)
    if used not in USED:USED.append(used)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])
    ns=dict(extra);exec(compile(ast.fix_missing_locations(module),'<pinned-original-Python>','exec'),ns);return ns[name]

def main():
    cases=[];logger=NS(debug=lambda *a:None,warning=lambda *a:None,exception=lambda *a:None)
    factory=original('vllm/vllm/v1/sample/logits_processor/__init__.py',None,'build_logitsprocs',dict(LogitsProcessors=lambda x=():list(x),MinTokensLogitsProcessor=lambda *a:'MIN_TOKENS',logger=logger,STR_SPEC_DEC_REJECTS_LOGITSPROCS='SPEC_CUSTOM_FORBIDDEN'))
    assert factory(NS(speculative_config=object()),None,False,False)==['MIN_TOKENS'];cases.append(dict(name='actual_spec_factory_only_MinTokens',result='PASS',kind='original_guard_with_constructor_tags'))
    try:factory(NS(speculative_config=object()),None,False,False,['custom'])
    except ValueError as e:assert str(e)=='SPEC_CUSTOM_FORBIDDEN'
    else:raise AssertionError('custom processor incorrectly admitted')
    cases.append(dict(name='spec_custom_processor_error',result='PASS',kind='original_error_guard'))
    trace=[]
    class Tag:
        def __init__(self,name,dtype='bf16'):self.name=name;self.dtype=dtype
        def to(self,dtype):trace.append(('to',self.name,dtype));return Tag(self.name+'.cast',dtype)
        def clone(self):trace.append(('clone',self.name));return Tag(self.name+'.copy',self.dtype)
        def long(self):trace.append(('long',self.name));return Tag(self.name+'.long','int64')
        def unsqueeze(self,d):trace.append(('unsqueeze',self.name,d));return self
    fwd=original('vllm/vllm/v1/sample/sampler.py','Sampler','forward',dict(torch=NS(float32='fp32',int32='int32',empty=lambda n:('empty',n)),SamplerOutput=lambda **kw:NS(**kw),LogprobsTensors=lambda *a:NS(logprobs=a[1])))
    sampler=NS(logprobs_mode='raw_logprobs',apply_logits_processors=lambda x,*a:x,sample=lambda *a:(Tag('sampled','int64'),None))
    metadata=NS(max_num_logprobs=-1,logprob_token_ids=None)
    output=fwd(sampler,Tag('bonus'),metadata,True,'raw_logits');assert output.logprobs_tensors.logprobs.dtype=='fp32';assert trace[0]==('to','bonus','fp32') and trace[1]==('to','bonus','fp32')
    cases.append(dict(name='bonus_forced_minus1_returns_full_raw_scores',result='PASS',kind='original_Sampler_forward_opaque_tensor_protocol',trace=trace.copy()))
    trace.clear();metadata.max_num_logprobs=None
    output=fwd(sampler,Tag('ordinary'),metadata);assert output.logprobs_tensors is None and trace[0]==('to','ordinary','fp32')
    cases.append(dict(name='ordinary_no_logprobs_has_no_score_output',result='PASS',kind='original_Sampler_forward_opaque_tensor_protocol',trace=trace.copy()))
    trace.clear();metadata.max_num_logprobs=-1;fwd(sampler,Tag('bonus_fp32','fp32'),metadata,True,'raw_logits');assert trace[0]==('clone','bonus_fp32')
    cases.append(dict(name='bonus_raw_fp32_clone_boundary',result='PASS',kind='original_Sampler_forward_opaque_tensor_protocol',trace=trace.copy()))
    class Event:
        def __init__(self):self.ready=False
        def synchronize(self):trace.append('copy_event_sync');self.ready=True
    class CPU:
        def __init__(self,rows,event):self.rows=rows;self.event=event
        def tolist(self):assert self.event.ready;trace.append('CPU_read');return copy.deepcopy(self.rows)
    repair=original('vllm/vllm/v1/worker/gpu_input_batch.py','InputBatch','update_async_output_token_ids',{})
    event=Event();trace.clear();state=NS(sampling_metadata=NS(output_token_ids=[[100,-1,-1,-1,-1,-1],[200,-1]]),sampled_token_ids_cpu=CPU([[8,-1,-1,-1,-1,-1],[3,4,-1,-1,-1,-1]],event),async_copy_ready_event=event,prev_req_id_to_index={'a':0,'b':1},req_ids=['b','a'])
    repair(state);assert state.sampling_metadata.output_token_ids==[[100,3,4],[200,8]] and trace==['copy_event_sync','CPU_read']
    cases.append(dict(name='reorder_and_optimistic_placeholder_repair',result='PASS',kind='original_Python_ownership_and_ID_lists',trace=trace.copy()))
    trace.clear();state.sampling_metadata.output_token_ids=[];repair(state);assert trace==[]
    cases.append(dict(name='no_output_reader_does_not_wait',result='PASS',kind='original_Python_guard'))
    spec_repair=original('vllm/vllm/v1/worker/gpu_input_batch.py','InputBatch','update_async_spec_token_ids',{})
    state=NS(prev_req_id_to_index={'a':0,'b':1},req_ids=['b','a'],sampling_metadata=NS(spec_token_ids=[[-1,-1],[-1]]));drafts=[[7,8,9],[1,2,3]]
    spec_repair(state,drafts);assert state.sampling_metadata.spec_token_ids==[[1,2],[7]] and drafts==[[7],[1,2]]
    cases.append(dict(name='draft_repair_trims_original_CPU_lists_in_place',result='PASS',kind='original_Python_alias_protocol'))
    class AsyncBase:pass
    class Ready(AsyncBase):
        def get_output(self):trace.append('get_output');return 'FULL_OUTPUT'
    class Failed(AsyncBase):
        def get_output(self):trace.append('get_output_failure');raise RuntimeError('controlled failure')
    enqueue=original('vllm/vllm/v1/executor/multiproc_executor.py','WorkerProc','enqueue_output',dict(AsyncModelRunnerOutput=AsyncBase,WorkerProc=NS(ResponseStatus=NS(SUCCESS='SUCCESS',FAILURE='FAILURE')),logger=logger))
    trace.clear();worker=NS(worker_response_mq=NS(enqueue=lambda x:trace.append(x)));enqueue(worker,Ready());assert trace==['get_output',('SUCCESS','FULL_OUTPUT')]
    cases.append(dict(name='executor_completes_full_async_output_before_publish',result='PASS',kind='original_Python_RPC_protocol',trace=trace.copy()))
    trace.clear();enqueue(worker,Failed());assert trace==['get_output_failure',('FAILURE','controlled failure')]
    cases.append(dict(name='async_get_output_error_becomes_failure_response',result='PASS',kind='original_Python_error_protocol',trace=trace.copy()))
    neg=float('-inf');masked=[neg,neg,4,neg];wrong_restore=[5,neg,4,neg]
    argmax=lambda row:max(range(len(row)),key=lambda i:row[i])
    assert argmax(masked)==2 and argmax(wrong_restore)==0
    cases.append(dict(name='local_all_negative_infinity_is_not_global_restore',result='PASS',kind='finite_counterexample_not_native',baseline_winner=2,incorrect_local_restore_winner=0))
    forced=[float('inf'),1e9];assert argmax(forced)==0
    cases.append(dict(name='thinking_1e9_write_is_not_forced_ID',result='PASS',kind='finite_counterexample_not_native',forced_token=1,baseline_winner=0))
    out=dict(status='ORIGINAL_PYTHON_GUARD_AND_DELIVERY_PROTOCOL_PASS_NOT_NUMERICAL_ADMISSION',cases=cases,original_methods=USED,total=len(cases),all_pass=True,qualification='Original function bodies extracted without rewrites except removed decorators; constructors/tensors/RPC/streams are opaque tags. No Torch/NumPy/NPU imports or native/compiler/numerical proof; finite examples reject unsafe simplifications only.',every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,remote_writes=[],script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'target-output-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(controls=len(cases),original_methods=len(USED),all_pass=True,native_admissions=0)))
if __name__=='__main__':main()
