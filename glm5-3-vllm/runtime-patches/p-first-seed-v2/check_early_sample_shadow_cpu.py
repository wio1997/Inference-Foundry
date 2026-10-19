"""Exercise actual hook ownership and fallback behavior with CPU stand-ins."""
from pathlib import Path
from types import SimpleNamespace as S
import importlib.util
import json
import os
import tempfile

os.environ['GLM53_P_FIRST_DRAFT_EXPORT'] = '0'
path = Path(__file__).with_name('early_sample_copy_shadow.py')
spec = importlib.util.spec_from_file_location('shadow', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
events = []

class Event:
    def __init__(self): self.ready = False; events.append(self)
    def query(self): return self.ready

class Tensor:
    def __init__(self, values, device='npu', dtype='i64'):
        self.values = [list(z) for z in values]; self.device = S(type=device)
        self.dtype = dtype; self.ndim = 2; self.shape = (len(values), len(values[0]))
    def is_contiguous(self): return True

class Async:
    copies = 0
    def __init__(self, model_runner_output, sampled_token_ids, **kwargs):
        Async.copies += 1
        self._model_runner_output = model_runner_output
        self._sampled_token_ids = sampled_token_ids
        self.sampled_token_ids_cpu = Tensor(sampled_token_ids.values, 'cpu', sampled_token_ids.dtype)
        self.async_copy_ready_event = Event()
    def get_output(self):
        for event in events: event.ready = True
        result = self._model_runner_output
        result.sampled_token_ids = self.sampled_token_ids_cpu.values
        del self._sampled_token_ids
        return result

class Runner:
    def __init__(self):
        self.parallel_config = S(tensor_parallel_size=16, data_parallel_size=1, pipeline_parallel_size=1,
                                 decode_context_parallel_size=1, prefill_context_parallel_size=1)
        self.model_config = S(max_model_len=8192, is_hybrid=False, use_mla=True)
        self.input_batch = S(req_ids=['r'], vocab_size=100)
        self.requests = {'r': S(sampling_params=S(max_tokens=64))}
        self.execute_model_state = object(); self.is_kv_consumer = True; self.is_kv_producer = False
        self.use_async_scheduling = True; self.num_spec_tokens = 5; self.dynamic_eplb = False
        self.tp_rank = 0; self.async_output_copy_stream = object(); self.mutate = False; self.fail = False
        self.sampler = S(sampled_token_ids=Tensor([[1,2,3,-1,-1,-1]]), logprobs_tensors=None)
    def _bookkeeping_sync(self, scheduler_output, sampler_output):
        if self.fail: raise ValueError('native bookkeeping failed')
        return (None, None, None, [], {}, ['r'], {'r':0}, [])
    def sample_tokens(self, grammar_output):
        self._bookkeeping_sync(None, self.sampler)
        if self.mutate: self.sampler.sampled_token_ids.values[0][0] = 99
        output = S(req_ids=['r'], sampled_token_ids=None)
        result = Async(model_runner_output=output, sampled_token_ids=self.sampler.sampled_token_ids,
                       logprobs_tensors=self.sampler.logprobs_tensors, invalid_req_indices=[],
                       async_output_copy_stream=self.async_output_copy_stream, vocab_size=100, num_nans=None)
        self.input_batch.native_event = result.async_copy_ready_event
        return result

torch = S(int32='i32', int64='i64', equal=lambda a,b: a.values == b.values)
checks = []
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    m.install(dict(torch=torch, NPUModelRunner=Runner, AsyncGPUModelRunnerOutput=Async, enable_sp=lambda:False), root)
    runner = Runner(); before = Async.copies; native = runner.sample_tokens(None)
    assert Async.copies-before == 1 and not hasattr(native, '_glm53_early_sample_shadow')
    checks.append('default_off_has_only_native_copy')
    (root/'enable').touch(); before = Async.copies; out = runner.sample_tokens(None)
    probe = out._glm53_early_sample_shadow
    assert Async.copies-before == 2 and probe['source'] is runner.sampler.sampled_token_ids
    assert runner.input_batch.native_event is out.async_copy_ready_event
    assert out.async_copy_ready_event is not probe['shadow'].async_copy_ready_event
    expected = out._model_runner_output; actual = out.get_output()
    assert actual is expected and actual.sampled_token_ids == [[1,2,3,-1,-1,-1]]
    assert not hasattr(out, '_glm53_early_sample_shadow')
    row = json.loads(next(root.glob('rank-*.jsonl')).read_text().splitlines()[-1])
    assert row['bitwise_equal'] and row['native_event_kept'] and not row['early_publication']
    checks += ['source_kept_until_native_completion', 'input_batch_keeps_distinct_native_late_event',
               'native_output_identity_and_ids_preserved', 'shadow_released_after_native_completion']
    for field,value in [('is_kv_producer',True),('num_spec_tokens',1),('use_async_scheduling',False)]:
        r = Runner(); setattr(r,field,value); before = Async.copies; out = r.sample_tokens(None)
        assert Async.copies-before == 1 and not hasattr(out,'_glm53_early_sample_shadow')
        checks.append(field+'_native_fallback')
    for why, change in [('grammar',lambda r:None),('logprobs',lambda r:setattr(r.sampler,'logprobs_tensors',object())),
                        ('shape',lambda r:setattr(r.sampler,'sampled_token_ids',Tensor([[1]])))]:
        r=Runner();change(r);before=Async.copies;out=r.sample_tokens(object() if why=='grammar' else None)
        assert Async.copies-before == 1 and not hasattr(out,'_glm53_early_sample_shadow')
        checks.append(why+'_native_fallback')
    r=Runner();r.fail=True
    try:r.sample_tokens(None);raise AssertionError('native error swallowed')
    except ValueError:pass
    r.fail=False;assert hasattr(r.sample_tokens(None),'_glm53_early_sample_shadow')
    checks.append('native_error_propagates_and_context_restores')
    r=Runner();r.mutate=True;out=r.sample_tokens(None)
    try:out.get_output();raise RuntimeError('missed changed source')
    except AssertionError as exc:assert 'differ' in str(exc)
    checks.append('post_copy_source_mutation_detected')
print(json.dumps(dict(completed=True, passed=len(checks), checks=checks,
                     scope='CPU hook ownership and refusal checks, not NPU stream safety or service performance.')))
