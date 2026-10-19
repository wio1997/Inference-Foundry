"""Compare an early read-only sample copy with the unchanged native completion.

This diagnostic never publishes from its early event. The real Async object,
InputBatch event, original output parsing and KV retirement stay native.
"""
from contextvars import ContextVar
import functools
import json
import os
from pathlib import Path
import time


def install(ns, root=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    torch, Runner, Async = ns['torch'], ns['NPUModelRunner'], ns['AsyncGPUModelRunnerOutput']
    root = Path('/out/early-sample-copy-shadow') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    frame_var = ContextVar('glm53_early_sample_copy_shadow', default=None)
    sample_native, book_native = Runner.sample_tokens, Runner._bookkeeping_sync
    init_native, get_native = Async.__init__, Async.get_output
    records = 0

    def eligible(runner, grammar):
        p = runner.parallel_config
        ids = list(runner.input_batch.req_ids)
        return bool(grammar is None and runner.execute_model_state is not None and ids
                    and runner.is_kv_consumer and not runner.is_kv_producer
                    and runner.use_async_scheduling and runner.num_spec_tokens == 5
                    and runner.model_config.max_model_len == 8192
                    and p.tensor_parallel_size == 16 and p.data_parallel_size == 1
                    and p.pipeline_parallel_size == 1 and p.decode_context_parallel_size == 1
                    and p.prefill_context_parallel_size == 1 and not ns['enable_sp']()
                    and not runner.model_config.is_hybrid and runner.model_config.use_mla
                    and not runner.dynamic_eplb and not getattr(runner, 'lora_config', None)
                    and all(rid in runner.requests and runner.requests[rid].sampling_params is not None
                            and runner.requests[rid].sampling_params.max_tokens > 1 for rid in ids))

    @functools.wraps(sample_native)
    def sample(self, grammar_output, *args, **kwargs):
        frame = None
        if (root / 'enable').exists() and eligible(self, grammar_output):
            frame = dict(runner=self, req_ids=list(self.input_batch.req_ids), book_calls=0)
        token = frame_var.set(frame)
        try:
            return sample_native(self, grammar_output, *args, **kwargs)
        finally:
            frame_var.reset(token)
    Runner.sample_tokens = sample

    @functools.wraps(book_native)
    def book(self, scheduler_output, sampler_output, *args, **kwargs):
        result = book_native(self, scheduler_output, sampler_output, *args, **kwargs)
        frame = frame_var.get()
        if frame is None or frame['runner'] is not self:
            return result
        frame['book_calls'] += 1
        assert frame['book_calls'] == 1 and len(result) == 8
        tensor = sampler_output.sampled_token_ids
        # Restrict the first diagnostic to the ordinary no-logprobs path.
        # Every refused case retains the original late copy unchanged.
        if (sampler_output.logprobs_tensors is not None or result[1] is not None
                or tensor.device.type != 'npu' or tensor.dtype not in (torch.int32, torch.int64)
                or tensor.ndim != 2 or tuple(tensor.shape) != (len(frame['req_ids']), 6)
                or not tensor.is_contiguous()):
            frame['refused'] = 'sample/logprobs/nans contract'
            return result
        # Call the exact saved native constructor, without this module's hook.
        # Its source references keep the GPU sample alive through both copies.
        shadow = object.__new__(Async)
        init_native(shadow, model_runner_output=None, sampled_token_ids=tensor,
                    logprobs_tensors=None, invalid_req_indices=list(result[7]),
                    async_output_copy_stream=self.async_output_copy_stream,
                    vocab_size=self.input_batch.vocab_size, num_nans=None)
        frame.update(shadow=shadow, source=tensor, invalid=list(result[7]),
                     early_queued_ns=time.monotonic_ns())
        return result
    Runner._bookkeeping_sync = book

    @functools.wraps(init_native)
    def init(self, *args, **kwargs):
        frame = frame_var.get()
        probe = None
        if frame is not None and 'shadow' in frame:
            tensor = kwargs.get('sampled_token_ids', args[1] if len(args) > 1 else None)
            assert tensor is frame['source'], 'Native constructor changed its source object'
            probe = dict(frame)
            probe['early_ready_at_late_constructor'] = bool(frame['shadow'].async_copy_ready_event.query())
            probe['late_constructor_ns'] = time.monotonic_ns()
        init_native(self, *args, **kwargs)
        if probe is not None:
            assert self.async_copy_ready_event is not probe['shadow'].async_copy_ready_event
            self._glm53_early_sample_shadow = probe
    Async.__init__ = init

    @functools.wraps(get_native)
    def get(self):
        nonlocal records
        probe = getattr(self, '_glm53_early_sample_shadow', None)
        if probe is None:
            return get_native(self)
        early_ready = bool(probe['shadow'].async_copy_ready_event.query())
        late_ready = bool(self.async_copy_ready_event.query())
        start_ns = time.monotonic_ns()
        result = get_native(self)
        native_return_ns = time.monotonic_ns()
        # Do not read the early destination until its own native event is ready.
        # Native late completion should already cover the FIFO copy stream.
        assert probe['shadow'].async_copy_ready_event.query()
        assert self.async_copy_ready_event.query()
        equal = bool(torch.equal(probe['shadow'].sampled_token_ids_cpu, self.sampled_token_ids_cpu))
        assert equal, 'Early sampled IDs differ from unchanged late native copy'
        assert result.req_ids == probe['req_ids']
        if records < 4000:
            records += 1
            row = dict(kind='early_sample_shadow', rank=probe['runner'].tp_rank, pid=os.getpid(),
                       req_ids=probe['req_ids'], shape=list(self.sampled_token_ids_cpu.shape),
                       dtype=str(self.sampled_token_ids_cpu.dtype), bitwise_equal=equal,
                       early_ready_at_late_constructor=probe['early_ready_at_late_constructor'],
                       early_ready_at_native_get=early_ready, late_ready_at_native_get=late_ready,
                       native_get_begin_ns=start_ns, native_get_return_ns=native_return_ns,
                       native_wait_parse_us=(native_return_ns-start_ns)/1000,
                       early_queued_ns=probe['early_queued_ns'], late_constructor_ns=probe['late_constructor_ns'],
                       native_event_kept=True, early_publication=False,
                       source_keepalive_until_native_completion=True)
            with (root / ('rank-%d-%d.jsonl' % (probe['runner'].tp_rank, os.getpid()))).open('a') as f:
                f.write(json.dumps(row, sort_keys=True) + '\n')
        del self._glm53_early_sample_shadow
        return result
    Async.get_output = get
