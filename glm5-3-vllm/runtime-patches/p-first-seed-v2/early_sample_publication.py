"""Publish verified sample data early, retaining native late retirement.

Early(k) and native Late(k-1) are FIFO on the same copy stream. A reply from
Early(k) retires only the preceding late sequence. Empty work drains the last
native event; KV-copy-only work gets an equivalent stream retirement event.
"""
from collections import deque
from contextvars import ContextVar
from copy import copy
import functools
import json
import os
from pathlib import Path
import time


def install(ns, root=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    torch, Runner, Async = ns['torch'], ns['NPUModelRunner'], ns['AsyncGPUModelRunnerOutput']
    root = Path('/out/early-sample-publication-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    frame_var = ContextVar('glm53_early_publication', default=None)
    execute_native, sample_native = Runner.execute_model, Runner.sample_tokens
    book_native, init_native, get_native = Runner._bookkeeping_sync, Async.__init__, Async.get_output

    def state(runner):
        if not hasattr(runner, '_glm53_publication_state'):
            assert runner.is_kv_consumer and not runner.is_kv_producer and runner.use_async_scheduling
            assert runner.vllm_config.max_concurrent_batches == 2
            runner._glm53_publication_state = dict(last_seq=0, last_event=None, current=None,
                                                 stream=runner.async_output_copy_stream,
                                                 retained=deque(), records=0)
        value = runner._glm53_publication_state
        assert value['stream'] is runner.async_output_copy_stream
        return value

    def emit(runner, q, **row):
        if (root / 'disabled').exists() or q['records'] >= 6000:
            return
        q['records'] += 1
        row.update(rank=runner.tp_rank, pid=os.getpid(), monotonic_ns=time.monotonic_ns())
        with (root / ('rank-%d-%d.jsonl' % (runner.tp_rank, os.getpid()))).open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')

    def prune(runner, q):
        while q['retained'] and q['retained'][0]['event'].query():
            entry = q['retained'].popleft()
            early = entry['early']
            equal = None
            if early is not None:
                # Late completion covers both FIFO destinations. This check
                # never waits and remains enabled during formal comparisons.
                equal = bool(torch.equal(early.sampled_token_ids_cpu, entry['late'].sampled_token_ids_cpu))
                assert equal, 'Published early sample differs from native late sample'
            emit(runner, q, kind='late_retired', seq=entry['seq'], early_copy=early is not None,
                 bitwise_equal=equal, source_kept_until_native_late=True)

    def annotate(output, seq, retired, early):
        assert isinstance(output, ns['ModelRunnerOutput'])
        output.glm53_gpu_step_seq = seq
        output.glm53_gpu_retired_through = retired
        output.glm53_gpu_early_sample = early
        output.glm53_gpu_retire_rank_count = 1
        return output

    @functools.wraps(execute_native)
    def execute(self, scheduled, *args, **kwargs):
        seq = scheduled.glm53_gpu_step_seq
        if seq is None:
            assert not self.requests, 'Real D work lacks retirement protocol metadata'
            return execute_native(self, scheduled, *args, **kwargs)
        q = state(self)
        assert q['current'] is None
        nonempty = scheduled.total_num_scheduled_tokens > 0
        writes = bool(scheduled.kv_cache_block_copies or scheduled.new_block_ids_to_zero)
        assert scheduled.glm53_gpu_step_has_work == (nonempty or writes)
        assert seq == q['last_seq'] + int(nonempty or writes)
        prune(self, q)
        if nonempty:
            # Early replies release EngineCore queue slots before the native
            # late tail retires. Preserve the original physical two-batch
            # budget before reusing CPU/device input preparation buffers.
            if len(q['retained']) >= self.vllm_config.max_concurrent_batches:
                waited = q['retained'][0]['seq']
                q['retained'][0]['event'].synchronize()
                prune(self, q)
                emit(self, q, kind='gpu_admission_wait', next_seq=seq,
                     waited_native_late_seq=waited, retained_after=len(q['retained']))
            assert len(q['retained']) < self.vllm_config.max_concurrent_batches
            q['current'] = seq
            result = execute_native(self, scheduled, *args, **kwargs)
            assert result is None, 'Only native PP1 generative execution is supported'
            return result
        # Keep actual original retirement separate from already published data.
        # An empty drain must continue after final completion or cancellation.
        if q['last_event'] is not None:
            q['last_event'].synchronize()
        prune(self, q)
        assert not q['retained']
        result = execute_native(self, scheduled, *args, **kwargs)
        if writes:
            event = torch.npu.Event(blocking=True)
            current = torch.npu.current_stream()
            with torch.npu.stream(q['stream']):
                q['stream'].wait_stream(current)
                event.record()
            event.synchronize()
            q.update(last_seq=seq, last_event=event)
        assert seq == q['last_seq']
        result = annotate(copy(result), seq, seq, False)
        emit(self, q, kind='empty_drain', seq=seq, kv_writes=writes,
             retained=0, original_late_waited=True)
        return result
    Runner.execute_model = execute

    def eligible(runner, grammar):
        p = runner.parallel_config
        ids = list(runner.input_batch.req_ids)
        return bool(grammar is None and runner.execute_model_state is not None and ids
                    and runner.num_spec_tokens == 5 and runner.model_config.max_model_len == 8192
                    and p.tensor_parallel_size == 16 and p.data_parallel_size == 1
                    and p.pipeline_parallel_size == 1 and p.decode_context_parallel_size == 1
                    and p.prefill_context_parallel_size == 1 and not ns['enable_sp']()
                    and runner.model_config.use_mla and not runner.model_config.is_hybrid
                    and not runner.dynamic_eplb and not getattr(runner, 'lora_config', None)
                    and all(rid in runner.requests and runner.requests[rid].sampling_params is not None
                            and runner.requests[rid].sampling_params.max_tokens > 1 for rid in ids))

    @functools.wraps(sample_native)
    def sample(self, grammar_output, *args, **kwargs):
        q = state(self)
        if q['current'] is None:
            assert self.execute_model_state is None
            return sample_native(self, grammar_output, *args, **kwargs)
        frame = dict(runner=self, q=q, seq=q['current'], previous=q['last_seq'],
                     req_ids=list(self.input_batch.req_ids), ctor_calls=0,
                     want=(root / 'enable').exists() and eligible(self, grammar_output))
        token = frame_var.set(frame)
        try:
            result = sample_native(self, grammar_output, *args, **kwargs)
            assert frame['ctor_calls'] == 1 and q['last_seq'] == frame['seq']
            return result
        finally:
            q['current'] = None
            frame_var.reset(token)
    Runner.sample_tokens = sample

    @functools.wraps(book_native)
    def book(self, scheduled, sampler_output, *args, **kwargs):
        result = book_native(self, scheduled, sampler_output, *args, **kwargs)
        frame = frame_var.get()
        if frame is None or not frame['want']:
            return result
        assert frame['runner'] is self and 'early' not in frame and len(result) == 8
        tensor = sampler_output.sampled_token_ids
        if (sampler_output.logprobs_tensors is not None or result[1] is not None
                or tensor.device.type != 'npu' or tensor.dtype not in (torch.int32, torch.int64)
                or tensor.ndim != 2 or tuple(tensor.shape) != (len(frame['req_ids']), 6)
                or not tensor.is_contiguous()):
            frame['refused'] = 'sample/logprobs/nans contract'
            return result
        early = object.__new__(Async)
        init_native(early, model_runner_output=None, sampled_token_ids=tensor,
                    logprobs_tensors=None, invalid_req_indices=list(result[7]),
                    async_output_copy_stream=self.async_output_copy_stream,
                    vocab_size=self.input_batch.vocab_size, num_nans=None)
        frame.update(early=early, source=tensor)
        return result
    Runner._bookkeeping_sync = book

    @functools.wraps(init_native)
    def init(self, *args, **kwargs):
        frame = frame_var.get()
        init_native(self, *args, **kwargs)
        if frame is None:
            return
        frame['ctor_calls'] += 1
        assert frame['ctor_calls'] == 1
        runner, q = frame['runner'], frame['q']
        stream = kwargs.get('async_output_copy_stream', args[4] if len(args) > 4 else None)
        assert stream is q['stream'] and q['last_seq'] == frame['previous']
        early = frame.get('early')
        if early is not None:
            source = kwargs.get('sampled_token_ids', args[1] if len(args) > 1 else None)
            assert source is frame['source']
            assert early.async_copy_ready_event is not self.async_copy_ready_event
        seq = frame['seq']
        self._glm53_publication = dict(seq=seq, previous=frame['previous'], early=early,
                                      rank=runner.tp_rank, req_ids=frame['req_ids'])
        q['retained'].append(dict(seq=seq, late=self, early=early, event=self.async_copy_ready_event))
        assert len(q['retained']) <= runner.vllm_config.max_concurrent_batches == 2
        q.update(last_seq=seq, last_event=self.async_copy_ready_event)
        emit(runner, q, kind='native_late_kept', seq=seq, previous=frame['previous'],
             early_copy=early is not None, shape=list(self.sampled_token_ids_cpu.shape),
             native_event_for_InputBatch=True, same_copy_stream=True,
             retained_gpu_steps=len(q['retained']), physical_gpu_limit=2,
             refused=frame.get('refused'))
    Async.__init__ = init

    @functools.wraps(get_native)
    def get(self):
        meta = getattr(self, '_glm53_publication', None)
        if meta is None:
            return get_native(self)
        early = meta['early']
        if early is None:
            result = get_native(self)
            retired = meta['seq']
        else:
            early._model_runner_output = self._model_runner_output
            result = get_native(early)
            # Same copy-stream FIFO proves only preceding native late work.
            # The current late object/event/source remain retained by runner.
            retired = meta['previous']
        assert result.req_ids == meta['req_ids']
        return annotate(result, meta['seq'], retired, early is not None)
    Async.get_output = get
