"""Native two-step overlap using the SDK's asynchronous sampled outputs.

Greedy ignore-EOS decode batches may run one step ahead. Ordinary EOS-aware
requests and the last length-capped step resolve inline, preserving native
termination and ensuring no cache is released while device work uses it.
"""
from dataclasses import dataclass
import math

from .sdk_dispatch import SDKBatchWorkerAdapter


@dataclass
class _Step:
    metadata: object
    scheduler_output: object
    is_prompt: bool
    output: object
    end_event: object


class SDKAsyncBatchWorkerAdapter(SDKBatchWorkerAdapter):
    expected_nextn = 0
    allow_speculative_metadata = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deferred = None
        self.async_stats = {'next_execute_before_previous_output': 0,
            'current_npu_pending_during_previous_cpu_output': 0,
            'deferred_decode_batches': 0, 'inline_batches': 0}

    def _resolve_step(self, worker, step, *, driver):
        import numpy as np
        import torch
        from deepinfer._C.engine import SequenceStatus
        from deepinfer.sequence import SampledInfo, SamplerOutput
        from deepinfer.worker.zos_status import WorkerStepState
        output = self.runtime.resolve_output(step.output)
        if not driver:
            return np.empty(0, dtype=np.uint64)
        emitted = self.controller.commit_completed(output)
        tokens, logprobs = [], []
        for meta in step.metadata:
            rid = meta.request_id
            if rid not in emitted:
                tokens.append(-1)
                logprobs.append(0.0)
                continue
            token = emitted[rid]
            index = output.req_id_to_index[rid]
            if output.logprobs is None:
                raise ValueError('native async output requires its actual sampled-token logprob')
            row = output.logprobs.slice_request(index, 1)
            ids = row.logprob_token_ids.reshape(-1).tolist()
            values = row.logprobs.reshape(-1).tolist()
            if len(ids) != 1 or ids[0] != token or not math.isfinite(values[0]):
                raise ValueError('async sampled-token logprob disagrees with output')
            tokens.append(token)
            logprobs.append(float(values[0]))
        sampled = SamplerOutput(sampled_token=SampledInfo(bs=len(tokens),
            token_ids=torch.tensor(tokens, dtype=torch.int32),
            token_logprobs=torch.tensor(logprobs, dtype=torch.float32)), post_funcs=[])
        state = WorkerStepState(seq_metadata_list=step.metadata,
            scheduler_outputs=step.scheduler_output, sampler_output=sampled,
            is_prompt=step.is_prompt, is_placeholder=False, sample_end_evt=step.end_event)
        result = worker.engine_callbacks.post_process_fn(state)
        if not isinstance(result, np.ndarray) or result.dtype != np.uint64:
            raise ValueError('native async callback returned an invalid output vector')
        terminal = {SequenceStatus.FINISHED_STOPPED, SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED, SequenceStatus.FINISHED_IGNORED,
            SequenceStatus.FINISHED_PREEMPTED, SequenceStatus.FINISHED_SO_FAILED,
            SequenceStatus.FINISHED_VE_FAILED}
        finished = [m.request_id for m, seq in
            zip(step.metadata, step.scheduler_output.scheduled_seqs, strict=True)
            if seq.status in terminal]
        self.controller.finish(finished)
        return result

    def _can_defer(self, metadata, is_prompt):
        if is_prompt:
            return False
        # A deferred request cannot finish when its result is delivered on the
        # next step. This avoids speculative extra execution or late cache free.
        return all((req := self.controller.requests[m.request_id].request).sampling_params.ignore_eos
            and req.num_output_tokens + req.num_output_placeholders < req.sampling_params.max_tokens
            and req.num_computed_tokens < self.controller.max_context
            for m in metadata)

    def execute_model_step(self, worker, is_prompt, is_placeholder,
                           seq_metadata_list, scheduler_outputs):
        import numpy as np
        import torch
        import torch.distributed as dist
        from deepinfer import envs
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        driver = dist.get_rank(self.group) == 0
        if (self.poisoned or worker.nextn != self.expected_nextn or worker.prev_step_state is not None
                or envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or is_placeholder
                or bool(worker.is_driver_worker) != driver):
            raise ValueError('unsupported native async worker lifecycle')
        message = None
        failure = None
        if driver:
            try:
                if (not seq_metadata_list or scheduler_outputs is None
                        or len(seq_metadata_list) != len(scheduler_outputs.scheduled_seqs)
                        or any(bool(m.is_prompt) != bool(is_prompt) for m in seq_metadata_list)
                        or (not self.allow_speculative_metadata and any(
                            m.last_next_n_token or m.last_accepted_token for m in seq_metadata_list))):
                    raise ValueError('native async scheduled metadata is incomplete or speculative')
                descriptor = self.controller.prepare(seq_metadata_list)
                self.batch_sizes.append(len(seq_metadata_list))
                message = {'ok': True, 'descriptor': descriptor}
            except Exception as error:
                failure = error
                message = {'ok': False, 'error': str(error)}
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('native async preparation failed: ' + message['error']) from failure
        try:
            output = self.runtime.execute(message['descriptor'])
            end = torch.npu.Event()
            end.record()
        except Exception:
            self.poisoned = True
            raise
        previous = self.deferred
        step = _Step(seq_metadata_list, scheduler_outputs, is_prompt, output, end)
        result = np.empty(0, dtype=np.uint64)
        message = None
        if driver:
            try:
                pieces = []
                self.controller.submitted()
                worker.engine_callbacks.pre_sampler_sync_fn(scheduler_outputs)
                worker.maybe_gc_collect()
                if previous is not None:
                    self.async_stats['next_execute_before_previous_output'] += 1
                    if not end.query():
                        self.async_stats['current_npu_pending_during_previous_cpu_output'] += 1
                    pieces.append(self._resolve_step(worker, previous, driver=True))
                self.deferred = None
                defer = self._can_defer(seq_metadata_list, is_prompt)
                if defer:
                    self.deferred = step
                    self.async_stats['deferred_decode_batches'] += 1
                else:
                    pieces.append(self._resolve_step(worker, step, driver=True))
                    self.async_stats['inline_batches'] += 1
                if pieces:
                    result = np.concatenate(pieces)
                message = {'ok': True, 'defer': defer}
            except Exception as error:
                self.poisoned = True
                failure = error
                message = {'ok': False, 'error': str(error)}
        # One acknowledgement at the same phase on every rank, including any
        # driver-only validation failure. It carries no sampled token.
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('native async execution failed: ' + message['error']) from failure
        if not driver:
            if previous is not None:
                self._resolve_step(worker, previous, driver=False)
            if message['defer']:
                self.deferred = step
                self.async_stats['deferred_decode_batches'] += 1
            else:
                self._resolve_step(worker, step, driver=False)
                self.deferred = None
                self.async_stats['inline_batches'] += 1
        return result

    def flush_finished(self):
        if self.deferred is not None:
            raise RuntimeError('native async deferred output remains at successful shutdown')
        return super().flush_finished()
