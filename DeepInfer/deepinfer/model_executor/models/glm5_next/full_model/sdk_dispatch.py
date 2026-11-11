"""Candidate native Worker dispatch for SDK batched computation.

DeepInfer still owns the C++ scheduler and output callbacks. SDK descriptors
are compute/cache inputs, not scheduling decisions. Not deployed yet.
"""
import math


class SDKBatchWorkerAdapter:
    def __init__(self, runtime, controller, group):
        self.runtime, self.controller, self.group = runtime, controller, group
        self.poisoned = False
        self.batch_sizes = []

    def register_request(self, request_id, prompt_ids, sampling_params):
        if self.controller is None:
            raise RuntimeError('request admission belongs to the native driver')
        self.controller.register(request_id, prompt_ids, sampling_params)
        return self.controller.prefill_checkpoints(request_id)

    def execute_model_step(self, worker, is_prompt, is_placeholder,
                           seq_metadata_list, scheduler_outputs):
        import numpy as np
        import torch
        import torch.distributed as dist
        from deepinfer import envs
        from deepinfer._C.engine import SequenceStatus
        from deepinfer.sequence import SampledInfo, SamplerOutput
        from deepinfer.worker.zos_status import WorkerStepState
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict

        driver = dist.get_rank(self.group) == 0
        if (self.poisoned or worker.nextn != 0 or worker.prev_step_state is not None
                or envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or is_placeholder
                or bool(worker.is_driver_worker) != driver):
            raise ValueError('candidate supports synchronous ordinary native text batches only')
        failure = None
        message = None
        if driver:
            try:
                if (not seq_metadata_list or scheduler_outputs is None
                        or len(seq_metadata_list) != len(scheduler_outputs.scheduled_seqs)
                        or any(bool(m.is_prompt) != bool(is_prompt) for m in seq_metadata_list)
                        or any(m.last_next_n_token or m.last_accepted_token for m in seq_metadata_list)):
                    raise ValueError('native scheduled metadata is missing, mixed or speculative')
                descriptor = self.controller.prepare(seq_metadata_list)
                self.batch_sizes.append(len(seq_metadata_list))
                message = {'ok': True, 'descriptor': descriptor}
            except Exception as error:
                failure = error
                message = {'ok': False, 'error': str(error)}
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('native batch preparation rejected: '+message['error']) from failure
        try:
            sdk_output = self.runtime.execute(message['descriptor'])
        except Exception:
            self.poisoned = True
            raise
        result = np.empty(0, dtype=np.uint64)
        message = None
        if driver:
            try:
                emitted = self.controller.commit(sdk_output)
                tokens, logprobs = [], []
                for meta in seq_metadata_list:
                    rid = meta.request_id
                    if rid not in emitted:
                        # Same unpublished placeholder contract as the existing
                        # native bridge for non-final prefill chunks.
                        tokens.append(-1)
                        logprobs.append(0.0)
                        continue
                    token = emitted[rid]
                    index = sdk_output.req_id_to_index[rid]
                    if sdk_output.logprobs is None:
                        raise ValueError('native C++ output requires the actual sampled logprob')
                    row = sdk_output.logprobs.slice_request(index, 1)
                    ids = row.logprob_token_ids.reshape(-1).tolist()
                    values = row.logprobs.reshape(-1).tolist()
                    if len(ids) != 1 or ids[0] != token or not math.isfinite(values[0]):
                        raise ValueError('SDK sampled-token logprob does not match native output token')
                    tokens.append(token)
                    logprobs.append(float(values[0]))
                sampled = SamplerOutput(sampled_token=SampledInfo(
                    bs=len(tokens), token_ids=torch.tensor(tokens, dtype=torch.int32),
                    token_logprobs=torch.tensor(logprobs, dtype=torch.float32)), post_funcs=[])
                worker.engine_callbacks.pre_sampler_sync_fn(scheduler_outputs)
                worker.maybe_gc_collect()
                event = torch.npu.Event(enable_timing=True)
                event.record()
                event.synchronize()
                state = WorkerStepState(seq_metadata_list=seq_metadata_list,
                    scheduler_outputs=scheduler_outputs, sampler_output=sampled,
                    is_prompt=is_prompt, is_placeholder=False, sample_end_evt=event)
                result = worker.engine_callbacks.post_process_fn(state)
                if not isinstance(result, np.ndarray) or result.dtype != np.uint64:
                    raise ValueError('native callback returned an invalid output vector')
                terminal = {SequenceStatus.FINISHED_STOPPED, SequenceStatus.FINISHED_LENGTH_CAPPED,
                    SequenceStatus.FINISHED_ABORTED, SequenceStatus.FINISHED_IGNORED,
                    SequenceStatus.FINISHED_PREEMPTED, SequenceStatus.FINISHED_SO_FAILED,
                    SequenceStatus.FINISHED_VE_FAILED}
                finished = [meta.request_id for meta, seq in
                    zip(seq_metadata_list, scheduler_outputs.scheduled_seqs, strict=True)
                    if seq.status in terminal]
                self.controller.finish(finished)
                message = {'ok': True}
            except Exception as error:
                failure = error
                message = {'ok': False, 'error': str(error)}
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('native batch commit failed: '+message['error']) from failure
        return result

    def flush_finished(self):
        import torch.distributed as dist
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        failure = None
        message = None
        if dist.get_rank(self.group) == 0:
            try:
                message = {'ok': True, 'descriptor': self.controller.prepare_cleanup()}
            except Exception as error:
                failure = error
                message = {'ok': False, 'error': str(error)}
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('batch cleanup preparation failed: '+message['error']) from failure
        output = self.runtime.resolve_output(self.runtime.execute(message['descriptor']))
        if self.controller is not None:
            self.controller.commit(output)
        self.runtime.drain()

    def close(self):
        # Successful callers have already flushed finished-request descriptors.
        # Failure teardown discards the whole worker/cache state; do not attempt
        # a new model collective when a peer may have failed.
        self.runtime.drain()
