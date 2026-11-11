"""Initial native MTP Worker bridge, with real variable acceptance outputs.

The CPU acceptance/draft dependency is resolved inline. This is a functional
integration stage, not a claim of cross-batch async parity or performance gain.
"""
import math

from .sdk_dispatch import SDKBatchWorkerAdapter


class SDKMTPBatchWorkerAdapter(SDKBatchWorkerAdapter):
    def _process_completed(self, worker, seq_metadata_list, scheduler_outputs, output, emitted,
                           *, is_prompt, end_event, prepare_native=True, refresh_metadata=False):
        import numpy as np
        import torch
        from deepinfer._C.engine import SequenceStatus
        from deepinfer.sequence import MTPPredictInfo, SampledInfo, SamplerOutput
        from deepinfer.worker.zos_status import WorkerStepState
        if refresh_metadata and not is_prompt:
            # Reuse native CPU bookkeeping after the previous output has been
            # applied. SDK already used its own actual device acceptance state
            # for model execution; this refresh performs no model computation.
            scheduler_outputs.update_metadata_for_sampling_zos(seq_metadata_list,
                torch.tensor([m.seqlen_q for m in seq_metadata_list], dtype=torch.int32),
                torch.arange(len(seq_metadata_list), dtype=torch.int32), worker.nextn)
        main, main_logp, verified, verified_logp, offsets, next_drafts = [], [], [], [], [], []
        for meta in seq_metadata_list:
            rid = meta.request_id
            tokens = emitted.get(rid, [])
            offsets.append(len(verified))
            next_drafts.append(output.draft_token_ids.get(rid, [-1] * worker.nextn))
            if not tokens:
                main.append(-1)
                main_logp.append(0.0)
                continue
            if output.logprobs is None:
                raise ValueError('native MTP output requires actual per-token logprobs')
            row = output.logprobs.slice_request(output.req_id_to_index[rid], len(tokens))
            ids = row.logprob_token_ids.reshape(-1).tolist()
            values = row.logprobs.reshape(-1).tolist()
            if ids != tokens or len(values) != len(tokens) or not all(map(math.isfinite, values)):
                raise ValueError('SDK MTP token/logprob rows disagree')
            main.append(tokens[-1])
            main_logp.append(float(values[-1]))
            if not is_prompt:
                # C++ consumes q-1 draft slots, followed by one unused
                # main slot. Rejected draft slots carry the real -1
                # acceptance sentinel, never an invented output token.
                accepted = tokens[:-1]
                padding = meta.seqlen_q - len(accepted)
                verified.extend(accepted + [-1] * padding)
                verified_logp.extend([float(v) for v in values[:-1]] + [0.0] * padding)
        sampled = SamplerOutput(
            sampled_token=SampledInfo(bs=len(main), token_ids=torch.tensor(main, dtype=torch.int32),
                                     token_logprobs=torch.tensor(main_logp, dtype=torch.float32)),
            verified_tokens=(SampledInfo() if is_prompt else SampledInfo(bs=len(main), nextn=worker.nextn, is_verified_token=True,
                token_ids=torch.tensor(verified, dtype=torch.int32),
                token_logprobs=torch.tensor(verified_logp, dtype=torch.float32))),
            first_token_positions=None if is_prompt else torch.tensor(offsets, dtype=torch.int32),
            predict_tokens=MTPPredictInfo(token_ids=torch.tensor(next_drafts, dtype=torch.int32)),
            post_funcs=[])
        if prepare_native:
            worker.engine_callbacks.pre_sampler_sync_fn(scheduler_outputs)
            worker.maybe_gc_collect()
        end_event.synchronize()
        state = WorkerStepState(seq_metadata_list=seq_metadata_list,
            scheduler_outputs=scheduler_outputs, sampler_output=sampled,
            is_prompt=is_prompt, is_placeholder=False, sample_end_evt=end_event)
        result = worker.engine_callbacks.post_process_fn(state)
        if not isinstance(result, np.ndarray) or result.dtype != np.uint64:
            raise ValueError('native MTP callback returned an invalid output vector')
        terminal = {SequenceStatus.FINISHED_STOPPED, SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED, SequenceStatus.FINISHED_IGNORED,
            SequenceStatus.FINISHED_PREEMPTED, SequenceStatus.FINISHED_SO_FAILED,
            SequenceStatus.FINISHED_VE_FAILED}
        finished = [m.request_id for m, seq in
            zip(seq_metadata_list, scheduler_outputs.scheduled_seqs, strict=True)
            if seq.status in terminal]
        self.controller.finish(finished)
        return result

    def execute_model_step(self, worker, is_prompt, is_placeholder,
                           seq_metadata_list, scheduler_outputs):
        import numpy as np
        import torch
        import torch.distributed as dist
        from deepinfer import envs
        from deepinfer._C.engine import SequenceStatus
        from deepinfer.sequence import MTPPredictInfo, SampledInfo, SamplerOutput
        from deepinfer.worker.zos_status import WorkerStepState
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict

        driver = dist.get_rank(self.group) == 0
        if (self.poisoned or worker.nextn != self.runtime.num_speculative_tokens or worker.prev_step_state is not None
                or envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or is_placeholder
                or bool(worker.is_driver_worker) != driver):
            raise ValueError('native MTP adapter requires its explicit one-draft lifecycle')
        failure, message = None, None
        if driver:
            try:
                if (not seq_metadata_list or scheduler_outputs is None
                        or len(seq_metadata_list) != len(scheduler_outputs.scheduled_seqs)
                        or any(bool(m.is_prompt) != bool(is_prompt) for m in seq_metadata_list)):
                    raise ValueError('native MTP scheduled metadata is incomplete or mixed')
                descriptor = self.controller.prepare(seq_metadata_list)
                self.batch_sizes.append(len(seq_metadata_list))
                message = {'ok': True, 'descriptor': descriptor}
            except Exception as error:
                failure = error
                message = {'ok': False, 'error': str(error)}
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('native MTP preparation failed: ' + message['error']) from failure
        try:
            output = self.runtime.resolve_output(self.runtime.execute(message['descriptor']))
        except Exception:
            self.poisoned = True
            raise
        result = np.empty(0, dtype=np.uint64)
        message = None
        if driver:
            try:
                emitted = self.controller.commit(output)
                end_event = torch.npu.Event()
                end_event.record()
                result = self._process_completed(worker, seq_metadata_list, scheduler_outputs,
                    output, emitted, is_prompt=is_prompt, end_event=end_event)
                message = {'ok': True}
            except Exception as error:
                self.poisoned = True
                failure = error
                message = {'ok': False, 'error': str(error)}
        message = broadcast_tensor_dict(message, src=0)
        if not message['ok']:
            self.poisoned = True
            raise RuntimeError('native MTP commit failed: ' + message['error']) from failure
        return result
