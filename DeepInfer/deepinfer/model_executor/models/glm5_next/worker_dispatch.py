# SPDX-License-Identifier: Apache-2.0
"""Explicit, synchronous, single-request GLM adapter for Worker dispatch.

Installation, model allocation, cache ownership, aborts and service lifecycle
remain the caller's responsibility. This is not a full model registry or engine.
All CP ranks must enter in the same order. Process/device loss and failures
inside collectives are not recovered.
"""

import numpy as np
import torch
import torch.distributed as dist


def validate_bounded_glm_sampling_params(p):
    """Fail before scheduling when this bounded adapter cannot honor sampling."""
    from deepinfer.sampling_params import SAMPLING_EPS

    if (p.temperature > SAMPLING_EPS or p.top_p != 1.0 or p.top_k != -1 or p.top_logprobs != 0
            or p.prompt_logprobs or p.first_n_token_no_eos or p.so_args is not None or p.golden_token_ids):
        raise ValueError('GLM Worker adapter only supports unfiltered greedy text sampling')


class GLMSingleRequestWorkerAdapter:
    def __init__(self, bridge, sampler):
        self.bridge = bridge
        self.sampler = sampler
        self.poisoned = False

    def _sampling_metadata(self, metadata):
        from vllm.v1.sample.metadata import SamplingMetadata
        p = metadata.sampling_params
        validate_bounded_glm_sampling_params(p)
        d = self.bridge._runner.boundary.norm_weight.device
        return SamplingMetadata(temperature=None, all_greedy=True, all_random=False, top_p=None, top_k=None,
            generators={}, max_num_logprobs=-1, no_penalties=True, prompt_token_ids=None,
            frequency_penalties=torch.zeros(1,device=d), presence_penalties=torch.zeros(1,device=d),
            repetition_penalties=torch.ones(1,device=d), output_token_ids=[[]], allowed_token_ids_mask=None,
            bad_words_token_ids={}, logitsprocs=_EmptyLogitsProcessors())

    @torch.inference_mode()
    def execute_model_step(self, worker, is_prompt, is_placeholder, seq_metadata_list, scheduler_outputs):
        """Synchronously run one ordinary scheduled step and the driver callback.

        Non-final prompt chunks carry an internal placeholder token/logprob;
        the scheduler must not publish them. Exceptions after model commit poison
        the adapter because request state is not rolled back.
        """
        from deepinfer import envs
        from deepinfer._C.engine import SequenceStatus
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        from deepinfer.sequence import SampledInfo, SamplerOutput
        from deepinfer.worker.zos_status import WorkerStepState

        cp, commit = self.bridge._groups()
        local_bad = (self.poisoned or worker.nextn != 0 or worker.prev_step_state is not None
                     or envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING or is_placeholder
                     or bool(worker.is_driver_worker) != (cp.rank() == 0))
        status = torch.tensor([int(local_bad)],dtype=torch.int32,device=self.bridge._runner.boundary.norm_weight.device)
        dist.all_reduce(status,op=dist.ReduceOp.MAX,group=commit)
        if int(status.item()):
            raise ValueError('GLM Worker adapter requires one non-placeholder non-ZOS non-speculative step')
        metadata = sampling_metadata = None
        failure = None
        payload = None
        if cp.rank() == 0:
            try:
                if (seq_metadata_list is None or len(seq_metadata_list)!=1 or scheduler_outputs is None
                        or len(scheduler_outputs.scheduled_seqs)!=1):
                    raise ValueError('exactly one live scheduled sequence required')
                metadata = seq_metadata_list[0]
                if bool(metadata.is_prompt)!=bool(is_prompt) or bool(scheduler_outputs.prompt_run)!=bool(is_prompt):
                    raise ValueError('scheduler and worker mode differ')
                sampling_metadata = self._sampling_metadata(metadata)
                if self.sampler is None:
                    raise ValueError('installed sampler required on driver')
                payload = {'ok':True}
            except Exception as exc:
                failure = exc
                payload = {'ok':False}
        payload = broadcast_tensor_dict(payload,src=0)
        if not payload['ok']:
            raise ValueError('GLM Worker adapter preflight failed') from failure
        try:
            _, output = self.bridge.step_and_sample(metadata,sampling_metadata,self.sampler)
        except Exception:
            # Execution or sampling may have committed recurrent state even
            # when no scheduler output was published. Never retry this owner.
            self.poisoned = True
            raise
        if output is None:
            output = SamplerOutput(sampled_token=SampledInfo(bs=1,token_ids=torch.tensor([-1],dtype=torch.int32),
                token_logprobs=torch.zeros(1,dtype=torch.float32)),post_funcs=[])
        result = np.empty(0,dtype=np.uint64)
        payload = None
        failure = None
        if cp.rank() == 0:
            try:
                worker.engine_callbacks.pre_sampler_sync_fn(scheduler_outputs)
                worker.maybe_gc_collect()
                device_module = torch.get_device_module(self.bridge._runner.boundary.norm_weight.device)
                event = device_module.Event(enable_timing=True)
                event.record();event.synchronize()
                step_state = WorkerStepState(seq_metadata_list=seq_metadata_list,scheduler_outputs=scheduler_outputs,
                    sampler_output=output,is_prompt=is_prompt,is_placeholder=False,sample_end_evt=event)
                result = worker.engine_callbacks.post_process_fn(step_state)
                if not isinstance(result,np.ndarray) or result.dtype != np.uint64:
                    raise ValueError('Worker callback must return uint64 request outputs')
                terminal = scheduler_outputs.scheduled_seqs[0].status in (
                    SequenceStatus.FINISHED_STOPPED,SequenceStatus.FINISHED_LENGTH_CAPPED,
                    SequenceStatus.FINISHED_ABORTED,SequenceStatus.FINISHED_IGNORED,
                    SequenceStatus.FINISHED_PREEMPTED,SequenceStatus.FINISHED_SO_FAILED,SequenceStatus.FINISHED_VE_FAILED)
                payload = {'ok':True,'terminal':terminal}
            except Exception as exc:
                failure = exc
                payload = {'ok':False}
        payload = broadcast_tensor_dict(payload,src=0)
        if not payload['ok']:
            self.poisoned = True
            raise RuntimeError('GLM Worker adapter post-processing failed after commit') from failure
        if payload['terminal']:
            self.bridge.close()
        return result


class _EmptyLogitsProcessors:
    non_argmax_invariant = ()
    argmax_invariant = ()
