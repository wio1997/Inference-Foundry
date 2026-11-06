# SPDX-License-Identifier: Apache-2.0
"""Bounded single-request bridge from driver scheduler metadata to text state.

This is an explicit seam for native worker integration, not a Worker, model
registry, cache allocator or serving implementation. Callers own sampling,
scheduler output processing and terminal cleanup. All CP ranks must call the
same methods in the same order; process loss and failures inside device
collectives are not recovered.
"""

import torch
import torch.distributed as dist

from .scheduler_contract import scheduled_text_span


class SingleRequestWorkerBridge:
    """Own at most one fresh single-sequence text request on every CP rank."""

    def __init__(self, runner, state_factory):
        if not callable(state_factory):
            raise TypeError('explicit initial-state factory required')
        self._runner = runner
        self._state_factory = state_factory
        self._request = None
        self._tokens_seen = 0

    def _groups(self):
        from deepinfer.model_executor.parallel_utils.parallel_state import PG
        cp = PG.context_parallel_group()
        commit = self._runner._rank_commit_group or dist.group.WORLD
        if not self._runner._rank_commit_enabled or dist.get_process_group_ranks(cp) != dist.get_process_group_ranks(commit):
            raise ValueError('worker bridge requires matching CP and rank-commit groups')
        return cp, commit

    def _open(self, metadata):
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        cp, commit = self._groups()
        payload = None
        failure = None
        if cp.rank() == 0:
            try:
                if metadata is None or not metadata.is_prompt:
                    raise ValueError('fresh request requires live prompt metadata')
                if metadata.last_next_n_token or metadata.last_accepted_token:
                    raise ValueError('speculative metadata is not supported')
                span = scheduled_text_span(metadata, request_id=metadata.request_id, seq_id=metadata.seq_id,
                                           tokens_seen=0, max_context_tokens=self._runner.max_context_tokens)
                if len(metadata.get_input_token_ids()) != span.token_count:
                    raise ValueError('scheduler metadata token span mismatch')
                payload = {'ok': True, 'request_id': metadata.request_id}
            except Exception as exc:
                failure = exc
                payload = {'ok': False}
        payload = broadcast_tensor_dict(payload, src=0)
        if not payload['ok']:
            raise ValueError('worker bridge request preparation failed') from failure
        request = None
        failure = None
        try:
            state = self._state_factory(payload['request_id'])
            if state.request_id != payload['request_id'] or state.tokens_seen != 0:
                raise ValueError('fresh initial state identity or position mismatch')
            request = self._runner.open_request(state)
        except Exception as exc:
            failure = exc
        status = torch.tensor([int(failure is not None)], dtype=torch.int32, device=self._runner.boundary.norm_weight.device)
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=commit)
        if int(status.item()):
            if request is not None:
                request.close()
            raise RuntimeError('worker bridge initial-state preparation failed') from failure
        self._request = request

    @torch.inference_mode()
    def step(self, driver_metadata):
        """Copy live metadata on driver, execute a coordinated prompt/decode."""
        if self._request is None:
            self._open(driver_metadata)
        result = self._request.step_driver_metadata(driver_metadata, expected_tokens_seen=self._tokens_seen)
        self._tokens_seen = result.token_stop
        return result

    @torch.inference_mode()
    def step_and_sample(self, driver_metadata, sampling_metadata, sampler):
        """Run a step, sampling only the last prompt chunk or a decode step.

        Unsupported sampling is rejected collectively before state is advanced.
        Runtime sampling or communication errors after the step are not rolled
        back. Driver owns scheduler post-processing and request termination.
        """
        from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict
        from .driver_sampling import sample_driver_output, validate_greedy_metadata
        cp, _ = self._groups();payload=None;failure=None
        if cp.rank()==0:
            try:
                if driver_metadata is None:raise ValueError('live scheduler metadata required')
                emit=not driver_metadata.is_prompt or driver_metadata.is_last_prompt_chunk
                if emit:validate_greedy_metadata(sampling_metadata,sampler)
                payload={'ok':True,'emit':emit}
            except Exception as exc:
                failure=exc;payload={'ok':False}
        payload=broadcast_tensor_dict(payload,src=0)
        if not payload['ok']:raise ValueError('worker bridge sampling preflight failed') from failure
        result=self.step(driver_metadata)
        output=sample_driver_output(result.logits[-1:] if cp.rank()==0 else None,sampling_metadata,sampler) if payload['emit'] else None
        return result,output

    def snapshot(self):
        if self._request is None:
            raise RuntimeError('worker bridge has no active request')
        return self._request.snapshot()

    def close(self):
        """Release this rank's owner; caller coordinates terminal lifecycle."""
        if self._request is not None:
            self._request.close()
            self._request = None
            self._tokens_seen = 0
