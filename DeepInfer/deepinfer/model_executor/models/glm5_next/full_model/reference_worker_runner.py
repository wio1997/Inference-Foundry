# SPDX-License-Identifier: Apache-2.0
"""Candidate runner protocol for existing DeepInfer Worker bridge/dispatcher.

The reference factory supplies model/cache/offloader bindings. Once a FullModelBackend
is constructed, build_worker_adapter reuses DeepInfer's greedy sampler callbacks
and single-request bridge. It does not create any vLLM request engine.
"""
from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import torch
import torch.distributed as dist


class TorchRankAgreement:
    """Same Worker ranks, verified reference TP order, bounded phase votes."""
    def __init__(self, group, device, get_tp_group):
        if not dist.is_initialized() or not callable(get_tp_group):
            raise ValueError('initialized Worker group and real get_tp_group required')
        self.group, self.device, self.get_tp_group = group, torch.device(device), get_tp_group
        self.validate_tp()

    def validate_tp(self):
        tp = self.get_tp_group()
        ranks = dist.get_process_group_ranks(self.group)
        if (list(tp.ranks) != ranks or tp.world_size != len(ranks)
                or tp.rank_in_group != dist.get_rank(self.group)):
            raise ValueError('reference TP and DeepInfer Worker membership/order differ')

    def __call__(self, phase, error, signature):
        digest = hashlib.sha256(json.dumps([phase, signature], sort_keys=True).encode()).digest()
        header = torch.tensor([int(error is not None)] +
                              [int.from_bytes(digest[i:i+7], 'big') for i in range(0, 28, 7)],
                              dtype=torch.int64, device=self.device)
        votes = [torch.empty_like(header) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(votes, header, group=self.group)
        if any(int(v[0].item()) for v in votes) or not all(torch.equal(votes[0], v) for v in votes[1:]):
            raise RuntimeError('Worker rank failure or signature mismatch at ' + phase) from error


@dataclass(frozen=True)
class FreshRequest:
    request_id: str
    tokens_seen: int = 0


class ReferenceWorkerRunner:
    """Only the protocol consumed by SingleRequestWorkerBridge is implemented."""
    def __init__(self, backend, *, group, broadcast_tensor_dict, scheduled_text_span):
        self.backend = backend
        self._rank_commit_enabled, self._rank_commit_group = True, group
        self.max_context_tokens = backend.max_context_tokens
        # Existing dispatcher reads this tensor solely for device placement.
        # It is the real reference final norm; no extra norm is performed here.
        self.boundary = SimpleNamespace(norm_weight=backend.model.model.norm.weight,
                                        max_tokens=backend.max_step_tokens)
        self.broadcast = broadcast_tensor_dict
        self.span_validator = scheduled_text_span
        if not callable(self.broadcast) or not callable(self.span_validator):
            raise TypeError('actual DeepInfer metadata broadcast and span validator required')

    def open_request(self, seed):
        if not isinstance(seed, FreshRequest) or seed.tokens_seen != 0:
            raise ValueError('fresh reference request seed required; archive/restore unsupported')
        return ReferenceWorkerRequest(self, self.backend.open(seed.request_id))


class ReferenceWorkerRequest:
    def __init__(self, runner, owner):
        self._runner, self.owner = runner, owner

    def step_driver_metadata(self, metadata, *, expected_tokens_seen):
        r = self._runner
        b = r.backend
        payload = None
        error = None
        if dist.get_rank(r._rank_commit_group) == 0:
            try:
                if metadata is None or metadata.last_next_n_token or metadata.last_accepted_token:
                    raise ValueError('live ordinary DeepInfer scheduler metadata required')
                span = r.span_validator(
                    metadata, request_id=self.owner.request_id,
                    seq_id=metadata.seq_id if self.owner.seq_id is None else self.owner.seq_id,
                    tokens_seen=self.owner.tokens_seen, max_context_tokens=r.max_context_tokens)
                ids = metadata.get_input_token_ids()
                if not isinstance(ids, list) or len(ids) != span.token_count or any(type(i) is not int for i in ids):
                    raise ValueError('live scheduler token vector differs from scheduled span')
                payload = {'ok': True, 'ids': ids, 'metadata': {
                    'request_id': metadata.request_id, 'seq_id': metadata.seq_id,
                    'is_prompt': metadata.is_prompt, 'is_placeholder': False,
                    'multi_modal_data': None, 'seqlen_q': metadata.seqlen_q,
                    'seqlen_k': metadata.seqlen_k, 'full_len': metadata.full_len}}
            except Exception as exc:
                error = exc
                payload = {'ok': False}
        payload = r.broadcast(payload, src=0)
        if not payload['ok']:
            raise ValueError('driver scheduler input preparation failed') from error
        span, ids, signature = None, None, None
        try:
            if type(expected_tokens_seen) is not int or expected_tokens_seen != self.owner.tokens_seen:
                raise ValueError('stale expected request position')
            copied = SimpleNamespace(**payload['metadata'])
            span = r.span_validator(
                copied, request_id=self.owner.request_id,
                seq_id=copied.seq_id if self.owner.seq_id is None else self.owner.seq_id,
                tokens_seen=self.owner.tokens_seen, max_context_tokens=r.max_context_tokens)
            ids = torch.tensor(payload['ids'], dtype=torch.int64)
            signature = (span.request_id, span.seq_id, span.token_start, span.token_stop,
                         span.is_prompt, payload['ids'])
        except Exception as exc:
            error = exc
        b._agree('scheduler-span', error, signature)
        return self.owner.step(ids, span)

    def snapshot(self):
        # Position only; this is deliberately not a canonical cache export.
        self.owner._require_open()
        return SimpleNamespace(request_id=self.owner.request_id,
                               seq_id=self.owner.seq_id, tokens_seen=self.owner.tokens_seen)

    def close(self):
        self.owner.close()


def build_worker_adapter(backend, *, sampler, worker_group):
    """Wire an already constructed backend; this is NOT a model/runtime factory.

    All actual CP ranks call this after verifying same-rank reference TP setup.
    Caller installs the returned adapter with Worker.install_glm_single_request_adapter.
    Driver supplies the existing installed greedy sampler. Engine remains the
    actual DeepInfer LLMEngine.from_bounded_glm_worker path.
    """
    from deepinfer.model_executor.models.glm5_next.worker_bridge import SingleRequestWorkerBridge
    from deepinfer.model_executor.models.glm5_next.worker_dispatch import GLMSingleRequestWorkerAdapter
    from deepinfer.model_executor.models.glm5_next.scheduler_contract import scheduled_text_span
    from deepinfer.model_executor.parallel_utils.communication_op import broadcast_tensor_dict

    backend.validate_tp()
    if backend.is_driver != (dist.get_rank(worker_group) == 0):
        raise ValueError('backend driver role differs from DeepInfer Worker rank zero')
    runner = ReferenceWorkerRunner(backend, group=worker_group,
                                  broadcast_tensor_dict=broadcast_tensor_dict,
                                  scheduled_text_span=scheduled_text_span)
    bridge = SingleRequestWorkerBridge(runner, FreshRequest)
    bridge._groups()  # Existing bridge verifies DeepInfer CP == commit membership.
    return GLMSingleRequestWorkerAdapter(bridge, sampler)
