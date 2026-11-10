# SPDX-License-Identifier: Apache-2.0
"""Full-model seam for pinned910C dependencies; acceptance is version-specific.

The caller constructs and loads the pinned Glm5NextForCausalLM, initializes its
TP group, installs its quantization/offload hooks, and supplies the real cache
allocator and attention metadata builders. This file never constructs an engine
or scheduler. Complete-model forward owns all deferred mHC materialization.
Only one request is live per model/backend; cache mutation is in place. A failed
execution poisons that owner, with no retry/rollback or process-loss recovery.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class LayerContract:
    metadata_type: type
    # Both checks are mandatory: the GDN dataclass alone omits Ascend fields.
    validate_metadata: Callable
    validate_cache: Callable


@dataclass(frozen=True)
class ForwardInputs:
    attn_metadata: Mapping
    slot_mapping: Mapping


@dataclass(frozen=True)
class StepResult:
    token_start: int
    token_stop: int
    hidden: torch.Tensor
    logits: torch.Tensor | None
    vocab_start: int
    vocab_end: int


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    else:
        raise TypeError('reference kv_cache must be a tensor/list/tuple tree')


def _storage_key(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


class FullModelBackend:
    """Serialize direct model forward and explicit request-owned cache binding.

    ``agree(phase, local_exception, signature)`` must be called on every Worker
    rank; it raises on any rank failure/signature mismatch. It cannot recover a
    failure inside a model collective. ``step_scope`` is the already-installed
    model/offloader's per-step scope; it must wait before exiting when needed.
    ``release_cache`` releases only this request's buffers after device work is
    safe. No default no-op implementations are selected here.
    """
    def __init__(self, model, *, model_type, contracts, cache_factory,
                 metadata_builder, set_forward_context, get_forward_context,
                 step_scope, release_cache, validate_tp, agree,
                 is_driver, max_context_tokens, max_step_tokens):
        if not isinstance(model_type, type) or not isinstance(model, model_type):
            raise TypeError('actual pinned complete-model class/instance required')
        if model.training:
            raise ValueError('loaded reference model must already be in eval mode')
        self.model = model
        self.config = model.vllm_config
        p = self.config.parallel_config
        if (p.pipeline_parallel_size != 1 or p.data_parallel_size != 1
                or p.use_sequence_parallel_moe or self.config.speculative_config is not None):
            raise ValueError('candidate supports PP1/DP1, no SP or speculation')
        if any(type(x) is not int or x < 1 for x in (max_context_tokens, max_step_tokens)):
            raise ValueError('positive explicit context and step bounds required')
        self.max_context_tokens, self.max_step_tokens = max_context_tokens, max_step_tokens
        if type(is_driver) is not bool:
            raise TypeError('explicit Worker driver role required')
        self.is_driver = is_driver
        self.vocab_size = model.config.vocab_size
        self.hidden_size = model.config.hidden_size
        self.device = model.model.norm.weight.device
        # Exact full set includes KDA, MLA, compressed indexer and tail layers.
        self.layers = {key: layer for key, layer in
                       self.config.compilation_config.static_forward_context.items()
                       if hasattr(layer, 'kv_cache')}
        if not self.layers or set(contracts) != set(self.layers):
            raise ValueError('contracts must cover every registered cache layer exactly')
        self.contracts = dict(contracts)
        for contract in self.contracts.values():
            if (not isinstance(contract, LayerContract)
                    or not isinstance(contract.metadata_type, type)
                    or not callable(contract.validate_metadata)
                    or not callable(contract.validate_cache)):
                raise TypeError('real metadata type and cache/metadata validators required')
        callbacks = (cache_factory, metadata_builder, set_forward_context,
                     get_forward_context, step_scope, release_cache, validate_tp, agree)
        if any(not callable(fn) for fn in callbacks):
            raise TypeError('explicit runtime bindings are required')
        self.cache_factory, self.metadata_builder = cache_factory, metadata_builder
        self.set_context, self.get_context = set_forward_context, get_forward_context
        self.step_scope, self.release_cache = step_scope, release_cache
        self.validate_tp, self.agree = validate_tp, agree
        self._lock, self._owner = Lock(), None
        validate_tp()  # Verify actual reference TP membership and rank order.

    def _agree(self, phase, error, signature):
        self.agree(phase, error, signature)
        if error is not None:
            raise RuntimeError(phase + ' failed locally') from error

    def open(self, request_id, seq_id=None):
        """All ranks allocate independent zero state before publishing an owner."""
        with self._lock:
            cache = None
            error = None
            try:
                if self._owner is not None:
                    raise RuntimeError('close the active request before opening another')
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError('nonempty request identity required')
                if seq_id is not None and (type(seq_id) is not int or seq_id < 0):
                    raise ValueError('invalid sequence identity')
                self.validate_tp()
            except Exception as exc:
                error = exc
            self._agree('open-preflight', error, (request_id, seq_id))
            try:
                cache = self.cache_factory(request_id)
                if not isinstance(cache, Mapping) or set(cache) != set(self.layers):
                    raise ValueError('request allocator omitted a registered cache layer')
                shared = {_storage_key(t) for layer in self.layers.values()
                          for t in _tensors(layer.kv_cache) if t.numel()}
                tensors = [t for value in cache.values() for t in _tensors(value)]
                if not tensors or not any(t.numel() for t in tensors):
                    raise ValueError('request cache is empty')
                if any(t.device != self.device for t in tensors):
                    raise ValueError('request cache is on a different device')
                if any(_storage_key(t) in shared for t in tensors if t.numel()):
                    raise ValueError('request cache aliases the model baseline cache')
                for prefix, contract in self.contracts.items():
                    contract.validate_cache(cache[prefix])
                # A fresh request never inherits previous request state, even
                # if an allocator safely recycles a released block.
                for tensor in tensors:
                    tensor.zero_()
            except Exception as exc:
                error = exc
            try:
                self._agree('open-cache', error, (request_id, seq_id))
            except Exception:
                if cache is not None:
                    self.release_cache(cache)
                raise
            owner = ReferenceRequest(self, request_id, seq_id, dict(cache))
            self._owner = owner
            return owner


class ReferenceRequest:
    def __init__(self, backend, request_id, seq_id, cache):
        self.backend, self.request_id, self.seq_id = backend, request_id, seq_id
        self.tokens_seen, self.poisoned, self.closed = 0, False, False
        self._cache = cache

    def _require_open(self):
        if self.closed or self.backend._owner is not self:
            raise RuntimeError('request owner is closed or no longer current')
        if self.poisoned:
            raise RuntimeError('failed execution owner must be closed; retry is forbidden')

    @torch.inference_mode()
    def step(self, token_ids, span):
        """Execute a validated DeepInfer ScheduledTextSpan on every rank.

        The scheduler bridge supplies identical CPU token vectors and spans.
        No sampling, final norm, mHC post, or head is applied outside the model.
        """
        b = self.backend
        with b._lock:
            error = None
            prepared = None
            signature = None
            try:
                self._require_open()
                if (span.request_id != self.request_id
                        or self.seq_id not in (None, span.seq_id)
                        or span.token_start != self.tokens_seen):
                    raise ValueError('stale scheduler span or different request/sequence')
                if (type(span.seq_id) is not int or span.seq_id < 0
                        or type(span.token_start) is not int or type(span.token_stop) is not int
                        or type(span.is_prompt) is not bool):
                    raise ValueError('explicit integer positions and prompt mode required')
                count = span.token_stop - span.token_start
                if (not 1 <= count <= b.max_step_tokens
                        or span.token_stop > b.max_context_tokens
                        or (not span.is_prompt and count != 1)
                        or (self.tokens_seen == 0 and not span.is_prompt)):
                    raise ValueError('invalid fresh/prompt/decode span or context bound')
                if (not isinstance(token_ids, torch.Tensor) or token_ids.device.type != 'cpu'
                        or token_ids.ndim != 1 or len(token_ids) != count
                        or token_ids.dtype not in (torch.int32, torch.int64)):
                    raise ValueError('scheduler must supply a matching CPU integer token vector')
                if bool(((token_ids < 0) | (token_ids >= b.vocab_size)).any()):
                    raise ValueError('token outside full reference vocabulary')
                signature = (self.request_id, span.seq_id, span.token_start,
                             span.token_stop, span.is_prompt, token_ids.tolist())
                b.validate_tp()
                prepared = b.metadata_builder(span, self._cache)
                if (not isinstance(prepared, ForwardInputs)
                        or not isinstance(prepared.attn_metadata, Mapping)
                        or set(prepared.attn_metadata) != set(b.layers)
                        or not isinstance(prepared.slot_mapping, Mapping)):
                    raise ValueError('real metadata and slot mappings required for every cache prefix')
                for prefix, contract in b.contracts.items():
                    metadata = prepared.attn_metadata[prefix]
                    if not isinstance(metadata, contract.metadata_type):
                        raise TypeError('metadata type differs from pinned backend type: ' + prefix)
                    contract.validate_metadata(metadata, span, self._cache[prefix])
            except Exception as exc:
                error = exc
            # Any rank preflight error is rejected before model/cache mutation.
            b._agree('step-preflight', error, signature)
            error = None
            result = None
            try:
                with ExitStack() as stack:
                    for prefix, layer in b.layers.items():
                        previous = layer.kv_cache
                        stack.callback(setattr, layer, 'kv_cache', previous)
                        layer.kv_cache = self._cache[prefix]
                    stack.enter_context(b.step_scope())
                    stack.enter_context(b.set_context(
                        dict(prepared.attn_metadata), b.config, num_tokens=count,
                        slot_mapping=dict(prepared.slot_mapping)))
                    active = b.get_context()
                    if (not isinstance(active.attn_metadata, dict)
                            or any(active.attn_metadata.get(key) is not value
                                   for key, value in prepared.attn_metadata.items())):
                        raise ValueError('active forward context lost required layer metadata')
                    positions = torch.arange(span.token_start, span.token_stop,
                                             device=b.device, dtype=torch.int64)
                    hidden = b.model(input_ids=token_ids.to(b.device), positions=positions,
                                     intermediate_tensors=None)
                    if (not isinstance(hidden, torch.Tensor)
                            or tuple(hidden.shape) != (count, b.hidden_size)
                            or hidden.device != b.device):
                        raise ValueError('complete PP1 model returned invalid hidden states')
                    logits = b.model.compute_logits(hidden)
                    # The pinned method explicitly allows None; non-driver
                    # ranks need not retain gathered logits for sampling.
                    if logits is None:
                        if b.is_driver:
                            raise ValueError('driver did not receive reference logits')
                    elif (not isinstance(logits, torch.Tensor)
                          or tuple(logits.shape) != (count, b.vocab_size)
                          or logits.device != b.device or not bool(torch.isfinite(logits).all())):
                        raise ValueError('model returned invalid full-vocabulary logits')
                    result = StepResult(span.token_start, span.token_stop, hidden, logits, 0, b.vocab_size)
            except Exception as exc:
                error = exc
            try:
                b._agree('step-result', error, signature)
            except Exception:
                self.poisoned = True
                raise
            self.seq_id, self.tokens_seen = span.seq_id, span.token_stop
            return result

    @torch.inference_mode()
    def close(self):
        """All Worker ranks close this owner; model/TP/offload teardown is external."""
        b = self.backend
        with b._lock:
            if self.closed:
                return
            if b._owner is not self:
                raise RuntimeError('cannot release another request owner')
            # release_cache must drain outstanding device uses before release.
            self.poisoned = True
            b.release_cache(self._cache)
            self._cache = None
            self.closed = True
            b._owner = None
