"""Two-step native scheduling bridge for the SDK's device-token feedback.

No SDK scheduler is constructed. Counts follow the pinned AsyncScheduler;
DeepInfer still schedules and decides when requests end. This is a candidate,
and physical execution/overlap must be verified separately.
"""
from collections import deque

from .sdk_schedule import SDKBatchSchedule


class SDKAsyncBatchSchedule(SDKBatchSchedule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inflight = deque()
        self.async_stats = {'max_inflight_batches': 0, 'device_token_feedback_uses': 0,
                            'submitted_batches': 0, 'resolved_batches': 0}

    def _validated_span(self, meta, state):
        req = state.request
        if not req.num_output_placeholders:
            return super()._validated_span(meta, state)
        from deepinfer.model_executor.models.glm5_next.scheduler_contract import ScheduledTextSpan
        # Exactly one prior sample is unknown on the host. The SDK runner feeds
        # it directly from its previous device sample, never from a guessed ID.
        if (meta.is_placeholder or meta.multi_modal_data is not None or meta.is_prompt
                or req.num_output_placeholders != 1 or meta.seq_id != state.seq_id
                or meta.seqlen_q != 1 or meta.seqlen_k != req.num_computed_tokens + 1
                or meta.full_len != req.num_tokens or meta.seqlen_k > self.max_context):
            raise ValueError('native async span differs from single device-token feedback contract')
        self.async_stats['device_token_feedback_uses'] += 1
        return ScheduledTextSpan(meta.request_id, meta.seq_id,
                                 req.num_computed_tokens, meta.seqlen_k, False)

    def _output_count(self, request):
        return request.num_output_tokens + request.num_output_placeholders

    def prepare(self, metadata_list):
        if len(self.inflight) >= 2:
            raise RuntimeError('two async batches already submitted;resolve oldest output first')
        return super().prepare(metadata_list)

    def submitted(self):
        """Advance optimistic counts; keep hashes and block pins unpublished."""
        if self.poisoned or self.pending is None or len(self.inflight) >= 2:
            raise RuntimeError('async submission exceeds the two-batch transaction bound')
        entries = []
        for rid, seq_id, stop, emit in self.pending:
            state = self.requests[rid]
            req = state.request
            count = stop - req.num_computed_tokens
            if count < 1:
                raise RuntimeError('async submission has no scheduled tokens')
            req.num_computed_tokens = stop
            req.num_in_flight_tokens += count
            req.num_output_placeholders += int(emit)
            state.seq_id, state.submitted = seq_id, True
            entries.append((rid, count, emit))
        self.inflight.append((entries, self.retained_copy_blocks))
        self.retained_copy_blocks = []
        self.pending = None
        self.async_stats['submitted_batches'] += 1
        self.async_stats['max_inflight_batches'] = max(
            self.async_stats['max_inflight_batches'], len(self.inflight))

    def commit_completed(self, output):
        """Called only after the SDK's asynchronous copy-ready event completes."""
        if self.poisoned or not self.inflight:
            raise RuntimeError('async output has no submitted transaction')
        entries, retained = self.inflight[0]
        emitted = {}
        try:
            for rid, _, emit in entries:
                tokens = output.sampled_token_ids[output.req_id_to_index[rid]]
                if emit:
                    if len(tokens) != 1 or not 0 <= tokens[0] < self.vocab_size:
                        raise ValueError('async greedy batch must emit exactly one valid token')
                    emitted[rid] = int(tokens[0])
                elif tokens:
                    raise ValueError('non-final async prefill emitted a token')
            for rid, count, emit in entries:
                req = self.requests[rid].request
                req.num_in_flight_tokens -= count
                req.num_output_placeholders -= int(emit)
                if min(req.num_in_flight_tokens, req.num_output_placeholders) < 0:
                    raise RuntimeError('async request counter underflow')
                if rid in emitted:
                    req.append_output_token_ids(emitted[rid])
                if self.request_block_hasher is not None:
                    # Same safe publication boundary as pinned AsyncScheduler.
                    self.cache.cache_blocks(req,
                        req.num_computed_tokens - req.num_output_placeholders)
            self.cache.block_pool.free_blocks(retained)
            self.inflight.popleft()
            self.async_stats['resolved_batches'] += 1
        except Exception:
            self.poisoned = True
            raise
        return emitted

    def finish(self, request_ids):
        request_ids = list(request_ids)
        busy = {rid for entries, _ in self.inflight for rid, _, _ in entries}
        if busy.intersection(request_ids):
            raise RuntimeError('cannot release an async cache still used by a submitted batch')
        return super().finish(request_ids)

    def prepare_cleanup(self):
        if self.inflight:
            raise RuntimeError('async compute must drain before finished-request cleanup')
        return super().prepare_cleanup()
