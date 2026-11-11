"""Two-inflight MTP accounting matching the pinned SDK AsyncScheduler.

Unknown drafts are SDK descriptor sentinels only. The device runner supplies
the actual IDs. Accepted outputs are checked against the preceding real draft
snapshot before the native C++ scheduler sees them.
"""
from collections import deque

from .sdk_async_schedule import SDKAsyncBatchSchedule
from .sdk_mtp_schedule import SDKMTPBatchSchedule


class SDKAsyncMTPBatchSchedule(SDKAsyncBatchSchedule, SDKMTPBatchSchedule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draft_inputs = deque()
        self.mtp_stats['native_host_output_overlap'] = True

    def _validated_span(self, meta, state):
        req = state.request
        if not req.num_output_placeholders:
            return SDKMTPBatchSchedule._validated_span(self, meta, state)
        from deepinfer.model_executor.models.glm5_next.scheduler_contract import ScheduledTextSpan
        q, k = meta.seqlen_q, meta.seqlen_k
        if (meta.is_placeholder or meta.multi_modal_data is not None or meta.is_prompt
                or req.num_output_placeholders not in (1, 2) or meta.seq_id != state.seq_id
                or type(q) is not int or type(k) is not int or q not in (1, 2)
                or k != req.num_computed_tokens + q or meta.full_len != req.num_tokens
                or k > self.max_context):
            raise ValueError('native async MTP span differs from optimistic SDK counts')
        self.async_stats['device_token_feedback_uses'] += 1
        return ScheduledTextSpan(meta.request_id, meta.seq_id, k - q, k, False)

    def submitted(self):
        if self.poisoned or self.pending is None or len(self.inflight) >= 2:
            raise RuntimeError('native MTP submission exceeds its two-batch bound')
        entries = []
        inputs = dict(self.scheduled_drafts)
        for rid, seq_id, stop, emit in self.pending:
            state = self.requests[rid]
            req = state.request
            count = stop - req.num_computed_tokens
            if count < 1:
                raise ValueError('MTP submission has no executable input')
            width = 1 + len(inputs.get(rid, ())) if emit else 0
            if not hasattr(state, 'committed_drafts'):
                state.committed_drafts = list(req.spec_token_ids)
            req.num_computed_tokens = stop
            req.num_in_flight_tokens += count
            req.num_output_placeholders += width
            if emit:
                req.spec_token_ids = [-1]
            state.seq_id, state.submitted = seq_id, True
            entries.append((rid, count, width))
        self.inflight.append((entries, self.retained_copy_blocks))
        self.draft_inputs.append(inputs)
        self.retained_copy_blocks, self.pending, self.scheduled_drafts = [], None, {}
        self.async_stats['submitted_batches'] += 1
        self.async_stats['max_inflight_batches'] = max(self.async_stats['max_inflight_batches'], len(self.inflight))

    def commit_completed(self, output):
        if self.poisoned or not self.inflight or len(self.inflight) != len(self.draft_inputs):
            raise RuntimeError('MTP result has no matching in-flight transaction')
        entries, retained = self.inflight[0]
        inputs = self.draft_inputs[0]
        emitted = {}
        try:
            for rid, _, width in entries:
                tokens = list(output.sampled_token_ids[output.req_id_to_index[rid]])
                if not width:
                    if tokens:
                        raise ValueError('non-final async MTP prefill emitted a token')
                    continue
                drafts = list(inputs.get(rid, ()))
                if drafts and drafts[0] == -1:
                    drafts = list(self.requests[rid].committed_drafts[:len(drafts)])
                if (len(drafts) != width - 1 or not 1 <= len(tokens) <= width
                        or any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens)
                        or tokens[:-1] != drafts[:len(tokens) - 1]):
                    raise ValueError('async MTP acceptance differs from the actual preceding draft')
                next_draft = output.draft_token_ids.get(rid)
                if (next_draft is None or len(next_draft) != 1 or type(next_draft[0]) is not int
                        or not 0 <= next_draft[0] < self.vocab_size):
                    raise ValueError('next actual async MTP draft missing')
                emitted[rid] = tokens
            for rid, count, width in entries:
                state = self.requests[rid]
                req = state.request
                tokens = emitted.get(rid, [])
                rejected = width - len(tokens) if width else 0
                req.num_computed_tokens -= rejected
                req.num_in_flight_tokens -= count
                req.num_output_placeholders -= width
                if min(req.num_computed_tokens, req.num_in_flight_tokens, req.num_output_placeholders) < 0:
                    raise RuntimeError('async MTP request counters underflowed')
                if tokens:
                    req.append_output_token_ids(tokens)
                    state.committed_drafts = list(output.draft_token_ids[rid])
                    req.spec_token_ids = [-1] if req.num_output_placeholders else list(state.committed_drafts)
                if self.request_block_hasher is not None:
                    self.cache.cache_blocks(req, req.num_computed_tokens - req.num_output_placeholders)
                self.mtp_stats['drafts_verified'] += max(width - 1, 0)
                self.mtp_stats['drafts_rejected'] += rejected
                self.mtp_stats['drafts_accepted'] += max(len(tokens) - 1, 0)
                self.mtp_stats['output_tokens'] += len(tokens)
            self.cache.block_pool.free_blocks(retained)
            self.inflight.popleft()
            self.draft_inputs.popleft()
            self.async_stats['resolved_batches'] += 1
            self.mtp_stats['resolved_batches'] += 1
        except Exception:
            self.poisoned = True
            raise
        return emitted
