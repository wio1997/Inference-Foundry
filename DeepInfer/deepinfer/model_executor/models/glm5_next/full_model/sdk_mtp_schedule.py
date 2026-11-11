"""One-token MTP descriptors and acceptance accounting for native scheduling.

DeepInfer decides admission, scheduled spans, stop conditions and request
lifetime. Only SDK physical-cache allocation and compute descriptors are used.
No SDK Scheduler or LLMEngine is constructed.
"""
from .sdk_schedule import SDKBatchSchedule


class SDKMTPBatchSchedule(SDKBatchSchedule):
    def __init__(self, *args, num_lookahead_tokens, num_speculative_tokens=1, **kwargs):
        super().__init__(*args, **kwargs)
        if num_speculative_tokens not in (1, 5):
            raise ValueError('only the validated-width candidate and pinned recipe width are supported')
        self.num_speculative_tokens = num_speculative_tokens
        if type(num_lookahead_tokens) is not int or num_lookahead_tokens < 1:
            raise ValueError('resolved SDK speculative lookahead required')
        self.num_lookahead_tokens = num_lookahead_tokens
        self.scheduled_drafts = {}
        self.mtp_stats = {'drafts_verified': 0, 'drafts_accepted': 0,
                          'drafts_rejected': 0, 'output_tokens': 0,
                          'resolved_batches': 0, 'native_host_output_overlap': False}

    def _allocation_kwargs(self, request):
        return {'num_lookahead_tokens': self.num_lookahead_tokens}

    def _validated_span(self, meta, state):
        if meta.is_prompt:
            return super()._validated_span(meta, state)
        from deepinfer.model_executor.models.glm5_next.scheduler_contract import ScheduledTextSpan
        req = state.request
        q, k = meta.seqlen_q, meta.seqlen_k
        expected_seq = meta.seq_id if state.seq_id is None else state.seq_id
        if (meta.is_placeholder or meta.multi_modal_data is not None
                or meta.request_id != req.request_id or meta.seq_id != expected_seq
                or type(q) is not int or type(k) is not int or not 1 <= q <= 1 + self.num_speculative_tokens
                or k - q != req.num_computed_tokens or k > self.max_context
                or meta.full_len != req.num_tokens
                or meta.last_output_token != req.all_token_ids[-1]):
            raise ValueError('native MTP span differs from committed SDK history')
        drafts = list(req.spec_token_ids[:q - 1])
        if (len(drafts) != q - 1 or list(meta.last_next_n_token[:q - 1]) != drafts
                or any(type(token) is not int or not 0 <= token < self.vocab_size for token in drafts)):
            raise ValueError('native MTP draft IDs differ from the actual preceding SDK draft')
        # get_input_token_ids() only supports committed spans. Draft positions
        # are deliberately validated through the real native MTP metadata above.
        return ScheduledTextSpan(meta.request_id, meta.seq_id, k - q, k, False)

    def prepare(self, metadata_list):
        result = super().prepare(metadata_list)
        self.scheduled_drafts = {m.request_id:
            list(self.requests[m.request_id].request.spec_token_ids[:m.seqlen_q - 1])
            for m in metadata_list if not m.is_prompt and m.seqlen_q > 1}
        result.scheduled_spec_decode_tokens = dict(self.scheduled_drafts)
        result.num_spec_tokens_to_schedule = self.num_speculative_tokens
        return result

    def commit(self, runner_output):
        if self.poisoned or self.pending is None:
            raise RuntimeError('no valid native MTP transaction')
        emitted = {}
        try:
            for rid, _, _, should_emit in self.pending:
                tokens = list(runner_output.sampled_token_ids[runner_output.req_id_to_index[rid]])
                drafts = self.scheduled_drafts.get(rid, [])
                if not should_emit:
                    if tokens:
                        raise ValueError('non-final MTP prefill emitted an output token')
                    continue
                if (not 1 <= len(tokens) <= len(drafts) + 1
                        or any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens)
                        or tokens[:-1] != drafts[:len(tokens) - 1]):
                    raise ValueError('SDK acceptance output violates the scheduled native draft contract')
                next_draft = runner_output.draft_token_ids.get(rid)
                if (next_draft is None or len(next_draft) != self.num_speculative_tokens
                        or any(type(t) is not int or not 0 <= t < self.vocab_size for t in next_draft)):
                    raise ValueError('actual next-step MTP draft missing or invalid')
                emitted[rid] = tokens
            for rid, seq_id, stop, _ in self.pending:
                state = self.requests[rid]
                req = state.request
                drafts = self.scheduled_drafts.get(rid, [])
                tokens = emitted.get(rid, [])
                rejected = len(drafts) - max(len(tokens) - 1, 0)
                state.seq_id, state.submitted = seq_id, True
                req.num_computed_tokens = stop - rejected
                if tokens:
                    req.append_output_token_ids(tokens)
                    req.spec_token_ids = list(runner_output.draft_token_ids[rid])
                if self.request_block_hasher is not None:
                    self.cache.cache_blocks(req, req.num_computed_tokens)
                self.mtp_stats['drafts_verified'] += len(drafts)
                self.mtp_stats['drafts_rejected'] += rejected
                self.mtp_stats['drafts_accepted'] += len(drafts) - rejected
                self.mtp_stats['output_tokens'] += len(tokens)
            self.cache.block_pool.free_blocks(self.retained_copy_blocks)
            self.retained_copy_blocks = []
            self.pending = None
            self.scheduled_drafts = {}
            self.mtp_stats['resolved_batches'] += 1
        except Exception:
            self.poisoned = True
            raise
        return emitted
