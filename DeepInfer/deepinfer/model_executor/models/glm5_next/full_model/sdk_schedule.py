"""Candidate: translate native DeepInfer batches into SDK compute descriptors.

No vLLM engine or scheduler is instantiated. DeepInfer owns admission, scheduling,
EOS/length decisions and callbacks; the SDK cache manager only allocates physical
cache groups consumed by its ModelRunner. This candidate is not deployed yet.
"""
from dataclasses import dataclass


@dataclass
class Registered:
    request: object
    seq_id: int | None = None
    submitted: bool = False


class SDKBatchSchedule:
    def __init__(self, cache_manager, *, max_context, max_batch_tokens, max_requests, vocab_size, num_cache_groups):
        if min(max_context, max_batch_tokens, max_requests, vocab_size, num_cache_groups) < 1:
            raise ValueError('positive explicit runtime limits required')
        self.cache = cache_manager
        self.max_context = max_context
        self.max_batch_tokens = max_batch_tokens
        self.max_requests = max_requests
        self.vocab_size = vocab_size
        self.num_cache_groups = num_cache_groups
        self.requests = {}
        self.finished = set()
        self.pending = None
        self.poisoned = False
        self.retained_copy_blocks = []

    def register(self, request_id, prompt_ids, native_params):
        from vllm import SamplingParams
        from vllm.v1.request import Request
        from deepinfer.model_executor.models.glm5_next.worker_dispatch import validate_bounded_glm_sampling_params
        validate_bounded_glm_sampling_params(native_params)
        if (self.poisoned or not isinstance(request_id, str) or not request_id
                or request_id in self.requests or request_id in self.finished):
            raise ValueError('invalid or already registered request identity')
        ids = list(prompt_ids)
        if (not ids or any(type(i) is not int or not 0 <= i < self.vocab_size for i in ids)
                or len(ids) + native_params.max_tokens > self.max_context):
            raise ValueError('registered prompt or output cap exceeds the runtime contract')
        # DeepInfer's C++ callback consumes the sampled token's real logprob.
        # Request only that value, rather than full-vocabulary logprobs.
        params = SamplingParams(temperature=0, top_p=1, top_k=-1,
            seed=native_params.seq_seed, max_tokens=native_params.max_tokens,
            ignore_eos=native_params.ignore_eos, logprobs=0)
        request = Request(request_id=request_id, prompt_token_ids=ids,
            sampling_params=params, pooling_params=None)
        self.requests[request_id] = Registered(request)

    def prepare(self, metadata_list):
        from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData, CachedRequestData
        from deepinfer.model_executor.models.glm5_next.scheduler_contract import scheduled_text_span
        if self.poisoned or self.pending is not None:
            raise RuntimeError('previous model transaction must complete before another batch')
        if not 1 <= len(metadata_list) <= self.max_requests:
            raise ValueError('native scheduled batch exceeds declared capacity')
        ids = [m.request_id for m in metadata_list]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate request in native batch')
        # All purely local metadata validation happens before cache mutation.
        spans = []
        for meta in metadata_list:
            if meta.request_id not in self.requests:
                raise ValueError('frontend must register the complete prompt before scheduling')
            state = self.requests[meta.request_id]
            req = state.request
            span = scheduled_text_span(meta, request_id=meta.request_id,
                seq_id=meta.seq_id if state.seq_id is None else state.seq_id,
                tokens_seen=req.num_computed_tokens, max_context_tokens=self.max_context)
            actual = meta.get_input_token_ids()
            expected = list(req.all_token_ids[span.token_start:span.token_stop])
            if actual != expected:
                raise ValueError('native input tokens differ from registered SDK request history')
            spans.append(span)
        total = sum(s.token_stop-s.token_start for s in spans)
        if total > self.max_batch_tokens:
            raise ValueError('native batch token budget exceeded')
        new, cached_ids, cached_blocks, computed, output_counts = [], [], [], [], []
        try:
            self.cache.new_step_starts()
            for meta, span in zip(metadata_list, spans, strict=True):
                state = self.requests[meta.request_id]
                req = state.request
                allocation = self.cache.allocate_slots(req, span.token_stop-span.token_start)
                if allocation is None:
                    raise RuntimeError('SDK physical cache exhausted;do not execute a partially prepared batch')
                if not state.submitted:
                    new.append(NewRequestData(req_id=req.request_id,
                        prompt_token_ids=list(req.prompt_token_ids), mm_features=[],
                        sampling_params=req.sampling_params, pooling_params=None,
                        block_ids=self.cache.get_block_ids(req.request_id),
                        num_computed_tokens=req.num_computed_tokens, lora_request=None))
                else:
                    cached_ids.append(req.request_id)
                    cached_blocks.append(allocation.get_block_ids())
                    computed.append(req.num_computed_tokens)
                    output_counts.append(req.num_output_tokens)
            cached = CachedRequestData(req_ids=cached_ids, resumed_req_ids=set(),
                new_token_ids=[[] for _ in cached_ids], all_token_ids={},
                new_block_ids=cached_blocks, num_computed_tokens=computed,
                num_output_tokens=output_counts)
            copies, retained = self.cache.take_kv_cache_block_copies()
            self.retained_copy_blocks = retained
            result = SchedulerOutput(scheduled_new_reqs=new, scheduled_cached_reqs=cached,
                num_scheduled_tokens={m.request_id: s.token_stop-s.token_start
                    for m, s in zip(metadata_list, spans, strict=True)},
                total_num_scheduled_tokens=total, scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={}, num_common_prefix_blocks=[0]*self.num_cache_groups,
                finished_req_ids=set(self.finished), free_encoder_mm_hashes=[],
                new_block_ids_to_zero=self.cache.take_new_block_ids(),
                kv_cache_block_copies=copies or None)
        except Exception:
            # Allocator mutation is not rolled back piecemeal. The owning worker
            # must fail the bounded run and discard the entire candidate state.
            self.poisoned = True
            raise
        self.finished.clear()
        self.pending = [(m.request_id, m.seq_id, s.token_stop,
            not m.is_prompt or m.is_last_prompt_chunk)
            for m, s in zip(metadata_list, spans, strict=True)]
        return result

    def commit(self, runner_output):
        if self.poisoned or self.pending is None:
            raise RuntimeError('no valid prepared model transaction')
        try:
            emitted = {}
            for request_id, seq_id, stop, should_emit in self.pending:
                index = runner_output.req_id_to_index[request_id]
                tokens = runner_output.sampled_token_ids[index]
                if should_emit:
                    if len(tokens) != 1 or not 0 <= tokens[0] < self.vocab_size:
                        raise ValueError('ordinary greedy batch must emit one token per ready request')
                    emitted[request_id] = int(tokens[0])
                elif tokens:
                    raise ValueError('non-final prefill chunk unexpectedly emitted a token')
            for request_id, seq_id, stop, _ in self.pending:
                state = self.requests[request_id]
                state.seq_id = seq_id
                state.submitted = True
                state.request.num_computed_tokens = stop
                if request_id in emitted:
                    state.request.append_output_token_ids(emitted[request_id])
        except Exception:
            self.poisoned = True
            raise
        # This adapter is synchronous: the CPU sampled output is available,
        # so the preceding copy/model stream has completed. The SDK scheduler
        # uses the same pool free when defer_block_free is false.
        self.cache.block_pool.free_blocks(self.retained_copy_blocks)
        self.retained_copy_blocks = []
        self.pending = None
        return emitted

    def prepare_cleanup(self):
        """Close the SDK's last finished-request states without model execution."""
        from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData
        if self.poisoned or self.pending is not None or self.requests:
            raise RuntimeError('cleanup requires no live or uncommitted native requests')
        result = SchedulerOutput(scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={}, total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0]*self.num_cache_groups,
            finished_req_ids=set(self.finished), free_encoder_mm_hashes=[])
        self.finished.clear()
        self.pending = []
        return result

    def finish(self, request_ids):
        if self.poisoned or self.pending is not None:
            raise RuntimeError('cannot release cache during an uncommitted model step')
        request_ids = list(request_ids)
        if len(set(request_ids)) != len(request_ids) or any(r not in self.requests for r in request_ids):
            raise ValueError('terminal notification must name distinct registered requests')
        try:
            for request_id in request_ids:
                state = self.requests.pop(request_id)
                self.cache.free(state.request)
                if state.submitted:
                    self.finished.add(request_id)
        except Exception:
            self.poisoned = True
            raise
        # The next SDK execute_model receives finished_req_ids before it reuses
        # slots. The worker must also submit an empty cleanup batch at shutdown.
