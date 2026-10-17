#!/usr/bin/env python3
"""Build the bounded producer-first publication prototype from the pinned scheduler."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

BASE_SHA = 'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    raw = args.base.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    before = raw.decode()
    code = before

    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old[:100]
        code = code.replace(old, new)

    replace('from collections import defaultdict, deque\n',
            'import os\n\nfrom collections import defaultdict, deque\n')
    replace('        self.requests: dict[str, Request] = {}\n', '''        self.requests: dict[str, Request] = {}
        # Publication state only: native request history and GPU work stay intact.
        self._glm53_p_first_enabled = os.environ.get("GLM53_P_FIRST_PUBLISH") == "1"
        self._glm53_p_first_pending: dict[str, int] = {}
        if self._glm53_p_first_enabled:
            logger.info("[glm53/p-first] enabled: KV-ready publication, native continuation")
''')
    replace('''            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
''', '''            # Keep all native history, async-placeholder, stop and spec accounting
            # above. Only suppress the already published duplicate in the wire output.
            if new_token_ids and req_id in self._glm53_p_first_pending:
                published = self._glm53_p_first_pending.pop(req_id)
                if new_token_ids[0] != published:
                    logger.error("[glm53/p-first-mismatch] request=%s P=%s D=%s",
                                 req_id, published, new_token_ids[0])
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True
                    new_token_ids = []
                else:
                    logger.info("[glm53/p-first-verified] request=%s token=%s",
                                req_id, published)
                    new_token_ids = new_token_ids[1:]

            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
''')
    replace('''        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)
''', '''        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)
            if self._glm53_p_first_enabled:
                for req_id in kv_connector_output.finished_recving or ():
                    request = self.requests.get(req_id)
                    if (request is None or req_id in failed_kv_load_req_ids
                            or not self._glm53_can_publish_p_first(request)):
                        continue
                    token = request.kv_transfer_params["last_token_id"]
                    self._glm53_p_first_pending[req_id] = token
                    outputs[request.client_index].append(EngineCoreOutput(
                        request_id=req_id, new_token_ids=[token],
                        events=request.take_events(), trace_headers=request.trace_headers))
                    logger.info("[glm53/p-first-published] request=%s token=%s KV=ready",
                                req_id, token)
''')
    replace('''    def _update_waiting_for_remote_kv(self, request: Request) -> None:
''', '''    def _glm53_can_publish_p_first(self, request: Request) -> bool:
        # Deliberately narrow prototype. Unsupported options retain native output.
        sp = request.sampling_params
        params = request.kv_transfer_params or {}
        return bool(
            request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and request.request_id in self.finished_recving_kv_req_ids
            and request.request_id not in self.failed_recving_kv_req_ids
            and request.request_id not in self._glm53_p_first_pending
            and request.num_output_tokens == 0 and request.num_in_flight_tokens == 0
            and request.prompt_token_ids is not None and request.prompt_embeds is None
            and not request.mm_features and not request.resumable
            and not request.lora_request and not request.pooling_params
            and not request.use_structured_output
            and not self.enable_return_routed_experts and not self.return_sampling_mask
            and sp is not None and sp.temperature == 0 and sp.ignore_eos
            and sp.min_tokens == 0 and not sp.stop and not sp.stop_token_ids
            and sp.logprobs is None and sp.prompt_logprobs is None
            and sp.repetition_penalty == 1 and sp.presence_penalty == 0
            and sp.frequency_penalty == 0 and not sp.allowed_token_ids
            and not sp.logit_bias and not getattr(sp, "_bad_words_token_ids", None)
            and not getattr(sp, "logits_processors", None)
            and not getattr(sp, "repetition_detection", None)
            and params.get("remote_num_prompt_tokens") == request.num_prompt_tokens
            and params.get("remote_block_ids") and params.get("remote_request_id")
            and not params.get("do_remote_decode")
            and type(params.get("last_token_id")) is int
            and params["last_token_id"] >= 0
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
''')
    replace('''        assert request.is_finished()

        self._inflight_prefills.discard(request)
''', '''        assert request.is_finished()
        self._glm53_p_first_pending.pop(request.request_id, None)

        self._inflight_prefills.discard(request)
''')
    ast.parse(code)
    args.output.mkdir(parents=True, exist_ok=True)
    result = code.encode()
    (args.output/'scheduler.py').write_bytes(result)
    (Path(__file__).parent/'scheduler.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), code.splitlines(True),
        fromfile='a/vllm/v1/core/sched/scheduler.py', tofile='b/vllm/v1/core/sched/scheduler.py')))
    report = {'base_sha256': BASE_SHA, 'sha256': hashlib.sha256(result).hexdigest(),
              'status': 'offline_built_not_model_validated',
              'scope': 'Greedy text-only ignore_eos/min0 without stop/logprob/penalty/custom '
                       'processing. Publish only after aggregate native KV receive completion. '
                       'Native D work/history/stop/placeholder accounting retained; compare and '
                       'suppress exactly one duplicate. Mismatch terminates that request as error. '
                       'Cancellation and failure retain native resource lifecycle. '
                       'Potential TTFT improvement only; no compute saving claimed.'}
    (Path(__file__).parent/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
