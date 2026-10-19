#!/usr/bin/env python3
"""Build a gated, per-request P-token history/query-advance prototype."""
import argparse
import ast
import base64
import difflib
import hashlib
import json
from pathlib import Path
import zlib

PINS = {
 'scheduler.py': ('vllm/vllm/v1/core/sched/scheduler.py', 'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884'),
 'output.py': ('vllm/vllm/v1/core/sched/output.py', '3a66c8efaefe10c572f6f33cf1f8aa4610758c945130cad62eb9f7331005677c'),
 'gpu_model_runner.py': ('vllm/vllm/v1/worker/gpu_model_runner.py', '87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51'),
 'model_runner_v1.py': ('vllm-ascend/vllm_ascend/worker/model_runner_v1.py', '568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c')}

ELIGIBILITY = '''    def _glm53_can_adopt_first(self, request):
        sp, params = request.sampling_params, request.kv_transfer_params or {}
        return bool(
            self._glm53_adopt_enabled and params.get("_glm53_adopt_first") is True
            and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and request.request_id in self.finished_recving_kv_req_ids
            and request.request_id not in self.failed_recving_kv_req_ids
            and request.num_computed_tokens == request.num_prompt_tokens
            and request.num_prompt_tokens in (256, 257) and request.max_tokens > 1
            and request.num_output_tokens == request.num_in_flight_tokens == request.num_output_placeholders == 0
            and request.prompt_token_ids is not None and request.prompt_embeds is None
            and not request.mm_features and not request.resumable
            and not request.lora_request and not request.pooling_params
            and not request.use_structured_output
            and not self.enable_return_routed_experts and not self.return_sampling_mask
            and sp is not None and sp.temperature == 0 and sp.ignore_eos and sp.min_tokens == 1
            and not sp.stop and not sp.stop_token_ids and sp.logprobs is None and sp.prompt_logprobs is None
            and sp.repetition_penalty == 1 and sp.presence_penalty == sp.frequency_penalty == 0
            and not sp.allowed_token_ids and not sp.logit_bias
            and not getattr(sp, "_bad_words_token_ids", None) and not getattr(sp, "logits_processors", None)
            and not getattr(sp, "repetition_detection", None)
            and params.get("remote_num_prompt_tokens") == request.num_prompt_tokens
            and params.get("remote_block_ids") and params.get("remote_request_id")
            and not params.get("do_remote_decode")
            and type(params.get("last_token_id")) is int and params["last_token_id"] >= 0)

'''


def digest(b): return hashlib.sha256(b).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, repr(old)
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True)
    raw = {name: (a.source_root/path).read_bytes() for name, (path, _) in PINS.items()}
    for name, (_, sha) in PINS.items(): assert digest(raw[name]) == sha, name
    contents = {name: data.decode() for name, data in raw.items()}
    s = contents['scheduler.py']
    s = replace_once(s, 'import itertools\n', 'import itertools\nimport os\n')
    s = replace_once(s, '        self.requests: dict[str, Request] = {}\n',
        '        self.requests: dict[str, Request] = {}\n        self._glm53_adopt_enabled = os.environ.get("GLM53_P_FIRST_ADOPT") == "1"\n        self._glm53_adopt_pending = {}\n')
    s = replace_once(s, '    def _update_waiting_for_remote_kv(self, request: Request) -> None:\n',
        ELIGIBILITY + '    def _update_waiting_for_remote_kv(self, request: Request) -> None:\n')
    s = replace_once(s, '            self._update_from_kv_xfer_finished(kv_connector_output)\n', '''            self._update_from_kv_xfer_finished(kv_connector_output)
            for req_id in kv_connector_output.finished_recving or ():
                request = self.requests.get(req_id)
                if request is None or req_id in failed_kv_load_req_ids or not self._glm53_can_adopt_first(request):
                    continue
                token = request.kv_transfer_params["last_token_id"]
                request.append_output_token_ids(token)
                request._glm53_adopted = True
                self._glm53_adopt_pending[req_id] = token
                logger.info("[glm53/adopt-first] request=%s N=%s token=%s computed=%s budget=%s",
                            req_id, request.num_prompt_tokens, token, request.num_computed_tokens, request.max_tokens)
''')
    s = replace_once(s, '            should_emit_output = bool(\n', '''            # Native history/placeholder/stop accounting above consumes ONLY
            # new D samples. y1 is already in history, so prepend only on wire.
            if new_token_ids and req_id in self._glm53_adopt_pending:
                first = self._glm53_adopt_pending.pop(req_id)
                new_token_ids = [first] + list(new_token_ids)
                logger.info("[glm53/adopt-wire] request=%s initial=%s returned=%s", req_id, first, len(new_token_ids))

            should_emit_output = bool(
''')
    s = replace_once(s, '        assert request.is_finished()\n\n        self._inflight_prefills.discard(request)\n',
        '        assert request.is_finished()\n        self._glm53_adopt_pending.pop(request.request_id, None)\n\n        self._inflight_prefills.discard(request)\n')
    contents['scheduler.py'] = s
    s = contents['output.py']
    s = replace_once(s, '    # Only used for v2 model runner.\n    prefill_token_ids: list[int] | None = None\n',
        '    # Explicit initial output history for the opt-in P-token adoption prototype.\n    initial_output_token_ids: list[int] | None = None\n\n    # Only used for v2 model runner.\n    prefill_token_ids: list[int] | None = None\n')
    s = replace_once(s, '            prefill_token_ids=prefill_token_ids,\n',
        '            prefill_token_ids=prefill_token_ids,\n            initial_output_token_ids=(list(request.output_token_ids) if getattr(request, "_glm53_adopted", False) else None),\n')
    contents['output.py'] = s
    s = contents['gpu_model_runner.py']
    s = replace_once(s, '                output_token_ids=[],\n', '                output_token_ids=list(new_req_data.initial_output_token_ids or ()),\n')
    contents['gpu_model_runner.py'] = s
    helper = Path(__file__).with_name('trace.py').read_text(); ast.parse(helper)
    contents['model_runner_v1.py'] += "\n\nif __import__('os').environ.get('GLM53_P_FIRST_ADOPT_TRACE') == '1':\n    import types as _glm53_types\n    _glm53_adopt_trace = _glm53_types.ModuleType('_glm53_adopt_trace')\n    exec(compile(" + repr(helper) + ", '<glm53-adopt-trace>', 'exec'), _glm53_adopt_trace.__dict__)\n    _glm53_adopt_trace.install(globals())\n"
    identity = digest(''.join(contents[n] for n in sorted(contents)).encode())
    remote = '/data/wio/glm5-3-vllm-execution/20261017-official-pd/execution-patches/p-first-adopt-v1/' + identity[:12]
    patches, payload = [], {}
    for name, s in contents.items():
        ast.parse(s); (a.output/name).write_text(s)
        (a.output/(name+'.patch')).write_text(''.join(difflib.unified_diff(raw[name].decode().splitlines(True), s.splitlines(True), fromfile=name, tofile=name)))
        target = '/vllm-workspace/' + PINS[name][0]
        patches.append({'source': remote+'/'+name, 'target': target, 'sha256': digest(s.encode()), 'base_sha256': PINS[name][1]})
        payload[name] = base64.b64encode(zlib.compress(s.encode())).decode()
    stage = '''import base64,hashlib,json,socket,zlib
from pathlib import Path
assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
root=Path(ROOT);root.mkdir(parents=True,exist_ok=True)
files=FILES
for name,value in PAYLOAD.items():
 data=zlib.decompress(base64.b64decode(value));assert hashlib.sha256(data).hexdigest()==files[name]
 dest=root/name
 if dest.exists():assert dest.read_bytes()==data
 else:
  with dest.open('xb') as out:out.write(data)
  dest.chmod(0o444)
print('GLM53_RESULT_JSON_BEGIN\\n'+json.dumps({'staged':True,'files':files,'root':str(root),'models_loaded':False})+'\\nGLM53_RESULT_JSON_END')
'''.replace('ROOT',repr(remote)).replace('FILES',repr({n: digest(s.encode()) for n,s in contents.items()})).replace('PAYLOAD',repr(payload))
    ast.parse(stage);(a.output/'stage_overlay.py').write_text(stage)
    manifest = {'identity':identity,'patches':patches,'remote_root':remote,'stage_sha256':digest(stage.encode()),'default_adoption':False,'request_opt_in_key':'_glm53_adopt_first','shape_scope':[256,257],'max1':'native fallback','wire_policy':'y1 prepended to first ordinary D sample output after native history/stop handling; no early publication','full_prefix_proved':False,'device_run':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
