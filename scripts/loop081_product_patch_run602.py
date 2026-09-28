#!/usr/bin/env python3
"""SHA-pinned reversible same-W0 Product ownership observer (diagnostic only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

import loop081_basis_patch_run597 as base

ROOT = Path('/data/wio/Inference_Foundry')
VLLM = Path('/data/wio/vllm_ascend_26/framework/vllm/vllm')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = dict(base.SOURCES)
SOURCES.update({
    'kvdelivery': ASC / 'patch/platform/patch_kv_delivery_preemption.py',
    'chat_serving': VLLM / 'entrypoints/openai/chat_completion/serving.py',
})
ORIGINAL = dict(base.ORIGINAL)
ORIGINAL.update({
    'kvdelivery': '5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d',
    'chat_serving': '860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067',
})
MARK = '# EXTREME_LOOP081_RUN602'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def one(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f'anchor count {src.count(old)}: {old[:90]!r}')
    return src.replace(old, new, 1)


ENTRY = '''        # EXTREME_LOOP081_RUN602_ENTRY
        _p602_dir = os.getenv("EXTREME_RUN602_DIR")
        _p602_enabled = bool(_p602_dir and os.path.exists(os.path.join(_p602_dir, "arm")))
        if _p602_enabled:
            _p602_rank = int(get_tp_group().rank_in_group)
            _p602_new = [{"req_id": x.req_id,
                          "prompt_len": len(x.prompt_token_ids) if x.prompt_token_ids is not None else None,
                          "num_computed_tokens": int(x.num_computed_tokens),
                          "scheduled_tokens": int(scheduler_output.num_scheduled_tokens.get(x.req_id, 0)),
                          "spec_tokens": len(scheduler_output.scheduled_spec_decode_tokens.get(x.req_id, []))}
                         for x in scheduler_output.scheduled_new_reqs]
            _p602_cached = [{"req_id": req_id, "num_computed_tokens": int(computed),
                             "scheduler_output_positions_including_placeholders": int(output),
                             "scheduled_tokens": int(scheduler_output.num_scheduled_tokens.get(req_id, 0)),
                             "spec_tokens": len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))}
                            for req_id, computed, output in zip(
                                scheduler_output.scheduled_cached_reqs.req_ids,
                                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                                scheduler_output.scheduled_cached_reqs.num_output_tokens)]
            self._p602_marks = getattr(self, "_p602_marks", [])
            self._p602_marks.append({"kind": "execute_entry", "rank": _p602_rank,
                                     "t_ns": time.monotonic_ns(),
                                     "total_scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens),
                                     "new": _p602_new, "cached": _p602_cached})
'''

BUILT = '''            # EXTREME_LOOP081_RUN602_BUILT
            if _p602_enabled:
                self._p602_marks.append({"kind": "runtime_built_host", "rank": _p602_rank,
                                         "t_ns": time.monotonic_ns(),
                                         "req_ids": list(_extreme_req_ids),
                                         "target_graph_mode": str(_target_inputs.aclgraph_runtime_mode)})
'''

PROPOSE_START = '''            # EXTREME_LOOP081_RUN602_PROPOSE_START
            _p602_prop_dir = os.getenv("EXTREME_RUN602_DIR")
            _p602_enabled = bool(_p602_prop_dir and os.path.exists(os.path.join(_p602_prop_dir, "arm")))
            _p602_prop_start = None
            if _p602_enabled:
                _p602_prop_start = torch.npu.Event(enable_timing=True)
                _p602_prop_end = torch.npu.Event(enable_timing=True)
                _p602_prop_host_start = time.monotonic_ns()
                _p602_prop_start.record()
'''

PROPOSE_END = '''            # EXTREME_LOOP081_RUN602_PROPOSE_END
            if _p602_prop_start is not None:
                _p602_prop_end.record()
                self._p602_calls = getattr(self, "_p602_calls", [])
                self._p602_calls.append((
                    {"kind": "propose_and_optional_copy", "execute_mark_count": len(self._p602_marks),
                     "host_start_ns": _p602_prop_host_start, "host_end_ns": time.monotonic_ns(),
                     "req_ids": list(self.input_batch.req_ids),
                     "copy_event_present": self.draft_token_ids_event is not None},
                    _p602_prop_start, _p602_prop_end))
'''

SERVE_START = '''                # EXTREME_LOOP081_RUN602_SERVE_START
                if _p602_enabled:
                    self._p602_marks.append({"kind": "serve_start_host", "rank": _p602_rank,
                                             "t_ns": time.monotonic_ns(), "req_ids": list(_extreme_req_ids)})
'''

SERVE_END = '''                # EXTREME_LOOP081_RUN602_SERVE_END
                if _p602_enabled:
                    self._p602_marks.append({"kind": "serve_synced_host", "rank": _p602_rank,
                                             "t_ns": time.monotonic_ns(), "req_ids": list(_extreme_req_ids)})
'''

FLUSH = '''                # EXTREME_LOOP081_RUN602_FLUSH_AFTER_EXISTING_SYNC
                if _p602_enabled:
                    _p602_calls = []
                    for _p602_row, _p602_start, _p602_end in getattr(self, "_p602_calls", []):
                        _p602_row["current_stream_elapsed_ms"] = float(_p602_start.elapsed_time(_p602_end))
                        _p602_calls.append(_p602_row)
                    _p602_payload = {"rank": _p602_rank, "cohort": _cohort_index,
                                     "run_ts": os.getenv("RUN_TS"),
                                     "time_namespace": os.readlink("/proc/self/ns/time"),
                                     "marks": self._p602_marks,
                                     "calls": _p602_calls,
                                     "scope": "Host marks and current-stream events only; side-stream writer readiness unproved"}
                    os.makedirs(_p602_dir, exist_ok=True)
                    with open(os.path.join(_p602_dir, f"rank{_p602_rank}_cohort{_cohort_index}.json"), "w") as _f:
                        json.dump(_p602_payload, _f, separators=(",", ":"))
                    self._p602_marks = []
                    self._p602_calls = []
'''

APPEND_RUNNER = '''
# EXTREME_LOOP081_RUN602_FORWARD_WRAPPER
if os.getenv("EXTREME_RUN602_DIR"):
    _p602_original_forward = NPUModelRunner._model_forward
    def _p602_forward(self, num_tokens_padded, *args, **kwargs):
        _p602_dir = os.environ["EXTREME_RUN602_DIR"]
        if (len(getattr(self, "_extreme_served_cohorts", ())) < 4
                or not os.path.exists(os.path.join(_p602_dir, "arm"))):
            return _p602_original_forward(self, num_tokens_padded, *args, **kwargs)
        _p602_start = torch.npu.Event(enable_timing=True)
        _p602_end = torch.npu.Event(enable_timing=True)
        _p602_host_start = time.monotonic_ns()
        _p602_start.record()
        try:
            return _p602_original_forward(self, num_tokens_padded, *args, **kwargs)
        finally:
            _p602_end.record()
            self._p602_calls = getattr(self, "_p602_calls", [])
            self._p602_calls.append((
                {"kind": "target_forward", "num_tokens_padded": int(num_tokens_padded),
                 "execute_mark_count": len(getattr(self, "_p602_marks", ())),
                 "host_start_ns": _p602_host_start, "host_end_ns": time.monotonic_ns()},
                _p602_start, _p602_end))
    NPUModelRunner._model_forward = _p602_forward
'''


def patch_runner(src):
    src = base.patch('runner', src)
    src = one(src,
        '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if self.vllm_config.model_config.enable_return_routed_experts:',
        '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n' + ENTRY +
        '        if self.vllm_config.model_config.enable_return_routed_experts:')
    src = one(src,
        '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )',
        '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n' + BUILT.rstrip())
    src = one(src,
        '            self._draft_token_ids = self.propose_draft_token_ids(\n',
        PROPOSE_START + '            self._draft_token_ids = self.propose_draft_token_ids(\n')
    src = one(src,
        '            self._copy_draft_token_ids_to_cpu(scheduler_output)\n',
        '            self._copy_draft_token_ids_to_cpu(scheduler_output)\n' + PROPOSE_END)
    src = one(src,
        '                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(',
        SERVE_START + '                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(')
    src = one(src,
        '                torch.npu.synchronize()\n                _extreme_wall_seconds = (',
        '                torch.npu.synchronize()\n' + SERVE_END + '                _extreme_wall_seconds = (')
    src = one(src,
        '                self._extreme_serving_output = ModelRunnerOutput(\n',
        FLUSH + '                self._extreme_serving_output = ModelRunnerOutput(\n')
    src += APPEND_RUNNER
    return src


def patch_kvdelivery(src):
    src = one(src, 'import time\nfrom collections import defaultdict',
        'import time\nimport os\nimport json\nfrom collections import defaultdict')
    src = one(src,
        '        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)\n',
        '        # EXTREME_LOOP081_RUN602_SCHED_INIT\n'
        '        _p602_dir = os.getenv("EXTREME_RUN602_DIR")\n'
        '        _p602_enabled = bool(_p602_dir and os.path.exists(os.path.join(_p602_dir, "arm"))\n'
        '                             and model_runner_output.extreme_bulk_output)\n'
        '        _p602_rows = []\n'
        '        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)\n')
    src = one(src,
        '            num_output_tokens_before = len(request._output_token_ids)\n',
        '            num_output_tokens_before = len(request._output_token_ids)\n'
        '            # EXTREME_LOOP081_RUN602_SCHED_PRE\n'
        '            _p602_before = ({"req_id": req_id, "t_ns": time.monotonic_ns(),\n'
        '                             "pre_ids": list(request._output_token_ids),\n'
        '                             "pre_placeholders": int(request.num_output_placeholders),\n'
        '                             "incoming_bulk_ids": list(generated_token_ids),\n'
        '                             "scheduled_tokens": int(num_tokens_scheduled)}\n'
        '                            if _p602_enabled else None)\n')
    src = one(src,
        '            if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids=new_token_ids):',
        '            # EXTREME_LOOP081_RUN602_SCHED_POST\n'
        '            if _p602_before is not None:\n'
        '                _p602_before.update(accepted_bulk_ids=list(new_token_ids),\n'
        '                                    post_ids=list(request._output_token_ids),\n'
        '                                    post_placeholders=int(request.num_output_placeholders),\n'
        '                                    stopped=bool(stopped), post_ns=time.monotonic_ns())\n'
        '                _p602_rows.append(_p602_before)\n'
        '            if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids=new_token_ids):')
    src = one(src,
        '        return engine_core_outputs\n\n    def _update_request_with_output(',
        '        # EXTREME_LOOP081_RUN602_SCHED_FLUSH\n'
        '        if _p602_enabled:\n'
        '            _p602_index = getattr(self, "_p602_bulk_index", 0) + 1\n'
        '            self._p602_bulk_index = _p602_index\n'
        '            _p602_path = os.path.join(_p602_dir, f"scheduler_bulk_{_p602_index}.json")\n'
        '            with open(_p602_path, "w") as _p602_f:\n'
        '                json.dump({"kind": "bulk_step", "run_ts": os.getenv("RUN_TS"),\n'
        '                           "time_namespace": os.readlink("/proc/self/ns/time"),\n'
        '                           "write_begin_ns": time.monotonic_ns(),\n'
        '                           "rows": _p602_rows}, _p602_f, separators=(",", ":"))\n'
        '        return engine_core_outputs\n\n    def _update_request_with_output(')
    return src


def patch_chat(src):
    src = one(src,
        '        created_time = int(time.time())\n        chunk_object_type: Final = "chat.completion.chunk"',
        '        # EXTREME_LOOP081_RUN602_API_INIT\n'
        '        import os, json, hashlib\n'
        '        _p602_dir = os.getenv("EXTREME_RUN602_DIR")\n'
        '        _p602_enabled = bool(_p602_dir and os.path.exists(os.path.join(_p602_dir, "arm")))\n'
        '        _p602_events = []\n'
        '        created_time = int(time.time())\n        chunk_object_type: Final = "chat.completion.chunk"')
    src = one(src,
        '                    previous_num_tokens[i] += len(output.token_ids)\n',
        '                    previous_num_tokens[i] += len(output.token_ids)\n'
        '                    # EXTREME_LOOP081_RUN602_API_OUTPUT\n'
        '                    if _p602_enabled:\n'
        '                        _p602_events.append({"choice": i, "engine_seen_ns": time.monotonic_ns(),\n'
        '                                             "token_ids": as_list(output.token_ids),\n'
        '                                             "cum_output_tokens": previous_num_tokens[i],\n'
        '                                             "finish_reason": output.finish_reason,\n'
        '                                             "sse_yield_ns": None})\n')
    src = one(src,
        '                    data = chunk.model_dump_json(exclude_unset=True)\n                    yield f"data: {data}\\n\\n"',
        '                    data = chunk.model_dump_json(exclude_unset=True)\n'
        '                    # EXTREME_LOOP081_RUN602_API_YIELD\n'
        '                    if _p602_enabled:\n'
        '                        _p602_events[-1]["sse_yield_ns"] = time.monotonic_ns()\n'
        '                        _p602_events[-1]["sse_payload_sha256"] = hashlib.sha256(data.encode()).hexdigest()\n'
        '                    yield f"data: {data}\\n\\n"')
    src = one(src,
        '        # Send the final done message after all response.n are finished\n        yield "data: [DONE]\\n\\n"',
        '        # EXTREME_LOOP081_RUN602_API_FLUSH\n'
        '        if _p602_enabled:\n'
        '            _p602_name = hashlib.sha256(request_id.encode()).hexdigest()[:24]\n'
        '            with open(os.path.join(_p602_dir, f"api_{_p602_name}.json"), "w") as _p602_f:\n'
        '                json.dump({"kind": "api_stream", "run_ts": os.getenv("RUN_TS"),\n'
        '                           "time_namespace": os.readlink("/proc/self/ns/time"),\n'
        '                           "request_id": request_id, "events": _p602_events,\n'
        '                           "previous_num_tokens": previous_num_tokens,\n'
        '                           "before_done_ns": time.monotonic_ns()},\n'
        '                          _p602_f, separators=(",", ":"))\n'
        '        # Send the final done message after all response.n are finished\n'
        '        yield "data: [DONE]\\n\\n"')
    return src


def patch(key, src):
    if MARK in src:
        raise ValueError('Run602 already patched')
    if key == 'runner':
        return patch_runner(src)
    if key == 'kvdelivery':
        return patch_kvdelivery(src)
    if key == 'chat_serving':
        return patch_chat(src)
    return base.patch(key, src)


def atomic(path, data):
    tmp = path.with_name(path.name + '.run602.tmp')
    if tmp.exists():
        raise ValueError('stale temporary file')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp, path)


def prepared():
    helper = base.HELPER.read_bytes()
    compile(helper, str(base.HELPER), 'exec')
    manifest = dict(helper_sha256=sha(helper), sources={})
    old, new = {}, {}
    for key, path in SOURCES.items():
        data = path.read_bytes()
        if sha(data) != ORIGINAL[key]:
            raise ValueError(f'{key} original SHA drift: {sha(data)}')
        edited = patch(key, data.decode()).encode()
        compile(edited, str(path), 'exec')
        old[key], new[key] = data, edited
        manifest['sources'][key] = dict(path=str(path), original_sha256=sha(data),
                                        patched_sha256=sha(edited))
    return manifest, old, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--offline-confirmed', action='store_true')
    args = parser.parse_args()
    if args.action != 'check' and not args.offline_confirmed:
        raise ValueError('offline confirmation required')
    if args.action in ('check', 'install'):
        manifest, old, new = prepared()
        if args.action == 'install':
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError('fresh state directory required')
            args.state_dir.mkdir(parents=True)
            for key, data in old.items():
                atomic(args.state_dir / f'{key}.orig', data)
            atomic(args.state_dir / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
            try:
                for key, path in SOURCES.items():
                    if path.read_bytes() != old[key]:
                        raise ValueError('install race ' + key)
                    atomic(path, new[key])
            except BaseException:
                for key, path in SOURCES.items():
                    if sha(path.read_bytes()) == manifest['sources'][key]['patched_sha256']:
                        atomic(path, old[key])
                raise
    else:
        if args.state_dir is None:
            raise ValueError('state dir required')
        manifest = json.loads((args.state_dir / 'manifest.json').read_text())
        if set(manifest['sources']) != set(SOURCES):
            raise ValueError('source set drift')
        for key, path in SOURCES.items():
            row = manifest['sources'][key]
            if row['path'] != str(path) or row['original_sha256'] != ORIGINAL[key]:
                raise ValueError('source manifest drift ' + key)
            backup = (args.state_dir / f'{key}.orig').read_bytes()
            if sha(backup) != ORIGINAL[key]:
                raise ValueError('backup drift ' + key)
            if sha(path.read_bytes()) not in (row['original_sha256'], row['patched_sha256']):
                raise ValueError('installed source drift ' + key)
        for key, path in SOURCES.items():
            if sha(path.read_bytes()) != ORIGINAL[key]:
                atomic(path, (args.state_dir / f'{key}.orig').read_bytes())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(dict(action=args.action, **manifest), indent=2) + '\n')
    print(json.dumps(dict(action=args.action, files=len(SOURCES))))


if __name__ == '__main__':
    main()
