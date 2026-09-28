#!/usr/bin/env python3
"""SHA-pinned source-only candidate for a full48 light stage-cost observer.

This script never edits live sources or starts a service.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
OUT = ROOT / 'evidence/20260928_loop081_bound/run655'
SOURCES = {
    'cycle': (ROOT / 'runtime/extreme_decode.py', 'eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499'),
    'serving': (ROOT / 'runtime/fixed_serving.py', '137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a'),
    'runner': (ASC / 'worker/model_runner_v1.py', '004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def one(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise AssertionError(f'anchor count={src.count(old)}: {old[:100]!r}')
    return src.replace(old, new, 1)


def cycle(src: str) -> str:
    src = one(src, '        self.diagnostic_events = []\n', '''        self.diagnostic_events = []
        # Run655 dormant full-cohort outer-stage observer, armed post-warmup.
        _light_ledger = os.getenv("EXTREME_LIGHT_LEDGER_DIR")
        if os.getenv("EXTREME_LIGHT_EVENT_DIR") and not _light_ledger:
            raise RuntimeError("light Event observer requires common ledger")
        self._light_armed = bool(_light_ledger and os.path.isfile(
            os.path.join(_light_ledger, "arm")))
        self._light_probe = None
        if self._light_armed and os.getenv("EXTREME_LIGHT_EVENT_DIR"):
            if self._profile_dag or self._diagnose or self._cycle_profile_dir:
                raise RuntimeError("light Event observer conflicts with profiler")
            if target_page_audit is not None or kv_slot_audit is not None:
                raise RuntimeError("light Event observer excludes page/KV audit")
''')
    src = one(src, '''                markers.append((label, event))
        with self._scope("extreme::cycle"):
''', '''                markers.append((label, event))
            if self._light_probe is not None and label in (
                "begin", "target", "proposer"
            ):
                self._light_probe.mark(
                    self.state.cycle_index,
                    {"begin": "cycle_begin", "target": "target_after",
                     "proposer": "proposer_after"}[label],
                )
        with self._scope("extreme::cycle"):
''')
    src = one(src, '''                with self._scope("extreme::target"):
                    if self._cycle_profiler''', '''                with self._scope("extreme::target"):
                    if self._light_probe is not None:
                        self._light_probe.mark(self.state.cycle_index, "target_before")
                    if self._cycle_profiler''')
    src = one(src, '''            with self._scope("extreme::proposer"):
                next_draft =''', '''            with self._scope("extreme::proposer"):
                if self._light_probe is not None:
                    self._light_probe.mark(self.state.cycle_index, "proposer_before")
                next_draft =''')
    return src


def serving(src: str) -> str:
    src = one(src, '''        for index in range(self.max_cycles):
            result = self.runtime.step()
''', '''        for index in range(self.max_cycles):
            _light_parked_before = sum(self._parked) if self.runtime._light_probe else 0
            result = self.runtime.step()
''')
    src = one(src, '''            self._park_completed(progress)
            if all(
''', '''            self._park_completed(progress)
            if self.runtime._light_probe is not None:
                self.runtime._light_probe.class_row(
                    index, _light_parked_before, sum(self._parked))
            if all(
''')
    src = one(src, '''                for slot in range(cfg.batch_size)
            ):
                break
''', '''                for slot in range(cfg.batch_size)
            ):
                if self.runtime._light_probe is not None:
                    self.runtime._light_probe.terminal(index)
                break
''')
    return one(src, '''        counts_cpu = count_history[:cycles].cpu()
        output: list[list[int]] =''', '''        counts_cpu = count_history[:cycles].cpu()
        if self.runtime._light_armed:
            self.runtime._light_cpu_history = (tokens_cpu, counts_cpu)
        output: list[list[int]] =''')


def runner(src: str) -> str:
    src = one(src, '''        # Run forward pass\n''', '''        # Run655 Event pool is created and first-recorded during service\n        # warmup, before the measured arm is opened. No new device drain.\n        if os.getenv("EXTREME_LIGHT_EVENT_DIR") and not hasattr(self, "_light_pool"):\n            _light_root = "/data/wio/Inference_Foundry"\n            if _light_root not in sys.path:\n                sys.path.insert(0, _light_root)\n            from runtime.light_stage_observer import LightStageRecorder\n            self._light_pool = LightStageRecorder(torch.npu.current_stream())\n        # Run forward pass\n''')
    src = one(src, '''            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n''', '''            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n            if _extreme_runtime._light_armed and os.getenv("EXTREME_LIGHT_EVENT_DIR"):\n                _light_pool = getattr(self, "_light_pool", None)\n                if _light_pool is None:\n                    raise RuntimeError("light Event pool was not warmed during service warmup")\n                _light_pool.begin_cohort(len(self._extreme_served_cohorts))\n                _extreme_runtime._light_probe = _light_pool\n''')
    src = one(src, '''                _extreme_wall_seconds = (
                    time.perf_counter() - _extreme_wall_start
                )
                _host_mirror_exact =''', '''                _extreme_wall_seconds = (
                    time.perf_counter() - _extreme_wall_start
                )
                # Existing terminal drain above precedes Event extraction.
                # ON extraction delays Product publication; OFF/ON/OFF needed.
                if _extreme_runtime._light_armed:
                    import hashlib
                    _light_ledger_dir = os.environ["EXTREME_LIGHT_LEDGER_DIR"]
                    os.makedirs(_light_ledger_dir, exist_ok=True)
                    _light_rank = int(get_tp_group().rank_in_group)
                    _light_cohort = len(self._extreme_served_cohorts)
                    _light_tokens, _light_counts = _extreme_runtime._light_cpu_history
                    _light_effective = hashlib.sha256()
                    for _light_i in range(_cohort_output.cycles):
                        for _light_slot in range(12):
                            _light_count = int(_light_counts[_light_i, _light_slot])
                            if not 0 <= _light_count <= 8:
                                raise RuntimeError("light acceptance count outside contract")
                            _light_effective.update(bytes((_light_count,)))
                            for _light_token in _light_tokens[
                                _light_i, _light_slot, :_light_count
                            ].tolist():
                                _light_effective.update(int(_light_token).to_bytes(
                                    4, "little", signed=True))
                    _light_identity = {
                        "rank": _light_rank, "cohort": _light_cohort,
                        "req_ids": list(_extreme_req_ids),
                        "run_ts": os.getenv("RUN_TS"),
                        "time_namespace": os.readlink("/proc/self/ns/time"),
                        "cycles": _cohort_output.cycles,
                        "runtime_scoped_wall_s": _extreme_wall_seconds,
                        "target_graph_requested": _extreme_target_graph,
                        "target_graph_mode": str(_target_inputs.aclgraph_runtime_mode),
                        "accepted_counts": _light_counts.tolist(),
                        "staged_output_counts": _cohort_output.staged_output_counts,
                        "overshoot_tokens": _cohort_output.overshoot_tokens,
                        "canonical_effective_staged_trajectory_sha256":
                            _light_effective.hexdigest(),
                        "count_history_sha256": hashlib.sha256(
                            _light_counts.contiguous().numpy().tobytes()).hexdigest(),
                        "raw_padded_token_history_sha256": hashlib.sha256(
                            _light_tokens.contiguous().numpy().tobytes()).hexdigest(),
                        "generated_output_counts": _cohort_output.generated_output_counts,
                        "runtime_bulk_output_sha256": hashlib.sha256(json.dumps(
                            _cohort_output.token_ids, separators=(",", ":")
                        ).encode()).hexdigest(),
                    }
                    with open(os.path.join(_light_ledger_dir,
                                           f"rank{_light_rank}_cohort{_light_cohort}.json"),
                              "w", encoding="utf-8") as _light_file:
                        json.dump(_light_identity, _light_file, separators=(",", ":"))
                    if _extreme_runtime._light_probe is not None:
                        _light_dir = os.environ["EXTREME_LIGHT_EVENT_DIR"]
                        os.makedirs(_light_dir, exist_ok=True)
                        _light_packet = _extreme_runtime._light_probe.export_after_existing_drain(
                            _light_identity)
                        with open(os.path.join(_light_dir,
                                               f"rank{_light_rank}_cohort{_light_cohort}.json"),
                                  "w", encoding="utf-8") as _light_file:
                            json.dump(_light_packet, _light_file, separators=(",", ":"))
                _host_mirror_exact =''')
    return src

PATCH = {'cycle': cycle, 'serving': serving, 'runner': runner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT / 'source_candidate.json')
    args = parser.parse_args()
    helper = ROOT / 'runtime/light_stage_observer.py'
    ast.parse(helper.read_text())
    manifest = {'status': 'source_only_NOT_INSTALLED_NOT_LIVE_READY',
                'helper_sha256': sha(helper.read_bytes()), 'sources': {}}
    diff_dir = OUT / 'candidate_diff'
    diff_dir.mkdir(parents=True, exist_ok=True)
    for key, (path, expected) in SOURCES.items():
        raw = path.read_bytes()
        if sha(raw) != expected:
            raise ValueError(f'live source drift: {key}')
        new = PATCH[key](raw.decode())
        ast.parse(new)
        for api in ('.synchronize(', '.wait_stream(', '.wait_event('):
            if new.count(api) != raw.decode().count(api):
                raise ValueError(f'added wait API: {key} {api}')
        delta = ''.join(difflib.unified_diff(raw.decode().splitlines(True),
                                             new.splitlines(True),
                                             fromfile=str(path),
                                             tofile=str(path) + '.candidate'))
        (diff_dir / f'{key}.diff').write_text(delta)
        manifest['sources'][key] = {'path': str(path), 'live_sha256': expected,
                                    'candidate_sha256': sha(new.encode()),
                                    'diff_sha256': sha(delta.encode())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'sources': list(SOURCES)}))


if __name__ == '__main__':
    main()
