#!/usr/bin/env python3
"""Reversible Host and current-stream marks for natural warmed refill calls."""
import argparse
import hashlib
import json
from pathlib import Path

from loop074_prefill_ready_patch import ENTRY, READY, SOURCE, replace_once

BACKUP = Path('/tmp/loop074_run341_model_runner_v1.py.orig')
BASE_SHA = '004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'
MARKER = 'EXTREME_LOOP074_RUN341'
ENTRY341 = ENTRY.replace('RUN332', 'RUN341').replace('run332', 'run341')
READY341 = READY.replace('RUN332', 'RUN341').replace('run332', 'run341')

APPEND = '''
# EXTREME_LOOP074_RUN341_FORWARD
if os.getenv("EXTREME_RUN341_DIR"):
    _run341_original_forward = NPUModelRunner._model_forward
    def _run341_forward(self, num_tokens_padded, *args, **kwargs):
        _dir = os.environ["EXTREME_RUN341_DIR"]
        if (len(getattr(self, "_extreme_served_cohorts", ())) < 4
                or not os.path.exists(os.path.join(_dir, "arm"))):
            return _run341_original_forward(self, num_tokens_padded, *args, **kwargs)
        _start = torch.npu.Event(enable_timing=True)
        _end = torch.npu.Event(enable_timing=True)
        _host_start = time.perf_counter_ns()
        _cpu_start = time.thread_time_ns()
        _start.record()
        try:
            return _run341_original_forward(self, num_tokens_padded, *args, **kwargs)
        finally:
            _end.record()
            _row = {"num_tokens_padded": int(num_tokens_padded),
                    "host_start_ns": _host_start,
                    "host_end_ns": time.perf_counter_ns(),
                    "thread_cpu_ns": time.thread_time_ns() - _cpu_start,
                    "execute_mark_count": len(getattr(self, "_run341_rows", ())) }
            self._run341_calls = getattr(self, "_run341_calls", [])
            self._run341_calls.append((_row, _start, _end))
    NPUModelRunner._model_forward = _run341_forward
'''

FLUSH = '''                # EXTREME_LOOP074_RUN341_FLUSH
                if _run341_dir and os.path.exists(os.path.join(_run341_dir, "arm")):
                    _run341_out = []
                    for _call, _start, _end in getattr(self, "_run341_calls", []):
                        _call["current_stream_elapsed_ms"] = float(_start.elapsed_time(_end))
                        _run341_out.append(_call)
                    _run341_file = os.path.join(
                        _run341_dir, f"rank{_rank}_cohort{_cohort_index}_calls.json")
                    with open(_run341_file, "w") as _f:
                        json.dump({"rank": _rank, "cohort": _cohort_index,
                                   "calls": _run341_out,
                                   "scope": "Host wall/CPU and current-stream events around original _model_forward; last serving synchronize completes events; side-stream work may extend beyond end event"}, _f)
                    self._run341_calls = []
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('preview', 'install', 'restore'))
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    if a.action in ('preview', 'install'):
        original = SOURCE.read_bytes()
        if sha(original) != BASE_SHA or BACKUP.exists() or MARKER.encode() in original:
            raise RuntimeError(f'unexpected source sha {sha(original)} or leftover backup')
        source = original.decode()
        source = replace_once(source,
            '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if self.vllm_config.model_config.enable_return_routed_experts:',
            '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n' + ENTRY341 +
            '        if self.vllm_config.model_config.enable_return_routed_experts:')
        source = replace_once(source,
            '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )',
            '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n' + READY341.rstrip('\n'))
        source = replace_once(source,
            '                return None\n            _initial_num_computed = (',
            FLUSH + '                return None\n            _initial_num_computed = (')
        source += APPEND
        compile(source, str(SOURCE), 'exec')
        result = {'action': a.action, 'original_sha256': sha(original),
                  'patched_sha256': sha(source.encode())}
        if a.action == 'install':
            BACKUP.write_bytes(original)
            SOURCE.write_text(source)
    else:
        if not BACKUP.exists() or MARKER not in SOURCE.read_text():
            raise RuntimeError('missing backup or marker')
        before = SOURCE.read_bytes()
        original = BACKUP.read_bytes()
        if sha(original) != BASE_SHA:
            raise RuntimeError('backup SHA mismatch')
        SOURCE.write_bytes(original)
        BACKUP.unlink()
        result = {'action': 'restore', 'patched_sha256': sha(before),
                  'restored_sha256': sha(SOURCE.read_bytes())}
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
