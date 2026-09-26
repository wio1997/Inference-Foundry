#!/usr/bin/env python3
"""Reversible no-profiler Host collective submission marks for one warm forward."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP = Path('/tmp/loop074_run337_model_runner_v1.py.orig')
MARKER = '# EXTREME_LOOP074_RUN337_HOST_COLLECTIVE'
APPEND = '''

# EXTREME_LOOP074_RUN337_HOST_COLLECTIVE
if os.getenv("EXTREME_RUN337_DIR"):
    _run337_original_forward = NPUModelRunner._model_forward
    def _run337_one_forward(self, num_tokens_padded, *args, **kwargs):
        _root = os.environ["EXTREME_RUN337_DIR"]
        if (len(getattr(self, "_extreme_served_cohorts", ())) == 4
                and not getattr(self, "_run337_traced", False)
                and os.path.exists(os.path.join(_root, "enable"))):
            self._run337_traced = True
            import torch.distributed as _dist
            from vllm.distributed.parallel_state import GroupCoordinator as _Group
            _rank = int(get_tp_group().rank_in_group)
            _events = []
            _orig_ag = _Group._all_gather_out_place
            _orig_rs = _Group._reduce_scatter_out_place
            _orig_a2a = _dist.all_to_all_single
            def _wrap(kind, fn):
                def _call(*call_args, **call_kwargs):
                    _input = call_args[1]
                    _input_bytes = int(_input.numel() * _input.element_size())
                    _dtype = str(_input.dtype)
                    _start = time.perf_counter_ns()
                    _cpu_start = time.thread_time_ns()
                    try:
                        return fn(*call_args, **call_kwargs)
                    finally:
                        _events.append((kind, _start, time.perf_counter_ns(),
                                        _cpu_start, time.thread_time_ns(),
                                        _input_bytes, _dtype))
                return _call
            _Group._all_gather_out_place = _wrap("all_gather", _orig_ag)
            _Group._reduce_scatter_out_place = _wrap("reduce_scatter", _orig_rs)
            _dist.all_to_all_single = _wrap("all_to_all", _orig_a2a)
            _start = time.perf_counter_ns()
            _cpu_start = time.thread_time_ns()
            try:
                _result = _run337_original_forward(self, num_tokens_padded, *args, **kwargs)
            finally:
                _end = time.perf_counter_ns()
                _cpu_end = time.thread_time_ns()
                _Group._all_gather_out_place = _orig_ag
                _Group._reduce_scatter_out_place = _orig_rs
                _dist.all_to_all_single = _orig_a2a
            _row = {"rank": _rank, "served_cohorts": 4,
                    "num_tokens_padded": int(num_tokens_padded),
                    "forward_start_ns": _start, "forward_end_ns": _end,
                    "forward_thread_cpu_ns": _cpu_end - _cpu_start,
                    "collectives": _events}
            with open(os.path.join(_root, f"rank{_rank}.json"), "w") as _out:
                json.dump(_row, _out, separators=(",", ":"))
            return _result
        return _run337_original_forward(self, num_tokens_padded, *args, **kwargs)
    NPUModelRunner._model_forward = _run337_one_forward
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('install', 'restore'))
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'install':
        original = SOURCE.read_bytes()
        if BACKUP.exists() or MARKER.encode() in original:
            raise RuntimeError('patch/backup exists')
        after = original + APPEND.encode()
        compile(after, str(SOURCE), 'exec')
        BACKUP.write_bytes(original)
        SOURCE.write_bytes(after)
        result = {'action': 'install', 'original_sha256': sha(original),
                  'patched_sha256': sha(after)}
    else:
        if not BACKUP.exists() or MARKER not in SOURCE.read_text():
            raise RuntimeError('missing backup or marker')
        before = SOURCE.read_bytes()
        original = BACKUP.read_bytes()
        SOURCE.write_bytes(original)
        BACKUP.unlink()
        result = {'action': 'restore', 'patched_sha256': sha(before),
                  'restored_sha256': sha(SOURCE.read_bytes())}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
