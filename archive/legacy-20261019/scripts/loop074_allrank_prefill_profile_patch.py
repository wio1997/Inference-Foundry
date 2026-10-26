#!/usr/bin/env python3
"""Reversible first warmed residual-prefill profiler on all eight TP ranks."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP = Path('/tmp/loop074_run333_model_runner_v1.py.orig')
MARKER = '# EXTREME_LOOP074_RUN333_PREFILL_ALLRANK'
APPEND = '''

# EXTREME_LOOP074_RUN333_PREFILL_ALLRANK
if os.getenv("EXTREME_RUN333_PROFILE_DIR"):
    _run333_original_forward = NPUModelRunner._model_forward
    def _run333_profile_one_prefill(self, num_tokens_padded, *args, **kwargs):
        _root = os.environ["EXTREME_RUN333_PROFILE_DIR"]
        if (len(getattr(self, "_extreme_served_cohorts", ())) == 4
                and not getattr(self, "_run333_profiled", False)
                and os.path.exists(os.path.join(_root, "enable"))):
            self._run333_profiled = True
            import torch_npu
            _rank = int(get_tp_group().rank_in_group)
            _dest = os.path.join(_root, f"rank{_rank}")
            os.makedirs(_dest, exist_ok=True)
            _handler = torch_npu.profiler.tensorboard_trace_handler(
                _dest, worker_name=f"rank{_rank}", analyse_flag=True, async_mode=False)
            _start_ns = time.perf_counter_ns()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
                on_trace_ready=_handler, record_shapes=False, profile_memory=False,
                with_stack=False, with_modules=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                    aic_metrics=None, data_simplification=True)) as _prof:
                _result = _run333_original_forward(self, num_tokens_padded, *args, **kwargs)
                _forward_return_ns = time.perf_counter_ns()
                torch.npu.synchronize()
                _synchronized_ns = time.perf_counter_ns()
                _prof.step()
            _row = {"rank": _rank, "served_cohorts": 4,
                    "num_tokens_padded": int(num_tokens_padded),
                    "start_ns": _start_ns,
                    "forward_return_ns": _forward_return_ns,
                    "synchronized_ns": _synchronized_ns,
                    "profiler_finished_ns": time.perf_counter_ns(),
                    "profile_dir": _dest}
            with open(os.path.join(_root, f"rank{_rank}_capture.json"), "w") as _out:
                json.dump(_row, _out, indent=2)
            return _result
        return _run333_original_forward(self, num_tokens_padded, *args, **kwargs)
    NPUModelRunner._model_forward = _run333_profile_one_prefill
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
