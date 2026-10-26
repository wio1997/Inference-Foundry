#!/usr/bin/env python3
"""Reversible EngineCore async-queue fence for the original bulk serving A path."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py')
BACKUP = Path('/tmp/loop074_run343_core.py.orig')
BASE_SHA = '3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad'
MARKER = 'EXTREME_LOOP074_RUN343_FENCE'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def replace_once(s, old, new):
    if s.count(old) != 1:
        raise RuntimeError(f'expected one anchor, got {s.count(old)}: {old[:80]}')
    return s.replace(old, new, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('preview', 'install', 'restore'))
    ap.add_argument('--record', type=Path, required=True)
    a = ap.parse_args()
    if a.action in ('preview', 'install'):
        original = SOURCE.read_bytes()
        if sha(original) != BASE_SHA or BACKUP.exists() or MARKER.encode() in original:
            raise RuntimeError('unexpected core SHA or leftover backup/patch')
        s = original.decode()
        s = replace_once(s,
            '        model_executed = False\n        deferred_scheduler_output = None\n        if self.scheduler.has_requests():',
            '''        # EXTREME_LOOP074_RUN343_FENCE: protect a possible c12 handoff
        # before any successor schedule() mutates state or submits worker RPC.
        _run343_enabled = os.getenv("EXTREME_RUN343_FENCE") == "1"
        _run343_pending = getattr(self, "_run343_pending", None)
        model_executed = False
        deferred_scheduler_output = None
        if (_run343_pending is None or not _run343_enabled) and self.scheduler.has_requests():''')
        s = replace_once(s,
            '            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())\n            with self.log_error_detail(scheduler_output):',
            '''            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            if _run343_enabled and scheduler_output.total_num_scheduled_tokens == 96:
                self._run343_pending = id(scheduler_output)
                self._run343_fence_seq = getattr(self, "_run343_fence_seq", 0) + 1
                print(f"EXTREME_RUN343_FENCE acquire seq={self._run343_fence_seq} "
                      f"queued_before_submit={len(batch_queue)} "
                      f"scheduled={scheduler_output.total_num_scheduled_tokens}",
                      flush=True)
            with self.log_error_detail(scheduler_output):''')
        s = replace_once(s,
            '        self._attach_iteration_details(engine_core_outputs, iteration_details)\n\n        # NOTE(nick): We can either handle the deferred tasks here or save',
            '''        self._attach_iteration_details(engine_core_outputs, iteration_details)
        if _run343_enabled and getattr(self, "_run343_pending", None) == id(scheduler_output):
            print(f"EXTREME_RUN343_FENCE release seq={self._run343_fence_seq} "
                  f"queued_after_consume={len(batch_queue)} "
                  f"bulk={bool(model_output.extreme_bulk_output)}", flush=True)
            self._run343_pending = None

        # NOTE(nick): We can either handle the deferred tasks here or save''')
        compile(s, str(SOURCE), 'exec')
        result = {'action': a.action, 'original_sha256': sha(original),
                  'patched_sha256': sha(s.encode()),
                  'scope': 'EngineCore step_with_batch_queue only; async flag retained; candidate total96 is conservative'}
        if a.action == 'install':
            BACKUP.write_bytes(original)
            SOURCE.write_text(s)
    else:
        if not BACKUP.exists() or MARKER not in SOURCE.read_text():
            raise RuntimeError('missing backup or marker')
        patched = SOURCE.read_bytes()
        original = BACKUP.read_bytes()
        if sha(original) != BASE_SHA:
            raise RuntimeError('backup SHA mismatch')
        SOURCE.write_bytes(original)
        BACKUP.unlink()
        result = {'action': 'restore', 'patched_sha256': sha(patched),
                  'restored_sha256': sha(SOURCE.read_bytes())}
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
