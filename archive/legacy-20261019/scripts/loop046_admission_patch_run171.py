#!/usr/bin/env python3
"""Temporary opt-in Scheduler admission trace for Run170; restore is mandatory."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/core/sched/scheduler.py')
BACKUP = Path('/tmp/loop046_run171_scheduler.py.orig')
MARKER = 'EXTREME_RUN171_ADMISSION_TRACE'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=['install', 'restore'])
    a.add_argument('--record', required=True)
    args = a.parse_args()
    before = SOURCE.read_bytes()
    if args.action == 'install':
        assert MARKER.encode() not in before
        assert not BACKUP.exists(), f'backup already exists: {BACKUP}'
        BACKUP.write_bytes(before)
        s = before.decode()
        s = s.replace('import itertools\n', 'import itertools\nimport json  # EXTREME_RUN171_ADMISSION_TRACE\n', 1)
        needle = '        # For logging.\n        scheduled_timestamp = time.monotonic()\n'
        insert = '''        # For logging.
        scheduled_timestamp = time.monotonic()
        _extreme_trace = os.getenv("EXTREME_RUN171_ADMISSION_TRACE")
        _extreme_waiting_before = len(self.waiting) + len(self.skipped_waiting) if _extreme_trace else 0
        _extreme_running_before = len(self.running) if _extreme_trace else 0
'''
        assert s.count(needle) == 1
        s = s.replace(needle, insert)
        needle = '        return scheduler_output\n\n    def _build_kv_connector_meta('
        insert = '''        if _extreme_trace:
            _row = {"kind": "schedule", "t": scheduled_timestamp,
                    "waiting_before": _extreme_waiting_before, "running_before": _extreme_running_before,
                    "waiting_after": len(self.waiting) + len(self.skipped_waiting),
                    "running_after": len(self.running), "new": len(scheduled_new_reqs),
                    "tokens": total_num_scheduled_tokens,
                    "by_req": num_scheduled_tokens,
                    "max_batched_tokens": self.max_num_scheduled_tokens,
                    "max_running_reqs": self.max_num_running_reqs}
            with open(_extreme_trace, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_row, separators=(",", ":")) + "\\n")
        return scheduler_output

    def _build_kv_connector_meta('''
        assert s.count(needle)==1
        s=s.replace(needle,insert)
        needle = '    def add_request(self, request: Request) -> None:\n        existing = self.requests.get(request.request_id)\n'
        insert = '''    def add_request(self, request: Request) -> None:
        _extreme_trace = os.getenv("EXTREME_RUN171_ADMISSION_TRACE")
        if _extreme_trace:
            _row = {"kind": "add", "t": time.monotonic(), "req": request.request_id,
                    "prompt_tokens": request.num_prompt_tokens,
                    "waiting_before": len(self.waiting) + len(self.skipped_waiting),
                    "running_before": len(self.running)}
            with open(_extreme_trace, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_row, separators=(",", ":")) + "\\n")
        existing = self.requests.get(request.request_id)
'''
        assert s.count(needle)==1
        s=s.replace(needle,insert)
        compile(s, str(SOURCE), 'exec')
        SOURCE.write_text(s)
        out={'action':'install','original_sha256':sha(before),'patched_sha256':sha(SOURCE.read_bytes()),'backup':str(BACKUP)}
    else:
        assert BACKUP.exists(), 'missing backup'
        original=BACKUP.read_bytes()
        assert MARKER.encode() in before, 'source no longer has marker'
        SOURCE.write_bytes(original)
        BACKUP.unlink()
        out={'action':'restore','patched_sha256':sha(before),'restored_sha256':sha(SOURCE.read_bytes())}
    Path(args.record).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':
    main()
