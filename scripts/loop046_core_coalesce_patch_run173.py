#!/usr/bin/env python3
"""Run173 guarded first-cohort input drain; borrowed Core source restored afterward."""
import argparse, hashlib, json
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py')
BACKUP=Path('/tmp/loop046_run173_core.py.orig')
MARKER='EXTREME_LOOP046_RUN173_COALESCE'
def sha(x): return hashlib.sha256(x).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 before=SOURCE.read_bytes()
 if a.action=='install':
  assert MARKER.encode() not in before and not BACKUP.exists()
  BACKUP.write_bytes(before)
  s=before.decode().replace('import gc\n','import gc\nimport json  # EXTREME_LOOP046_RUN173_COALESCE\n',1)
  needle='''            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
'''
  insert='''            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # EXTREME_LOOP046_RUN173_COALESCE: bounded frozen c12 cohort admission.
            _coalesce_trace = os.getenv("EXTREME_RUN173_COALESCE_TRACE")
            if (_coalesce_trace and not self.scheduler.running
                    and 0 < sum(self.scheduler.get_request_counts()) < 12):
                _start = time.monotonic()
                _before = self.scheduler.get_request_counts()
                while (time.monotonic() - _start < 0.500
                       and sum(self.scheduler.get_request_counts()) < 12
                       and self.is_running()):
                    time.sleep(0.005)
                    self._process_input_queue()
                _row = {"t": _start, "wait_s": time.monotonic() - _start,
                        "counts_before": _before,
                        "counts_after": self.scheduler.get_request_counts()}
                with open(_coalesce_trace, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(_row, separators=(",", ":")) + "\\n")
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
'''
  assert s.count(needle)==1
  s=s.replace(needle,insert)
  compile(s,str(SOURCE),'exec');SOURCE.write_text(s)
  result={'action':'install','original_sha256':sha(before),'patched_sha256':sha(SOURCE.read_bytes()),'backup':str(BACKUP)}
 else:
  assert BACKUP.exists() and MARKER.encode() in before
  original=BACKUP.read_bytes();SOURCE.write_bytes(original);BACKUP.unlink()
  result={'action':'restore','patched_sha256':sha(before),'restored_sha256':sha(SOURCE.read_bytes())}
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
