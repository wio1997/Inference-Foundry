#!/usr/bin/env python3
"""Temporary diagnostic wrapper for NPUModelRunner execute/sample wall time."""
import argparse, hashlib, json
from pathlib import Path
MODEL=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP=Path('/tmp/extreme_loop045_run160_model_runner_v1.py.boundary')
MARKER='# EXTREME_LOOP045_RUN160_EXEC_TIMING'
APPEND='''\n\n# EXTREME_LOOP045_RUN160_EXEC_TIMING\nif os.getenv("EXTREME_EXEC_TIMING_DIR"):\n    _extreme_original_execute_model = NPUModelRunner.execute_model\n    _extreme_original_sample_tokens = NPUModelRunner.sample_tokens\n    def _extreme_log_timing(self, label, start_ns, end_ns, scheduled):\n        try:\n            _rank = int(get_tp_group().rank_in_group)\n        except Exception:\n            _rank = -1\n        _path = os.path.join(os.environ["EXTREME_EXEC_TIMING_DIR"], f"rank{_rank}_pid{os.getpid()}.jsonl")\n        os.makedirs(os.path.dirname(_path), exist_ok=True)\n        with open(_path, "a", encoding="utf-8") as _out:\n            _out.write(json.dumps({"rank":_rank,"pid":os.getpid(),"method":label,"start_ns":start_ns,"end_ns":end_ns,"scheduled_tokens":scheduled,"stage_marks":getattr(self,"_extreme_stage_marks",[])},separators=(",",":"))+"\\n")\n    def _extreme_timed_execute_model(self, scheduler_output, *args, **kwargs):\n        _start = time.perf_counter_ns()\n        try:\n            return _extreme_original_execute_model(self, scheduler_output, *args, **kwargs)\n        finally:\n            _end = time.perf_counter_ns()\n            self._extreme_log_timing("execute_model", _start, _end, int(scheduler_output.total_num_scheduled_tokens))\n    def _extreme_timed_sample_tokens(self, *args, **kwargs):\n        _start = time.perf_counter_ns()\n        try:\n            return _extreme_original_sample_tokens(self, *args, **kwargs)\n        finally:\n            _end = time.perf_counter_ns()\n            self._extreme_log_timing("sample_tokens", _start, _end, 0)\n    NPUModelRunner._extreme_log_timing = _extreme_log_timing\n    NPUModelRunner.execute_model = _extreme_timed_execute_model\n    NPUModelRunner.sample_tokens = _extreme_timed_sample_tokens\n'''
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 if a.action=='install':
  assert not BACKUP.exists() and MARKER not in MODEL.read_text()
  original=MODEL.read_bytes();BACKUP.write_bytes(original);MODEL.write_bytes(original+APPEND.encode())
  result={'action':'install','original_sha256':sha(original),'patched_sha256':sha(MODEL.read_bytes())}
 else:
  assert BACKUP.exists() and MARKER in MODEL.read_text()
  original=BACKUP.read_bytes();MODEL.write_bytes(original);BACKUP.unlink()
  result={'action':'restore','original_sha256':sha(original),'restored_sha256':sha(MODEL.read_bytes())}
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
