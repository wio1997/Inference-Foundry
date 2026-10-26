#!/usr/bin/env python3
"""Borrowed model-runner target scope event probe for Run179."""
import argparse,hashlib,json
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP=Path('/tmp/loop047_run179_model_runner.py.orig')
MARKER='# EXTREME_LOOP047_RUN179_TARGET_SLOPE'
APPEND=r'''

# EXTREME_LOOP047_RUN179_TARGET_SLOPE
if os.getenv("EXTREME_RUN179_SLOPE_DIR"):
    _extreme_run179_orig_forward = NPUModelRunner._model_forward
    def _extreme_run179_forward(self, num_tokens_padded, *args, **kwargs):
        _arm = os.getenv("EXTREME_RUN179_ARM_FILE")
        try:
            with open(_arm, "r", encoding="utf-8") as _f:
                _arm_value = _f.read().strip()
            _size_text, _epoch = _arm_value.split(":", 1)
            _size = int(_size_text)
        except (OSError, ValueError, TypeError):
            return _extreme_run179_orig_forward(self, num_tokens_padded, *args, **kwargs)
        if int(num_tokens_padded) != _size:
            return _extreme_run179_orig_forward(self, num_tokens_padded, *args, **kwargs)
        _counts = getattr(self, "_extreme_run179_counts", None)
        if _counts is None:
            _counts = self._extreme_run179_counts = {}
        _key = (_size, _epoch)
        _index = _counts.get(_key, 0)
        _counts[_key] = _index + 1
        if not 64 <= _index < 88:
            return _extreme_run179_orig_forward(self, num_tokens_padded, *args, **kwargs)
        _context = get_forward_context()
        _mode = str(getattr(_context, "cudagraph_runtime_mode", "unknown"))
        _actual = int(getattr(_context, "num_actual_tokens", num_tokens_padded) or num_tokens_padded)
        _reqs = int(getattr(self.input_batch, "num_reqs", -1))
        _lens = getattr(self.input_batch, "num_computed_tokens_cpu", None)
        _min_len = int(min(_lens[:_reqs])) if _lens is not None and _reqs > 0 else None
        _max_len = int(max(_lens[:_reqs])) if _lens is not None and _reqs > 0 else None
        torch.npu.synchronize()
        _begin = torch.npu.Event(enable_timing=True)
        _end = torch.npu.Event(enable_timing=True)
        _begin.record()
        _host_start = time.perf_counter_ns()
        _output = _extreme_run179_orig_forward(self, num_tokens_padded, *args, **kwargs)
        _end.record()
        _end.synchronize()
        _host_end = time.perf_counter_ns()
        torch.npu.synchronize()
        _rank = int(get_tp_group().rank_in_group)
        _row = {"rank":_rank,"size":_size,"epoch":_epoch,"call_index":_index,
                "mode":_mode,"num_actual_tokens":_actual,"num_reqs":_reqs,
                "min_computed_tokens":_min_len,"max_computed_tokens":_max_len,
                "event_ms":_begin.elapsed_time(_end),
                "host_wall_ms":(_host_end-_host_start)/1e6}
        _path = os.path.join(os.environ["EXTREME_RUN179_SLOPE_DIR"], f"rank{_rank}_size{_size}_{_epoch}.jsonl")
        os.makedirs(os.path.dirname(_path),exist_ok=True)
        with open(_path,"a",encoding="utf-8") as _f:
            _f.write(json.dumps(_row,separators=(",",":"))+"\n")
        return _output
    NPUModelRunner._model_forward = _extreme_run179_forward
'''
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 before=SOURCE.read_bytes()
 if a.action=='install':
  assert not BACKUP.exists() and MARKER.encode() not in before
  BACKUP.write_bytes(before)
  s=before+APPEND.encode();compile(s,str(SOURCE),'exec');SOURCE.write_bytes(s)
  result={'action':'install','original_sha256':sha(before),'patched_sha256':sha(s),'backup':str(BACKUP)}
 else:
  assert BACKUP.exists() and MARKER.encode() in before
  original=BACKUP.read_bytes();SOURCE.write_bytes(original);BACKUP.unlink()
  result={'action':'restore','patched_sha256':sha(before),'restored_sha256':sha(SOURCE.read_bytes())}
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
