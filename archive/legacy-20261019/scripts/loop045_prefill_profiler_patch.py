#!/usr/bin/env python3
"""Profile one rank0 measured prefill forward in borrowed model runner."""
import argparse,hashlib,json
from pathlib import Path
MODEL=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP=Path('/tmp/extreme_loop045_run162_model_runner_v1.py.boundary')
MARKER='# EXTREME_LOOP045_RUN162_PREFILL_PROFILE'
APPEND='''\n\n# EXTREME_LOOP045_RUN162_PREFILL_PROFILE\nif os.getenv("EXTREME_PREFILL_PROFILE_DIR"):\n    _extreme_original_model_forward = NPUModelRunner._model_forward\n    def _extreme_profile_one_prefill(self, *args, **kwargs):\n        _root = os.environ["EXTREME_PREFILL_PROFILE_DIR"]\n        if (len(getattr(self, "_extreme_served_cohorts", ())) == 4\n                and not getattr(self, "_extreme_prefill_profiled", False)\n                and os.path.exists(os.path.join(_root, "enable"))\n                and int(get_tp_group().rank_in_group) == 0):\n            self._extreme_prefill_profiled = True\n            import torch_npu\n            _dest = os.path.join(_root, "rank0")\n            os.makedirs(_dest, exist_ok=True)\n            _handler = torch_npu.profiler.tensorboard_trace_handler(\n                _dest, worker_name="rank0", analyse_flag=True, async_mode=False)\n            with torch_npu.profiler.profile(\n                activities=[torch_npu.profiler.ProfilerActivity.CPU,\n                            torch_npu.profiler.ProfilerActivity.NPU],\n                schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),\n                on_trace_ready=_handler, record_shapes=False, profile_memory=False,\n                with_stack=False, with_modules=False,\n                experimental_config=torch_npu.profiler._ExperimentalConfig(\n                    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,\n                    aic_metrics=None, data_simplification=True)) as _prof:\n                _result = _extreme_original_model_forward(self, *args, **kwargs)\n                torch.npu.synchronize()\n                _prof.step()\n            with open(os.path.join(_root, "capture.json"), "w", encoding="utf-8") as _out:\n                json.dump({"rank":0,"served_cohorts":4,"profile_dir":_dest},_out)\n            return _result\n        return _extreme_original_model_forward(self, *args, **kwargs)\n    NPUModelRunner._model_forward = _extreme_profile_one_prefill\n'''
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
