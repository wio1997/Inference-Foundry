#!/usr/bin/env python3
"""Temporarily add common-clock service boundary marks and park history."""
import argparse,hashlib,json
from pathlib import Path
MODEL=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
SERVE=Path('/data/wio/Inference_Foundry/runtime/fixed_serving.py')
BACKUPS=(Path('/tmp/extreme_loop045_model_runner_v1.py.orig'),Path('/tmp/extreme_loop045_fixed_serving.py.orig'))
MARKER='EXTREME_LOOP045_BOUNDARY'
def sha(x):return hashlib.sha256(x).hexdigest()
def once(s,old,new):
    if s.count(old)!=1:raise RuntimeError(f'expected one anchor, found {s.count(old)}: {old[:80]}')
    return s.replace(old,new,1)
def install():
    for path,backup in zip((MODEL,SERVE),BACKUPS):
        if backup.exists() or MARKER in path.read_text():raise RuntimeError('existing patch/backup')
    model=MODEL.read_text()
    model=once(model,
       '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if self.vllm_config.model_config.enable_return_routed_experts:',
       '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        # EXTREME_LOOP045_BOUNDARY\n        _extreme_boundary_dir = os.getenv("EXTREME_BOUNDARY_DIR")\n        if _extreme_boundary_dir:\n            _calls = getattr(self, "_extreme_boundary_calls", [])\n            _calls.append({"t_ns": time.perf_counter_ns(), "scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens)})\n            self._extreme_boundary_calls = _calls\n        if self.vllm_config.model_config.enable_return_routed_experts:')
    model=once(model,
       '                self._extreme_served_cohorts.add(_extreme_req_ids)\n            _runtime_root = "/data/wio/Inference_Foundry"',
       '                self._extreme_served_cohorts.add(_extreme_req_ids)\n            _boundary_handoff_ns = time.perf_counter_ns() if _extreme_boundary_dir else 0\n            _runtime_root = "/data/wio/Inference_Foundry"')
    model=once(model,
       '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )',
       '            _boundary_build_start_ns = time.perf_counter_ns() if _extreme_boundary_dir else 0\n            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n            _boundary_build_end_ns = time.perf_counter_ns() if _extreme_boundary_dir else 0')
    model=once(model,
       '                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(',
       '                _boundary_serve_start_ns = time.perf_counter_ns() if _extreme_boundary_dir else 0\n                _extreme_wall_start = time.perf_counter()\n                _cohort_output = FixedCohortServing(')
    model=once(model,
       '                _extreme_wall_seconds = (\n                    time.perf_counter() - _extreme_wall_start\n                )',
       '                _extreme_wall_seconds = (\n                    time.perf_counter() - _extreme_wall_start\n                )\n                _boundary_serve_end_ns = time.perf_counter_ns() if _extreme_boundary_dir else 0')
    model=once(model,
       '                    extreme_bulk_output=True,\n                )\n                return None',
       '                    extreme_bulk_output=True,\n                )\n                if _extreme_boundary_dir:\n                    _boundary_publication_ns = time.perf_counter_ns()\n                    os.makedirs(_extreme_boundary_dir, exist_ok=True)\n                    with open(os.path.join(_extreme_boundary_dir, f"rank{_rank}_cohort{_cohort_index}.json"), "w", encoding="utf-8") as _boundary_file:\n                        json.dump({"rank": _rank, "cohort": _cohort_index, "calls": self._extreme_boundary_calls, "handoff_ns": _boundary_handoff_ns, "build_start_ns": _boundary_build_start_ns, "build_end_ns": _boundary_build_end_ns, "serve_start_ns": _boundary_serve_start_ns, "serve_end_ns": _boundary_serve_end_ns, "publication_ns": _boundary_publication_ns, "park_log": getattr(_extreme_runtime, "_extreme_park_log", []), "counts_cpu": getattr(_extreme_runtime, "_extreme_counts_cpu", torch.empty(0)).tolist()}, _boundary_file, separators=(",", ":"))\n                    self._extreme_boundary_calls = []\n                return None')
    serve=SERVE.read_text()
    serve=once(serve,
       '        self._parked = [False] * self.config.batch_size',
       '        # EXTREME_LOOP045_BOUNDARY\n        self._parked = [False] * self.config.batch_size\n        self._extreme_park_log = []')
    serve=once(serve,
       '        if not slots:\n            return\n        # Fixed shape requires parked slots to keep executing.',
       '        if not slots:\n            return\n        self._extreme_park_log.append({"cycle": self._extreme_cycle_index, "slots": list(slots)})\n        # Fixed shape requires parked slots to keep executing.')
    serve=once(serve,
       '            progress = self._committed_progress()\n            self._park_completed(progress)',
       '            progress = self._committed_progress()\n            self._extreme_cycle_index = index\n            self._park_completed(progress)')
    serve=once(serve,
       '        counts_cpu = count_history[:cycles].cpu()',
       '        counts_cpu = count_history[:cycles].cpu()\n        self.runtime._extreme_counts_cpu = counts_cpu\n        self.runtime._extreme_park_log = self._extreme_park_log')
    record=[]
    for path,backup,patched in ((MODEL,BACKUPS[0],model),(SERVE,BACKUPS[1],serve)):
        original=path.read_bytes();backup.write_bytes(original);path.write_text(patched)
        record.append({'path':str(path),'original_sha256':sha(original),'patched_sha256':sha(path.read_bytes())})
    return record
def restore():
    record=[]
    for path,backup in zip((MODEL,SERVE),BACKUPS):
        if not backup.exists():raise RuntimeError(f'missing backup {backup}')
        original=backup.read_bytes();path.write_bytes(original);backup.unlink()
        record.append({'path':str(path),'restored_sha256':sha(path.read_bytes()),'original_sha256':sha(original)})
    return record
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('install','restore'));p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();r=install() if a.action=='install' else restore()
    a.record.parent.mkdir(parents=True,exist_ok=True);a.record.write_text(json.dumps({'action':a.action,'files':r},indent=2)+'\n')
    print(json.dumps({'action':a.action,'files':r}))
if __name__=='__main__':main()
