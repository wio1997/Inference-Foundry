#!/usr/bin/env python3
"""Reversible passive worker marks for original warmed request preparation."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP = Path('/tmp/loop074_run332_model_runner_v1.py.orig')
MARKER = 'EXTREME_LOOP074_RUN332'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'anchor count {source.count(old)} for {old[:60]}')
    return source.replace(old, new, 1)


ENTRY = '''        # EXTREME_LOOP074_RUN332
        _run332_dir = os.getenv("EXTREME_RUN332_DIR")
        if _run332_dir and os.path.exists(os.path.join(_run332_dir, "arm")):
            _run332_rank = int(get_tp_group().rank_in_group)
            _run332_new = [
                {"req_id": x.req_id,
                 "prompt_len": len(x.prompt_token_ids) if x.prompt_token_ids is not None else None,
                 "num_computed_tokens": int(x.num_computed_tokens),
                 "scheduled_tokens": int(scheduler_output.num_scheduled_tokens.get(x.req_id, 0)),
                 "spec_tokens": len(scheduler_output.scheduled_spec_decode_tokens.get(x.req_id, []))}
                for x in scheduler_output.scheduled_new_reqs
            ]
            _run332_cached = [
                {"req_id": req_id, "num_computed_tokens": int(computed),
                 "num_output_tokens": int(output),
                 "scheduled_tokens": int(scheduler_output.num_scheduled_tokens.get(req_id, 0)),
                 "spec_tokens": len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))}
                for req_id, computed, output in zip(
                    scheduler_output.scheduled_cached_reqs.req_ids,
                    scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                    scheduler_output.scheduled_cached_reqs.num_output_tokens)
            ]
            _run332_row = {"kind": "execute_entry", "rank": _run332_rank,
                           "t_ns": time.perf_counter_ns(),
                           "total_scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens),
                           "new": _run332_new, "cached": _run332_cached,
                           "finished_req_ids": sorted(scheduler_output.finished_req_ids)}
            self._run332_rows = getattr(self, "_run332_rows", [])
            self._run332_rows.append(_run332_row)
'''

READY = '''            # EXTREME_LOOP074_RUN332_READY
            if _run332_dir and os.path.exists(os.path.join(_run332_dir, "arm")):
                _run332_row = {"kind": "runtime_built", "rank": _run332_rank,
                               "t_ns": time.perf_counter_ns(),
                               "req_ids": list(_extreme_req_ids),
                               "target_graph_mode": str(_target_inputs.aclgraph_runtime_mode)}
                self._run332_rows.append(_run332_row)
                with open(os.path.join(_run332_dir, f"rank{_run332_rank}.jsonl"), "a") as _f:
                    for _row in self._run332_rows:
                        _f.write(json.dumps(_row, separators=(",", ":")) + "\\n")
                self._run332_rows = []
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('install', 'restore'))
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'install':
        if BACKUP.exists():
            raise RuntimeError('backup already exists')
        original = SOURCE.read_bytes()
        source = original.decode()
        if MARKER in source:
            raise RuntimeError('patch already present')
        source = replace_once(source,
            '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n        if self.vllm_config.model_config.enable_return_routed_experts:',
            '    ) -> ModelRunnerOutput | IntermediateTensors | None:\n' + ENTRY +
            '        if self.vllm_config.model_config.enable_return_routed_experts:')
        source = replace_once(source,
            '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )',
            '            _extreme_runtime = build_extreme_runtime(\n                _extreme_runtime_inputs,\n                _cfg,\n            )\n' + READY.rstrip('\n'))
        compile(source, str(SOURCE), 'exec')
        BACKUP.write_bytes(original)
        SOURCE.write_text(source)
        result = {'action': 'install', 'original_sha256': sha(original),
                  'patched_sha256': sha(SOURCE.read_bytes())}
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
