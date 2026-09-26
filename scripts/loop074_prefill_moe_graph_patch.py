#!/usr/bin/env python3
"""Reversible opt-in first warm prefill private MoE Graph fixture hook."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BACKUP = Path('/tmp/loop074_run338_model_runner_v1.py.orig')
BASE_SHA = '004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba'
MARKER = '# EXTREME_LOOP074_RUN338_PREFILL_MOE_GRAPH'
APPEND = '''

# EXTREME_LOOP074_RUN338_PREFILL_MOE_GRAPH
if os.getenv("EXTREME_RUN338_DIR"):
    _run338_original_forward = NPUModelRunner._model_forward
    def _run338_one_forward(self, num_tokens_padded, *args, **kwargs):
        _root = os.environ["EXTREME_RUN338_DIR"]
        if (len(getattr(self, "_extreme_served_cohorts", ())) == 4
                and not getattr(self, "_run338_traced", False)
                and os.path.exists(os.path.join(_root, "enable"))):
            self._run338_traced = True
            from scripts.loop074_prefill_moe_graph_fixture import install_for_one_forward, restore
            _module, _original, _report = install_for_one_forward(self, _root)
            try:
                return _run338_original_forward(self, num_tokens_padded, *args, **kwargs)
            finally:
                restore(_module, _original)
        return _run338_original_forward(self, num_tokens_padded, *args, **kwargs)
    NPUModelRunner._model_forward = _run338_one_forward
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('preview', 'install', 'restore'))
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    if args.action in ('preview', 'install'):
        original = SOURCE.read_bytes()
        if sha(original) != BASE_SHA or BACKUP.exists() or MARKER.encode() in original:
            raise RuntimeError(f'unexpected current source sha={sha(original)} or leftover backup')
        after = original + APPEND.encode()
        compile(after, str(SOURCE), 'exec')
        result = {'source': str(SOURCE), 'base_sha': BASE_SHA, 'patched_sha': sha(after)}
        if args.action == 'install':
            BACKUP.write_bytes(original)
            SOURCE.write_bytes(after)
            args.record.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    else:
        if not BACKUP.exists() or MARKER not in SOURCE.read_text():
            raise RuntimeError('missing backup or marker')
        before = SOURCE.read_bytes()
        original = BACKUP.read_bytes()
        if sha(original) != BASE_SHA:
            raise RuntimeError('backup does not match base SHA')
        SOURCE.write_bytes(original)
        BACKUP.unlink()
        result = {'action': 'restore', 'patched_sha': sha(before),
                  'restored_sha': sha(SOURCE.read_bytes())}
        args.record.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
