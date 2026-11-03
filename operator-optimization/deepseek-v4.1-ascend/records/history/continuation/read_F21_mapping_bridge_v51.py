"""Resolve two exact import/delegation edges in the frozen core reference, without importing it."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib

from finalize_origin_coverage_v9 import N, O, W, read, write


def main():
    root = W / 'frozen-source-snapshot/vllm/vllm/model_executor/layers/fused_moe'
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v51.json')
    files = []
    for name in ['__init__.py', 'layer.py']:
        p = root / name;raw = p.read_bytes();sha = hashlib.sha256(raw).hexdigest()
        descriptor = dict(path=str(p), sha256=sha,
            symbols=['fused_moe_make_expert_params_mapping'],
            scope='Frozen originalsource snapshot reference;individual historical166filehash not present in v19pins.')
        if descriptor not in gate['allowed_source']:gate['allowed_source'].append(descriptor)
        write(O / 'PROFILE-CODE-QUESTION-GATE-v51.json', gate)
        tree = ast.parse(raw);lines = raw.decode().splitlines()
        if name == '__init__.py':
            selected = [n for n in tree.body if isinstance(n, ast.ImportFrom) and
                        any(a.name == 'fused_moe_make_expert_params_mapping' for a in n.names)]
        else:
            selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_moe_make_expert_params_mapping']
        assert len(selected) == 1
        frozen = N / 'F21-checkpoint-source-v51/vllm/vllm/model_executor/layers/fused_moe' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():assert hashlib.sha256(frozen.read_bytes()).hexdigest() == sha
        else:frozen.write_bytes(raw)
        parts = [dict(name='fused_moe_make_expert_params_mapping/import_or_wrapper', start=n.lineno, end=n.end_lineno,
                      source=[str(i + 1) + ': ' + lines[i] for i in range(n.lineno - 1, n.end_lineno)]) for n in selected]
        files.append(dict(path=str(p), frozen_path=str(frozen), sha256=sha,
            historical166_hash_matched=False, scope=descriptor['scope'], parts=parts))
    write(O / 'F21-mapping-import-bridge-source-v51.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), files=files,
        no_model_import_remote_or_device_work=True, target_changed=False,
        scope='Two exact bridge statements only. Captured-model import is known;actual166bridge bytes/global rebinding remain unverified.',
        whole_module_closed=False, overall_complete=False))
    for f in files:
        print('FILE',f['path'])
        for p in f['parts']:print('\n'.join(p['source']))


if __name__ == '__main__':
    main()
