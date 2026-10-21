#!/usr/bin/env python3
"""Apply the frozen deployment patches to pristine pinned source archives."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = [
    ('deepselect-ascend-build-import.patch', ROOT / 'src/DeepInfer'),
    ('flashmla-cann-rint-api.patch', ROOT / 'src/DeepInfer'),
    ('engram-direct-storage.patch', ROOT),
    ('server-local-ubmem.patch', ROOT),
    ('npu-from-blob.patch', ROOT),
    ('deep-gemm-assert-include.patch', ROOT),
    ('candidate-short-context.patch', ROOT / 'src/DeepInfer'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Requires an already patched source tree')
    args = parser.parse_args()
    state_path = ROOT / 'patch-state.json'
    patch_hashes = {name: hashlib.sha256((ROOT / 'patches' / name).read_bytes()).hexdigest() for name, _ in ORDER}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['patches'] != patch_hashes:
            raise SystemExit('Patch set changed; use a separate pristine source checkout.')
        for relative, expected in state['files'].items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise SystemExit('Patched source drift: ' + relative)
        print('PATCH_SET_VERIFIED')
        return
    if args.check_only:
        raise SystemExit('No patch-state.json; apply patches to a pristine pinned tree first.')
    touched = set()
    for name, cwd in ORDER:
        patch = ROOT / 'patches' / name
        base = ['git', 'apply', '-p1']
        # These source archives do not need .git metadata.
        subprocess.run(base + ['--check', str(patch)], cwd=cwd, check=True)
        subprocess.run(base + [str(patch)], cwd=cwd, check=True)
        for line in patch.read_text().splitlines():
            if line.startswith('+++ b/'):
                touched.add((cwd / line[6:]).relative_to(ROOT).as_posix())
        print('APPLIED', name)
    state = {'patches': patch_hashes,
             'files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(touched)}}
    state_path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
