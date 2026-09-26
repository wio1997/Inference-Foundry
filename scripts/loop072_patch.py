"""SHA-guarded reversible opt-in parked MoE diagnostic hook."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path('/data/wio/Inference_Foundry/bootstrap/vllm_extreme_handoff.py')
BASE_SHA = 'f644bd14ac1cb9c8365ba2464abbff989d7f2c4c2d58279a8e18a5bce918716f'
ANCHOR = '    target_handoff = DirectTargetHandoff(inputs.target)\n'
INSERT = '''    if os.getenv("EXTREME_PARKED_MOE_DIR"):
        from scripts.loop072_moe_census import install as install_parked_moe_probe
        install_parked_moe_probe(target_handoff, state, inputs.target_tp_rank)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('preview', 'install', 'restore'))
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    current = SOURCE.read_bytes()
    if args.action in ('preview', 'install'):
        if sha(current) != BASE_SHA:
            raise SystemExit(f'bootstrap source SHA mismatch: {sha(current)}')
        text = current.decode()
        if text.count(ANCHOR) != 1:
            raise SystemExit('target handoff anchor not unique')
        modified = text.replace(ANCHOR, ANCHOR + INSERT).encode()
        record = {'source': str(SOURCE), 'base_sha': BASE_SHA,
                  'patched_sha': sha(modified),
                  'backup': str(args.record.with_suffix('.original.py'))}
        if args.action == 'install':
            args.record.parent.mkdir(parents=True, exist_ok=True)
            Path(record['backup']).write_bytes(current)
            args.record.write_text(json.dumps(record, indent=2) + '\n')
            SOURCE.write_bytes(modified)
        print(json.dumps(record))
        return
    record = json.loads(args.record.read_text())
    if record['source'] != str(SOURCE) or record['base_sha'] != BASE_SHA:
        raise SystemExit('patch record does not match source contract')
    if sha(current) == BASE_SHA:
        print('already restored')
        return
    if sha(current) != record['patched_sha']:
        raise SystemExit(f'unknown modified source SHA: {sha(current)}')
    original = Path(record['backup']).read_bytes()
    if sha(original) != BASE_SHA:
        raise SystemExit('backup source SHA mismatch')
    SOURCE.write_bytes(original)
    print(f'restored {BASE_SHA}')


if __name__ == '__main__':
    main()
