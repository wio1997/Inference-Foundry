#!/usr/bin/env python3
"""Rebuild frozen input files and check every byte against saved SHA256 values.

Uses only vocabulary IDs, frozen seeds and the original workload algorithms.
No network, model loading or benchmark execution. Existing output is refused.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
import e27_prepare as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['E27-L32K-C32', 'E27-L32K-C64',
        'E27-L32K-C128', 'E29-L80K-C16', 'E29-L80K-C32'], required=True)
    parser.add_argument('--out', type=Path, help='New directory; default is the frozen case directory')
    args = parser.parse_args()
    source = ROOT / 'data/workloads'
    manifest = json.loads((source / args.case / 'manifest.json').read_text())
    pool = json.loads((source / 'seed-token-pool.json').read_text())
    provenance = json.loads((source / 'seed-token-pool.provenance.json').read_text())
    assert pool == sorted(set(pool))
    assert hashlib.sha256(json.dumps(pool, separators=(',', ':')).encode()).hexdigest() == provenance['pool_payload_sha256']
    target = args.out or source / args.case
    # The shipped case directory contains only its manifest. Never replace inputs.
    if target.exists() and any(p.name != 'manifest.json' for p in target.iterdir()):
        raise FileExistsError('Refusing to replace existing workload files: ' + str(target))
    target.mkdir(parents=True, exist_ok=True)
    seed = manifest['seed']
    length = manifest.get('length', manifest.get('input_tokens'))
    requests = manifest.get('requests', manifest.get('total_requests'))
    cached = manifest.get('cached_tokens', manifest.get('planned_cached_tokens'))
    warm = manifest.get('v_warm_prefix', cached + 128)
    handles = {name: (target / name).open('x', encoding='utf-8', newline='\n') for name in shared.WORKLOAD_FILES}
    digests = {name: hashlib.sha256() for name in shared.WORKLOAD_FILES}
    try:
        for q in range(requests):
            ids = random.Random(seed + q).choices(pool, k=length)
            shape = shared.make_shape_warm_ids(q, ids, pool, length, seed)
            sentinel = shared.make_v_sentinel(q, ids[warm], pool, seed)
            payloads = {'corpus.jsonl': ids, 'shape-warm.jsonl': shape,
                        'di-prefix.jsonl': ids[:cached], 'v-prefix.jsonl': ids[:warm] + [sentinel]}
            family = 'E29-L80K' if length == 81920 else 'E27-L32K'
            for name, tokens in payloads.items():
                shared.write_json_line(handles[name], digests[name], dict(request_index=q,
                    logical_id=family + '-q' + str(q), input_token_ids=tokens))
            shared.write_json_line(handles['questions.jsonl'], digests['questions.jsonl'], dict(question=str(q), answer=''))
    finally:
        for handle in handles.values(): handle.close()
    checked = {}
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((target / name).read_bytes()).hexdigest()
        if actual != expected['sha256']:
            raise ValueError('Frozen workload hash mismatch: ' + name)
        checked[name] = actual
    print(json.dumps(dict(case=args.case, files=checked, status='hashes_match_frozen_inputs')))


if __name__ == '__main__': main()
