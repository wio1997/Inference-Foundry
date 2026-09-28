#!/usr/bin/env python3
"""Read-only source/observer reuse gate for the narrowed Framework packet."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
PLAN = E / 'run636/runtime_packet_preflight.json'
PATCH = E / 'run637/observer_patch_preflight.json'
OUT = E / 'run650/source_reuse_preflight.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan = json.loads(PLAN.read_text())
    patch = json.loads(PATCH.read_text())
    sources = {}
    for key in ('cycle', 'target', 'serving', 'draft', 'target_handoff',
                'runner', 'graph'):
        spec = plan['source'][key]
        path = Path(spec['path'])
        raw = path.read_text()
        actual = sha(path)
        assert actual == spec['sha256'], (key, actual)
        assert all(anchor in raw for anchor in spec['validated_anchors']), key
        sources[key] = {'path': str(path), 'sha256': actual,
                        'anchors_checked': len(spec['validated_anchors'])}
    observer = ROOT / 'runtime/bound_observer.py'
    assert sha(observer) == patch['observer_module_sha256']
    assert patch['status'] == 'candidate_only_NOT_INSTALLED_NOT_LIVE_READY'
    assert plan['status'] == 'source_preflight_only_NOT_LIVE_READY'
    result = {
        'status': 'source_reuse_pass_NOT_LIVE_READY',
        'run636_plan_sha256': sha(PLAN),
        'run637_patch_manifest_sha256': sha(PATCH),
        'observer_sha256': sha(observer),
        'sources': sources,
        'p0_reuse': 'Run606 existing 1214-cycle/Product final-ID ledger; do not recollect solely for P0',
        'unmet': ['implemented bounded true-producer/consumer packet',
                  'Event stream/generation and no-new-sync preflight',
                  'observer and fixed-W0/shape-cost transfer controls',
                  'live all8 ready/issue/resource evidence and correctness admission'],
        'numeric_framework_product_tps_bound': None,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(OUT)

if __name__ == '__main__':
    main()
