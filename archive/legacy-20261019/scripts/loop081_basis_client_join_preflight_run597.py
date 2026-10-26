#!/usr/bin/env python3
"""Exercise Run597 client/request prefix join on prior admitted Run589 files."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
ARM=ROOT/'evidence/20260928_loop081_bound/run589/live/b'
OUT=ROOT/'evidence/20260928_loop081_bound/run597_preflight/client_join.json'
if OUT.exists():raise ValueError('preflight output exists')
client_path=ARM/'client_admission.json'
client=json.loads(client_path.read_text())
assert client['status']=='client_two_phase_admitted'
by_response={x['response_id']:x for x in client['request_index']}
assert len(by_response)==96
seen=set();keys={};hashes={str(client_path.relative_to(ROOT)):
                        hashlib.sha256(client_path.read_bytes()).hexdigest()}
for cohort in range(1,9):
    p=ARM/'runtime'/f'rank0_cohort{cohort}.json'
    runtime=json.loads(p.read_text())
    assert runtime['cohort']==cohort and runtime['pass'] and len(runtime['req_ids'])==12
    hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[]
    for request_id in runtime['req_ids']:
        matches=[x for response_id,x in by_response.items()
                 if request_id.startswith(response_id+'-')]
        assert len(matches)==1 and request_id not in seen
        seen.add(request_id)
        item=matches[0]
        assert item['phase']==('warmup' if cohort<=4 else 'measured')
        rows.append(item['dataset_index'])
    keys[str(cohort)]=rows
assert len(seen)==96
for cohorts in (range(1,5),range(5,9)):
    assert sorted(v for c in cohorts for v in keys[str(c)])==list(range(48))
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(dict(status='prior_client_prefix_join_pass',
    source_run='Run589 admitted two-phase diagnostic',
    request_count=96,cohort_dataset_indices=keys,source_sha256=hashes),indent=2)+'\n')
print('Run597 client join preflight PASS')
