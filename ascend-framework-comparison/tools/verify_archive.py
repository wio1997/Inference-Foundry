#!/usr/bin/env python3
"""Verify publication bytes, JSON and archived SHA lists; no network or model execution."""
from pathlib import Path
import hashlib
import json
from collections import Counter

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'FILE_MANIFEST.json').read_text())
for rel, expected in manifest['files'].items():
    p = root / rel
    assert not p.is_symlink() and p.is_file(), rel
    assert p.stat().st_size == expected['bytes'], rel
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['sha256'], rel
    if p.suffix == '.json': json.loads(p.read_text())

archive = root / 'archive/20261016'
for n in [33,34,35,36,37,41]:
    node = archive / f'current/records/node-{n}'
    runtime = json.loads((node/'runtime.json').read_text())
    for section in ['sources','host_files']:
        for rel, meta in runtime[section].items():
            p = node / section / rel
            expected = meta if isinstance(meta,str) else meta.get('sha256')
            if expected:
                assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, str(p)
    assert len(runtime['schedulers']) == 16, n
    assert runtime['image_id'] == 'sha256:88d9988736fd0e0ba51b5f5e5b4b0274b97e57d0e3500a0daf4bcb242cde68d0'
    assert any(m['Destination']=='/model' and not m['RW'] for m in runtime['mounts'])

sdk = json.loads((archive/'b103/DOWNLOAD_RESULT.json').read_text())
config = json.loads((root/'RESOURCES.json').read_text())
assert sdk['all_uncompressed_diff_ids_verified'] and len(sdk['layer_checks']) == 21
nodes = [n for model in ['flash','glm53'] for group in config[model].values() for role in ['P','D'] for n in group[role]]
assert len(nodes)==16
counts=Counter(nodes)
duplicates={n for n,count in counts.items() if count>1}
conflicts={c['node'] for c in config.get('allocation_conflicts',[])}
assert duplicates==conflicts=={47}
assert config['glm53']['vllm_ascend']['P']==[30,31,32,47]
assert 38 not in nodes
assert all(c['do_not_deploy_both'] for c in config['allocation_conflicts'])
print(json.dumps({'status':'PASS','files':len(manifest['files']),'snapshot_schedulers':96,'proposed_role_slots':16,'unique_assigned_nodes':len(set(nodes)),'unresolved_role_conflicts':sorted(duplicates),'model_requests':0,'network_calls':0}))
