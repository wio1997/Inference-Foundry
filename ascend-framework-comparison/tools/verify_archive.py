#!/usr/bin/env python3
"""Verify publication bytes, JSON and archived SHA lists; no network or model execution."""
from pathlib import Path
import hashlib
import json

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
assert sdk['all_uncompressed_diff_ids_verified'] and len(sdk['layer_checks']) == 21
index=json.loads((root/'RESOURCES.json').read_text());assert index['kind']=='INDEPENDENT_PROJECT_INDEX_ONLY'
f=json.loads((root/'glm53-flash/RESOURCES.json').read_text());g=json.loads((root/'glm53/RESOURCES.json').read_text())
assert f['project_id']=='glm53-flash' and g['project_id']=='glm53' and not f['depends_on_other_model_project'] and not g['depends_on_other_model_project']
assert f['model']['model_type']=='glm5_next' and f['model']['architecture']=='Glm5NextForConditionalGeneration' and f['model']['files']==97 and f['model']['bytes']==332970562139
assert g['model']['model_type']=='glm_moe_dsa' and g['model']['architecture']=='GlmMoeDsaForCausalLM' and g['model']['files']==193 and g['model']['bytes']==764240829290
assert f['model']['weight_path']!=g['model']['weight_path'] and f['runtime_namespace']!=g['runtime_namespace']
def nodes(x):return {n for group in x['frameworks'].values() for role in ['P','D'] for n in group[role]}
assert nodes(f)=={14,15,46,47} and nodes(g)=={30,31,34,35,36,41,43,45} and not nodes(f)&nodes(g)
assert f['frameworks']['vllm_ascend']['P']==[14] and f['frameworks']['vllm_ascend']['D']==[15] and f['frameworks']['sglang']['P']==[46] and f['frameworks']['sglang']['D']==[47]
assert g['frameworks']['vllm_ascend']['P']==[30,31] and g['frameworks']['vllm_ascend']['D']==[43,45] and g['frameworks']['sglang']['P']==[34,35] and g['frameworks']['sglang']['D']==[36,41]
assert not f['supplementary_assets'] and not g['supplementary_assets']
assert g['model']['precision']=='W8A8' and g['model']['weight_path']=='/data/wio/models/GLM-5.3-w8a8' and g['model']['source_path']=='/data/wio/GLM53/GLM-5.3-w8a8'
assert g['weight_policy']['allowed_variants']==['W8A8'] and g['weight_policy']['c8_experiments_in_scope'] is False and g['weight_policy']['same_weight_directory_for_both_frameworks']
assert not (root/'glm53/C8_RESOURCE_EVIDENCE.json').exists() and not (root/'glm53/C8_SOURCE_SHA256.json').exists()
for project in ['glm53-flash','glm53']:
 state=json.loads((root/project/'STATE.json').read_text());assert state['project_id']==project and not state['depends_on_other_project']
 assert all('/' not in p for p in state['required_context'])
 for file in ['AGENTS.md','HANDOFF.md','START_PROMPT.md','PLAN.md','BENCHMARK_CONTRACT.md','EXECUTION_CONTRACT.md','RESOURCES.json','ASSETS.md','RESOURCE_EVIDENCE.json']:
  assert (root/project/file).is_file()
 evidence=json.loads((root/project/'RESOURCE_EVIDENCE.json').read_text());assert {x['node'] for x in evidence['nodes']}==nodes(f if project=='glm53-flash' else g)
 assert all(x['model']['path']==(f if project=='glm53-flash' else g)['model']['weight_path'] for x in evidence['nodes'])
print(json.dumps({'status':'PASS','files':len(manifest['files']),'historical_snapshot_schedulers':96,'independent_model_projects':2,'model_paths_and_contracts_separate':True,'primary_node_sets_disjoint':True,'original_model_w8a8_only':True,'model_requests':0,'network_calls':0}))
