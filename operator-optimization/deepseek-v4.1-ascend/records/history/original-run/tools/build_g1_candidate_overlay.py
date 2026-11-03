#!/usr/bin/env python3
"""Build a conditional source-derived overlay without modifying frozen graphs."""
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
g1_path = root / 'graphs/G1.json'
g5_path = root / 'graphs/source-templates-G5.json'
config_path = root / 'evidence/model-config.json'
def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()

g1, g1_sha = read(g1_path)
g5, g5_sha = read(g5_path)
config, config_sha = read(config_path)
assert g1_sha == '9bece982c89be6320c4f3ef986ec33030a2e71327f0f0a96ae1159a19bb187cc'
assert g5_sha == 'a3a4e82afb8fed8894300148791a22ed1ca954288a0ed8fd79ef9a3879c56731'
assert config_sha == '43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
source_hashes = {
    'models/deepseek_v41/model.py': '101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',
    'models/deepseek_v41/indexer.py': 'd5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33',
    'attention/dsa_v41.py': '3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',
}
source_root = root / 'sources/vllm-workspace/vllm-ascend/vllm_ascend'
for path, expected in source_hashes.items():
    assert hashlib.sha256((source_root / path).read_bytes()).hexdigest() == expected
text = config['text_config']
source = int(text['candidate_source_layer_id'])
targets = [int(i) for i in text['index_source_layer_ids'] if int(i) > source and int(i) < int(text['num_hidden_layers'])]
assert source == 20 and targets == [24, 28, 32, 36]
nodes = {n['id'] for n in g1['nodes']}
templates = {t['id']: t for t in g5['templates']}
for template, node in [('ATTN_C1_SOURCE', 'qli'), ('ATTN_C1_INDEX_SOURCE', 'candidate_state_and_lengths')]:
    assert sum(n['id'] == node for n in templates[template]['nodes']) == 1
old = {'producer': 'embedding', 'consumer': 'target.layers.0.hc_attn_pre', 'value': 'initial pre_mix', 'dependency_kind': 'RAW'}
assert g1['edges'].count(old) == 1
assert any(e['producer'] == 'embedding' and e['consumer'] == 'target.layers.0.hc_attn_pre' and e['value'] == 'HC residual' and e['dependency_kind'] == 'RAW' for e in g1['edges'])
cross = []
for i in targets:
    a, b = f'target.layers.{source}.attention', f'target.layers.{i}.attention'
    assert a in nodes and b in nodes
    assert any(e['producer'] == a and e['consumer'] == b and e['value'] == 'shared kv cache/state' for e in g1['edges'])
    assert not any(e['producer'] == a and e['consumer'] == b and 'candidate' in e['value'].lower() for e in g1['edges'])
    cross.append({'producer': a, 'consumer': b, 'value': 'candidate block IDs, relevant active rows',
                  'dependency_kind': 'RAW_PERSISTENT_SUBRANGE', 'whole_producer_barrier_required': None,
                  'condition': 'non-empty query and cache, non-packed source mode 1 and filter mode 2; packed and empty branches need separate modeling',
                  'publication_granularity': 'unknown; logical row relevance alone does not prove partial publication or synchronization',
                  'source_detail': 'ATTN_C1_SOURCE.qli candidate_out -> conditional shared.candidates copy',
                  'consumer_detail': 'ATTN_C1_INDEX_SOURCE.candidate_state_and_lengths -> qli candidate input'})
assert not {'target.initial_pre_mix', 'target.hidden_after_optional_sp_shard_metadata'} & nodes
overlay = {
    'id': 'G1-source-conditional-overlay-candidate-and-premix',
    'parents': {'G1.json': g1_sha, 'source-templates-G5.json': g5_sha, 'evidence/model-config.json': config_sha},
    'source_sha256': source_hashes,
    'scope': 'Frozen-source conditional dependency overlay; not an observed runtime graph and not a full fusion enumeration.',
    'remove_g1_edges': [old],
    'add_g1_nodes': [{'id': 'target.hidden_after_optional_sp_shard_metadata', 'kind': 'EXTERNAL_SHAPE_DEVICE_METADATA', 'module': 'target',
                      'source': 'models/deepseek_v41/model.py:1519,1543-1553'},
                     {'id': 'target.initial_pre_mix', 'kind': 'SHAPE_DEVICE_DEPENDENT_INITIALIZATION', 'module': 'target',
                      'source': 'models/deepseek_v41/model.py:1543-1553'}],
    'add_g1_edges': [
        {'producer': 'target.hidden_after_optional_sp_shard_metadata', 'consumer': 'target.initial_pre_mix', 'value': 'local row shape and device, not embedding contents', 'dependency_kind': 'SHAPE_DEVICE_METADATA'},
        {'producer': 'target.initial_pre_mix', 'consumer': 'target.layers.0.hc_attn_pre', 'value': 'initial FP32 pre_mix', 'dependency_kind': 'RAW'},
        *cross,
    ],
    'source_scope': ['models/deepseek_v41/model.py:490-502,510-549,1543-1553',
                     'attention/dsa_v41.py:575-610', 'models/deepseek_v41/indexer.py:274-298'],
    'conditions_and_gaps': ['Assumes frozen configuration and indicated source branches; historical binary and tensor identities are unbound.',
                            'Candidate buffer publication and consumption require correct active rows, metadata, lifetime, packed/unpacked paths and empty-cache behavior.',
                            'G5 templates are not instance graphs; the cross-layer mapping above is an overlay, not a modification of either template.',
                            'Shape/device metadata follows optional SP shard and may originate from input embeddings rather than the embedding node; runtime allocation, stream order and numerical behavior still need proof.',
                            'Existing G1 enumeration and fusion inventory were not regenerated for this overlay; raw G1 edge indices must never be reused after deletion.']
}
out = root / 'graphs/G1-source-conditional-overlay.json'
adj = defaultdict(set)
for edge in g1['edges']:
    if edge != old:
        a, b = edge['producer'], edge['consumer']
        adj[a].add(b)
        adj[b].add(a)
assert all(e['consumer'] in adj[e['producer']] for e in cross)
for edge in overlay['add_g1_edges']:
    a, b = edge['producer'], edge['consumer']
    adj[a].add(b)
    adj[b].add(a)
sets = {frozenset(['target.initial_pre_mix'])}
containing_init = {}
for size in (2, 3, 4):
    sets = {s | {neighbor} for s in sets for n in s for neighbor in adj[n] if neighbor not in s}
    containing_init[str(size)] = len(sets)
payload = (json.dumps(overlay, ensure_ascii=False, indent=2) + '\n').encode()
fd, temp_name = tempfile.mkstemp(prefix='.g1-overlay-', dir=out.parent)
try:
    with os.fdopen(fd, 'wb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_name, out)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
print(json.dumps({'path': str(out.relative_to(root)), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                  'added_cross_layer_edges': len(cross), 'premix_content_edges_reclassified': 1,
                  'candidate_pairs_already_connected_by_other_edge': len(cross),
                  'connected_node_sets_involving_init_2_to_4': containing_init,
                  'note': 'Structural sets include metadata edges, are not feasible fusions; original graphs retained and historical enumeration not regenerated.'}))
