#!/usr/bin/env python3
"""Audit old G1 edge-index references before applying its conditional overlay."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
def load(path, expected):
    data = path.read_bytes()
    assert sha256(data).hexdigest() == expected
    return json.loads(data)
g1_hash = '9bece982c89be6320c4f3ef986ec33030a2e71327f0f0a96ae1159a19bb187cc'
g1 = load(root / 'graphs/G1.json', g1_hash)
inventory = load(root / 'candidates/cross-module-G1.json', 'f375d77bc12836c4666c1123c5c432dd1f20211204a2a9db599c21546dc462fc')
overlay = json.loads((root / 'graphs/G1-source-conditional-overlay.json').read_text())
assert inventory['G1_sha256'] == overlay['parents']['G1.json'] == g1_hash
assert len(overlay['remove_g1_edges']) == 1
index = g1['edges'].index(overlay['remove_g1_edges'][0])
assert index == 644 and len(inventory['items']) == 22802
counts, affected = Counter(), Counter()
source = 'target.layers.20.attention'
targets = {f'target.layers.{i}.attention' for i in (24, 28, 32, 36)}
semantic_review = 0
for item in inventory['items']:
    cats = set()
    if source in item['nodes'] and targets.intersection(item['nodes']):
        semantic_review += 1
    for key in ('internal_edges_G1_edge_indices', 'external_inputs_G1_edge_indices', 'external_outputs_G1_edge_indices'):
        for i in item[key]:
            assert isinstance(i, int) and 0 <= i < len(g1['edges'])
            category = 'equal' if i == index else 'higher' if i > index else 'lower'
            counts[category] += 1
            cats.add(category)
    affected.update(cats)
assert counts['equal'] == 62 and counts['higher'] == 5673 and semantic_review == 1226
print(json.dumps({'removed_edge_index': index, 'old_index_references': dict(counts),
                  'old_items_by_reference_category': dict(affected), 'old_items_for_candidate_semantic_review': semantic_review,
                  'warning': 'Index positions are valid only for their parent G1; rebind or regenerate before using an overlaid graph. Item count is not a count of feasible fusions.'}, indent=2))
