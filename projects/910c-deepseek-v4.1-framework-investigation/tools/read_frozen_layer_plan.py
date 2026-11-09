#!/usr/bin/env python3
"""Execute only the pure topology definitions from pinned model source."""
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/Users/wio/work/branch-fusion-audit-20261104/full-review')
CONFIG = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/evidence/model-config.json')
names = {'DeepseekV41LayerRole', 'DeepseekV41Topology', '_latest_source', 'build_layer_plan'}
index = json.loads((ARCHIVE / 'flow-index.json').read_text())
pin = next(x for x in index['files'] if x['runtime_path'].endswith('/models/deepseek_v41/model.py'))
source = Path(pin['local_path'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == pin['sha256']
tree = ast.parse(source.read_text())
selected = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
assert {x.name for x in selected} == names
body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected
ns = {'dataclass': dataclasses.dataclass, 'Any': Any}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), 'exec'), ns)
config = json.loads(CONFIG.read_text())['text_config']
plan = ns['build_layer_plan'](SimpleNamespace(**config))
aux_one_based = tuple(i + 1 for i in config['dspark_target_layer_ids'])
assert 'if layer.layer_idx + 1 in self.aux_hidden_state_layers:' in source.read_text()
rows = []
for role in plan.layers:
    row = dataclasses.asdict(role)
    row.update(aux_before_layer=role.layer_idx + 1 in aux_one_based,
               body='Attention-HcPre/norm/CP/attention/RS/Post -> FFN-HcPre/normcast/MoE/Post',
               body_value_review='PENDING',
               engram_value_review='S04' if role.engram_slot is not None else 'not-present')
    rows.append(row)
result = {
    'status': 'CODE_DERIVED_TOPOLOGY_NOT_COMPLETED_BODY_REVIEW',
    'source': {'path': str(source), 'sha256': pin['sha256']},
    'config': {'path': str(CONFIG), 'sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest()},
    'layers': rows,
    'aux_capture_contract': {'checkpoint_zero_based_ids': list(config['dspark_target_layer_ids']),
                             'runner_selected_one_based_ids': list(aux_one_based),
                             'zero_based_entry_layers': [r['layer_idx'] for r in rows if r['aux_before_layer']],
                             'source': 'NPUModelRunner._get_eagle3_aux_layers_from_config:797 adds1; setter stores unchanged; model.forward:1560 tests layer_idx+1',
                             'correction': 'Retract model-only36/37/38 mapping which omitted runner+1. Actual entries37/38/39; body38 full output is aux at entry39. Original hardcoded37/38/39 was correct by coincidence.'},
    'KV_source_consumers': {str(s): list(plan.kv_consumers(s)) for s in plan.kv_source_layer_ids},
    'index_source_consumers': {str(s): list(plan.index_consumers(s)) for s in plan.index_source_layer_ids},
    'common_body_calls_per_target': {'Attention_HcPre': 40, 'FFN_HcPre': 40, 'HcPost': 80, 'SWA_write': 40,
                                     'long_KV_write': 4, 'index_selection': 8, 'Engram': 2, 'aux_mean': 3},
    'excluded_by_fixed_config': ['PCP/MRV2', 'A5 packed cache', 'unrotated Engram', 'true MTP raw numeric consumption (raw snapshot still allocated)',
                                 'hash router (num_hash_layers=0)', 'PP transfer', 'multimodal request contents'],
    'no_candidate_import': True,
}
out = ROOT / 'audit/code-led/target-layer-plan.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'layers': len(rows), 'KV_source_consumers': result['KV_source_consumers'],
                  'index_source_consumers': result['index_source_consumers'], 'out': str(out)}, ensure_ascii=False))
