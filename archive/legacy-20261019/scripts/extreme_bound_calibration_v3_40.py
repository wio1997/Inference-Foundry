#!/usr/bin/env python3
"""V3.40 fixed-algorithm Runtime DAG ownership, still no finite Bound."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
R12 = ROOT / 'evidence/20260928_loop081_bound/run612'
R13 = ROOT / 'evidence/20260928_loop081_bound/run613'
FILES = {
    'base': R12 / 'bound_calibration_v3_39.json',
    'flow': R13 / 'eager_flow_join.json',
    'review': R13 / 'astra_kv_review.md',
}
PINS = {
    'base': 'fba742dc27aaff0758abaebecd28b8ef66690fc60a5f56a11d664a5e39f418c2',
    'flow': 'dc17ab40f85d38d2d5f87d921d44d1b1f4debb51dea6aceddd9b9ae0ad9c99b9',
    'review': 'a829aac896b1d8cf45c9f4e3623b015f781fd982bd7b0c0ac37c02f399bda8be',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


for key, path in FILES.items():
    need(sha(path) == PINS[key], f'SHA drift {key}')
model = json.loads(FILES['base'].read_text())
flow = json.loads(FILES['flow'].read_text())
need(model['bound_semantics_v3_39']['formal_current_tps'] == 571.681 and
     flow['status'] == 'all8_eager_scatter_sparse_attention_exact_flow_join_pass' and
     len(flow['rows']) == 24 and
     sum(r['host_to_native_exact_flow_join_count'] for r in flow['rows']) == 216,
     'fixed-W0 flow identity')
need(all(r['scatter_count'] == 6 and r['sparse_attention_count'] == 3 for r in flow['rows']) and
     {r['rank'] for r in flow['rows']} == set(range(8)),
     'all8 per-occurrence cardinality')
model['model_revision'] = 'v3.40_exported_eager_flow_ownership_not_generation'
model['bound_semantics_v3_40'] = {
    'active_objective': 'minimum Product E2E Runtime for unchanged DSpark7 algorithm, acceptance/cycle trajectory, output semantics and model work',
    'formal_current_tps': 571.681,
    'same_W0_exported_flow': {
        'CPU_to_native_unique_pairs': 216,
        'scatter_pairs': 144,
        'sparse_attention_pairs': 72,
        'native_stream': 47,
        'three_first_scatter_per_proposer': 'source-consistent context KV writes; no direct per-layer context tag',
        'three_later_scatter_plus_attention': 'Host layer43/44/45 scopes and exact native torch_to_npu s/f flow identity',
        'rank5_cross_cycle_observation': 'first six native scatter starts follow next Target Host scope start on exported trace; no inference of next Target device-work overlap',
        'source_alias_relation': 'source-level SWA cache object flows through query scatter and ori_kv attention argument for relevant prefill/decode branches; exact Run610 allocation and row generation not exported',
        'missing_typed_edge': 'actual cache allocation-generation, slot validity/overwrite, per-row last writer, first query read set and predecessor/successor retained state'
    },
    'strict_resource_hardware_floor_s': None,
    'strict_scheduling_execution_floor_s': None,
    'strict_product_e2e_tps_ceiling': None,
    'numeric_current_to_credible_limit_gap': None,
    'resource_proof_required': 'fixed-W0 compulsory arithmetic/traffic plus certified exact-board upper cumulative compute/HBM/HCCL C_plus,B; no promotional use of attained current service',
    'scheduling_proof_required': 'typed actual row/storage-generation DAG, legal alternative schedule under shared-resource contention, observer OFF/ON/OFF timing transfer',
    'highest_information_next_probe': 'one bounded all8 adjacent active ordinary cycle pair: preallocated typed context/query scatter and pre-attention witness with cache allocation-generation, valid slot payload, block table/window/mask/sparse indices, predecessor retained KV and successor consumer. No algorithm change or sync; then judge whether a legal relaxation exists.',
    'independent_review': 'Run613 Astra High source/native flow ownership SCOPED PASS; no tensor generation or numerical Bound'
}
need(all(model['bound_semantics_v3_40'][key] is None for key in
         ('strict_resource_hardware_floor_s', 'strict_scheduling_execution_floor_s',
          'strict_product_e2e_tps_ceiling', 'numeric_current_to_credible_limit_gap')),
     'strict endpoints fail closed')
out = R13 / 'bound_calibration_v3_40.json'
out.write_text(json.dumps(model, indent=2) + '\n')
print(json.dumps({'status': 'v3_40_flow_ownership_not_generation',
                  'strict_endpoints': 'null', 'joins': 216}))
