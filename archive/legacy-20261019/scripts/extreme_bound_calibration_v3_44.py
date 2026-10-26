#!/usr/bin/env python3
"""V3.44: structural Runtime DAG without fabricated timing or capacity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
PINS = {
    'v3_43': (E / 'run623/bound_calibration_v3_43.json',
              '6235199ed583c0961ab3f7207fd011d985d3a9c6139c2d7611e2f7933ec299c5'),
    'dag': (E / 'run625/runtime_dag_v0.json',
            'c2b80720c0501a50fb605335d69a5d3657d6afe5b5f9a66e349046d007ac5b74'),
    'dag_review': (E / 'run625/astra_dag_review.md',
                   '1df84b5abad72c97bec5cb6703a8dee9fdcdc59d1ff4b39c9bd758401de57bfd'),
}
for name, (path, expected) in PINS.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(name + ' SHA drift')

model = json.loads(PINS['v3_43'][0].read_text())
dag = json.loads(PINS['dag'][0].read_text())
if model['bound_semantics_v3_43']['formal_current_tps'] != 571.681:
    raise ValueError('formal Current drift')
if dag['validations']['node_count'] != 82 or len(dag['edges']) != 106:
    raise ValueError('DAG topology drift')
if dag['metadata_operand_readiness_ancestry'] is not None or dag['cross_cycle_retained_cache_edges'] is not None:
    raise ValueError('unbound ancestry falsely closed')
metadata_roots = [name for name, row in dag['nodes'].items()
                  if row['category'] == 'unbound_operand_generation']
if len(metadata_roots) != 18 or any(
    edge['to'] == root for root in metadata_roots for edge in dag['edges']
):
    raise ValueError('metadata readiness scope drift')
if len([edge for edge in dag['edges'] if edge['kind'] == 'conditional_reader']) != 12:
    raise ValueError('conditional reader scope drift')
if any(value is not None for row in dag['nodes'].values() for key, value in row.items()
       if key in ('current_cost_s', 'engineering_cost_s', 'aggressive_cost_s', 'duration_lower_s')):
    raise ValueError('unmeasured DAG time')

model['model_revision'] = 'v3.44_fixed_work_partial_runtime_dag_unbound'
model['bound_semantics_v3_44'] = {
    'active_objective': 'fixed DSpark7 acceptance/cycles/output/model work; minimum full Product Runtime execution time',
    'formal_current_tps': 571.681,
    'source_structural_dag': {
        'nodes': 82,
        'edges': 106,
        'conditional_reader_edges': 12,
        'unbound_metadata_operand_roots': 18,
        'scope': 'two nominal diagnostic cycles on one rank; value uses conditional on current operation sites, not compulsory physical nodes or complete all8 Product DAG',
        'candidate_reordering': 'three layer-specific context projections share input but are Python-issued serially; Target logits and auxiliary hidden readiness and query Q/KV are separated; benefit is unknown under resource contention',
        'unknowns': 'actual metadata readiness ancestry, writer-reader row generation, cross-cycle retained KV/Host/count-copy, full Target/HCCL/DSpark/tails and Product publication',
        'unbound_root_rule': 'unknown roots are not time-zero available or zero cost; no timing solver admission until credited as entry values or linked to producers',
    },
    'resource_hardware_floor_s': None,
    'scheduling_execution_floor_s': None,
    'product_e2e_ceiling_tps': None,
    'conditional_engineering_interval_tps': None,
    'numeric_current_to_credible_limit_gap': None,
    'next_decisive_acquisition': 'combined same-W0 typed actual invocation writer/reader + metadata readiness/row generation and conditional fresh projection; use admitted all8 mixed service only for later resource-constrained DAG sensitivity',
}
v = model['bound_semantics_v3_44']
if any(v[key] is not None for key in ('resource_hardware_floor_s', 'scheduling_execution_floor_s',
                                      'product_e2e_ceiling_tps', 'conditional_engineering_interval_tps',
                                      'numeric_current_to_credible_limit_gap')):
    raise ValueError('unsupported finite endpoint')

out = E / 'run625/bound_calibration_v3_44.json'
out.write_text(json.dumps(model, indent=2) + '\n')
print(json.dumps({'status': model['model_revision'], 'strict_endpoints': 'null', 'dag_nodes': 82}))
