#!/usr/bin/env python3
"""Partial fixed-work two-cycle Runtime DAG, with unresolved edges kept open."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
PINS = {
    'v3_43': (E / 'run623/bound_calibration_v3_43.json',
              '6235199ed583c0961ab3f7207fd011d985d3a9c6139c2d7611e2f7933ec299c5'),
    'source_review': (E / 'run623/astra_next_packet_review.md',
                      'f42965ff744c86587f984f7ed85a7341e99519e16617aac87b6d68d1b72c35dc'),
    'reader_review': (E / 'run616/astra_reader_abi_review.md',
                      '10ec319edb9ee02368c6b341fdf761328e561c31d9003485a82479ba08bda2b8'),
    'flow': (E / 'run613/eager_flow_join.json',
             'dc17ab40f85d38d2d5f87d921d44d1b1f4debb51dea6aceddd9b9ae0ad9c99b9'),
    'interval_census': (E / 'run616/eager_interval_census.json',
                        'a6030a53c42f7683f1323c61494387aa525b1dc40cdc7a830b94318e191c323e'),
}
for name, (path, digest) in PINS.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(name + ' source SHA drift')

model = json.loads(PINS['v3_43'][0].read_text())
intervals = json.loads(PINS['interval_census'][0].read_text())
assert intervals['record_count'] == 72
assert model['bound_semantics_v3_43']['formal_current_tps'] == 571.681
assert model['bound_semantics_v3_43']['strict_scheduling_execution_floor_s'] is None

nodes: dict[str, dict] = {}
edges: list[dict] = []


def node(name: str, category: str, work: str) -> None:
    if name in nodes:
        raise ValueError('duplicate node ' + name)
    nodes[name] = {
        'category': category,
        'work': work,
        'current_cost_s': None,
        'engineering_cost_s': None,
        'aggressive_cost_s': None,
        'duration_lower_s': None,
    }


def edge(src: str, dst: str, kind: str, basis: str) -> None:
    if src not in nodes or dst not in nodes or src == dst:
        raise ValueError('bad edge')
    edges.append({'from': src, 'to': dst, 'kind': kind, 'basis': basis})


node('entry_state_64', 'boundary', 'retained target/draft/KV/Host state and prior proposal; entry credit unknown')
node('product_arrival', 'boundary', 'frozen c12 request arrival/permit conditions; not timestamped here')
for c in (64, 65):
    p = str(c)
    for suffix, category, work in (
        ('target', 'composite_value_scope_not_completion', 'required Target aux/logit producer scope; remainder of FULL Graph and internal HCCL unresolved, not a Graph-completion barrier'),
        ('target_logits', 'value', 'Target logits/value for frozen acceptance'),
        ('target_aux_hidden', 'value', 'Target auxiliary hidden for draft context'),
        ('accept', 'device_host_boundary', 'fixed DSpark7 acceptance decision/count; no acceptance optimization'),
        ('state_advance', 'value', 'fixed target/KV/state update for successor'),
        ('draft_prepare', 'host_device_boundary', 'prepare actual context/query inputs and metadata'),
        ('draft_context_input', 'value', 'shared context hidden and position; slot operand readiness separate'),
        ('draft_query_input', 'value', 'query input for sequential draft layers'),
        ('draft_output', 'value', 'proposal input to successor Target'),
        ('output_ledger', 'boundary', 'generated/retained IDs; API publication may be later'),
    ):
        node(f'{suffix}_{p}', category, work)
    for layer in (43, 44, 45):
        for suffix, category, work in (
            ('context_project', 'compute', 'layer-specific context WKV/RoPE projection'),
            ('context_slot_args', 'unbound_operand_generation', 'actual formatted context scatter slots, readiness ancestry unresolved'),
            ('context_scatter', 'memory', 'formatted-slot SWA cache update'),
            ('query_layer_input', 'value', 'query hidden input from preceding draft layer; output projection/residual not expanded'),
            ('query_q_project', 'compute', 'Q projection; scheduling relative to KV prolog unproved'),
            ('query_kv_project', 'compute', 'query KV projection/prolog; scheduling relative to Q unproved'),
            ('query_slot_args', 'unbound_operand_generation', 'actual query scatter slots, readiness ancestry unresolved'),
            ('query_scatter', 'memory', 'query SWA cache update, branch-specific'),
            ('reader_metadata_args', 'unbound_operand_generation', 'actual sparse/dense reader indices, table, lengths and metadata; ancestry unresolved'),
            ('swa_reader', 'memory_compute', 'sparse attention with actual branch-specific eligible rows'),
        ):
            node(f'{suffix}_{layer}_{p}', category, work)

    edge(f'target_{p}', f'target_logits_{p}', 'value_required', 'Target required logits producer; not full Graph completion')
    edge(f'target_{p}', f'target_aux_hidden_{p}', 'value_required', 'Target auxiliary producer; may have distinct readiness')
    edge(f'target_logits_{p}', f'accept_{p}', 'value_required', 'acceptance consumes Target logits')
    edge(f'accept_{p}', f'state_advance_{p}', 'value_required', 'fixed accepted output updates state')
    edge(f'state_advance_{p}', f'draft_prepare_{p}', 'value_required',
         'handoff uses advanced last_sampled_tokens/num_sampled state')
    edge(f'target_aux_hidden_{p}', f'draft_prepare_{p}', 'value_required', 'auxiliary hidden/input selects draft context')
    edge(f'accept_{p}', f'draft_prepare_{p}', 'value_required', 'sample/count controls actual proposer preparation')
    edge(f'draft_prepare_{p}', f'draft_context_input_{p}', 'value_required', 'prepared actual context inputs')
    edge(f'draft_prepare_{p}', f'draft_query_input_{p}', 'value_required', 'prepared actual query inputs')
    edge(f'accept_{p}', f'output_ledger_{p}', 'value_required', 'accepted token ledger; not API send time')
    for layer in (43, 44, 45):
        k = f'{layer}_{p}'
        edge(f'draft_context_input_{p}', f'context_project_{k}', 'value_required', 'same context input, independent layer weights')
        edge(f'context_project_{k}', f'context_scatter_{k}', 'value_required', 'scatter consumes projected KV')
        edge(f'context_slot_args_{k}', f'context_scatter_{k}', 'value_required', 'scatter consumes actual formatted slot operands')
        edge(f'query_layer_input_{k}', f'query_q_project_{k}', 'value_required', 'Q consumes layer input')
        edge(f'query_layer_input_{k}', f'query_kv_project_{k}', 'value_required', 'KV consumes layer input')
        edge(f'query_kv_project_{k}', f'query_scatter_{k}', 'value_required', 'query scatter consumes projected KV')
        edge(f'query_slot_args_{k}', f'query_scatter_{k}', 'value_required', 'query scatter consumes actual slot operands')
        edge(f'query_q_project_{k}', f'swa_reader_{k}', 'value_required', 'reader consumes projected Q')
        edge(f'reader_metadata_args_{k}', f'swa_reader_{k}', 'value_required', 'reader consumes branch-specific metadata operands')
        edge(f'query_scatter_{k}', f'swa_reader_{k}', 'conditional_reader',
             'actual executed query-written row membership, writer epoch and branch required')
        edge(f'context_scatter_{k}', f'swa_reader_{k}', 'conditional_reader',
             'actual executed context-written row membership, last-writer generation and coverage required; Resource freshness is separate')
    edge(f'context_scatter_45_{p}', f'query_layer_input_43_{p}', 'current_python_order',
         'current first-pass context precompute completes before query forward; not a global necessary barrier')
    edge(f'draft_query_input_{p}', f'query_layer_input_43_{p}', 'value_required', 'first draft query layer input')
    for lo, hi in ((43, 44), (44, 45)):
        edge(f'swa_reader_{lo}_{p}', f'query_layer_input_{hi}_{p}', 'partial_value_required',
             'next layer hidden depends on preceding layer output; other same-layer work unresolved')
        edge(f'context_scatter_{lo}_{p}', f'context_project_{hi}_{p}', 'current_python_order',
             'current context precompute loop; independent layer-specific projection inputs')
    edge(f'swa_reader_45_{p}', f'draft_output_{p}', 'partial_value_required',
         'final draft layer contributes to proposal; head/residual not expanded')
    edge(f'target_aux_hidden_{p}', f'draft_context_input_{p}', 'partial_value_required',
         'Target auxiliary hidden enters context projection')

edge('entry_state_64', 'target_64', 'value_required', 'prior retained state/proposal')
edge('state_advance_64', 'target_65', 'value_required', 'next Target state')
edge('draft_output_64', 'target_65', 'value_required', 'next Target draft proposal')
edge('entry_state_64', 'draft_prepare_64', 'partial_value_required', 'prior retained proposal/state credit unexpanded')


def topological(include: set[str]) -> tuple[list[str], list[str]]:
    use = [e for e in edges if e['kind'] in include]
    indeg = {n: 0 for n in nodes}
    out = defaultdict(list)
    for e in use:
        indeg[e['to']] += 1
        out[e['from']].append(e['to'])
    ready = deque(sorted(n for n, d in indeg.items() if d == 0))
    order = []
    while ready:
        u = ready.popleft()
        order.append(u)
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if len(order) != len(nodes):
        raise ValueError('cycle in declared DAG')
    return order, sorted(n for n, d in indeg.items() if d != 0)


required = {'value_required', 'partial_value_required'}
topological(required)
topological(required | {'current_python_order', 'conditional_reader'})
assert len(nodes) == 2 + 2 * (10 + 3 * 10)
assert len([e for e in edges if e['kind'] == 'conditional_reader']) == 12

out = {
    'status': 'two_cycle_partial_dependency_template_not_timed_bound',
    'contract': 'fixed DSpark7 W0 acceptance/cycles/output/model work; DeepSeek V4 Flash W4A8 8x910B3 DP1TP8 48x32K->1024 c12',
    'source_pins': {k: {'path': str(v[0].relative_to(ROOT)), 'sha256': v[1]} for k, v in PINS.items()},
    'cycles': [64, 65],
    'rank_scope': 'per-rank structural template; all8 collective and Host/Product joins unresolved',
    'node_semantics': 'Nodes name current computation/materialization sites to locate input value dependencies. They are not asserted compulsory physical operations: fusion, recomputation, renaming, reuse and alternate layout remain legal. All duration lower fields null.',
    'metadata_operand_readiness_ancestry': None,
    'metadata_scope': 'Per-layer actual context/query slot and reader metadata operands are explicit, but predecessor values may derive from entry state, accepted state or prior cycle; no global prepare barrier or prefetch legality asserted',
    'cross_cycle_retained_cache_edges': None,
    'cross_cycle_scope': 'Only draft proposal/state edges are shown; retained cache, Host count-copy and metadata edges across cycles remain unmeasured and must not be treated as absent',
    'nodes': nodes,
    'edges': edges,
    'edge_classes': {
        'value_required': 'input value dependence conditional on executing these current operation sites; it does not prove either physical site compulsory or its kernel-duration lower bound',
        'partial_value_required': 'input contribution conditional on this decomposition; omitted subgraph may add other dependencies and physical work may be reorganized',
        'conditional_reader': 'requires same-acquisition actual ABI row/generation/coverage witness; not admitted as necessary yet',
        'current_python_order': 'observed source issue order only, removable if resources/values permit',
    },
    'observed_current_interval_reference': {
        'source': 'Run616 Level0 native start spacing, separate W0; do not assign node durations',
        'context_to_query_median_us': intervals['distributions_us']['context_to_query_native_start_us']['median'],
        'query_to_reader_median_us': intervals['distributions_us']['query_to_reader_native_start_us']['median'],
    },
    'validations': {
        'required_graph_acyclic': True,
        'augmented_current_candidate_graph_acyclic': True,
        'node_count': len(nodes),
        'conditional_reader_edges': 12,
    },
    'current_costs': None,
    'attainable_resource_service': None,
    'all8_collective_message_bytes_topology': None,
    'legal_full_product_critical_path': None,
    'resource_hardware_floor_s': None,
    'scheduling_execution_floor_s': None,
    'product_e2e_ceiling_tps': None,
    'numeric_current_to_credible_limit_gap': None,
    'next_data_for_dag': 'same-acquisition actual writer/reader branch, row generation and flow; all8 Target/Draft/HCCL/Host boundary and attainable mixed-service durations before any critical-path numerical claim',
}
assert all(out[k] is None for k in ('resource_hardware_floor_s', 'scheduling_execution_floor_s',
                                    'product_e2e_ceiling_tps', 'numeric_current_to_credible_limit_gap'))
path = E / 'run625/runtime_dag_v0.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'status': out['status'], **out['validations']}))
