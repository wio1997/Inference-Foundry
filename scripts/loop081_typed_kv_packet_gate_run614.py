#!/usr/bin/env python3
"""Fail-closed design for a bounded fixed-DSpark7 typed-KV witness.

Read-only design check. This does not arm a profiler or a live service.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
R09 = ROOT / 'evidence/20260928_loop081_bound/run609'
R13 = ROOT / 'evidence/20260928_loop081_bound/run613'
R14 = ROOT / 'evidence/20260928_loop081_bound/run614'
PIN = {
    'v3_40': (R13 / 'bound_calibration_v3_40.json',
              'b48c02d069cd31209b94f2bc08b9f53f3e41be3281d80ba0542b153f187c5bfd'),
    'flow': (R13 / 'eager_flow_join.json',
             'dc17ab40f85d38d2d5f87d921d44d1b1f4debb51dea6aceddd9b9ae0ad9c99b9'),
    'kv_review': (R13 / 'astra_kv_review.md',
                  'a829aac896b1d8cf45c9f4e3623b015f781fd982bd7b0c0ac37c02f399bda8be'),
    'resource_review': (R14 / 'astra_resource_gate.md',
                        'f04c711bb1c12f23106c112156bd936fc99d8f6d71a3d1807bb031d3355a2291'),
    'resource_inputs': (R14 / 'astra_resource_inputs.json',
                        'dded1eb489f7b51ee688e04a9395ba35fda986f63d8fc7d3bed01dddc76401f2'),
    'prior_pair': (R09 / 'two_cycle_packet_gate.json',
                   '5727e9e6738108e515e46c3ef72b42e33b03b1131d075ac03ed9181ac3416097'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for key, (path, expected) in PIN.items():
    if sha(path) != expected:
        raise ValueError(f'input SHA drift {key}')
base = json.loads(PIN['v3_40'][0].read_text())
flow = json.loads(PIN['flow'][0].read_text())
prior = json.loads(PIN['prior_pair'][0].read_text())
if (base['bound_semantics_v3_40']['formal_current_tps'] != 571.681 or
        len(flow['rows']) != 24 or
        prior['status'] != 'prior_W0_two_cycle_packet_design_gate_pass'):
    raise ValueError('fixed contract / prior identity')

packet = {
    'status': 'typed_kv_two_cycle_design_not_live_armed',
    'contract': {
        'model': 'DeepSeek V4 Flash W4A8', 'hardware': '8xAscend 910B3',
        'parallel': 'DP1xTP8', 'draft': 'DSpark7',
        'product': 'warm48 -> measured48, 32K->1024, c12',
        'algorithm_acceptance_cycle_output': 'frozen; measurements descriptive only',
        'formal_current_tps': 571.681,
    },
    'source_evidence': {key: {'path': str(path.relative_to(ROOT)), 'sha256': digest}
                        for key, (path, digest) in PIN.items()},
    'selection': {
        'candidate': 'one all8 measured cohort5 adjacent ordinary active pair, nominal cycles64/65 plus predecessor retained KV and successor Target66/first reader',
        'new_W0_rule': 'do not gate on Run606 accepted47/59, 49 numeric slot intersections, or any prior output trajectory',
        'fallback': 'if current W0 lacks active/read-valid context row, return NO_WITNESS; do not force acceptance, cycle count or another model call',
        'join': 'rank/cohort/cycle/request/layer/role, ContextVar ownership, exact installed source hashes and existing Basis/Product/dispatch lineage',
    },
    'typed_capture': {
        'context_scatter': ['layer43/44/45 call tag', 'request and canonical context input/prefix/position',
                            'SWA cache object token and separate allocation lifetime generation, content generation, per-row last-writer generation',
                            'storage base/offset/nbytes/dtype/shape/stride',
                            'slot-buffer allocation-generation and actual device slot values',
                            'valid row/slot generation, stream and native launch correlation'],
        'query_scatter': ['same cache/storage/slot/generation fields', 'overwrite rows and generation',
                          'actual prefill/decode branch and quant method'],
        'pre_attention': ['ori_kv actual storage generation', 'block table', 'seq lengths',
                          'window/mask/sparse indices and complete ABI-semantic eligible read domain',
                          'physical first-read claim requires independently validated loaded sparse-op implementation',
                          'per-row last-writer after query overwrite'],
        'boundaries': ['Target auxiliary producer and acceptance/state readiness',
                       'Host count-copy63->64/64->65', 'Draft64->Target65',
                       'Draft65->Target66/first consumer', 'HCCL/stream waits'],
        'freshness': ['window-entry reusable results and retained cache credit',
                      'cross-request/state/prefix semantic equality audit',
                      'one actually consumed unoverwritten layer43 context projection',
                      'input, actual weight/quant/ownership and output identity',
                      'frozen output ledger to next Target use'],
        'resource_scratch': 'extra selected-row operand copies <=128KiB/rank; scheduling metadata budget separate and preflighted',
        'asynchronous_capture_order': 'snapshot on each original producer/consumer stream after producer and before overwrite; no new cross-stream wait; copy to Host only after the existing drain',
        'ownership_link': 'freeze Host role/layer/request/submission token at enqueue and join its CANN torch_to_npu flow to the exact native task; never assign device work by Host cycle timestamp bin',
    },
    'hard_gates': [
        'new source patch must be reversible, SHA-pinned, compile/selftest before install and restore after stop',
        'preallocated bounded device witness; no new stream synchronization, model call, HCCL or algorithm change',
        'all8 exact current-W0 client/Basis/Product/dispatch joins and bounded typed records',
        'correctness includes request IDs/output ledger plus KV/state/acceptance semantic checks; any mismatch INVALID',
        'negative fixtures reject false freshness promotion for stale generation, pre-entry reuse, cross-request duplicate, query overwrite, invalid/masked row and missing successor consumer',
        'Astra preflight prior to live; diagnostic timing never formal; clean tagged stop/all8 idle and source/script restoration',
    ],
    'bound_admission': {
        'scheduling_packet': 'can PASS exact identity even when a legal reused or overwritten row produces no fresh-work witness; preserve typed current DAG evidence',
        'actual_storage_hazard': 'requires typed allocation+content+row-writer generation and actual ABI-semantic consumed rows; numeric slots or source alias insufficient',
        'conditional_partial_W_minus': 'separate PASS/NO_WITNESS/UNKNOWN gate in explicitly declared online fixed-expression class; requires consumed fresh contraction, input/prefix semantic key and complete entry-credit proof, actual quant/ownership units. A single row does not by itself yield a positive time floor if W_minus<=B',
        'compulsory_HBM_bytes': None,
        'exact_board_certified_C_plus_B': None,
        'strict_resource_floor_s': None,
        'strict_scheduling_floor_s': None,
        'strict_product_tps_ceiling': None,
        'numeric_current_to_credible_limit_gap': None,
    },
    'decision_after_packet': 'A proved hazard/independence chooses the smallest legal one-edge schedule intervention with same-state correctness and observer OFF/ON/OFF, then repeated formal E2E only if a validated intervention exists. Keep Resource C_plus,B search separate.',
}
R14.mkdir(parents=True, exist_ok=True)
out = R14 / 'typed_kv_packet_gate.json'
out.write_text(json.dumps(packet, indent=2) + '\n')
print(json.dumps({'status': packet['status'], 'strict_endpoints': 'null'}))
