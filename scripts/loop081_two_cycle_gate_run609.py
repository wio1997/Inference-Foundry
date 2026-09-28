#!/usr/bin/env python3
"""Read-only two-cycle packet gate for the next fixed-DSpark7 Bound acquisition.

This binds a measured prior W0 stratum and source contracts. It does not
measure a new schedule, device readiness, compulsory work, or a time floor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path('/data/wio/Inference_Foundry')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
ARM = ROOT / 'evidence/20260928_loop081_bound/run606/live/b'
OUT = ROOT / 'evidence/20260928_loop081_bound/run609/two_cycle_packet_gate.json'
COHORT = 5
CYCLES = (64, 65)
BASIS_SHA = '5bbbbb3492d75c7c37679d911d1986174ddf8d3ebb599e982a90b894847defe7'
SOURCE_PINS = {
    ROOT / 'runtime/extreme_decode.py': ('eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499',
        ('target_output = self.target.execute(self.state)', 'acceptance_output = self.acceptance.execute(',
         'self._state_machine.advance_state(acceptance_output)', 'next_draft = self.proposer.execute(',
         'self.state.draft_tokens.copy_(next_draft)')),
    ROOT / 'runtime/target_adapter.py': ('c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5',
        ('model_output = self.binding.forward(', 'logits = self.binding.compute_logits(sample_hidden)')),
    ROOT / 'runtime/fixed_acceptance.py': ('824289bae3bd31676e3a697cc59b4169a345c70e10063c5660763e3e3557c6ae',
        ('predicted = target.logits.argmax(dim=-1)', 'accepted, counts = greedy_accept(')),
    ROOT / 'runtime/fixed_decode.py': ('7d74f4cbba380ff0a9296cc06d86e0cb9c9c921b6213b7965700831a29a6bafa',
        ('state.last_sampled_tokens.copy_(', 'state.num_computed_tokens.add_(state.num_sampled)')),
    ROOT / 'bootstrap/vllm_dspark_handoff.py': ('fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e',
        ('self._commit_host_mirrors()', 'self._launch_host_count_copy(state.num_sampled)',
         'self.proposer.prepare_inputs_padded(', 'hidden[token_indices]', 'next_draft = self.proposer._propose(')),
    ASC / 'spec_decode/dspark_proposer.py': ('e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f',
        ('self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states',
         'num_query_total = batch_size * self.num_query_per_req')),
    ASC / 'spec_decode/dflash_proposer.py': ('6be46559f9f869861c676efb4531b0c01eef8a0445521ecadf5c1fec1c4412f8',
        ('self.model.precompute_and_store_context_kv(',)),
    ASC / 'spec_decode/llm_base_proposer.py': ('e3e1ff579e67f845f155c983294ae5546e6ee76ba330098d7867e870c3f9f6e4',
        ('target_hidden_states = self.model.combine_hidden_states(', 'self.set_inputs_first_pass(')),
    ASC / 'models/deepseek_v4_dspark.py': ('b459fa373e89da1722085cffa63a9959502dd29fb1a0cf4b8f073e431d811867',
        ('def precompute_and_store_context_kv(', 'if context_states.numel() == 0 or context_slot_mapping is None:',
         'if slot_mapping is None or slot_mapping.numel() == 0:',
         'DeviceOperator.dsa_kv_compress_scatter(')),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reduce_rank_basis(rank: int, admission: dict) -> dict:
    path = ARM / 'basis' / f'rank{rank}_cohort{COHORT}.json'
    relative = str(path.relative_to(ROOT))
    need(admission['source_sha256'][relative] == sha(path), f'basis raw SHA drift rank {rank}')
    record = load(path)
    need((record['rank'], record['cohort'], record['run_ts']) ==
         (rank, COHORT, 'LOOP081-RUN606-B'), f'identity rank {rank}')
    need(record['cycles'] > max(CYCLES) and len(record['count_history']) == record['cycles']
         and len(record['branch_history']) == record['cycles'], f'cycle ledger rank {rank}')
    need(len(record['req_ids']) == 12, f'c12 request cardinality rank {rank}')
    selected = []
    for cycle in CYCLES:
        branch = record['branch_history'][cycle]
        counts = record['count_history'][cycle]
        need(branch == {'cycle': cycle, 'scheduled_target': False,
                        'schedule_mode': 'off'}, f'prior branch stratum rank {rank} cycle {cycle}')
        need(len(counts) == 12 and all(isinstance(x, int) and 1 <= x <= 8 for x in counts),
             f'prior active/count stratum rank {rank} cycle {cycle}')
        selected.append({'cycle': cycle, 'accepted_counts': counts,
                         'accepted_sum': sum(counts), 'branch': branch})
    witnesses = [w for w in record['draft_model_witnesses'] if w['cycle'] == 64]
    need(len(witnesses) == 1, f'sparse cycle64 witness rank {rank}')
    witness = witnesses[0]
    need(witness['num_query_total'] == 84 and witness['num_context'] == 96 and
         witness['num_query_per_req'] == 7 and witness['sample_from_anchor'] is True,
         f'fixed Draft geometry rank {rank}')
    contexts, queries = witness['group_context_slots'], witness['group_query_slots']
    need(set(contexts) == set(queries) == {'2', '3'}, f'prior group labels rank {rank}')
    overlap = {}
    for group in sorted(contexts):
        c, q = contexts[group], queries[group]
        need(len(c) == 96 and len(q) == 84 and all(x >= 0 for x in c + q),
             f'prior slot shape/validity rank {rank} group {group}')
        per_request_overlap = [len(set(c[8*i:8*(i+1)]) & set(q[7*i:7*(i+1)]))
                               for i in range(12)]
        need(per_request_overlap == [8 - n for n in selected[0]['accepted_counts']],
             f'prior same-request slot relation rank {rank} group {group}')
        overlap[group] = {'context_slots': len(c), 'query_slots': len(q),
                          'same_numeric_slot_ids': len(set(c) & set(q)),
                          'context_unique': len(set(c)), 'query_unique': len(set(q)),
                          'same_request_intersection': per_request_overlap}
    return {'rank': rank, 'basis_path': relative, 'basis_sha256': sha(path),
            'req_ids': record['req_ids'], 'selected': selected,
            'cycle64_slot_labels': overlap,
            'cycle65_slot_witness_available': any(w['cycle'] == 65 for w in record['draft_model_witnesses'])}


def build() -> dict:
    source = {}
    for path, (expected, predicates) in SOURCE_PINS.items():
        need(sha(path) == expected, f'source SHA drift {path}')
        code = path.read_text()
        need(all(fragment in code for fragment in predicates), f'source anchor drift {path}')
        source[str(path)] = expected
    need(sha(ARM / 'basis_admission.json') == BASIS_SHA, 'Basis admission drift')
    admission = load(ARM / 'basis_admission.json')
    need(admission['status'] == 'diagnostic_compact_basis_all8_admitted' and
         admission['run_ts'] == 'LOOP081-RUN606-B', 'Basis admission identity')
    rows = [reduce_rank_basis(rank, admission) for rank in range(8)]
    for row in rows[1:]:
        need(row['req_ids'] == rows[0]['req_ids'] and
             row['selected'] == rows[0]['selected'] and
             row['cycle64_slot_labels'] == rows[0]['cycle64_slot_labels'],
             'all8 prior stratum mismatch')
    need(all(not row['cycle65_slot_witness_available'] for row in rows),
         'cycle65 sparse witness status changed')
    return {
        'status': 'prior_W0_two_cycle_packet_design_gate_pass',
        'scope': 'Run606 prior W0 cohort5 cycles64/65; source-typed packet design only',
        'source_sha256': source,
        'basis_admission_sha256': BASIS_SHA,
        'allrank': rows,
        'selected_prior_w0': {'cohort': COHORT, 'cycles': list(CYCLES),
                              'req_ids': rows[0]['req_ids'],
                              'accepted_sums': [x['accepted_sum'] for x in rows[0]['selected']]},
        'slot_overlap_interpretation': 'numeric slot labels overlap within cycle64, with per-request count 8 minus accepted count in this prior W0; actual per-layer write, cache storage identity, generation, consumer, and mandatory hazard remain unproved; alternate storage/layout is admissible',
        'next_live_packet_required_nodes': [
            'Target FULL replay with Graph generation and all8 native task IDs',
            'Target forward/aux last writer and actual stream-ready event',
            'Target logits and greedy acceptance raw counts',
            'state advance and next-token seed',
            'Draft hidden combine and per-layer KV projection/scatter with slot valid count, cache identity and version',
            'Draft query first actual consumer and head/commit',
            'next cycle Target input/collective first use; Target66 boundary if Draft65 successor-consumption closure is claimed',
            'Host enqueue/return and stream wait plus HCCL message identity',
        ],
        'admission_gates': [
            'new run all8 warm48/measured48 Product/Basis parity and exact output IDs',
            'predetermined cohort5 cycle64/65 active normal FULL96 stratum; verify actual new-W0 Graph replay and shape, else reject without substitution',
            'prior W0 sums47/59 and slot overlap49 are study priors, not acceptance gates for new W0',
            'canonical source/Graph generation and typed actual last writer/first consumer',
            'cache and slot storage generation before pointer reuse; cross-stream joins retained',
            'no added synchronization, algorithm change, collective or model work',
            'raw native/Host traces saved and rehashed before parsing; source restore and owned service stop',
            'separate A0/probe/A1 timing gate; prior W0 is not same-state timing control',
        ],
        'strict_resource_latency_floor_s': None,
        'strict_scheduling_latency_floor_s': None,
        'product_tps_ceiling': None,
        'formal_current_tps': 571.681,
    }


def main():
    if OUT.exists():
        raise ValueError('Run609 output already exists')
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'],
                      'accepted_sums': result['selected_prior_w0']['accepted_sums'],
                      'cycle64_slot_labels': result['allrank'][0]['cycle64_slot_labels']}))


if __name__ == '__main__':
    main()
