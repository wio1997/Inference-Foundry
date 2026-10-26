#!/usr/bin/env python3
"""CPU preflight for one actual-cache-row writer/eligible-reader lineage packet.

This composes only after a separate source-pinned ABI oracle admits the actual
reader eligible set. It never proves physical HBM reads or fresh necessary work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
PINS = {
    'run619_oracle': (E / 'run619/swa_eligible_domain.json',
                      '51869101193a2efa2240f83f2abb19d0063e19dd5dd00d6f5e4527d0eb6d37a0'),
    'run623_source': (E / 'run623/astra_next_packet_review.md',
                      'f42965ff744c86587f984f7ed85a7341e99519e16617aac87b6d68d1b72c35dc'),
    'run625_dag': (E / 'run625/runtime_dag_v0.json',
                   'c2b80720c0501a50fb605335d69a5d3657d6afe5b5f9a66e349046d007ac5b74'),
}
for key, (path, digest) in PINS.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(key + ' SHA drift')


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def nonneg_int(x: object, why: str) -> int:
    need(type(x) is int and 0 <= x <= (1 << 63) - 1, why)
    return x


def result(status: str, reason: str, *, selected_slot: int | None = None,
           writer_call: str | None = None) -> dict:
    return {
        'current_eligible_row_lineage': status,
        'reason': reason,
        'selected_physical_slot': selected_slot,
        'selected_context_writer_call': writer_call,
        'physical_first_read': None,
        'fresh_unavoidable_arithmetic': None,
        'compulsory_HBM_bytes': None,
        'strict_scheduling_floor_s': None,
        'strict_resource_floor_s': None,
    }


def classify(packet: dict) -> dict:
    """PASS means selected current context writer is last before eligible reader.

    Native order must come from same-acquisition exact flow. A caller cannot
    declare it from Host enqueue ordinal, pointer equality or equal row bytes.
    The gate validates the packet's shape/logic but does not authenticate flags.
    """
    need(isinstance(packet, dict), 'packet dict')
    capacity = nonneg_int(packet['cache_slot_capacity'], 'capacity integer')
    slot = nonneg_int(packet['selected_physical_slot'], 'selected slot integer')
    need(capacity > 0 and slot < capacity, 'selected slot outside cache')
    cache = packet['cache_identity']
    need(isinstance(cache, str) and cache, 'cache identity')
    selected_call = packet['selected_context_call']
    need(isinstance(selected_call, str) and selected_call, 'selected call')
    reader = packet['reader']
    need(isinstance(reader, dict), 'reader dict')
    rseq = nonneg_int(reader['native_seq'], 'reader native sequence')
    rstream = reader['native_stream']
    need(isinstance(rstream, str) and rstream, 'reader stream')
    need(isinstance(reader['cache_identity'], str) and reader['cache_identity'],
         'reader cache identity')
    need(isinstance(reader['eligible_slots'], list), 'eligible slot list')
    for x in reader['eligible_slots']:
        need(type(x) is int and 0 <= x < capacity, 'eligible slot outside cache')
    flags = ('native_flow_verified', 'allocation_lifetime_verified',
             'all_alias_writers_captured', 'reader_abi_oracle_verified',
             'reader_execution_coverage_verified')
    need(all(type(packet.get(k)) is bool for k in flags), 'certificate flags bool')
    snapshot = packet.get('snapshot_post_context_equals_pre_reader')
    need(snapshot is None or type(snapshot) is bool, 'snapshot flag bool or unknown')

    writers = packet['writers']
    need(isinstance(writers, list), 'writer list')
    normalized = []
    duplicate_dest_call = None
    invalid_formatted_slot_seen = False
    call_ids, native_keys = set(), {(rstream, rseq)}
    for event in writers:
        need(isinstance(event, dict), 'writer dict')
        call = event['call_id']
        need(isinstance(call, str) and call, 'writer call ID')
        need(call not in call_ids, 'duplicate writer call ID')
        call_ids.add(call)
        need(event['role'] in ('context', 'query', 'other'), 'writer role')
        seq = nonneg_int(event['native_seq'], 'writer native sequence')
        stream = event['native_stream']
        need(isinstance(stream, str) and stream, 'writer stream')
        need((stream, seq) not in native_keys, 'duplicate native stream/order key')
        native_keys.add((stream, seq))
        need(isinstance(event['cache_identity'], str) and event['cache_identity'],
             'writer cache identity')
        need(isinstance(event['slots'], list), 'writer actual formatted slot pairs')
        pair_values = []
        for pair in event['slots']:
            need(isinstance(pair, list) and len(pair) == 2 and
                 all(type(x) is int for x in pair), 'actual [block,offset] pair')
            block, offset = pair
            if block < 0 or offset < 0:
                # Actual non-A5 format of flat -1 under //32 and %32.
                need(block == -1 and offset == 31, 'unsupported invalid slot pair')
                invalid_formatted_slot_seen = True
                continue
            need(offset < 32 and block * 32 + offset < capacity,
                 'writer formatted slot outside cache')
            pair_values.append(block * 32 + offset)
        if len(pair_values) != len(set(pair_values)):
            duplicate_dest_call = call
        normalized.append((event, pair_values))

    # All schema and scope checks above precede any evidence-class return.
    if duplicate_dest_call is not None:
        return result('UNKNOWN', 'duplicate destination within one writer call',
                      selected_slot=slot, writer_call=duplicate_dest_call)
    if invalid_formatted_slot_seen:
        return result('UNKNOWN', 'formatted [-1,31] sentinel sink semantics not certified',
                      selected_slot=slot)
    if reader['cache_identity'] != cache:
        return result('UNKNOWN', 'reader cache allocation differs', selected_slot=slot)
    missing = [k for k in flags if not packet[k]]
    if missing:
        return result('UNKNOWN', 'missing certificates: ' + ','.join(missing), selected_slot=slot)
    if slot not in reader['eligible_slots']:
        return result('NO_CONTEXT_WITNESS', 'selected slot outside eligible reader set', selected_slot=slot)
    if snapshot is not True:
        return result('UNKNOWN', 'selected row snapshot parity absent or contradictory',
                      selected_slot=slot)

    selected = []
    hits = []
    for event, pair_values in normalized:
        call = event['call_id']
        seq = event['native_seq']
        stream = event['native_stream']
        if event['cache_identity'] != cache:
            if call == selected_call:
                return result('UNKNOWN', 'selected writer allocation differs', selected_slot=slot)
            continue
        if stream != rstream:
            return result('UNKNOWN', 'cross-stream order lacks existing dependency certificate',
                          selected_slot=slot)
        if seq > rseq:
            continue
        if call == selected_call:
            selected.append(event)
        if slot in pair_values:
            hits.append((seq, call, event['role']))
    if len(selected) != 1:
        return result('UNKNOWN', 'selected writer missing or duplicated', selected_slot=slot)
    chosen = selected[0]
    if chosen['role'] != 'context':
        return result('UNKNOWN', 'selected call is not context writer', selected_slot=slot)
    selected_seq = chosen['native_seq']
    if not any(seq == selected_seq and call == selected_call for seq, call, _ in hits):
        return result('NO_CONTEXT_WITNESS', 'selected context call did not write selected row',
                      selected_slot=slot, writer_call=selected_call)
    if max(hits)[1] != selected_call:
        return result('NO_CONTEXT_WITNESS', 'later writer replaced selected context row',
                      selected_slot=slot, writer_call=selected_call)
    return result('PASS_CURRENT_ELIGIBLE_LINEAGE',
                  'selected context writer is last certified same-stream writer before eligible reader',
                  selected_slot=slot, writer_call=selected_call)


BASE = {
    'cache_slot_capacity': 1090880,
    'selected_physical_slot': 37,
    'cache_identity': 'rank0-layer43-allocationG7',
    'selected_context_call': 'context64-layer43',
    'native_flow_verified': True,
    'allocation_lifetime_verified': True,
    'all_alias_writers_captured': True,
    'reader_abi_oracle_verified': True,
    'reader_execution_coverage_verified': True,
    'snapshot_post_context_equals_pre_reader': True,
    'reader': {'native_seq': 30, 'native_stream': 'stream47',
               'cache_identity': 'rank0-layer43-allocationG7', 'eligible_slots': [37, 38]},
    'writers': [
        {'call_id': 'context64-layer43', 'role': 'context', 'native_seq': 10,
         'native_stream': 'stream47', 'cache_identity': 'rank0-layer43-allocationG7',
         'slots': [[1, 5]]},
        {'call_id': 'query64-layer43', 'role': 'query', 'native_seq': 20,
         'native_stream': 'stream47', 'cache_identity': 'rank0-layer43-allocationG7',
         'slots': [[1, 6]]},
    ],
}


def variant(change) -> dict:
    p = json.loads(json.dumps(BASE))
    change(p)
    return p


fixtures = []


def check(name: str, expected: str, p: dict) -> None:
    got = classify(p)
    need(got['current_eligible_row_lineage'] == expected, name + ' status drift')
    need(all(got[k] is None for k in ('physical_first_read', 'fresh_unavoidable_arithmetic',
                                     'compulsory_HBM_bytes', 'strict_scheduling_floor_s',
                                     'strict_resource_floor_s')), name + ' overpromotion')
    fixtures.append({'name': name, 'status': expected})


def reject(name: str, p: dict) -> None:
    try:
        classify(p)
    except (ValueError, KeyError):
        fixtures.append({'name': name, 'status': 'REJECTED_SCHEMA'})
        return
    raise AssertionError(name + ' must reject')


check('selected_context_last_writer', 'PASS_CURRENT_ELIGIBLE_LINEAGE', BASE)
check('reader_mask_excludes_slot', 'NO_CONTEXT_WITNESS',
      variant(lambda p: p['reader'].update(eligible_slots=[38])))
check('query_overwrites_context', 'NO_CONTEXT_WITNESS',
      variant(lambda p: p['writers'][1].update(slots=[[1, 5]])))
check('context_did_not_write_selected', 'NO_CONTEXT_WITNESS',
      variant(lambda p: p['writers'][0].update(slots=[[1, 4]])))
check('missing_flow', 'UNKNOWN', variant(lambda p: p.update(native_flow_verified=False)))
check('missing_alias_coverage', 'UNKNOWN',
      variant(lambda p: p.update(all_alias_writers_captured=False)))
check('missing_lifetime', 'UNKNOWN',
      variant(lambda p: p.update(allocation_lifetime_verified=False)))
check('missing_reader_oracle', 'UNKNOWN',
      variant(lambda p: p.update(reader_abi_oracle_verified=False)))
check('missing_reader_execution_coverage', 'UNKNOWN',
      variant(lambda p: p.update(reader_execution_coverage_verified=False)))
check('cross_stream_without_edge', 'UNKNOWN',
      variant(lambda p: p['writers'][0].update(native_stream='stream48')))
check('different_reader_allocation', 'UNKNOWN',
      variant(lambda p: p['reader'].update(cache_identity='different-G')))
check('duplicate_destinations', 'UNKNOWN',
      variant(lambda p: p['writers'][0].update(slots=[[1, 5], [1, 5]])))
check('snapshot_mismatch', 'UNKNOWN',
      variant(lambda p: p.update(snapshot_post_context_equals_pre_reader=False)))
check('snapshot_unknown', 'UNKNOWN',
      variant(lambda p: p.update(snapshot_post_context_equals_pre_reader=None)))
check('actual_formatter_negative_slot_unsupported', 'UNKNOWN',
      variant(lambda p: p['writers'][0].update(slots=[[1, 5], [-1, 31]])))
reject('snapshot_non_boolean',
       variant(lambda p: p.update(snapshot_post_context_equals_pre_reader='False')))
reject('duplicate_native_key_disjoint_rows',
       variant(lambda p: p['writers'][1].update(native_seq=10)))
reject('selected_call_repeated_after_reader',
       variant(lambda p: p['writers'].append({**p['writers'][0], 'native_seq': 40})))
reject('malformed_other_cache_identity',
       variant(lambda p: p['writers'][1].update(cache_identity=None)))
reject('out_of_range_selected', variant(lambda p: p.update(selected_physical_slot=1090880)))
reject('malformed_pair', variant(lambda p: p['writers'][0].update(slots=[[1]])))
reject('partial_invalid_pair', variant(lambda p: p['writers'][0].update(slots=[[-1, 0]])))
reject('synthetic_invalid_encoding_not_installed_format',
       variant(lambda p: p['writers'][0].update(slots=[[-1, -1]])))
reject('out_of_range_reader', variant(lambda p: p['reader'].update(eligible_slots=[1090880])))
reject('non_boolean_certificate', variant(lambda p: p.update(native_flow_verified=1)))

out = {
    'status': 'CPU_selected_row_lineage_gate_preflight_only',
    'source_pins': {k: {'path': str(path.relative_to(ROOT)), 'sha256': digest}
                    for k, (path, digest) in PINS.items()},
    'fixture_count': len(fixtures),
    'fixtures': fixtures,
    'pass_meaning': 'one selected current writer is last before an ABI-eligible reader, conditional on externally authenticated same-acquisition native/coverage/lifetime/ABI flags; booleans and Run619 toy pin do not authenticate themselves; no physical read or mandatory work',
    'actual_W0_packet': None,
    'formal_current_tps': 571.681,
    'strict_resource_floor_s': None,
    'strict_scheduling_floor_s': None,
    'product_ceiling_tps': None,
    'numeric_current_to_credible_limit_gap': None,
}
path = E / 'run627/row_lineage_gate.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'status': out['status'], 'fixtures': len(fixtures)}))
