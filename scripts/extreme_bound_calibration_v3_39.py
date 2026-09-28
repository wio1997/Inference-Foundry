#!/usr/bin/env python3
"""V3.39 native replay identity, without promoting instrumented time to Bound."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
RUN611 = ROOT / 'evidence/20260928_loop081_bound/run611'
RUN612 = ROOT / 'evidence/20260928_loop081_bound/run612'
FILES = {
    'v3_38': RUN611 / 'bound_calibration_v3_38.json',
    'replay': RUN611 / 'native_graph_replay.json',
    'replay_review': RUN611 / 'astra_native_replay_review.md',
    'bound_review': RUN611 / 'astra_bound_review_v3_38.md',
}
PINS = {
    'v3_38': '8206db47c4a36ee9c61090d8c3415f1cb48a1397d7f0bdbe2ef2eebaedcb546b',
    'replay': '9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d',
    'replay_review': 'b2816262812aa8dee3077a256e821236acb4a53fe827057c272734c83332b45a',
    'bound_review': 'cf086cacf39137f5f6f33643929cd5070c520f66de5adbae8b83042123ca21e5',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


for key, path in FILES.items():
    need(sha(path) == PINS[key], f'input SHA drift {key}')
model = json.loads(FILES['v3_38'].read_text())
replay = json.loads(FILES['replay'].read_text())
need(model['bound_semantics_v3_38']['current_formal_tps'] == 571.681 and
     replay['status'] == 'all8_three_native_model45_replay_occurrences_pass' and
     len(replay['rows']) == 24, 'base/replay scope')
need({r['rank'] for r in replay['rows']} == set(range(8)) and
     {r['ordinal_in_profile'] for r in replay['rows']} == {0, 1, 2} and
     all(r['static_task_keys'] == r['native_event_count'] == 5412 and
         r['hcom_named_native_events'] == 260 and
         r['last_device_after_target_host_scope_us'] > 0 for r in replay['rows']),
     'all8 three-replay invariants')
spans = [r['native_span_us_instrumented'] for r in replay['rows']]
after = [r['last_device_after_target_host_scope_us'] for r in replay['rows']]
model['model_revision'] = 'v3.39_native_graph_occurrence_current_only'
model['bound_semantics_v3_39'] = {
    'active_objective': 'minimum Product E2E Runtime for fixed DSpark7 algorithm, acceptance/cycle trajectory, output and model work',
    'formal_current_tps': 571.681,
    'native_graph_identity': {
        'scope': 'instrumented Run610/611 same-W0 Level0 exported Model45 native X events',
        'ranks': 8, 'replays_per_rank': 3,
        'static_task_keys_per_replay': 5412,
        'hcom_named_events_per_replay': 260,
        'exported_native_event_envelope_us_range': [min(spans), max(spans)],
        'last_exported_event_after_target_host_scope_us_range': [min(after), max(after)],
        'method': 'static task tuple × occurrence and stream1/task0 temporal anchor; connection_id alone is invalid because it is reused across replays',
        'limit': 'last event may be zero-duration NOTIFY_RECORD; exported envelope is not certified physical completion or removable wall-time; no typed KV/device-ready/Host correlation certificate'
    },
    'strict_resource_hardware_floor_s': None,
    'strict_scheduling_execution_floor_s': None,
    'strict_product_e2e_tps_ceiling': None,
    'numeric_current_to_credible_limit_gap': None,
    'resource_proof_required': 'matched fixed-W0 compulsory fresh work/traffic plus certified exact-board upper cumulative compute/HBM/HCCL capacity envelopes C_plus,B; attained service is conditional Engineering evidence only',
    'scheduling_proof_required': 'actual typed KV writer-generation→first-consumer and collective/Host/device-ready edges, legal alternate schedule with shared-resource contention, observer OFF/ON/OFF transfer',
    'next_action': 'use same-W0 parsed eager op ranges and native tasks to type three-layer context vs query KV scatter generations and first query reader; audit rank7 eager flow shortfall before claiming complete Host→native pairing',
    'independent_review': 'Astra High Run611 V3.38 Bound review plus native replay scoped review'
}
need(all(model['bound_semantics_v3_39'][key] is None for key in
         ('strict_resource_hardware_floor_s', 'strict_scheduling_execution_floor_s',
          'strict_product_e2e_tps_ceiling', 'numeric_current_to_credible_limit_gap')),
     'strict endpoints fail closed')
RUN612.mkdir(parents=True, exist_ok=True)
out = RUN612 / 'bound_calibration_v3_39.json'
out.write_text(json.dumps(model, indent=2) + '\n')
print(json.dumps({'status': 'v3_39_native_identity_current_only',
                  'strict_endpoints': 'null', 'rows': 24}))
