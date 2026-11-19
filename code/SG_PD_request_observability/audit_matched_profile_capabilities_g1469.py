"""Verify which request-level evidence the fixed G1317 capture can produce."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace


P_RECEIPT_SHA = 'dab24421a4d8c30bcaba354a3ffee604a4023acc4a896e8ce4a2124dfdc9d613'
D_RECEIPT_SHA = '815f8c26121e089374e745173cdd12d2d435c8ec629927000e756ebcd9316b59'
DRIVER_SHA = 'ab831dabbff37bd6c4337737807c074a3bc78d0407350f927f4335209c336635'
MARKER = 'G1469_TRACE_CAPABILITY '
TIME_STATS = 'observability/req_time_stats.py'
REQUEST = 'managers/schedule_batch.py'
STREAMER = 'managers/scheduler_components/output_streamer.py'
FIELDS = 'arg_groups/fields/observability.py'


def pinned_bytes(path, digest):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == digest
    return raw


def receipt(path, digest, role):
    packet = json.loads(pinned_bytes(path, digest))
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    line, = (x.removeprefix(MARKER) for x in packet['output'].splitlines() if x.startswith(MARKER))
    data = json.loads(line)
    assert data['role'] == role and data['epoch'] == 1317
    return data


def request_log_control(sources):
    modes = SimpleNamespace(PREFILL='prefill', DECODE='decode', NULL='unified')
    payloads = []
    logger = SimpleNamespace(info=payloads.append)
    functions = [sources[REQUEST]['functions']['log_time_stats']['source']]
    functions.extend(sources[TIME_STATS]['functions'][fn]['source']
                     for fn in ('convert_to_duration', 'disagg_mode_str'))
    namespace = {'logger': logger, 'DisaggregationMode': modes, 'SGLANG_TEST_REQUEST_TIME_STATS': True}
    source = '\n\n'.join(textwrap.dedent(fn) for fn in functions)
    exec(compile(source, '<fixed-request-logger>', 'exec'), namespace)

    for role in (modes.PREFILL, modes.DECODE):
        stats = SimpleNamespace(
            disagg_mode=role, get_queueing_time=lambda: 2, duration_between=lambda begin, end: end - begin,
            format_duration=lambda duration: f'{duration:.2f} s', format_wallclock=lambda stamp: str(stamp),
            wait_queue_entry_time=11, forward_entry_time=13, completion_time=21,
            bootstrap_done_time=4, prefill_bootstrap_queue_entry_time=1,
            decode_prealloc_queue_entry_time=2, decode_transfer_queue_entry_time=5,
            transfer_speed_gb_s=0.5, transfer_total_mb=96.25,
        )
        stats.convert_to_duration = lambda: namespace['convert_to_duration'](stats)
        stats.disagg_mode_str = lambda: namespace['disagg_mode_str'](stats)
        req = SimpleNamespace(has_log_time_stats=False, bootstrap_room=123456, rid='test-' + role,
                              origin_input_ids=range(32768), output_ids=[1], cached_tokens=0,
                              prefill_attempt_count=1, time_stats=stats)
        namespace['log_time_stats'](req)
        namespace['log_time_stats'](req)
    assert len(payloads) == 2
    assert all('bootstrap_room=123456' in row for row in payloads)
    assert 'rid=test-prefill' in payloads[0] and 'type=prefill' in payloads[0]
    assert 'rid=test-decode' in payloads[1] and 'type=decode' in payloads[1]
    assert 'forward_duration=8.00 s' in payloads[0] and 'transfer_speed=0.50 GB/s' in payloads[0]
    assert 'transfer_duration=6.00 s' in payloads[1] and 'alloc_wait_duration=1.00 s' in payloads[1]
    return {'both_roles_can_emit_bootstrap_room': True,
            'role_specific_request_IDs_appear_alongside_room': True,
            'overlap_duplicate_log_call_suppressed_within_each_request': True}


def audit(repository, p, d, driver):
    pin_path = repository / 'code/SG_PD_wait_recovery_G1317/G1317_runtime_source_index.json'
    launch_index = json.loads(pin_path.read_bytes())
    for row in (p, d):
        assert launch_index['roles'][row['role']]['candidate_config_SHA256'] == row['candidate_sha256']
        assert row['startup_resolved_trace_settings'] == {
            'enable_trace': ['false'], 'enable_request_time_stats_logging': ['false']}
        assert not row['trace_switches'] and not row['trace_environment_keys']
    assert d['profile_driver_sha256'] == DRIVER_SHA
    tag = repository / 'evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_stage.json'
    previous = json.loads(tag.read_bytes())
    assert previous['staged_source_and_reader'] and not previous['annotation_installed_in_service']
    controls = []
    for node in ast.walk(ast.parse(driver)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'control':
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                controls.append(node.args[1].value)
    assert set(controls) == {'start_profile', 'stop_profile'}
    assert "'detailed_annotations':True" in driver.decode()
    assert not any(name in driver.decode() for name in
                   ('decode_prefix_prompt_annotation', 'enable_request_time_stats_logging', 'enable_trace'))

    sources = p['source_summary']
    pins = {}
    for name in (TIME_STATS, REQUEST, STREAMER, FIELDS):
        left, right = sources[name], d['source_summary'][name]
        assert left['sha256'] == right['sha256']
        pins[name] = left['sha256']
    return {
        'epoch': 1317,
        'actual_startup_settings': {row['role']: row['startup_resolved_trace_settings'] for row in (p, d)},
        'actual_planned_profile_driver_sha256': d['profile_driver_sha256'],
        'planned_torch_profiler_actions': sorted(set(controls)),
        'planned_detailed_annotations': True,
        'existing_decode_prompt_marker_staged_only': True,
        'same_request_PD_transfer_boundary_covered_by_planned_capture': False,
        'fixed_native_log_output_CPU_controls': request_log_control(sources),
        'native_log_duration_boundaries': {
            'prefill_forward_duration': 'first forward entry to request completion',
            'decode_transfer_duration': 'Decode transfer queue entry to wait queue entry',
        },
        'source_sha256': pins,
        'staged_decode_marker_evidence_sha256': sha256(tag.read_bytes()).hexdigest(),
        'scope': ('Startup and current staged-driver verification. CPU controls use the fixed log formatting '
                  'functions with isolated request metadata. Real cross-role identifiers and transport timing '
                  'still require a separate instrumented run.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'prefill_receipt', 'decode_receipt', 'driver'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    p = receipt(args.prefill_receipt, P_RECEIPT_SHA, 'prefill')
    d = receipt(args.decode_receipt, D_RECEIPT_SHA, 'decode')
    result = audit(args.repository, p, d, pinned_bytes(args.driver, DRIVER_SHA))
    result['runtime_source_receipts_sha256'] = {'prefill': P_RECEIPT_SHA, 'decode': D_RECEIPT_SHA}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
