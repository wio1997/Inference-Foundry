"""Review the closed P/D trace bound to the independent GSM8K regrade."""

from datetime import datetime
from hashlib import sha256
import json


def summarize(quality_path, driver, closed, identity, traces):
    raw = quality_path.read_bytes()
    proof_sha = sha256(raw).hexdigest()
    assert proof_sha == driver['quality_proof_SHA256']
    quality = json.loads(raw)
    binding = quality['actual_G1317_independent_terminal_binding']
    upstream = driver['quality_terminal']
    assert binding['epoch'] == identity['epoch'] == 1317
    assert binding['original_scored_attempt_accepted'] is False
    assert binding['quality_terminal_controls_sha256']['terminal.json'] == closed['quality_terminal_SHA256']
    assert upstream['controller_PID'] == closed['quality_controller_PID'] == 3118006
    assert upstream['terminal_at'] == closed['quality_terminal_at'] and upstream['worker_exit_code'] == 0

    assert driver['driver_PID'] == closed['driver_PID'] and closed['exact_capture_driver_has_exited']
    assert identity['capture_prefix'] == 'matched_32k_g1498'
    assert identity['control_record_sha256']['gate.json'] == driver['gate_SHA256']
    assert identity['capture_record_sha256']['state.json'] == closed['capture_state_SHA256']
    assert datetime.fromisoformat(closed['capture_end_UTC']) >= datetime.fromisoformat(upstream['terminal_at'])
    assert identity['bound_input_records'] == 16
    packed_identity = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    identity_sha = sha256(packed_identity).hexdigest()
    assert traces['capture_identity_sha256'] == identity_sha
    assert set(traces['role_status']) == {'prefill', 'decode'}

    quality_complete = (quality['both_runs_all_request_bytes_verified'] == 1319
                        and quality['SGLang_all_transport_and_no_length_limit']
                        and quality['paired_official_grades']['valid_untruncated_pairs'] == 1319)
    raw_complete = traces['paired_8_plus_8_raw_files_present'] is True
    return {
        'epoch': 1317,
        'original_scored_attempt_accepted': False,
        'quality_terminal_SHA256': closed['quality_terminal_SHA256'],
        'quality_report_SHA256': proof_sha,
        'quality_all_1319_comparable': bool(quality_complete),
        'profile_driver_PID': driver['driver_PID'],
        'capture_identity_SHA256': identity_sha,
        'trace_inventory': traces,
        'independent_quality_and_raw_trace_evidence_complete': bool(quality_complete and raw_complete),
        'parsed_event_coverage_requires_review': True,
    }
