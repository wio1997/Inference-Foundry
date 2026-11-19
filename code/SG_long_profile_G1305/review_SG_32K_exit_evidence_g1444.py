"""Bind accepted 32K measurements to the same quality worker and closed P/D capture."""

from datetime import datetime
from hashlib import sha256
import json

from intake_SG_matched_profile_g1431 import identity_hash, pair_receipt


def summarize(formal, quality, driver, closed, identity, roles):
    quality_link = quality['actual_G1317_terminal_binding']
    upstream_quality = driver['quality_terminal']
    assert formal['epoch'] == quality_link['epoch'] == identity['epoch'] == 1317
    assert formal['run_id'] == 'sg_pd_g1317_aisbench_32k_g1324'
    assert formal['accepted_cold_and_replay_requests'] == [32, 32]
    assert formal['prefill_cold_local_hit_delta'] == 0
    assert quality_link['formal_audit_SHA256'] == formal['accepted_formal_audit_SHA256']
    assert quality_link['terminal_control_SHA256']['terminal.json'] == closed['quality_terminal_SHA256']
    assert upstream_quality['controller_PID'] == closed['quality_controller_PID']
    assert upstream_quality['terminal_at'] == closed['quality_terminal_at']
    assert upstream_quality['worker_exit_code'] == 0
    assert driver['driver_PID'] == closed['driver_PID'] and closed['exact_capture_driver_has_exited']
    assert identity['control_record_sha256']['gate.json'] == driver['gate_SHA256']
    assert identity['capture_record_sha256']['state.json'] == closed['capture_state_SHA256']
    assert datetime.fromisoformat(closed['capture_end_UTC']) >= datetime.fromisoformat(upstream_quality['terminal_at'])
    assert identity['bound_input_records'] == 16

    traces = pair_receipt(identity, roles)
    quality_complete = (quality['both_runs_all_request_bytes_verified'] == 1319
                        and quality['SGLang_all_transport_and_no_length_limit']
                        and quality['paired_official_grades']['valid_untruncated_pairs'] == 1319)
    raw_complete = traces['paired_8_plus_8_raw_files_present']
    return {
        'epoch': 1317,
        'formal_audit_SHA256': formal['accepted_formal_audit_SHA256'],
        'quality_terminal_SHA256': closed['quality_terminal_SHA256'],
        'quality_report_SHA256': sha256(json.dumps(quality, sort_keys=True).encode()).hexdigest(),
        'quality_cases_verified': quality['both_runs_all_request_bytes_verified'],
        'quality_all_1319_comparable': bool(quality_complete),
        'profile_driver_PID': driver['driver_PID'],
        'capture_identity_SHA256': identity_hash(identity),
        'trace_inventory': traces,
        'prior_phase_evidence_complete': bool(quality_complete and raw_complete),
    }
