"""Validate a wide campaign's continuation and compare its Prefill boundaries."""

from datetime import datetime
from hashlib import sha256
import json

from compare_SG_prefill_compile_g1414 import compare


def remaining_seconds(stage, next_warmup, warm, previous, plan, now):
    epoch = plan['epoch']
    workload = plan['workload']
    width = workload['width_tokens']
    guard_budget = workload['remaining_CPU_and_pair_lease_required_at_formal_submit_seconds']
    old_gate = previous['gate']
    assert previous['run_id'] == warm['run_id'] == f'sg_pd_g{epoch}_aisbench_{width // 1024}k'
    assert previous['lengths'] == [width] and previous['warmup_passes'] == 5
    assert old_gate['profile_epoch'] == epoch and old_gate['input_tokens'] == width
    assert previous['phase_timeout_seconds'] == workload['warmup_phase_seconds']
    assert previous['total_timeout_seconds'] == workload['warmup_campaign_seconds']
    elapsed = (now-datetime.fromisoformat(previous['benchmark_started_at'])).total_seconds()
    assert 0 <= elapsed < workload['warmup_campaign_seconds']
    if stage == 'formal':
        assert next_warmup is None and warm['phase'] == 'warmup_diagnostics_complete_pending_compile_audit'
        assert warm['readiness_passed'] is True
        return guard_budget
    assert stage == 'remaining' and next_warmup in (4, 5)
    assert warm['phase'] == 'warmup_paused_for_P_snapshot' and warm['completed_warmups'] == next_warmup-1
    warm_budget = (6-next_warmup) * (workload['warmup_phase_seconds']+90) + 300
    assert workload['warmup_campaign_seconds']-elapsed >= warm_budget
    return guard_budget + warm_budget


def bound_prefill_snapshot(path, before_after, plan, role, previous):
    raw = path.read_bytes()
    data = json.loads(raw)
    summary = data['summary']
    width = plan['workload']['width_tokens']
    assert summary['boundary_label'] == f'after_warmup{before_after}_{width}'
    assert summary['epoch'] == plan['epoch'] and summary['P_controller_PID'] == role['handle']['pid']
    assert summary['P_container_ID'] == role['container']['Id']
    assert summary['verified_guard_PID'] == role['guard_PID'] and summary['guard_FD_verified'] is True
    assert summary['ready'] == 200 and summary['idle']
    assert all(values and all(v == 0 for v in values) for values in summary['idle'].values())
    assert summary['compiler_file_count'] == len(data['compiler_files'])
    assert datetime.fromisoformat(summary['at']) >= datetime.fromisoformat(previous['benchmark_started_at'])
    return data, sha256(raw).hexdigest()


def bound_prefill_comparison(before, after, plan, role, previous):
    p3, h3 = bound_prefill_snapshot(before, 3, plan, role, previous)
    p5, h5 = bound_prefill_snapshot(after, 5, plan, role, previous)
    result = compare(p3, p5)
    assert result['persistent_compiler_artifacts_unchanged']
    return {'before_snapshot_SHA256':h3, 'after_snapshot_SHA256':h5, **result}
