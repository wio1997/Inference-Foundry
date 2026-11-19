"""Check a read-only P/D clock journal against the actual matched profile lifecycle."""

import argparse
from calendar import timegm
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path


ROLES = ('prefill', 'decode')
BEFORE_REASONS = {'checking_actual_idle_roles_before_profile',
                  'submitting_independent_matched_profile_driver_once'}
TERMINAL_PHASE = 'profile_workload_complete_pending_allrank_review'


def ns(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError('A clock boundary requires an explicit time zone')
    return timegm(stamp.utctimetuple()) * 1_000_000_000 + stamp.microsecond * 1000


def sample_window(row):
    raw = row['raw_clock_samples']
    if set(raw['roles']) != set(ROLES):
        raise ValueError('Expected two observed host clocks')
    measurements = raw['roles']['decode']['samples']
    if len(measurements) != 12 or len(raw['roles']['prefill']['samples']) != 12:
        raise ValueError('Unexpected number of calibrated round trips')
    role_ranges = {}
    for role,diag in raw['roles'].items():
        limits = [max(m['offset_lower_ns'] for m in diag['samples']),
                  min(m['offset_upper_ns'] for m in diag['samples'])]
        if limits != diag['remote_minus_collector_ns'] or limits[0] > limits[1]:
            raise ValueError('Invalid per-role offset bounds')
        role_ranges[role] = limits
    p,d = (role_ranges[role] for role in ROLES)
    low, high = raw['prefill_minus_decode_realtime_ns']
    if [low,high] != [p[0]-d[1],p[1]-d[0]] or low > high:
        raise ValueError('Invalid P-minus-D offset bounds')
    return {
        'lower_D_realtime_ns': min(value['remote_receive_realtime_ns'] for value in measurements),
        'upper_D_realtime_ns': max(value['remote_send_realtime_ns'] for value in measurements),
        'P_minus_D_bounds_ns': [low, high],
    }


def check_witness(value, gate, pid):
    return (isinstance(value, dict) and value['driver_PID']==pid
            and value['driver_gate_sha256']==gate)


def evaluate(rows):
    armed, = (row for row in rows if row['event']=='observer_armed')
    samples = [row for row in rows if row['event']=='paired_realtime_sample']
    errors = [row for row in rows if row['event']=='read_only_observation_error']
    windows = [(row,sample_window(row)) for row in samples]
    arm_verified = any(row['reason']=='observer_arm_transport_verification' for row,_ in windows)
    result = {'actual_observer_armed':True,'both_transport_paths_observed':arm_verified,
              'sample_interval_s':armed['interval_s'],'actual_paired_sample_count':len(samples),
              'read_error_count':len(errors),'profile_clock_windows_verified':False}
    handles = [r for r in rows if r['event']=='bound_driver_receipt']
    finished = [r for r in rows if r['event']=='observer_finished']
    if not handles or not finished:
        result['evidence_status']='awaiting_actual_driver_and_terminal_clock_sample'
        return result
    handle, = handles
    terminal, = finished
    pid,gate=handle['driver_PID'],handle['gate_sha256']
    if terminal['original_driver_PID'] != pid or terminal['actual_terminal']['phase'] != TERMINAL_PHASE:
        result['evidence_status']='actual_driver_did_not_report_completed_profile_workload'
        return result

    terminal_samples = []
    for row in samples:
        before, after = row['receiver_state_before'], row['receiver_state_after']
        for witness in (before,after):
            if witness is not None and not check_witness(witness,gate,pid):
                raise ValueError('Clock sample belongs to a different capture driver')
        if row['reason']=='actual_driver_terminal' and before and after and not (
            before['driver_exact_live'] or after['driver_exact_live']
        ):
            terminal_samples.append(row)
    if not terminal_samples:
        result['evidence_status']='missing_exact_post_exit_witness'
        return result
    closed=terminal_samples[-1]['receiver_state_after']
    if closed['capture_state'] != terminal['actual_terminal']:
        raise ValueError('Terminal phase does not match the final clock sample')
    controls=closed['durable_control_observations']
    wanted={role+'_'+act for role in ROLES for act in ('start_profile','stop_profile')}
    if set(controls) != wanted or not all(row['ok'] for row in controls.values()):
        result['evidence_status']='one_or_more_original_role_controls_unverified'
        return result
    start=[ns(controls[role+'_start_profile']['at_D_UTC']) for role in ROLES]
    stop=[ns(controls[role+'_stop_profile']['at_D_UTC']) for role in ROLES]
    closed_ns=ns(closed['capture_state']['at_D_UTC'])
    if not min(start) <= max(start) < min(stop) <= max(stop) <= closed_ns:
        raise ValueError('Profile control timestamps are inconsistent')
    samples_before=[]
    samples_during=[]
    samples_after=[]
    for row, observed in windows:
        lower,upper=observed['lower_D_realtime_ns'],observed['upper_D_realtime_ns']
        before,after=row['receiver_state_before'],row['receiver_state_after']
        if row['reason'] in BEFORE_REASONS and upper<=min(start):
            samples_before.append(observed)
        if before and after and before['driver_exact_live'] and after['driver_exact_live']:
            if (before['capture_state']['phase']==after['capture_state']['phase']=='profile_workload_running'
                and max(start)<=lower<=upper<=min(stop)):
                samples_during.append(observed)
        if row in terminal_samples and closed_ns<=lower:
            samples_after.append(observed)
    coverage = (len(samples_before),len(samples_during),len(samples_after))
    result.update(samples_before_control_start=coverage[0],samples_with_both_profiler_controls_open=coverage[1],
                  samples_after_original_driver_exit=coverage[2],actual_capture_state_sha256=closed['capture_state']['sha256'],
                  profile_clock_windows_verified=all(coverage),
                  evidence_status='three_windows_observed' if all(coverage) else 'one_or_more_clock_windows_missing')
    bounded=sorted(samples_before[-1:]+samples_during+samples_after[:1],key=lambda s:s['lower_D_realtime_ns'])
    if bounded:
        offsets=[row['P_minus_D_bounds_ns'] for row in bounded]
        result['observed_P_minus_D_offset_envelope_ns']=[min(l for l,_ in offsets),max(u for _,u in offsets)]
    if len(bounded)>1:
        gaps=[b['lower_D_realtime_ns']-a['upper_D_realtime_ns'] for a,b in zip(bounded,bounded[1:])]
        result['maximum_unsampled_gap_s']=max(gaps)/1e9
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path)
    args=parser.parse_args()
    raw=args.journal.read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    value=evaluate(rows)
    value['journal_sha256']=sha256(raw).hexdigest()
    value['scope']=('Sampled host realtime windows tied to the exact existing driver and control receipts. '
                    'No continuous bound on unobserved clock drift, no device-clock alignment and no profile acceptance claim.')
    print(json.dumps(value,indent=2,sort_keys=True))
    return 0 if value['profile_clock_windows_verified'] else 1


if __name__=='__main__':
    raise SystemExit(main())
