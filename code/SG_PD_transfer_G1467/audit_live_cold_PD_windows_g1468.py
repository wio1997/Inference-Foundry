"""Compare coarse timing windows in actual G1317 P/D logs during the cold run."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re


P_LOG_SHA = '218923928cbecc0b9f1b12a0017eab3c582da1998077551ef29990b513f1221e'
D_LOG_SHA = '4b89ed7555060698afa0b04e72d7c525cc64610d2e11a746a7d6fe3aa666b3f1'
EARLY_CLOCK_SHA = 'c77901e0ad6934e421ea7a97853f4ef61e6c64e0f94af5d27da4fbc1fe37f7db'
LATE_CLOCK_SHA = '7ca7fe067ce94334f9567cd33ce33d8140cbfb176daa07a9ca77eec9150c9ca3'
COLD_LIVE_SHA = '9ee90c92bc27692c3ec6216672956d3efa77d2a3eefefdabe11f0f4129af2a01'
D_REPORT_GAP = 60
TIME = re.compile(r'^\[([\d-]+ [\d:]+) TP0 EP0\]')
ID_FIELD = re.compile(r'\b(?:request_id|rid|bootstrap_room|transfer_id)\s*[:=]', re.I)


def load(path, expected):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == expected
    return json.loads(raw)


def unwrap(path, expected, marker):
    packet = load(path, expected)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    record, = (row.removeprefix(marker+' ') for row in packet['output'].splitlines()
               if row.startswith(marker+' '))
    return json.loads(record)


def batch_reports(rows, title, cutoff):
    reports = []
    for row in rows:
        if title not in row:
            continue
        match = TIME.search(row)
        assert match
        stamp = datetime.fromisoformat(match[1]).replace(tzinfo=timezone.utc)
        if stamp > cutoff:
            fields = {key: int(value) for key, value in re.findall(r'#([a-z-]+): (\d+)', row)}
            reports.append((stamp, fields))
    assert all(a[0] < b[0] for a, b in zip(reports, reports[1:]))
    return reports


def same_log_window(prefill, decode, cold, early_clock, late_clock):
    assert prefill['epoch'] == decode['epoch'] == cold['epoch'] == 1317
    assert prefill['role'] == 'prefill' and decode['role'] == 'decode'
    assert cold['runner_PID'] == 3090471 and cold['current_phase'] == 'cold'
    offsets = [data['prefill_minus_decode_realtime_ns'] for data in (early_clock, late_clock)]
    for data in (early_clock, late_clock):
        assert all(len(node['samples']) == 12 for node in data['roles'].values())
    lower_ns = min(lower for lower, _ in offsets)
    upper_ns = max(upper for _, upper in offsets)
    lower, upper = lower_ns / 1e9, upper_ns / 1e9
    assert 19 < lower <= upper < 21
    cold_seen = datetime.fromisoformat(cold['at_UTC'])
    p_rows = prefill['reports']['engine']['selected_lines']
    d_rows = decode['reports']['engine']['selected_lines']
    p = batch_reports(p_rows, 'Prefill batch', cold_seen + timedelta(seconds=upper))
    d = batch_reports(d_rows, 'Decode batch', cold_seen)
    assert all(row['new-seq'] == 1 and row['new-token'] == 8192 and row['cached-token'] == 0 for _, row in p)

    inflight_runs = []
    for begin, row in p:
        if row['inflight-req'] == 0:
            continue
        assert row['inflight-req'] == 1
        if not inflight_runs or (begin - inflight_runs[-1][-1]).total_seconds() > 180:
            inflight_runs.append([])
        inflight_runs[-1].append(begin)
    bursts = []
    for begin, row in d:
        assert row['running-req'] == 1 and row['transfer-req'] == 15
        if not bursts or (begin - bursts[-1][-1]).total_seconds() > D_REPORT_GAP:
            bursts.append([])
        bursts[-1].append(begin)
    assert len(inflight_runs) == len(bursts) == 5 and all(len(run) == 2 for run in inflight_runs)
    rows = []
    for ordinal, ((p_first, p_second), burst) in enumerate(zip(inflight_runs, bursts, strict=True), 1):
        d_first = burst[0]
        from_first = (d_first - p_first).total_seconds()
        until_second = (p_second - d_first).total_seconds()
        # Log timestamps have whole-second precision. Assume only the offsets
        # seen in the two short sampled windows for this approximate alignment.
        relative = [from_first + lower - 1, from_first + upper + 1]
        before_second = [until_second - upper - 1, until_second - lower + 1]
        assert -1 < relative[0] and relative[1] < 80
        assert before_second[0] > 70
        rows.append({
            'chronological_window': ordinal,
            'prefill_first_and_second_positive_inflight_report_host_times': [p_first.isoformat(), p_second.isoformat()],
            'decode_first_report_host_time': d_first.isoformat(),
            'decode_reports': len(burst),
            'decode_first_vs_prefill_first_report_approx_seconds': [round(x, 3) for x in relative],
            'decode_first_before_prefill_second_report_approx_seconds': [round(x, 3) for x in before_second],
        })
    actual = p_rows + d_rows + prefill['reports']['router']['selected_lines']
    return {
        'epoch': 1317, 'formal_phase': 'cold', 'prefill_sched_reports': len(p), 'decode_sched_reports': len(d),
        'host_offset_two_observed_windows_ns': offsets, 'alignment_offset_envelope_ns': [lower_ns, upper_ns],
        'decode_burst_separation_threshold_s': D_REPORT_GAP, 'chronological_windows': rows,
        'selected_lines_with_request_or_room_key': sum(bool(ID_FIELD.search(row)) for row in actual),
        'same_request_transport_latency_or_link_bandwidth_verified': False,
        'scope': ('Coarse host log time comparison using two short clock checks. No continuous clock tracking. '
                  'The windows describe aggregate activity and lack a request-level join. '
                  'Log cadence and scheduler metric snapshots can delay visibility.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefill_log', 'decode_log', 'early_clock', 'late_clock', 'cold_live'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prefill = unwrap(args.prefill_log, P_LOG_SHA, 'G1468_COLD_LOGS')
    decode = unwrap(args.decode_log, D_LOG_SHA, 'G1468_COLD_LOGS')
    cold = unwrap(args.cold_live, COLD_LIVE_SHA, 'G1459_ACTUAL_SG_COLD_LIVE_JSON')
    result = same_log_window(prefill, decode, cold, load(args.early_clock, EARLY_CLOCK_SHA),
                             load(args.late_clock, LATE_CLOCK_SHA))
    result['input_receipts_sha256'] = {'prefill_log': P_LOG_SHA, 'decode_log': D_LOG_SHA,
                                       'early_clock': EARLY_CLOCK_SHA, 'late_clock': LATE_CLOCK_SHA,
                                       'cold_process': COLD_LIVE_SHA}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
