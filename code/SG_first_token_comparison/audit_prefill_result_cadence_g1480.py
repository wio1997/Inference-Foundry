"""Compare actual Prefill result-report cadence with the first SSE on the client."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from statistics import median

from compare_first_wave_g1479 import digest, load_decoder, load_sglang, read_pinned


SOURCES = {
    'prefill_order': '80f8796f6249e3452c9eb433aa42d78d067d257cfe438e470586ab8d6b18790b',
    'prefill_logs': 'c7bffb96831bb8206c457e87b5007a89b1486ba1f780b0b716e77141a0821855',
    'pd_clock_early': 'c77901e0ad6934e421ea7a97853f4ef61e6c64e0f94af5d27da4fbc1fe37f7db',
    'pd_clock_middle': '7ca7fe067ce94334f9567cd33ce33d8140cbfb176daa07a9ca77eec9150c9ca3',
    'pd_clock_late': '00a640cf8d34d56a636223e350a5be72fe390b5e14fe74bb486f9f447e68fff1',
    'client_clock_early': '5a50547318260acebbd7680e2a4fed4294979885a47d707825e3fbe21d5af031',
    'client_clock_late': '9718d4b9482cfba1f578c2fb9b7394a0496024b8eb05b759ab4056851f4235b8',
}
TIME = re.compile(r'^\[([\d-]+ [\d:]+) TP0 EP0\]')


def receipt(path, key, marker):
    packet = read_pinned(path, SOURCES[key])
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    line, = (line.removeprefix(marker+' ') for line in packet['output'].splitlines()
             if line.startswith(marker+' '))
    return json.loads(line)


def source_order(report):
    assert report['epoch'] == 1317 and report['role'] == 'prefill'
    assert report['startup_disable_overlap_schedule'] == ['False']
    files = report['source_files']
    prefill = files['disaggregation/prefill.py']
    functions = prefill['functions']

    def point(function, label):
        index, = (i for i, line in functions[function]['selected'] if label in line)
        return index

    output = 'process_batch_result_disagg_prefill'
    ordering = [point(output, source) for source in ('copy_done.synchronize()',
                 'self.send_kv_chunk(req, last_chunk=True)', 'self.metrics_reporter.report_prefill_stats(')]
    assert ordering == sorted(ordering)
    overlap = 'event_loop_overlap_disagg_prefill'
    overlap_order = [point(overlap, source) for source in ('self.run_batch(batch)',
          'self.process_batch_result(oldest_batch, oldest_batch_result)',
          'self.process_disagg_prefill_transfer_queue()')]
    assert overlap_order == sorted(overlap_order)
    logger = files['managers/scheduler_components/metrics_reporter.py']
    assert any('Prefill batch' in line for _, line in logger['functions']['report_prefill_stats']['selected'])
    return {'runtime_disable_overlap_schedule': False, 'prefill_result_source_sha256': prefill['source_sha256'],
            'result_source_rows_copy_sync_last_chunk_send_report': ordering,
            'overlap_loop_rows_dispatch_process_old_result_poll_transfer': overlap_order,
            'reporter_source_sha256': logger['source_sha256']}


def aligned_cadence(logs, times, pd_clock_reports, client_clock_reports):
    assert logs['role'] == 'prefill' and logs['epoch'] == 1317
    assert len(pd_clock_reports) == 3 and len(client_clock_reports) == 2
    offsets = [item['prefill_minus_decode_realtime_ns'] for item in pd_clock_reports]
    assert all(len(role['samples']) == 12 for item in pd_clock_reports for role in item['roles'].values())
    low, high = min(item[0] for item in offsets)/1e9, max(item[1] for item in offsets)/1e9
    assert 19 < low < high < 21
    command_at = {item['formal_cold_command_UTC'] for item in client_clock_reports}
    command, = command_at
    command_s = datetime.fromisoformat(command).timestamp()
    assert all(item['client_container_matches_host_time_namespace'] and item['same_runner_live']
               and item['perf_counter_impl'] == 'clock_gettime(CLOCK_MONOTONIC)'
               for item in client_clock_reports)
    mono_offsets = [[item['wall_seconds']-item['monotonic_bracket_s'][1],
                     item['wall_seconds']-item['monotonic_bracket_s'][0]] for item in client_clock_reports]
    mono_lo, mono_hi = min(item[0] for item in mono_offsets), max(item[1] for item in mono_offsets)
    assert mono_hi - mono_lo < 0.001

    completed = []
    for message in logs['reports']['engine']['selected_lines']:
        if 'Prefill batch' not in message:
            continue
        stamp = datetime.fromisoformat(TIME.match(message)[1]).replace(tzinfo=timezone.utc).timestamp()
        if stamp - high <= command_s:
            continue
        fields = {name: int(value) for name, value in re.findall(r'#([a-z-]+): (\d+)', message)}
        assert fields['new-seq'] == 1 and fields['new-token'] == 8192 and fields['cached-token'] == 0
        completed.append(stamp)
    assert len(completed) == 64 and all(a < b for a, b in zip(completed, completed[1:]))
    first = sorted(item[1] for item in times.values())
    assert len(first) == 14
    earliest_submit = min(item[0] for item in times.values())
    assert 0 < earliest_submit + mono_lo - command_s < 60
    reports_between = [b - a for a, b in zip(completed, completed[1:])]
    result = []
    for n, first_sse in enumerate(first, 1):
        first_sse_lo, first_sse_hi = first_sse + mono_lo, first_sse + mono_hi

        def difference(index):
            stamp = completed[index]
            return [round(first_sse_lo - (stamp + 1 - low), 3),
                    round(first_sse_hi - (stamp - high), 3)]

        quartet = 4 * n
        previous = difference(quartet - 1)
        following = difference(quartet)
        assert 118 < previous[0] <= previous[1] < 123
        assert -2 < following[0] <= following[1] < 2
        result.append({'client_first_SSE_ordinal': n, 'result_report_ordinals': [quartet, quartet + 1],
                       'seconds_after_quartet_report': previous,
                       'seconds_relative_to_following_report': following})
    return {'epoch': 1317, 'observed_first_SSE_events': len(first),
            'prefill_result_reports_observed': len(completed), 'new_input_tokens_per_prefill_report': 8192,
            'one_request_per_report': True, 'zero_reported_cached_input_tokens': True,
            'P_minus_D_observed_realtime_envelope_s': [low, high],
            'P_report_interval_range_s': [min(reports_between), max(reports_between)],
            'P_report_interval_median_s': median(reports_between),
            'client_earliest_submit_to_first_P_report_approx_s': [
                round(completed[0] - high - earliest_submit - mono_hi, 3),
                round(completed[0] + 1 - low - earliest_submit - mono_lo, 3)],
            'ordinal_windows': result,
            'scope': 'Approximate join of aggregate reports and client ordinals using sampled host offsets, '
                     'one-second log precision and a pinned client time-namespace mapping. '
                     'Individual GPU/NPU spans and request-level KV latency need independent events. '
                     'The Prefill result report follows conditional copy synchronization and final-chunk send invocation.',
            'formal_phase_score_accepted': False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in (*SOURCES, 'sglang_receipt', 'sglang_db', 'decoder'):
        cli.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    args = cli.parse_args()
    decoder = load_decoder(args.decoder)
    times, _, sample = load_sglang(args.sglang_receipt, args.sglang_db, decoder)
    order = source_order(receipt(args.prefill_order, 'prefill_order', 'G1480_P_LIVE_SOURCE_JSON'))
    logs = receipt(args.prefill_logs, 'prefill_logs', 'G1468_COLD_LOGS')
    pd_keys = ('pd_clock_early','pd_clock_middle','pd_clock_late')
    d_keys = ('client_clock_early','client_clock_late')
    report = aligned_cadence(logs, times, [read_pinned(getattr(args,key), SOURCES[key]) for key in pd_keys],
                     [receipt(getattr(args,key), key, 'G1476_D_CLIENT_CLOCK_JSON') for key in d_keys])
    report.update(prefill_source_order=order, private_receipts_sha256=SOURCES, client_observation=sample,
                  analysis_source_sha256=digest(Path(__file__).read_bytes()))
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
