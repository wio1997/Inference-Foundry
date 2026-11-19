"""Read request clocks from a verified copy of the live AISBench timing database."""

import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from statistics import median

import raw_sqlite_timing_audit_g751 as timing


RECEIPT_SHA = 'ff1830386ba6b15dcb30cfb6ea9fb6cdd38fb51393a5ab90c7bc59bfb3bcaa40'
DECODER_SHA = 'f812cb9da44b96cdf941645f95897354c38404c936bd63a81e8b28b0890b7d30'
MARKER = 'G1474_CLIENT_TEMP_TIMING '


def read_receipt(path):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == RECEIPT_SHA
    packet = json.loads(raw)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    value, = (line.removeprefix(MARKER) for line in packet['output'].splitlines()
              if line.startswith(MARKER))
    result = json.loads(value)
    assert result['epoch'] == 1317 and result['runner_PID'] == 3090471
    assert result['aisbench_exec_PID'] == 3090587
    return result, packet['script_sha256']


def timestamps(database, bundle):
    src = Path(timing.__file__).read_bytes()
    assert sha256(src).hexdigest() == DECODER_SHA
    raw = database.read_bytes()
    assert len(raw) == bundle['database_bytes'] and sha256(raw).hexdigest() == bundle['database_sha256']
    refs = bundle['timing_refs']
    assert len({row['reference'] for row in refs}) == len({row['data_id'] for row in refs}) == len(refs)
    output = []
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        for row in refs:
            match, = db.execute('SELECT arr_blob FROM numpy_store WHERE id=?',
                                (row['reference'],)).fetchall()
            output.append(timing.npy_float64(match[0]))
    assert database.read_bytes() == raw
    return output


def summarize(receipt, times):
    observation = receipt['observation']
    assert not observation['formal_terminal_acceptance']
    assert observation['all_complete_temp_rows_match_frozen_inputs']
    assert len(times) == observation['verified_complete_temp_rows'] == 9
    assert observation['temp_rows_by_outcome']['recorded_success_with_exact_usage'] == len(times)
    caps = observation['status_before_and_after']
    assert caps[0] == caps[1] and caps[0]['finish_count'] == len(times)
    start = min(row[0] for row in times)
    starts = [row[0] - start for row in times]
    sequence = sorted(times, key=lambda row: row[1])
    first = [row[1] - start for row in sequence]
    gaps = [b - a for a, b in zip(first, first[1:])]
    tail = [row[-1] - row[1] for row in sequence]
    idle_visible = [b[1] - a[-1] for a, b in zip(sequence, sequence[1:])]
    assert min(idle_visible) > 0
    return {
        'epoch': 1317, 'phase': 'cold_in_progress', 'observed_at_UTC': receipt['at_UTC'],
        'actual_completed_temp_rows': len(times), 'formal_requests_configured': caps[0]['total_count'],
        'completed_subset_issue_time_span_s': max(starts),
        'first_SSE_offsets_from_earliest_submit_s': first,
        'adjacent_first_SSE_gap_s': gaps,
        'median_adjacent_first_SSE_gap_s': median(gaps),
        'first_to_last_SSE_duration_range_s': [min(tail), max(tail)],
        'previous_last_to_next_first_SSE_gap_range_s': [min(idle_visible), max(idle_visible)],
        'temporary_timing_db_sha256': observation['timing_bundle']['database_sha256'],
        'verified_client_temp_prefix_sha256': observation['complete_temp_prefix_sha256'],
        'timing_decoder_sha256': DECODER_SHA, 'private_receipt_sha256': RECEIPT_SHA,
        'scope': ('Timing of the currently completed subset on one client clock. SSE events can carry '
                  'multiple output tokens. Full-cohort score and per-operator timing are pending.'),
        'formal_result_accepted': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('copied_database', type=Path)
    args = parser.parse_args()
    receipt, observer_digest = read_receipt(args.receipt)
    times = timestamps(args.copied_database, receipt['observation']['timing_bundle'])
    report = summarize(receipt, times)
    report['copy_observer_source_sha256'] = observer_digest
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
