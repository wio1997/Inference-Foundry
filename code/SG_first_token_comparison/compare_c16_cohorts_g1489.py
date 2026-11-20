"""Compare exact-input client timing before and after the first C16 admissions."""

import argparse
from base64 import b64decode
import gzip
import io
import json
from pathlib import Path
from statistics import mean, median
from struct import pack

import compare_first_wave_g1479 as base


RECEIPT_SHA = '7fee436186bd973a689405b1b44c20e61c01e525a8050a98da79ccbf17c4eeef'
WIDTH = 16
COMPLETE_ROWS = 24


def load_current(receipt, database, decoder):
    packet = base.read_pinned(receipt, RECEIPT_SHA)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    raw, = (line.removeprefix(base.MARKER) for line in packet['output'].splitlines()
            if line.startswith(base.MARKER))
    sample = json.loads(raw)
    assert sample['epoch'] == 1317 and sample['runner_PID'] == 3090471
    assert sample['aisbench_exec_PID'] == 3090587
    observed = sample['observation']
    assert observed['dataset_sha256'] == base.DATA_SHA
    assert observed['all_complete_temp_rows_match_frozen_inputs'] is True
    assert observed['formal_terminal_acceptance'] is False
    assert observed['verified_complete_temp_rows'] == COMPLETE_ROWS
    assert observed['temp_rows_by_outcome'] == {'recorded_success_with_exact_usage': COMPLETE_ROWS}
    assert observed['configured_requests'] == 32
    timing = observed['timing_bundle']
    compressed = b64decode(timing['database_gzip_b64'], validate=True)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as source:
        data = source.read(1024 * 1024 + 1)
    assert len(data) == timing['database_bytes'] <= 1024 * 1024
    assert base.digest(data) == timing['database_sha256']
    assert database.read_bytes() == data
    ids = {row['data_id'] for row in timing['timing_refs']}
    assert len(ids) == len(timing['timing_refs']) == COMPLETE_ROWS
    refs = {row['data_id']: row['reference'] for row in timing['timing_refs']}
    prompts = {row['data_id']: row['exact_input_utf8_sha256'] for row in timing['timing_refs']}
    info = {'receipt_sha256': RECEIPT_SHA, 'timing_database_sha256': timing['database_sha256'],
            'at_UTC': sample['at_UTC'], 'validated_tmp_rows': COMPLETE_ROWS}
    return base.clocks(database, refs, decoder), prompts, info


def first_admissions(times):
    order = sorted(times, key=lambda item: times[item][0])
    initial = order[:WIDTH]
    assert len(initial) == WIDTH
    assert all(row[0] < row[1] <= row[-1] for row in times.values())
    assert times[initial[-1]][0] < min(row[-1] for row in times.values())
    subsequent = order[WIDTH:]
    assert subsequent and times[subsequent[0]][0] >= min(times[item][-1] for item in initial)
    return initial, subsequent


def elapsed(times, ids):
    start = min(row[0] for row in times.values())
    pre_submit = [times[item][0] - start for item in ids]
    intervals = [base.time_parts(times[item]) for item in ids]
    before_first, after_first, aggregate = map(mean, zip(*intervals))
    firsts = sorted(times[item][1] for item in ids)
    return {
        'client_earliest_to_latest_submit_span_s': max(pre_submit) - min(pre_submit),
        'client_submit_offset_from_first_batch_submit_mean_s': mean(pre_submit),
        'elapsed_submit_to_first_SSE_mean_s': before_first,
        'elapsed_first_to_terminal_SSE_mean_s': after_first,
        'elapsed_submit_to_terminal_SSE_mean_s': aggregate,
        'adjacent_first_SSE_spacing_median_s': median(b - a for a, b in zip(firsts, firsts[1:])),
    }


def compare(sg, sg_prompts, native, native_prompts):
    assert set(sg) <= set(native)
    assert all(sg_prompts[item] == native_prompts[item] for item in sg)
    sg_first, sg_later = first_admissions(sg)
    native_first, native_later = first_admissions(native)
    assert set(sg_first) == set(native_first) and set(sg_later) <= set(native_later)
    results = {}
    for name, ids in (('first_C16', sg_first), ('observed_later_admissions', sg_later)):
        s = elapsed(sg, ids)
        v = elapsed(native, ids)
        first = s['elapsed_submit_to_first_SSE_mean_s'] - v['elapsed_submit_to_first_SSE_mean_s']
        total = s['elapsed_submit_to_terminal_SSE_mean_s'] - v['elapsed_submit_to_terminal_SSE_mean_s']
        assert total > 0
        results[name] = {'exact_matched_inputs': len(ids), 'SGLang': s, 'vLLM': v,
                         'mean_client_elapsed_gap_s': total,
                         'fraction_of_client_elapsed_gap_before_first_SSE': first / total}
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('receipt', 'database', 'earlier-receipt', 'earlier-database',
                 'native-run', 'native-reference', 'decoder'):
        cli.add_argument('--' + name, required=True, type=Path)
    args = cli.parse_args()
    decoder = base.load_decoder(args.decoder)
    sg, sg_prompts, current = load_current(args.receipt, args.database, decoder)
    old, old_prompts, earlier = base.load_sglang(args.earlier_receipt, args.earlier_database, decoder)
    for item, times in old.items():
        fmt = f'<{len(times)}d'
        assert pack(fmt, *sg[item]) == pack(fmt, *times)
        assert sg_prompts[item] == old_prompts[item]
    native, native_prompts, baseline = base.load_vllm(args.native_run, args.native_reference, decoder)
    result = {'SGLang_current_receipt': current, 'vLLM_accepted_reference': baseline,
              'earlier_SGLang_receipt_sha256': earlier['receipt_sha256'],
              'earlier_SGLang_timing_arrays_unchanged': len(old),
              'matched_cohorts': compare(sg, sg_prompts, native, native_prompts),
              'dataset_sha256': base.DATA_SHA, 'SGLang_cold_formal_score_accepted': False,
              'comparison_source_sha256': base.digest(Path(__file__).read_bytes()),
              'scope': 'Both cohorts use exact same inputs. The later cohort has only the SGLang rows already '
                       'completed at this read. Intervals use the client clock within each run. '
                       'Submit offsets are separate from elapsed TTFT. Attribution to a server stage needs '
                       'server traces. This is an observation before the SGLang cold phase terminal audit.'}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
