"""Compare client timing for identical requests with retained first-wave timestamps."""

import argparse
from contextlib import closing
from hashlib import sha256
import importlib.util
import json
from math import isclose
from pathlib import Path
import sqlite3
from statistics import mean, median


DATA_SHA = '2484713fafb09065e6426717d1f07fa26cd3e7de4d771e734c60ec2511b94ffe'
NATIVE_RUN_SHA = '07e9ad3d8b07830d10e59bf86065b26fc5f0f4e9471fde687cad58aa735fd549'
DECODER_SHA = 'f812cb9da44b96cdf941645f95897354c38404c936bd63a81e8b28b0890b7d30'
SG_RECEIPT_SHA = '74da1ff0fe5cdb23a24e1aac285572f3ac54ab7a81952ec57b0db90cbbf0a77a'
MARKER = 'G1479_COLD_RAW_TIMING_AND_PROMPT_IDENTITY '


def digest(raw):
    return sha256(raw).hexdigest()


def read_pinned(path, pin):
    raw = path.read_bytes()
    assert digest(raw) == pin, 'Source digest mismatch'
    return json.loads(raw)


def load_decoder(path):
    assert digest(path.read_bytes()) == DECODER_SHA
    spec = importlib.util.spec_from_file_location('fixed_AISBench_decoder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clocks(database, references, decoder):
    assert len(set(references.values())) == len(references)
    assert all(type(ref) is int for ref in references.values())
    assert not any(database.with_name(database.name + suffix).exists()
                   for suffix in ('-journal', '-wal', '-shm'))
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        values = {}
        for data_id, ref in references.items():
            result, = db.execute('SELECT arr_blob FROM numpy_store WHERE id=?', (ref,)).fetchall()
            values[data_id] = decoder.npy_float64(result[0])
    return values


def load_sglang(receipt, db, decoder):
    packet = read_pinned(receipt, SG_RECEIPT_SHA)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    line, = (line.removeprefix(MARKER) for line in packet['output'].splitlines() if line.startswith(MARKER))
    sample = json.loads(line)
    assert sample['epoch'] == 1317
    observation = sample['observation']
    assert observation['dataset_sha256'] == DATA_SHA
    assert observation['all_complete_temp_rows_match_frozen_inputs']
    assert not observation['formal_terminal_acceptance']
    count = observation['verified_complete_temp_rows']
    assert count == 14 and observation['configured_requests'] == 32
    assert observation['temp_rows_by_outcome'] == {'recorded_success_with_exact_usage': count}
    timing = observation['timing_bundle']
    data = db.read_bytes()
    assert len(data) == timing['database_bytes'] and digest(data) == timing['database_sha256']
    refs = timing['timing_refs']
    assert count == len({row['data_id'] for row in refs}) == len(refs)
    prompts = {row['data_id']: row['exact_input_utf8_sha256'] for row in refs}
    times = clocks(db, {row['data_id']: row['reference'] for row in refs}, decoder)
    return times, prompts, {'receipt_sha256': SG_RECEIPT_SHA, 'db_sha256': timing['database_sha256'],
                            'time_at_UTC': sample['at_UTC'], 'observer_sha256': packet['script_sha256']}


def load_vllm(raw_run, reference, decoder):
    native = read_pinned(reference, NATIVE_RUN_SHA)
    assert native['engine'] == 'vllm' and native['all_required_raw_metrics_and_usage_validated']
    profile, = native['profiles']
    assert profile['input_tokens'] == 32768 and profile['dataset_sha256'] == DATA_SHA
    current = profile['cold']
    assert current['fully_validated'] and current['requests'] == current['successful_requests'] == 32
    assert current['configured_concurrency'] == current['observed_max_concurrency'] == 16
    local = json.loads((raw_run / 'cold_32768_audit.json').read_bytes())
    assert local == current
    command = json.loads((raw_run / 'cold_32768_command.json').read_bytes())
    assert command['dataset_sha256'] == DATA_SHA
    rolled = decoder.audit_phase(raw_run, 'cold', 32768)
    assert rolled['all_raw_request_counts_and_reported_metrics_match']
    folder = raw_run / 'cold_32768'
    relative, = (path for path in current['evidence'] if path.endswith('_details.jsonl'))
    raw_file = folder / relative
    raw = raw_file.read_bytes()
    detail_sha = digest(raw)
    assert detail_sha == current['evidence'][relative]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert {row['data_id'] for row in rows} == set(range(32))
    basename, = {row['db_name'] for row in rows}
    db = raw_file.parent / 'db_data' / basename
    db_sha = digest(db.read_bytes())
    assert rolled['database_sha256'][basename] == db_sha
    prompts = {row['data_id']: digest(row['input'].encode('utf-8')) for row in rows}
    times = clocks(db, {row['data_id']: row['time_points']['__db_ref__'] for row in rows}, decoder)
    return times, prompts, {'accepted_run_audit_sha256': NATIVE_RUN_SHA, 'details_sha256': detail_sha,
                            'db_sha256': db_sha}


def time_parts(times):
    submit, first = times[:2]
    last = times[-1]
    return first - submit, last - first, last - submit


def summary(sg, native, sg_prompts, native_prompts):
    ids = set(sg)
    assert len(ids) == 14
    assert all(sg_prompts[data_id] == native_prompts[data_id] for data_id in ids)
    first_16 = sorted(native, key=lambda i: native[i][0])[:16]
    assert ids <= set(first_16)
    sg_order = sorted(sg, key=lambda i: sg[i][0])
    sg_submit = [sg[i][0] for i in sg_order]
    sg_submit_span = sg_submit[-1] - sg_submit[0]
    assert sg_submit_span < 0.2 and max(sg_submit) < min(times[-1] for times in sg.values())
    assert max(native[i][0] for i in first_16) < min(times[-1] for times in native.values())

    def parts_table(times):
        columns = list(zip(*(time_parts(times[i]) for i in ids)))
        return dict(zip(('submit_to_first_sse_s','first_to_last_sse_s','submit_to_last_sse_s'),
                        map(mean, columns)))

    sg_parts, native_parts = parts_table(sg), parts_table(native)
    full_gap = sg_parts['submit_to_last_sse_s'] - native_parts['submit_to_last_sse_s']
    first_gap = sg_parts['submit_to_first_sse_s'] - native_parts['submit_to_first_sse_s']
    tail_gap = sg_parts['first_to_last_sse_s'] - native_parts['first_to_last_sse_s']
    assert full_gap > 0 and isclose(full_gap, first_gap + tail_gap, abs_tol=1e-10)

    first_rows = sorted(times[1] for times in sg.values())
    first_sg_data_id = min(sg, key=lambda i: sg[i][1])
    return {
        'exact_matching_input_count': len(ids), 'completed_sglang_temporary_rows': len(ids),
        'configured_both_phases_requests': 32, 'native_first_admission_count': len(first_16),
        'all_matched_SGLang_submissions_precede_first_observed_completion': True,
        'all_matched_vLLM_requests_in_initial_C16_wave': True,
        'completed_sglang_submission_span_s': sg_submit_span,
        'SGLang_mean_client_intervals_s': sg_parts,
        'vLLM_mean_client_intervals_s': native_parts,
        'SGLang_minus_vLLM_mean_end_to_end_s': full_gap,
        'portion_of_mean_gap_before_first_SSE': first_gap / full_gap,
        'portion_of_mean_gap_after_first_SSE': tail_gap / full_gap,
        'hypothetical_max_gap_removed_by_zero_post_first_SSE_at_fixed_TTFT': sg_parts[
            'first_to_last_sse_s'] / full_gap,
        'SGLang_median_adjacent_first_sse_gap_s': median(
            right - left for left, right in zip(first_rows, first_rows[1:])),
        'first_returning_SGLang_request_TTFT_s': time_parts(sg[first_sg_data_id])[0],
        'same_input_vLLM_request_TTFT_s': time_parts(native[first_sg_data_id])[0],
        'scope': 'Client elapsed intervals for the completed SGLang subset joined by exact input SHA256 and dataset ID. '
                 'First-SSE latency includes admission, Prefill, transfer and any Decode work before the first output. '
                 'Operator and rank attribution, remaining requests and terminal SGLang score require their own evidence.',
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('sglang-receipt', 'sglang-db', 'vllm-raw-run', 'vllm-reference', 'decoder'):
        cli.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    decoder = load_decoder(args.decoder)
    sg, sg_prompts, sg_sources = load_sglang(args.sglang_receipt, args.sglang_db, decoder)
    native, native_prompts, native_sources = load_vllm(args.vllm_raw_run, args.vllm_reference, decoder)
    result = summary(sg, native, sg_prompts, native_prompts)
    result.update(frozen_dataset_sha256=DATA_SHA, sg_observation=sg_sources, vllm_observation=native_sources,
                  decoder_sha256=DECODER_SHA, comparison_source_sha256=digest(Path(__file__).read_bytes()),
                  sglang_formal_phase_score_accepted=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
