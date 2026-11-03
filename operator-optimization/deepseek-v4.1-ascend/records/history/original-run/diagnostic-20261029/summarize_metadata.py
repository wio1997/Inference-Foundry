#!/usr/bin/env python3
"""Offline bounded summary of this diagnostic runtime; never contacts a device."""
import argparse
from collections import Counter, defaultdict
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / 'runtime'
KEY = b'wio-synthetic-recapture-169-20261029'
HEX = re.compile(r'^[0-9a-f]{64}$')
FILE_PATTERN = re.compile(r'^events-([0-9]+)\.jsonl$')
REQUEST_PATTERN = re.compile(r'^wio-diag169-20261029-measure1-[0-7]$')
MAX_FILES, MAX_FILE_BYTES, MAX_LINE_BYTES = 128, 100_000_000, 1_000_000
MAX_RESPONSE_BYTES = 200_000_000


def integer(x):
    return x if type(x) is int and x >= 0 else None


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_schedule(schedule):
    total, rows = schedule.get('scheduled_total'), schedule.get('requests')
    if not isinstance(rows, list) or (total is not None and integer(total) is None):
        return None
    fields = {'request_hmac_sha256', 'scheduled_tokens', 'new_computed_tokens',
              'cached_computed_tokens', 'spec_token_count'}
    if not all(isinstance(row, dict) and set(row) == fields for row in rows):
        return None
    canonical = json.dumps({'scheduled_total': total, 'requests': rows},
                           sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(canonical).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help='Runtime directory containing metadata/ and profile-client-results/')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    metadata = root / 'metadata'
    responses_path = root / 'profile-client-results' / 'measure1-responses.json'
    output = args.output or root / 'metadata-summary.json'
    if output.exists() or output.is_symlink():
        raise RuntimeError('output already exists')
    if not metadata.is_dir() or not responses_path.is_file():
        raise RuntimeError('metadata directory or completed response file missing')
    if responses_path.stat().st_size > MAX_RESPONSE_BYTES:
        raise RuntimeError('response file exceeds size bound')
    with responses_path.open(encoding='utf-8') as f:
        responses = json.load(f)
    if not isinstance(responses, list) or len(responses) != 8:
        raise RuntimeError('expected eight completed synthetic responses')
    expected = {}
    for response in responses:
        rid = response.get('request_id') if isinstance(response, dict) else None
        if not isinstance(rid, str) or not REQUEST_PATTERN.fullmatch(rid):
            raise RuntimeError('unexpected synthetic request ID')
        index = int(rid.rsplit('-', 1)[1])
        digest = hmac.new(KEY, ('cmpl-' + rid + '-0').encode(), hashlib.sha256).hexdigest()
        if digest in expected or index in expected.values():
            raise RuntimeError('duplicate synthetic request ID')
        expected[digest] = index
    if set(expected.values()) != set(range(8)):
        raise RuntimeError('synthetic response indices incomplete')
    del responses

    paths = sorted(metadata.glob('events-*.jsonl'))
    if not paths or len(paths) > MAX_FILES:
        raise RuntimeError('metadata event file count missing or exceeds bound')
    processes = {}
    schedules = defaultdict(list)
    kinds = Counter()
    errors = Counter()
    digest_problems = Counter()
    match_hmac = Counter()
    per_request = {i: {'scheduler_rows': 0, 'runner_rows': 0, 'scheduled_tokens_sum': 0,
        'new_descriptor_rows': 0, 'cached_descriptor_rows': 0, 'both_descriptor_rows': 0,
        'no_descriptor_rows': 0, 'new_computed_min': None, 'new_computed_max': None,
        'cached_computed_min': None, 'cached_computed_max': None,
        'post_computed_min': None, 'post_computed_max': None,
        'post_equals_start_plus_scheduled_rows': 0, 'post_differs_from_start_plus_scheduled_rows': 0,
        'spec_count_known_rows': 0, 'spec_count_unknown_rows': 0,
        'spec_token_count_sum_known': 0, 'post_prompt_count_known_rows': 0,
        'post_prompt_count_32768_rows': 0, 'conditional_prompt_intersection_tokens': 0,
        'conditional_prompt_intersection_rows': 0} for i in range(8)}
    known_digests = set(expected)
    seen_event_ids = Counter()
    request_rows = Counter()
    for path in paths:
        file_match = FILE_PATTERN.fullmatch(path.name)
        if not file_match or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError('invalid metadata file')
        file_pid = int(file_match.group(1))
        proc = {'event_count': 0, 'kind_counts': Counter(), 'labels': Counter(),
                'schedule_events': 0, 'runner_events': 0, 'json_errors': 0,
                'pid_mismatches': 0, 'sha256': fingerprint(path), 'declared_event_cap': None}
        with path.open('rb') as f:
            for raw in f:
                if len(raw) > MAX_LINE_BYTES:
                    raise RuntimeError('metadata line exceeds bound')
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    proc['json_errors'] += 1
                    continue
                if not isinstance(event, dict):
                    proc['json_errors'] += 1
                    continue
                if event.get('pid') != file_pid:
                    proc['pid_mismatches'] += 1
                    continue
                proc['event_count'] += 1
                kind, label = event.get('kind'), event.get('label')
                if kind not in ('call_begin', 'call_end', 'instrumentation_error', 'instrumentation_loaded',
                                'profiler_factory', 'profiler_start_anchor', 'profiler_stop_anchor'):
                    kind = 'other_or_invalid'
                kinds[kind] += 1
                proc['kind_counts'][kind] += 1
                if label in ('Scheduler.schedule', 'NPUModelRunner.execute_model',
                             'DeepseekV41DecoderLayer.forward', 'DeepseekV41DSparkModel.forward'):
                    proc['labels'][label] += 1
                if kind == 'instrumentation_loaded':
                    proc['declared_event_cap'] = integer(event.get('max_events'))
                if kind == 'instrumentation_error':
                    errors['count'] += 1
                wanted = ((kind == 'call_end' and label == 'Scheduler.schedule' and event.get('outcome') == 'ok') or
                          (kind == 'call_begin' and label == 'NPUModelRunner.execute_model'))
                if not wanted:
                    continue
                side = 'scheduler' if label == 'Scheduler.schedule' else 'runner'
                proc['schedule_events' if side == 'scheduler' else 'runner_events'] += 1
                call_id = event.get('call_id')
                if isinstance(call_id, str):
                    seen_event_ids[(file_pid, side, call_id)] += 1
                schedule = event.get('schedule')
                if not isinstance(schedule, dict):
                    digest_problems[side + '_schedule_missing'] += 1
                    continue
                claimed = schedule.get('schedule_digest_sha256')
                computed = digest_schedule(schedule)
                if not isinstance(claimed, str) or not HEX.fullmatch(claimed):
                    digest_problems[side + '_digest_missing'] += 1
                    continue
                if computed != claimed:
                    digest_problems[side + '_digest_mismatch'] += 1
                    continue
                schedules[claimed].append((side, file_pid))
                states = schedule.get('scheduler_state_after_schedule', [])
                states_by_hmac = {}
                duplicate_state = set()
                if isinstance(states, list):
                    for state in states:
                        if isinstance(state, dict):
                            h = state.get('request_hmac_sha256')
                            if h in states_by_hmac:
                                duplicate_state.add(h)
                            states_by_hmac[h] = state
                for row in schedule['requests']:
                    rid_hmac = row.get('request_hmac_sha256')
                    if not isinstance(rid_hmac, str) or not HEX.fullmatch(rid_hmac):
                        request_rows[side + '_missing_or_invalid_hmac'] += 1
                        continue
                    if rid_hmac not in known_digests:
                        request_rows[side + '_not_in_eight_expected'] += 1
                        continue
                    request_rows[side + '_expected'] += 1
                    index = expected[rid_hmac]
                    match_hmac[(side, index)] += 1
                    z = per_request[index]
                    z[side + '_rows'] += 1
                    if side != 'scheduler':
                        continue
                    n = integer(row.get('scheduled_tokens'))
                    new = integer(row.get('new_computed_tokens'))
                    cached = integer(row.get('cached_computed_tokens'))
                    spec = integer(row.get('spec_token_count'))
                    z['scheduled_tokens_sum'] += n or 0
                    category = ('both' if new is not None and cached is not None else
                                'new' if new is not None else 'cached' if cached is not None else 'no')
                    z[category + '_descriptor_rows'] += 1
                    for name, value in (('new_computed', new), ('cached_computed', cached)):
                        if value is not None:
                            z[name + '_min'] = value if z[name + '_min'] is None else min(z[name + '_min'], value)
                            z[name + '_max'] = value if z[name + '_max'] is None else max(z[name + '_max'], value)
                    if spec is None:
                        z['spec_count_unknown_rows'] += 1
                    else:
                        z['spec_count_known_rows'] += 1
                        z['spec_token_count_sum_known'] += spec
                    state = states_by_hmac.get(rid_hmac)
                    prompt = integer(state.get('num_prompt_tokens')) if isinstance(state, dict) and rid_hmac not in duplicate_state else None
                    post = integer(state.get('num_computed_tokens_post_schedule')) if isinstance(state, dict) and rid_hmac not in duplicate_state else None
                    if post is not None:
                        z['post_computed_min'] = post if z['post_computed_min'] is None else min(z['post_computed_min'], post)
                        z['post_computed_max'] = post if z['post_computed_max'] is None else max(z['post_computed_max'], post)
                    if prompt is not None:
                        z['post_prompt_count_known_rows'] += 1
                        if prompt == 32768:
                            z['post_prompt_count_32768_rows'] += 1
                    # This is interval arithmetic, not proof of execution/prefill.
                    start = new if category == 'new' else cached if category == 'cached' else None
                    if n is not None and start is not None and post is not None:
                        z['post_equals_start_plus_scheduled_rows' if post == start + n else 'post_differs_from_start_plus_scheduled_rows'] += 1
                    if n is not None and prompt is not None and start is not None:
                        z['conditional_prompt_intersection_rows'] += 1
                        z['conditional_prompt_intersection_tokens'] += max(0, min(n, prompt - start))
        proc['potential_cap_hit'] = (proc['declared_event_cap'] is not None and
                                     proc['event_count'] >= proc['declared_event_cap'])
        proc['kind_counts'] = dict(proc['kind_counts'])
        proc['labels'] = dict(proc['labels'])
        processes[str(file_pid)] = proc

    groups = Counter()
    sched_events = runner_events = 0
    for entries in schedules.values():
        s = sum(side == 'scheduler' for side, _ in entries)
        r = sum(side == 'runner' for side, _ in entries)
        sched_events += s
        runner_events += r
        if s == 1 and r == 1: group = 'unique_one_scheduler_one_runner'
        elif s == 0: group = 'runner_only'
        elif r == 0: group = 'scheduler_only'
        elif s == 1: group = 'one_scheduler_multiple_runner'
        else: group = 'multiple_scheduler_ambiguous'
        groups[group] += 1
    result = {'scope': {'scheduler': 'PROFILE_ACTIVE gate; may outlast worker profiler iterations',
        'runner': 'hook-enabled worker interval; configuration requests at most 12 profiler iterations',
        'matching': 'identical canonical schedule digest, not unique replay or cross-rank clock identity',
        'coverage': 'request coverage counts only events with valid canonical schedule digests'},
        'input': {'response_file_sha256': fingerprint(responses_path), 'event_file_count': len(paths)},
        'synthetic_request_hmac_coverage': {'expected_eight': len(expected),
          'scheduler_distinct': sum(match_hmac[('scheduler', i)] > 0 for i in range(8)),
          'runner_distinct': sum(match_hmac[('runner', i)] > 0 for i in range(8)),
          'scheduler_and_runner_distinct': sum(match_hmac[('scheduler', i)] > 0 and match_hmac[('runner', i)] > 0 for i in range(8)),
          'request_rows': dict(request_rows)},
        'per_request_by_synthetic_index': [{'synthetic_index': i, **per_request[i]} for i in range(8)],
        'digest_groups': dict(groups), 'valid_digest_scheduler_events': sched_events,
        'valid_digest_runner_events': runner_events, 'digest_problems': dict(digest_problems),
        'duplicate_call_ids_extra': sum(v - 1 for v in seen_event_ids.values() if v > 1),
        'event_kind_counts': dict(kinds), 'instrumentation_error_count': errors['count'],
        'per_pid': processes,
        'limitations': ['No raw request text, generated content, original IDs or environment is emitted.',
          'Identical digest can recur in separate steps and can be broadcast to many workers; counts do not prove a unique request/step or cross-device execution.',
          'new/cached descriptors refer to scheduler output structures, not newly computed work or prefix hits.',
          'num_computed_tokens_post_schedule is post-update; absent spec count is unknown, not zero.',
          'Conditional prompt intersection assumes reported computed count is the scheduled interval start and scheduling is contiguous; not proof of completed work, no deduplication/preemption adjustment, nor an end-to-end count.',
          'Positive/negative timing, NPU actual work and profiler perturbation require independent evidence; workers and scheduler have different hook windows.',
          'No events can result from truncation, missing hooks or sampling, not proof no work took place.'],
        'no_device_work': True}
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if len(serialized.encode()) > 1_000_000:
        raise RuntimeError('summary exceeds output bound')
    with output.open('x', encoding='utf-8') as f:
        f.write(serialized)
    print(json.dumps({'status': 'SUMMARY_WRITTEN', 'bytes': len(serialized.encode()),
                      'sha256': hashlib.sha256(serialized.encode()).hexdigest()}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('SUMMARY_FAILED: ' + str(exc), file=sys.stderr)
        sys.exit(1)
