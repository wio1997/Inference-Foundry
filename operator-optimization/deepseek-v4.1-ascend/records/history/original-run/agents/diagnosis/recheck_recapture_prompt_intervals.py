#!/usr/bin/env python3
"""Read fixed recapture metadata; report anonymous scheduler interval arithmetic."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'))
root = parser.parse_args().root
summary_path = root / 'metadata-summary-early.json'
expected_summary_hash = '2e322b98e2f972e35d74cb2d5fc98dd7071a0ff48f38f8b47a62d91a2b48538a'
blob = summary_path.read_bytes()
if hashlib.sha256(blob).hexdigest() != expected_summary_hash:
    raise ValueError('summary hash mismatch')
summary = json.loads(blob)
reqs = defaultdict(list)
events = 0
calls = set()
hex64 = re.compile(r'^[0-9a-f]{64}$')
for pid, meta in summary['per_pid'].items():
    if not pid.isdigit():
        raise ValueError('invalid pid')
    path = root / 'runtime' / 'metadata' / ('events-' + pid + '.jsonl')
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000_000:
        raise ValueError('invalid file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError('event file hash mismatch')
    for lineno, line in enumerate(raw.splitlines(), 1):
        if len(line) > 1_000_000:
            raise ValueError('oversized line')
        e = json.loads(line)
        if e.get('pid') != int(pid):
            raise ValueError('pid mismatch')
        if e.get('kind') != 'call_end' or e.get('label') != 'Scheduler.schedule' or e.get('outcome') != 'ok':
            continue
        callspec = (pid, e.get('call_id'))
        if callspec in calls:
            raise ValueError('duplicate call id')
        calls.add(callspec)
        events += 1
        s = e['schedule']
        payload = json.dumps({'scheduled_total': s['scheduled_total'], 'requests': s['requests']}, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(payload).hexdigest() != s['schedule_digest_sha256']:
            raise ValueError('digest mismatch')
        states = {}
        for state in s.get('scheduler_state_after_schedule', []):
            h = state.get('request_hmac_sha256')
            if h in states:
                raise ValueError('duplicate state')
            states[h] = state
        seen = set()
        for row in s['requests']:
            h = row.get('request_hmac_sha256')
            if not isinstance(h, str) or not hex64.fullmatch(h) or h in seen:
                raise ValueError('bad or duplicate request hmac')
            seen.add(h)
            reqs[h].append((pid, lineno, row, states.get(h, {})))
if events != 453 or len(reqs) != 8:
    raise ValueError('unexpected scheduler coverage')

items = []
for index, h in enumerate(sorted(reqs), 1):
    rows = reqs[h]
    flags = []
    if len({r[0] for r in rows}) != 1:
        flags.append('multiple_scheduler_pids')
    new_indices = [i for i, (_, _, r, _) in enumerate(rows) if type(r.get('new_computed_tokens')) is int]
    if new_indices != [0]:
        flags.append('new_descriptor_missing_repeated_or_not_first')
    starts = []
    for pid, line, row, state in rows:
        new, cached, n = row.get('new_computed_tokens'), row.get('cached_computed_tokens'), row.get('scheduled_tokens')
        valid_new = type(new) is int and new >= 0
        valid_cached = type(cached) is int and cached >= 0
        if valid_new == valid_cached or type(n) is not int or n < 0:
            flags.append('invalid_descriptor')
            continue
        starts.append((new if valid_new else cached, n, state, row.get('spec_token_count')))
    if not starts or len(starts) != len(rows):
        items.append({'anonymous_index': index, 'status': 'unknown', 'flags': sorted(set(flags))})
        continue
    start0 = starts[0][0]
    cursor = start0
    prompt = 32768
    chunks = []
    finished = False
    gap = overlap = repeats = 0
    for start, n, state, spec in starts:
        if state.get('num_prompt_tokens') != prompt:
            flags.append('prompt_length_unknown_or_inconsistent')
        if state.get('num_computed_tokens_post_schedule') != start + n:
            flags.append('post_state_missing_or_not_start_plus_scheduled')
        if finished:
            if start < prompt and n > 0:
                flags.append('later_prompt_reentry')
            continue
        if start > cursor:
            gap += start - cursor
        if start < cursor:
            overlap += min(cursor-start, n)
            if any(start == x[0] and min(start+n, prompt) == x[1] for x in chunks):
                repeats += 1
        stop = min(start + n, prompt)
        if stop > start:
            chunks.append((start, stop))
        cursor = max(cursor, stop)
        if spec is not None and (type(spec) is not int or spec < 0):
            flags.append('invalid_spec_count')
        if type(spec) is int and spec > 0 and start < prompt:
            flags.append('spec_count_present_in_prompt_interval')
        if cursor >= prompt:
            finished = True
    if not finished:
        flags.append('prompt_end_not_observed')
    if gap: flags.append('gap')
    if overlap: flags.append('overlap')
    if repeats: flags.append('repeated_interval')
    if start0 > prompt:
        flags.append('initial_start_past_prompt')
    chunk_sum = sum(b-a for a,b in chunks)
    items.append({'anonymous_index': index, 'status': 'unknown' if flags else 'contiguous_to_prompt_end',
                  'initial_computed': start0, 'expected_remaining': max(0,prompt-start0),
                  'prompt_intervals': [list(c) for c in chunks], 'scheduled_prompt_interval_sum': chunk_sum,
                  'gap_tokens': gap, 'overlap_tokens': overlap, 'repeated_interval_count': repeats,
                  'later_scheduler_rows': len(rows)-len(chunks), 'flags': sorted(set(flags))})

out = {'summary_sha256': expected_summary_hash, 'metadata_files_verified': len(summary['per_pid']),
       'scheduler_events': events, 'anonymous_request_hmac_count': len(reqs),
       'initial_computed_sum': sum(x.get('initial_computed',0) for x in items),
       'expected_remaining_sum': sum(x.get('expected_remaining',0) for x in items),
       'contiguous_scheduled_prompt_sum': sum(x.get('scheduled_prompt_interval_sum',0) for x in items if x['status'] == 'contiguous_to_prompt_end'),
       'all_eight_contiguous': all(x['status'] == 'contiguous_to_prompt_end' for x in items),
       'requests': items,
       'limitations': ['Scheduler descriptor interval arithmetic is conditional on saved source semantics, not hardware completion.',
                       'HMAC request identities cannot be bound to synthetic external request IDs without an actual mapping.',
                       'New descriptor is not necessarily a never-before-computed request; local and external prefix hits are not distinguished.',
                       'A null spec count is unknown.']}
progress = root / 'runtime' / 'profile-client-results' / 'progress.json'
if progress.is_file() and not progress.is_symlink() and progress.stat().st_size < 5_000_000:
    raw = progress.read_bytes()
    p = json.loads(raw)
    out['progress_sha256'] = hashlib.sha256(raw).hexdigest()
    delta = p.get('measure1_counter_deltas', {})
    allowed = ['prefix_cache_queries_total','prefix_cache_hits_total','prompt_tokens_total','num_preemptions_total']
    out['measure1_counter_deltas'] = {k:delta[k] for k in allowed if isinstance(delta,dict) and type(delta.get(k)) in (float,int)}
print(json.dumps(out, sort_keys=True, separators=(',', ':')))
