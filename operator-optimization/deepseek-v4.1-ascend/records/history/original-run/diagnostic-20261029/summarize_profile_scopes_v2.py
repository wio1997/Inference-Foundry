#!/usr/bin/env python3
"""Read-only bounded scope attribution for this diagnostic runtime.

Run inside the diagnostic container after export, e.g.:
  python3 summarize_profile_scopes.py /data/tiankuan/wio/projects/910c-deepseek-v4.1-flash
Prints JSON only; no trace or metadata is modified. No NPU work.
"""
import collections
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

MAX_JSON_BYTES = 200 * 1024 * 1024
MAX_METADATA_BYTES = 30 * 1024 * 1024
MAX_SCOPES = 200000
SAMPLE_LIMIT = 24
NAME = re.compile(r'^wio_diag:([A-Za-z0-9_]+\.(?:forward|execute_model|schedule)):(\d+-\d+)$')
FIELDS = {'record_shapes', 'with_stack', 'with_modules', 'data_simplification', 'msprof_tx', 'op_attr', 'record_op_args', 'profiler_level'}
try:
    import ijson
except ImportError:
    ijson = None

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def configurations(rank):
    result = []
    for path in sorted(rank.glob('profiler_info*.json')):
        item = {'file': path.name}
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 8*1024*1024:
                item['status'] = 'unsupported_file'
            else:
                obj = json.loads(path.read_text())
                found = []
                def visit(v, prefix=''):
                    if isinstance(v, dict):
                        for key, val in v.items():
                            location = prefix + '.' + key if prefix else key
                            if key.lstrip("_") in FIELDS and isinstance(val, (str, bool, int, float, type(None))):
                                found.append({'path': location, 'value': val})
                            if isinstance(val, (dict, list)):
                                visit(val, location)
                    elif isinstance(v, list):
                        for i, val in enumerate(v[:100]):
                            if isinstance(val, (dict, list)):
                                visit(val, prefix + '[%d]' % i)
                visit(obj)
                item.update(status='read', sha256=file_sha(path), fields=found,
                            limitation='missing keys are unknown; requested flags do not prove captured stacks')
        except Exception as e:
            item.update(status='error', error_type=type(e).__name__)
        result.append(item)
    return result

def read_metadata(root):
    result = {}; stats = collections.Counter(); errors = []
    folder = root / 'metadata'
    for p in sorted(folder.glob('events-*.jsonl')):
        try:
            if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_METADATA_BYTES:
                stats['files_skipped'] += 1; continue
            stats['files_read'] += 1
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except (ValueError, UnicodeError):
                        stats['invalid_lines'] += 1; continue
                    kind = item.get('kind'); stats[str(kind)] += 1
                    if kind == 'instrumentation_error':
                        errors.append({'pid': item.get('pid'), 'where': item.get('where'), 'error_type': item.get('error_type')})
                    if kind not in ('call_begin', 'call_end'):
                        continue
                    call_id = item.get('call_id')
                    if not isinstance(call_id, str) or not re.fullmatch(r'\d+-\d+', call_id):
                        stats['invalid_call_id'] += 1; continue
                    slot = result.setdefault(call_id, {'begin': [], 'end': []})
                    if len(slot[kind[5:]]) < 3:
                        slot[kind[5:]].append({k:item.get(k) for k in ('pid','tid','label','layer_index','outcome')})
                    else:
                        stats['metadata_rows_clipped'] += 1
        except Exception as e:
            stats['files_error'] += 1
            errors.append({'file': p.name, 'error_type': type(e).__name__})
    return result, dict(stats), errors[:SAMPLE_LIMIT]

def events(path):
    with path.open('rb') as f:
        prefix = f.read(1024).lstrip()
    if not prefix:
        raise ValueError('empty_trace')
    if prefix[:1] == b'[':
        selector = 'item'
        top_type = 'array'
    elif prefix[:1] == b'{':
        selector = 'traceEvents.item'
        top_type = 'object'
    else:
        raise ValueError('unsupported_trace_root')
    if ijson:
        with path.open('rb') as f:
            yield from ijson.items(f, selector)
        return
    if path.stat().st_size > MAX_JSON_BYTES:
        raise RuntimeError('ijson_unavailable_file_over_200MiB')
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f, parse_float=Decimal)
    if top_type == 'array' and isinstance(obj, list):
        yield from obj
    elif top_type == 'object' and isinstance(obj, dict) and isinstance(obj.get('traceEvents'), list):
        yield from obj['traceEvents']
    else:
        raise ValueError('missing_traceEvents_array')

def read_scopes(path):
    scopes = []; counts = collections.Counter()
    for e in events(path):
        counts['trace_events_seen'] += 1
        if not isinstance(e, dict):
            continue
        name = e.get('name')
        if not isinstance(name, str) or not name.startswith('wio_diag:'):
            continue
        counts['named_events'] += 1
        match = NAME.fullmatch(name)
        if not match:
            counts['unrecognized_scope_name'] += 1; continue
        try:
            raw_ts, raw_dur = e.get('ts'), e.get('dur')
            if isinstance(raw_ts, bool) or isinstance(raw_dur, bool):
                raise ValueError('boolean_time')
            ts, dur = Decimal(str(raw_ts)), Decimal(str(raw_dur))
            if e.get('ph') != 'X' or not ts.is_finite() or not dur.is_finite() or dur < 0:
                raise ValueError('unsupported_phase_or_time')
        except (ValueError, InvalidOperation, TypeError):
            counts['unsupported_phase_or_time'] += 1; continue
        if len(scopes) >= MAX_SCOPES:
            counts['scope_limit_clipped'] += 1; continue
        scopes.append({'label': match.group(1), 'call_id': match.group(2), 'pid': e.get('pid'), 'tid': e.get('tid'), 'ts': ts, 'dur': dur})
    return scopes, counts

def analyze(scopes, metadata):
    stats = collections.Counter(); samples = []; grouped = collections.defaultdict(list)
    ids = collections.Counter(s['call_id'] for s in scopes)
    stats['scopes'] = len(scopes)
    stats['duplicate_call_ids'] = sum(1 for n in ids.values() if n > 1)
    stats['duplicate_scope_rows'] = sum(n - 1 for n in ids.values() if n > 1)
    for s in scopes:
        m = metadata.get(s['call_id'])
        if not m:
            stats['no_metadata_call_id'] += 1; status = 'missing'
        elif len(m['begin']) != 1 or len(m['end']) != 1:
            stats['metadata_not_unique_or_unpaired'] += 1; status = 'not_unique_or_unpaired'
        else:
            begin, end = m['begin'][0], m['end'][0]
            if begin.get('label') != s['label'] or end.get('label') != s['label']:
                stats['label_mismatch'] += 1; status = 'label_mismatch'
            else:
                stats['metadata_label_matched'] += 1; status = 'label_matched'
                if begin.get('pid') == s['pid'] and begin.get('tid') == s['tid']:
                    stats['metadata_numeric_pid_tid_equal'] += 1
                else:
                    stats['metadata_numeric_pid_tid_differ_or_encoded'] += 1
                s['layer_index'] = begin.get('layer_index')
        s['metadata_status'] = status
        grouped[(str(s['pid']), str(s['tid']))].append(s)
    for group in grouped.values():
        group.sort(key=lambda s: (s['ts'], -(s['ts'] + s['dur'])))
        active = []
        for s in group:
            st, en = s['ts'], s['ts'] + s['dur']
            while active and active[-1]['end'] <= st:
                active.pop()
            if active:
                parent = active[-1]
                if en <= parent['end']:
                    stats['same_trace_thread_nested'] += 1
                    s['nested_parent_call_id'] = parent['call_id']
                else:
                    stats['same_trace_thread_partial_overlap'] += 1
                    # Crossing intervals do not establish a parent. Resetting
                    # prevents a crossing range being used as a nesting stack.
                    active.clear()
            active.append({'end': en, 'call_id': s['call_id']})
    for s in scopes[:SAMPLE_LIMIT]:
        sample = dict(s)
        sample['ts'], sample['dur'] = str(sample['ts']), str(sample['dur'])
        samples.append(sample)
    stats['metadata_call_ids_absent_from_trace'] = sum(1 for k in metadata if k not in ids)
    return {'counts': dict(stats), 'samples': samples,
            'limits': ['Only wio_diag record_function scopes; nesting is on exported pid/tid and trace clock.',
                       'CPU scope duration is not NPU execution time; no NPU fraction or request critical-path claim.',
                       'PID/TID equality is a raw check, not proof of time-domain or worker identity semantics.',
                       'Per-rank metadata absence counts include metadata from other processes/ranks and are not global missing counts.',
                       'Capture may omit graph-replay layer scopes; omitted or clipped data is not zero activity.']}

def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: summarize_profile_scopes.py DIAGNOSTIC_RUNTIME_ROOT')
    root = Path(sys.argv[1]).resolve(strict=True)
    if not (root / 'metadata').is_dir() or not (root / 'diagnostic-traces').is_dir():
        raise SystemExit('Expected metadata/ and diagnostic-traces/ under diagnostic runtime root')
    meta, meta_stats, errors = read_metadata(root)
    output = {'metadata_counts': meta_stats, 'instrumentation_error_samples': errors,
              'parser': 'ijson' if ijson else 'json_limited_200MiB', 'ranks': []}
    for rank in sorted((root / 'diagnostic-traces').glob('*_ascend_pt')):
        if rank.is_symlink() or not rank.is_dir():
            continue
        item = {'rank_directory': rank.name, 'profiler_info': configurations(rank)}
        path = rank / 'ASCEND_PROFILER_OUTPUT' / 'trace_view.json'
        if path.is_symlink() or not path.is_file():
            item['trace_status'] = 'missing_or_unsafe'
        else:
            item['trace_bytes'] = path.stat().st_size
            try:
                scopes, counts = read_scopes(path)
                item.update(trace_status='read', trace_sha256=file_sha(path), parse_counts=dict(counts), analysis=analyze(scopes, meta))
                if not counts['trace_events_seen']:
                    item['trace_status'] = 'empty_or_unrecognized_traceEvents'
                elif counts['scope_limit_clipped']:
                    item['trace_status'] = 'incomplete_scope_limit'
            except Exception as e:
                item.update(trace_status='incomplete_or_error', error_type=type(e).__name__, reason=str(e)[:150])
        output['ranks'].append(item)
    print(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False))

if __name__ == '__main__':
    main()
