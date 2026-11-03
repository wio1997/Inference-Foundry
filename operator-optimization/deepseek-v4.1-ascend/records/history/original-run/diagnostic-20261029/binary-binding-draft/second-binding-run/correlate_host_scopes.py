#!/usr/bin/env python3
"""Audit captured C hooks against Python scopes; never infer a device execution.

Input must be a stable, task-owned snapshot of the select2 metadata directory.
Output is deliberately conditional on independently established identity/clock
contracts; this script cannot itself prove they applied during the capture.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def read_lines(path):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('input changed during read: ' + str(path))
    rows, malformed = [], 0
    for line in data.splitlines():
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError('not an object')
            rows.append(item)
        except (ValueError, UnicodeDecodeError):
            malformed += 1
    return rows, {'path': str(path), 'bytes': len(data),
                  'sha256': hashlib.sha256(data).hexdigest(),
                  'final_newline': data.endswith(b'\n'), 'malformed_lines': malformed}


def integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metadata', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    meta = args.metadata
    crows, cfile = read_lines(meta / 'binding-trace.jsonl')
    pyrows, pyfiles = [], []
    for path in sorted(meta.glob('events-*.jsonl')):
        r, info = read_lines(path)
        pyrows.extend(r)
        pyfiles.append(info)
    if not pyfiles:
        raise RuntimeError('no Python sidecar files')

    # IDs are unique only within their process lifetime. Refuse duplicate or
    # unpaired keys instead of choosing a convenient event.
    begins, ends = collections.defaultdict(list), collections.defaultdict(list)
    pycounts = collections.Counter()
    for r in pyrows:
        pycounts[r.get('kind', '<missing>')] += 1
        if r.get('kind') in ('call_begin', 'call_end'):
            (begins if r['kind'] == 'call_begin' else ends)[(r.get('pid'), r.get('call_id'))].append(r)
    windows = collections.defaultdict(list)
    scopeaudit = collections.Counter()
    for key in begins.keys() | ends.keys():
        bs, es = begins[key], ends[key]
        if len(bs) != 1 or len(es) != 1:
            scopeaudit['unpaired_or_duplicate_call_id'] += 1
            continue
        b, e = bs[0], es[0]
        if (not all(integer(b.get(x)) for x in ('pid', 'tid', 'perf_counter_ns'))
            or not all(integer(e.get(x)) for x in ('pid', 'tid', 'perf_counter_ns'))
            or b['pid'] != e['pid'] or b['tid'] != e['tid']
            or b.get('label') != e.get('label') or b['perf_counter_ns'] >= e['perf_counter_ns']):
            scopeaudit['invalid_pair'] += 1
            continue
        windows[(b['pid'], b['tid'])].append((b['perf_counter_ns'], e['perf_counter_ns'], b.get('label'), b.get('layer_index'), key[1]))
        scopeaudit['complete_pairs'] += 1

    cb, ce = collections.defaultdict(list), collections.defaultdict(list)
    ccounts = collections.Counter()
    cpid = collections.defaultdict(list)
    backfills = []
    field_sets = collections.defaultdict(collections.Counter)
    required = {
        'target_launch_begin': ('pid', 'tid', 'call', 'monotonic_ns', 'func', 'bin', 'generation', 'entry', 'pre_call_sha256', 'stream', 'kind', 'state_incomplete'),
        'target_launch_end': ('pid', 'tid', 'call', 'monotonic_ns', 'func', 'stream', 'rc'),
        'target_preclassification_launch': ('pid', 'original_tid', 'original_call', 'original_begin_ns', 'monotonic_ns', 'bin', 'func', 'entry', 'generation', 'stream', 'rc', 'state_incomplete'),
        'target_graph_begin': ('pid', 'tid', 'call', 'monotonic_ns', 'model_ri', 'stream', 'state_incomplete'),
        'target_graph_end': ('pid', 'tid', 'call', 'monotonic_ns', 'model_ri', 'stream', 'rc')}
    missing_fields = collections.Counter()
    for r in crows:
        ev = r.get('event', '<missing>')
        ccounts[ev] += 1
        field_sets[ev][','.join(sorted(r))] += 1
        for name in required.get(ev, ()):
            if name not in r:
                missing_fields[ev + ':' + name] += 1
        cpid[r.get('pid')].append(r)
        if ev in ('target_launch_begin', 'target_graph_begin', 'target_launch_end', 'target_graph_end'):
            typ = 'launch' if 'launch' in ev else 'graph'
            key = (r.get('pid'), r.get('tid'), r.get('call'), typ)
            (cb if ev.endswith('_begin') else ce)[key].append(r)
        elif ev == 'target_preclassification_launch':
            backfills.append(r)

    audit = collections.Counter()
    matched = []
    for key in cb.keys() | ce.keys():
        bs, es = cb[key], ce[key]
        if len(bs) != 1 or len(es) != 1:
            audit['unpaired_or_duplicate_c_call'] += 1
            continue
        b, e = bs[0], es[0]
        if (not all(integer(v) for v in key[:3]) or not integer(b.get('monotonic_ns'))
            or not integer(e.get('monotonic_ns')) or b['monotonic_ns'] >= e['monotonic_ns']
            or b.get('stream') != e.get('stream') or e.get('rc') != 0
            or (key[3] == 'launch' and b.get('func') != e.get('func'))
            or (key[3] == 'graph' and b.get('model_ri') != e.get('model_ri'))):
            audit['invalid_or_failed_c_pair'] += 1
            continue
        hits = [w for w in windows.get(key[:2], ()) if w[0] < b['monotonic_ns'] and e['monotonic_ns'] < w[1]]
        target = [w for w in hits if w[2] == 'DeepseekV41DecoderLayer.forward']
        # All enclosing scopes, including other labels, are provided; an
        # overlapping layer pair is ambiguous, never picked as innermost.
        if len(target) == 1:
            audit['one_layer_scope'] += 1
        elif len(target) > 1:
            audit['ambiguous_layer_scope'] += 1
        else:
            audit['no_layer_scope'] += 1
        matched.append({'c_pid': key[0], 'c_tid': key[1], 'c_call': key[2], 'type': key[3],
                        'c_begin_ns': b['monotonic_ns'], 'c_end_ns': e['monotonic_ns'],
                        'kind': b.get('kind'), 'state_incomplete_at_begin': b.get('state_incomplete'),
                        'generation': b.get('generation'), 'pre_call_sha256': b.get('pre_call_sha256'),
                        'model_ri': b.get('model_ri'), 'layer_scope_count': len(target),
                        'enclosing_scopes': [{'label': w[2], 'layer_index': w[3], 'call_id': w[4]} for w in hits]})

    # Backfilled events carry the original entry time only, not a complete
    # launch interval. Never elevate to a full-interval scope association.
    backfill_audit = collections.Counter()
    for r in backfills:
        pid, tid, t = r.get('pid'), r.get('original_tid'), r.get('original_begin_ns')
        if not all(integer(v) for v in (pid, tid, t)) or not t:
            backfill_audit['invalid_original_point'] += 1
            continue
        target = [w for w in windows.get((pid, tid), ()) if w[2] == 'DeepseekV41DecoderLayer.forward' and w[0] < t < w[1]]
        backfill_audit['one_layer_point_only' if len(target) == 1 else 'ambiguous_or_no_layer_point'] += 1

    procs = {}
    for pid, rows in cpid.items():
        stats = [r for r in rows if r.get('event') == 'selective_stats']
        seqs = [r['seq'] for r in rows if integer(r.get('seq'))]
        procs[str(pid)] = {'events': len(rows), 'stats': stats,
                           'duplicate_seq_count': len(seqs) - len(set(seqs)),
                           'dropped_reported': sum(r.get('dropped_since_last', 0) for r in rows if integer(r.get('dropped_since_last'))),
                           'has_process_exit_stats': any(r.get('reason') == 'process_exit' for r in stats),
                           'record_cap': any(r.get('record_cap_reached') is True for r in rows)}
    result = {'status': 'conditional_same_process_thread_and_clock_contract_required',
              'input_binding': cfile, 'input_python': pyfiles,
              'c_event_counts': dict(ccounts), 'python_event_counts': dict(pycounts),
              'scope_pair_audit': dict(scopeaudit), 'c_pair_audit': dict(audit),
              'backfill_audit': dict(backfill_audit), 'process_audit': procs,
              'event_field_sets': {k: dict(v) for k, v in field_sets.items()},
              'required_field_missing': dict(missing_fields),
              'identity_audit': {
                  'python_all_scope_pid_tid_pairs': len(windows),
                  'python_layer_scope_pid_tid_pairs': len({k for k, ws in windows.items() if any(w[2] == 'DeepseekV41DecoderLayer.forward' for w in ws)}),
                  'matched_c_pid_tid_pairs': len({(y['c_pid'], y['c_tid']) for y in matched}),
                  'matched_c_and_any_python_scope_shared_pid_tid_pairs': len({(y['c_pid'], y['c_tid']) for y in matched} & set(windows)),
                  'matched_c_and_layer_scope_shared_pid_tid_pairs': len({(y['c_pid'], y['c_tid']) for y in matched} & {k for k, ws in windows.items() if any(w[2] == 'DeepseekV41DecoderLayer.forward' for w in ws)}),
                  'matched_c_events_on_python_layer_thread': sum((y['c_pid'], y['c_tid']) in {k for k, ws in windows.items() if any(w[2] == 'DeepseekV41DecoderLayer.forward' for w in ws)} for y in matched)},
              'trace_cap_seen': ccounts['trace_cap'] > 0,
              'matches': sorted(matched, key=lambda x: (x['c_pid'], x['c_tid'], x['c_begin_ns'])),
              'limitations': [
                  'Candidate relationships assume independently established same PID/TID namespace and identical monotonic clock semantics; neither is proven by this script.',
                  'Nested host timestamps are not an explicit causal or tensor edge and cannot identify asynchronous worker tasks.',
                  'A graph handle is a model-level candidate, not a graph node or proof of device execution.',
                  'By-entry and pre-call SHA are candidates; state_incomplete and dropped/capped events prevent negative or exhaustive claims.',
                  'Backfill is only a point, using original_tid and original_begin_ns, and does not establish whole-launch nesting.',
                  'File stability during read does not prove recording finished; use a quiescent independently identified snapshot.',
                  'Logs lack process lifetime IDs; PID reuse, fork resets, and missing final stats require independent lifecycle audit.',
                  'Sidecar can silently hit the per-process 10000-event cap or encounter a log write error; absence is not proof of no event.']}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output), 'scope_pair_audit': result['scope_pair_audit'],
                      'c_pair_audit': result['c_pair_audit'], 'backfill_audit': result['backfill_audit'],
                      'trace_cap_seen': result['trace_cap_seen']}))


if __name__ == '__main__':
    main()
