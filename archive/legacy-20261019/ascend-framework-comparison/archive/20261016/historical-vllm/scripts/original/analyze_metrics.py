#!/usr/bin/env python3
"""Offline aggregation of capture_metrics.py Prometheus snapshots; emits no raw metrics."""
import argparse
import json
import math
import pathlib
import re
import sys
from collections import defaultdict

DIRECT_COUNTERS = (
    'vllm:request_success_total',
    'vllm:prompt_tokens_total',
    'vllm:generation_tokens_total',
    'vllm:prefix_cache_queries_total',
    'vllm:prefix_cache_hits_total',
    'vllm:external_prefix_cache_queries_total',
    'vllm:external_prefix_cache_hits_total',
    'vllm:spec_decode_num_drafts_total',
    'vllm:spec_decode_num_draft_tokens_total',
    'vllm:spec_decode_num_accepted_tokens_total',
)
HISTOGRAMS = {
    'prefill': 'vllm:request_prefill_time_seconds',
    'queue': 'vllm:request_queue_time_seconds',
    'decode': 'vllm:request_decode_time_seconds',
    'ttft': 'vllm:time_to_first_token_seconds',
}
GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
VALUE_RE = re.compile(r'^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$')
LABEL_NAME_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def parse_labels(text):
    """Parse Prometheus label assignments, respecting escaped quotes and commas."""
    labels = {}
    i = 0
    n = len(text)
    while i < n:
        while i < n and (text[i].isspace() or text[i] == ','):
            i += 1
        if i >= n:
            break
        start = i
        while i < n and (text[i].isalnum() or text[i] == '_'):
            i += 1
        name = text[start:i]
        if not LABEL_NAME_RE.match(name):
            raise ValueError('bad label name')
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '=':
            raise ValueError('missing label equals')
        i += 1
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '"':
            raise ValueError('label value must be quoted')
        i += 1
        raw = []
        escaped = False
        while i < n:
            ch = text[i]
            if escaped:
                raw.extend(('\\', ch))
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                break
            else:
                raw.append(ch)
            i += 1
        if i >= n or text[i] != '"':
            raise ValueError('unterminated label value')
        i += 1
        labels[name] = json.loads('"' + ''.join(raw) + '"')
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] != ',':
            raise ValueError('bad label separator')
    return labels


def parse_sample_line(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    brace = line.find('{')
    if brace < 0:
        parts = line.split()
        if len(parts) < 2:
            return None
        name, label_text, rest = parts[0], '', parts[1:]
    else:
        name = line[:brace]
        quoted = False
        escaped = False
        close = None
        for i in range(brace + 1, len(line)):
            ch = line[i]
            if escaped:
                escaped = False
            elif ch == '\\' and quoted:
                escaped = True
            elif ch == '"':
                quoted = not quoted
            elif ch == '}' and not quoted:
                close = i
                break
        if close is None:
            raise ValueError('unterminated labels')
        label_text = line[brace + 1:close]
        rest = line[close + 1:].split()
    if not rest:
        return None
    value_text = rest[0]
    if not VALUE_RE.match(value_text):
        return None
    value = float(value_text)
    if not math.isfinite(value):
        return None
    labels = parse_labels(label_text)
    return name, labels, value


def is_selected(name):
    if name in DIRECT_COUNTERS or name in GAUGES:
        return True
    for base in HISTOGRAMS.values():
        if name in (base + '_count', base + '_sum'):
            return True
    return False


def parse_prometheus(text):
    series = {}
    malformed = 0
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        try:
            parsed = parse_sample_line(line)
        except (ValueError, json.JSONDecodeError):
            malformed += 1
            continue
        if parsed is None:
            continue
        name, labels, value = parsed
        if not is_selected(name):
            continue
        key = (name, tuple(sorted(labels.items())))
        # Duplicate identical series are invalid exposition; retain their sum and count it.
        series[key] = series.get(key, 0.0) + value
    return series, malformed


def load_snapshot(path):
    path = pathlib.Path(path)
    if not path.exists():
        return {'status': 'file_missing', 'time_ns': None, 'endpoints': {}}
    raw = path.read_text(errors='replace').strip()
    if raw == 'NOT_CAPTURED':
        return {'status': 'not_captured', 'time_ns': None, 'endpoints': {}}
    if not raw:
        return {'status': 'empty_file', 'time_ns': None, 'endpoints': {}}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # capture_metrics.py sampling mode writes one compact JSON object per line.
        rows = []
        for line in raw.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            return {'status': 'invalid_json', 'time_ns': None, 'endpoints': {}}
        obj = rows[-1]
    if not isinstance(obj, dict):
        return {'status': 'invalid_shape', 'time_ns': None, 'endpoints': {}}
    return {'status': 'ok', 'time_ns': obj.get('time_ns'),
            'endpoints': obj.get('endpoints', {}) if isinstance(obj.get('endpoints'), dict) else {}}


def endpoint_parse(snapshot, role):
    if snapshot.get('status') != 'ok':
        return {'status': snapshot.get('status', 'invalid_snapshot'), 'series': {}, 'malformed_lines': 0}
    entry = snapshot.get('endpoints', {}).get(role)
    if not isinstance(entry, dict):
        return {'status': 'endpoint_missing', 'series': {}, 'malformed_lines': 0}
    if not entry.get('ok'):
        return {'status': 'endpoint_not_ready', 'series': {}, 'malformed_lines': 0,
                'error_class': (entry.get('error', '').split(':', 1)[0] or 'fetch_error')[:80]}
    body = entry.get('text')
    if not isinstance(body, str) or not body.strip():
        return {'status': 'empty_metrics', 'series': {}, 'malformed_lines': 0}
    series, malformed = parse_prometheus(body)
    if not series:
        return {'status': 'no_selected_series', 'series': {}, 'malformed_lines': malformed}
    return {'status': 'ok', 'series': series, 'malformed_lines': malformed}


def engine_of(labels):
    return labels.get('engine', 'unknown')


def metric_names():
    names = list(DIRECT_COUNTERS) + list(GAUGES)
    for base in HISTOGRAMS.values():
        names.extend((base + '_count', base + '_sum'))
    return names


def role_engine_series(parsed):
    grouped = defaultdict(dict)
    for (name, label_tuple), value in parsed['series'].items():
        labels = dict(label_tuple)
        role_engine = engine_of(labels)
        # Preserve series-level labels for reset/missing-series detection.
        grouped[role_engine][(name, label_tuple)] = value
    return grouped


def detect_endpoint_aliases(start_parsed, end_parsed, roles):
    """Find P endpoints exposing identical selected counters at both snapshots.

    Identical values alone cannot prove shared ownership, so this is reported as
    an alias warning and duplicate endpoint deltas are excluded from sums.
    """
    p_roles = [r for r in roles if re.fullmatch(r'P\d+', r)]
    if len(p_roles) < 2:
        return {}

    def signature(parsed):
        if parsed.get('status') != 'ok':
            return None
        return {key: value for key, value in parsed['series'].items()
                if key[0] not in GAUGES}

    starts = {r: signature(start_parsed[r]) for r in p_roles}
    ends = {r: signature(end_parsed[r]) for r in p_roles}
    if any(v is None for v in starts.values()) or any(v is None for v in ends.values()):
        return {}
    representative = p_roles[0]
    if all(starts[r] == starts[representative] and ends[r] == ends[representative]
           for r in p_roles[1:]):
        return {r: representative for r in p_roles[1:]}
    return {}


def delta_for_series(start, end, valid_endpoints):
    if not valid_endpoints:
        return None, 'endpoint_not_ready'
    if start is None and end is None:
        return None, 'metric_missing'
    if start is None and end is not None:
        return end, 'new_series_assumed_zero_baseline'
    if start is not None and end is None:
        return None, 'series_missing_after'
    if end < start:
        return None, 'counter_reset'
    return end - start, 'ok'


def phase_delta(start_snap, end_snap, phase_name, start_name, end_name, expected_p=4, expected_d=16):
    roles = sorted(set(start_snap.get('endpoints', {})) | set(end_snap.get('endpoints', {})),
                   key=lambda x: (x != 'D', x))
    start_parsed = {r: endpoint_parse(start_snap, r) for r in roles}
    end_parsed = {r: endpoint_parse(end_snap, r) for r in roles}
    endpoint_aliases = detect_endpoint_aliases(start_parsed, end_parsed, roles)
    out = {'phase': phase_name, 'from': start_name, 'to': end_name,
           'duration_s': None, 'endpoint_status': {}, 'by_role_engine': {},
           'by_role': {}, 'totals': {}, 'success_increment_checks': {}, 'phase_means': {},
           'endpoint_aliases': endpoint_aliases}
    if isinstance(start_snap.get('time_ns'), (int, float)) and isinstance(end_snap.get('time_ns'), (int, float)):
        out['duration_s'] = (end_snap['time_ns'] - start_snap['time_ns']) / 1e9
    for role in roles:
        out['endpoint_status'][role] = {
            'from': start_parsed[role]['status'], 'to': end_parsed[role]['status'],
            'malformed_lines_from': start_parsed[role]['malformed_lines'],
            'malformed_lines_to': end_parsed[role]['malformed_lines'],
        }
        if role in endpoint_aliases:
            out['by_role_engine'][role] = {'unknown': {name: {
                'delta': None, 'status': 'duplicate_global_alias',
                'alias_of': endpoint_aliases[role]} for name in metric_names()}}
            out['by_role'][role] = {name: {'delta': None,
                'status': 'duplicate_global_alias', 'alias_of': endpoint_aliases[role]}
                for name in metric_names()}
            continue
        s_groups = role_engine_series(start_parsed[role])
        e_groups = role_engine_series(end_parsed[role])
        if start_parsed[role]['status'] != 'ok' or end_parsed[role]['status'] != 'ok':
            bad = end_parsed[role]['status'] if end_parsed[role]['status'] != 'ok' else start_parsed[role]['status']
            status = 'endpoint_not_ready' if bad in ('endpoint_not_ready', 'endpoint_missing', 'empty_metrics') else bad
            out['by_role_engine'][role] = {'unknown': {name: {'delta': None, 'status': status}
                                                       for name in metric_names()}}
            out['by_role'][role] = {name: {'delta': None, 'status': status} for name in metric_names()}
            continue
        engines = sorted(set(s_groups) | set(e_groups), key=lambda x: (x == 'unknown', int(x) if x.isdigit() else x))
        if not engines:
            engines = ['unknown']
        role_engine_results = {}
        for engine in engines:
            s_series = s_groups.get(engine, {})
            e_series = e_groups.get(engine, {})
            results = {}
            for name in metric_names():
                relevant_s = {k: v for k, v in s_series.items() if k[0] == name}
                relevant_e = {k: v for k, v in e_series.items() if k[0] == name}
                if not relevant_s and not relevant_e:
                    results[name] = {'delta': None, 'status': 'metric_missing'}
                    continue
                keys = set(relevant_s) | set(relevant_e)
                deltas = []
                statuses = []
                for key in keys:
                    delta, status = delta_for_series(relevant_s.get(key), relevant_e.get(key),
                                                     start_parsed[role]['status'] == 'ok' and end_parsed[role]['status'] == 'ok')
                    statuses.append(status)
                    if delta is not None:
                        deltas.append(delta)
                if any(s in ('endpoint_not_ready', 'endpoint_missing', 'empty_metrics', 'no_selected_series') for s in statuses):
                    results[name] = {'delta': None, 'status': 'endpoint_not_ready'}
                elif 'counter_reset' in statuses:
                    results[name] = {'delta': None, 'status': 'counter_reset'}
                elif 'series_missing_after' in statuses:
                    results[name] = {'delta': None, 'status': 'series_missing_after'}
                elif deltas:
                    status = 'new_series_assumed_zero_baseline' if 'new_series_assumed_zero_baseline' in statuses else 'ok'
                    results[name] = {'delta': sum(deltas), 'status': status}
                else:
                    results[name] = {'delta': None, 'status': 'metric_missing'}
            role_engine_results[engine] = results
        out['by_role_engine'][role] = role_engine_results
        role_summary = {}
        for name in metric_names():
            present = [v[name] for v in role_engine_results.values() if name in v]
            bad = [x['status'] for x in present if x['status'] not in ('ok', 'new_series_assumed_zero_baseline')]
            vals = [x['delta'] for x in present if x['delta'] is not None]
            if bad:
                role_summary[name] = {'delta': None, 'status': 'counter_reset' if 'counter_reset' in bad else ('endpoint_not_ready' if 'endpoint_not_ready' in bad else 'partial_or_missing')}
            elif vals:
                role_summary[name] = {'delta': sum(vals), 'status': 'ok'}
            else:
                role_summary[name] = {'delta': None, 'status': 'metric_missing'}
        out['by_role'][role] = role_summary

    # Totals are provided only when all observed endpoint-role values are valid.
    for name in metric_names():
        vals, statuses = [], []
        for role, metrics in out['by_role'].items():
            item = metrics[name]
            statuses.append(item['status'])
            if item['delta'] is not None:
                vals.append(item['delta'])
        if any(s not in ('ok',) for s in statuses):
            status = 'counter_reset' if 'counter_reset' in statuses else 'partial_or_missing'
            delta = None
        elif vals:
            status, delta = 'ok', sum(vals)
        else:
            status, delta = 'metric_missing', None
        out['totals'][name] = {'delta': delta, 'status': status}

    success = 'vllm:request_success_total'
    p_roles = ['P%d' % i for i in range(expected_p)]
    p_checks = {}
    for role in p_roles:
        item = out['by_role'].get(role, {}).get(success, {'delta': None, 'status': 'endpoint_missing'})
        p_checks[role] = {'delta': item['delta'], 'status': item['status'],
                          'positive_increment': (item['delta'] > 0 if item['delta'] is not None else None),
                          **({'alias_of': item['alias_of']} if 'alias_of' in item else {})}
    p_values = [p_checks[r]['positive_increment'] for r in p_roles]
    p4_check = {
        'per_role': p_checks,
        'all_roles_positive': (all(p_values) if all(v is not None for v in p_values) else None),
        'roles_observed': sum(v is not None for v in p_values), 'expected_roles': expected_p,
    }
    if endpoint_aliases:
        representative = next(iter(endpoint_aliases.values()))
        engines = out['by_role_engine'].get(representative, {})
        per_engine = {}
        for i in range(expected_p):
            engine = str(i)
            item = engines.get(engine, {}).get(success, {'delta': None, 'status': 'engine_missing'})
            per_engine[engine] = {'delta': item['delta'], 'status': item['status'],
                                  'positive_increment': (item['delta'] > 0 if item['delta'] is not None else None)}
        engine_values = [per_engine[str(i)]['positive_increment'] for i in range(expected_p)]
        p4_check.update({
            'all_roles_positive': None,
            'global_engine_check': {
                'representative_role': representative,
                'per_engine': per_engine,
                'all_engines_positive': (all(engine_values) if all(v is not None for v in engine_values) else None),
                'engines_observed': sum(v is not None for v in engine_values),
                'expected_engines': expected_p,
            },
        })
    out['success_increment_checks']['P4'] = p4_check
    d_engines = {}
    for i in range(expected_d):
        engine = str(i)
        item = out['by_role_engine'].get('D', {}).get(engine, {}).get(success,
                    {'delta': None, 'status': 'engine_missing'})
        d_engines[engine] = {'delta': item['delta'], 'status': item['status'],
                             'positive_increment': (item['delta'] > 0 if item['delta'] is not None else None)}
    d_values = [d_engines[str(i)]['positive_increment'] for i in range(expected_d)]
    out['success_increment_checks']['D16'] = {
        'per_engine': d_engines,
        'all_engines_positive': (all(d_values) if all(v is not None for v in d_values) else None),
        'engines_observed': sum(v is not None for v in d_values), 'expected_engines': expected_d,
    }

    for phase, base in HISTOGRAMS.items():
        count_name, sum_name = base + '_count', base + '_sum'
        role_values, engine_values = {}, {}
        for role, engines in out['by_role_engine'].items():
            counts, sums, statuses = [], [], []
            engine_values[role] = {}
            for engine, metrics in engines.items():
                c, s = metrics[count_name], metrics[sum_name]
                statuses.extend((c['status'], s['status']))
                if c['delta'] is not None: counts.append(c['delta'])
                if s['delta'] is not None: sums.append(s['delta'])
                if c['delta'] is not None and s['delta'] is not None and c['delta'] > 0:
                    engine_values[role][engine] = {
                        'count_delta': c['delta'], 'sum_delta_seconds': s['delta'],
                        'mean_seconds': s['delta']/c['delta'], 'mean_milliseconds': 1000*s['delta']/c['delta'],
                        'status': 'ok',
                    }
                elif c['delta'] == 0 and s['delta'] == 0:
                    engine_values[role][engine] = {
                        'count_delta': 0, 'sum_delta_seconds': 0, 'mean_seconds': None,
                        'mean_milliseconds': None, 'status': 'no_observations',
                    }
                else:
                    engine_values[role][engine] = {
                        'count_delta': c['delta'], 'sum_delta_seconds': s['delta'],
                        'mean_seconds': None, 'mean_milliseconds': None,
                        'status': ('counter_reset' if 'counter_reset' in (c['status'], s['status']) else
                                   'endpoint_not_ready' if 'endpoint_not_ready' in (c['status'], s['status']) else
                                   'metric_missing'),
                    }
            if any(x not in ('ok', 'new_series_assumed_zero_baseline', 'metric_missing') for x in statuses):
                role_values[role] = {'count_delta': None, 'sum_delta_seconds': None, 'mean_seconds': None,
                                     'status': 'counter_reset' if 'counter_reset' in statuses else 'partial_or_missing'}
            elif 'metric_missing' in statuses and (counts or sums):
                role_values[role] = {'count_delta': None, 'sum_delta_seconds': None, 'mean_seconds': None,
                                     'status': 'partial_or_missing'}
            elif counts and sums:
                count, total = sum(counts), sum(sums)
                role_values[role] = {'count_delta': count, 'sum_delta_seconds': total,
                                     'mean_seconds': (total/count if count > 0 else None),
                                     'mean_milliseconds': (1000*total/count if count > 0 else None), 'status': 'ok'}
            else:
                role_values[role] = {'count_delta': None, 'sum_delta_seconds': None, 'mean_seconds': None,
                                     'status': 'metric_missing'}
        out['phase_means'][phase] = {'by_role_engine': engine_values, 'by_role': role_values}
    return out


def load_samples(path):
    path = pathlib.Path(path)
    if not path.exists():
        return {'status': 'not_found'}
    rows, invalid = [], 0
    for line in path.read_text(errors='replace').splitlines():
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: invalid += 1
    if not rows:
        return {'status': 'no_valid_samples', 'invalid_lines': invalid}
    valid_shape_rows = [row for row in rows if isinstance(row, dict)]
    invalid += len(rows) - len(valid_shape_rows)
    rows = valid_shape_rows
    if not rows:
        return {'status': 'no_valid_samples', 'invalid_lines': invalid}
    rows.sort(key=lambda x: x.get('time_ns') or 0)
    parsed_rows = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('time_ns'), (int, float)):
            invalid += 1; continue
        entries = {}
        for role in row.get('endpoints', {}):
            ep = endpoint_parse({'status': 'ok', 'endpoints': row.get('endpoints', {})}, role)
            entries[role] = ep
        parsed_rows.append((float(row['time_ns'])/1e9, entries))
    if not parsed_rows:
        return {'status': 'no_valid_samples', 'invalid_lines': invalid}
    gauges = {}
    for role in sorted(set(r for _, eps in parsed_rows for r in eps), key=lambda x: (x != 'D', x)):
        sample_values = []
        all_keys = set()
        for ts, eps in parsed_rows:
            ep = eps.get(role, {'status': 'endpoint_missing', 'series': {}})
            values = defaultdict(float)
            if ep['status'] == 'ok':
                for (name, label_tuple), value in ep['series'].items():
                    if name in GAUGES:
                        values[(engine_of(dict(label_tuple)), name)] += value
            all_keys.update(values)
            sample_values.append((ts, ep['status'], values))
        for key in all_keys:
            ts_values = [(ts, values.get(key)) for ts, _, values in sample_values]
            span_start, span_end = parsed_rows[0][0], parsed_rows[-1][0]
            observations = [v for _, v in ts_values if v is not None]
            peak = max(observations) if observations else None
            area, covered, gap_intervals = 0.0, 0.0, 0
            for (t0, value), (t1, _) in zip(ts_values, ts_values[1:]):
                dt = t1 - t0
                if dt > 10:
                    gap_intervals += 1
                elif dt > 0 and value is not None:
                    area += value * dt
                    covered += dt
            total_span = max(0.0, span_end-span_start)
            label = key[1]
            gauge_name = 'running' if label == 'vllm:num_requests_running' else 'waiting'
            engine = key[0]
            gauges.setdefault(role, {}).setdefault(engine, {})[gauge_name] = {
                'peak': peak, 'time_weighted_mean': (area/covered if covered > 0 else None),
                'covered_seconds': covered, 'sample_span_seconds': total_span,
                'coverage_fraction': (covered/total_span if total_span > 0 else None),
                'sample_count': len(observations), 'excluded_long_gap_intervals': gap_intervals,
            }
    return {'status': 'ok', 'sample_count': len(parsed_rows), 'invalid_lines': invalid,
            'duration_seconds': max(0.0, parsed_rows[-1][0]-parsed_rows[0][0]),
            'sampling_note': 'Left-continuous time weighting; intervals over 10 seconds are excluded and counted as gaps.',
            'by_role_engine': gauges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=pathlib.Path, help='run directory or metrics directory')
    parser.add_argument('--output', type=pathlib.Path, help='output JSON path (default: metrics/metrics_analysis.json)')
    parser.add_argument('--expected-p-roles', type=int, default=4)
    parser.add_argument('--expected-d-engines', type=int, default=16)
    args = parser.parse_args()
    metrics_dir = args.run_dir / 'metrics'
    names = ('before.metrics', 'before_warmup.metrics', 'after_warmup.metrics',
             'before_full.metrics', 'after.metrics', 'after_full.metrics', 'samples.jsonl')
    if not metrics_dir.is_dir() or not any((metrics_dir / name).exists() for name in names):
        metrics_dir = args.run_dir

    def choose(*basenames):
        for basename in basenames:
            candidate = metrics_dir / basename
            if candidate.exists():
                return candidate
        return metrics_dir / basenames[0]

    # Accept the simple before/after_warmup/after naming and run_prepared.sh's
    # before_warmup/after_warmup/before_full/after_full naming.
    paths = {
        'before': choose('before.metrics', 'before_warmup.metrics'),
        'after_warmup': choose('after_warmup.metrics'),
        'formal_before': choose('before_full.metrics', 'after_warmup.metrics', 'before.metrics'),
        'after': choose('after.metrics', 'after_full.metrics'),
    }
    snaps = {name: load_snapshot(path) for name, path in paths.items()}
    analysis = {
        'source_files': {k: str(v) for k, v in paths.items()},
        'snapshot_status': {k: v['status'] for k, v in snaps.items()},
        'endpoint_status_by_snapshot': {},
        'phases': {},
        'samples': {'status': 'pending'},
        'raw_metrics_included': False,
    }
    for name, snap in snaps.items():
        analysis['endpoint_status_by_snapshot'][name] = {
            role: endpoint_parse(snap, role)['status'] for role in sorted(set(['P%d' % i for i in range(args.expected_p_roles)] + ['D']))
        }
    if snaps['before']['status'] == 'ok' and snaps['after_warmup']['status'] == 'ok':
        analysis['phases']['warmup'] = phase_delta(snaps['before'], snaps['after_warmup'], 'warmup', 'before', 'after_warmup', args.expected_p_roles, args.expected_d_engines)
    else:
        analysis['phases']['warmup'] = {'status': 'snapshot_not_ready', 'from': 'before', 'to': 'after_warmup'}
    if snaps['formal_before']['status'] == 'ok' and snaps['after']['status'] == 'ok':
        analysis['phases']['formal'] = phase_delta(snaps['formal_before'], snaps['after'], 'formal', 'formal_before', 'after', args.expected_p_roles, args.expected_d_engines)
    else:
        analysis['phases']['formal'] = {'status': 'snapshot_not_ready', 'from': 'formal_before', 'to': 'after'}
    analysis['samples'] = load_samples(metrics_dir / 'samples.jsonl')
    out = args.output or (metrics_dir / 'metrics_analysis.json')
    out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + '\n')
    # Console output is deliberately limited to aggregate readiness and delta summaries.
    safe = {
        'snapshot_status': analysis['snapshot_status'],
        'endpoint_status_by_snapshot': analysis['endpoint_status_by_snapshot'],
        'warmup_success_checks': analysis['phases'].get('warmup', {}).get('success_increment_checks'),
        'formal_totals': analysis['phases'].get('formal', {}).get('totals'),
        'sample_status': analysis['samples'].get('status'),
        'output': str(out),
    }
    print(json.dumps(safe, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
