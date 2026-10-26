#!/usr/bin/env python3
"""Prepare or collect read-only C32-L20 paired native profile evidence.

--prepare renders the exact host/container Python sent in --collect, writes the
SHA manifest, and performs no SSH/HTTP/model/SDK call.

--collect is refused unless the master-saved review SHA pins this script, the
prepared manifest, the frozen settings, and every rendered code file.  It first
requires a true coordinator terminal (DONE, client_rc 0, no host_wait_timeout)
and a diagnostic report whose SHA/framework/run_id match.  Model request success
is not a gate.  Each framework has one unique collect run directory and no retry.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PREPARE_RID = '20261019T-L20-native-evidence-prepare-v1'
REVIEW_PATH = 'results/C32-L20-native-evidence-collect-review-v1.json'
SOCKET = '/Users/wio/work/sglang-glm53-pd/ssh-control-l20-foreground-20261019'
INVENTORY_SHA = 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
COLLECT_RUN = {
    'vllm': '20261019T-L20-native-evidence-collect-vllm-v1',
    'sglang': '20261019T-L20-native-evidence-collect-sglang-v1',
}
SETTINGS = {
    'vllm': {
        'path': 'config/C32-L20-paired-diagnostic-v2/vllm-cohort-settings-v2.json',
        'sha256': 'e66efa21f706c53326bcc892bdf4849da75f11a2a362b92649cd6f534b301df6',
    },
    'sglang': {
        'path': 'config/C32-L20-paired-diagnostic-v2/sglang-cohort-settings-v2.json',
        'sha256': '7afc1f15076e5fd69a2278c16ae51842b4135d74b690f507788117f2bf4cdc1e',
    },
}
D_NODES = {'vllm': [43, 45], 'sglang': [36, 41]}
EXPECTED_CIDS = {
    'vllm': {
        43: '0b3266fcc1f72005a763fdbd5191cecb564d67b448508c2e1d42e965a2973779',
        45: 'c1f460d681f216677d48e3ca881c72bd3a325f10b19d2d83cba299d47dae8a99',
    },
    'sglang': {
        36: 'fe95e2fca4f0e2832685f688d0616c060fa5774ad5baf0cad375f9bd4c89bb37',
        41: 'c5e28348c3ede16066218a3729602f7e409d84c55088d719386e8e90dc8c82b9',
    },
}
EXPECTED_IMAGE = {
    'vllm': 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83',
    'sglang': 'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263',
}
PREFERRED_RANK = {'vllm': {43: 0, 45: 16}, 'sglang': {36: 0, 41: 16}}
VLLM_CONTAINER_ROOT = '/run/glm53/profile'
SGLANG_HOST_ROOT = (
    '/data/wio/inference-foundry/glm53/runs/20261019T-L17-D-native-eager-v2/profile/'
    '20261019T-L20-sglang-C32-paired-diagnostic-v2'
)
EXPECTED_WORKERS = 16
NODE_TIMEOUT = 150
MAPPING_TIMEOUT = 10
PROBE_TIMEOUT = 140
INNER_TIMEOUT = 125
MAX_CONCURRENT_SSH = 2

CONTAINER_TEMPLATE = r'''import base64, collections, csv, gzip, hashlib, io, json, os, pathlib, re, time
from decimal import Decimal, InvalidOperation, getcontext

getcontext().prec = 50

NODE = __NODE__
FRAMEWORK = __FRAMEWORK__
RUN_ID = __RUN_ID__
PROFILE_ROOT = __PROFILE_ROOT_LITERAL__
EXPECTED_FOLDERS = __EXPECTED_FOLDERS__
PREFERRED_RANK = __PREFERRED_RANK__
WORKER_FOLDER_RULE = __WORKER_FOLDER_RULE__

CSV_NAMES = ['api_statistic.csv', 'kernel_details.csv', 'op_statistic.csv',
             'operator_details.csv', 'step_trace_time.csv', 'task_time.csv']
CSV_FILE_CAP = 128 * 1024 ** 2
CSV_NODE_CAP = 512 * 1024 ** 2
TRACE_RAW_CAP = 192 * 1024 ** 2
TRACE_GZIP_CAP = 32 * 1024 ** 2
STABILITY_INTERVAL = 3.0
STABILITY_MAX = 60.0
DEADLINE_SECONDS = 115.0
CSV_PHASE_SECONDS = 60.0
TRACE_RESERVE_SECONDS = 45.0
NAME_RE = re.compile(r'forward|draft|verify', re.I)
FOLDER_RE = re.compile(r'^(?P<worker_name>.+)_(?P<pid>[0-9]+)_(?P<timestamp>[0-9]{17,20})_ascend_pt$')
DP_RE = re.compile(r'(?i)(?:^|[_-])(?:dp|data[_-]?parallel)[_-]?([0-9]+)(?:[_-]|$)')
TP_RE = re.compile(r'(?i)(?:^|[_-])(?:tp|tensor[_-]?parallel)[_-]?([0-9]+)(?:[_-]|$)')

CSV_SPECS = {
    'api_statistic.csv': {'name': 'API Name', 'sums': ['Time(us)'], 'counts': ['Count']},
    'kernel_details.csv': {'name': 'Name', 'sums': ['Duration(us)'], 'counts': [],
                           'time_range': ('Start Time(us)', None)},
    'op_statistic.csv': {'name': 'OP Type', 'sums': ['Total Time(us)'], 'counts': ['Count']},
    'operator_details.csv': {'name': 'Name',
                             'sums': ['Host Self Duration(us)', 'Host Total Duration(us)',
                                      'Device Self Duration(us)', 'Device Total Duration(us)'],
                             'counts': []},
    'step_trace_time.csv': {'name': 'Step', 'sums': ['Computing', 'Communication', 'Free'],
                            'counts': [], 'step_rows': True},
    'task_time.csv': {'name': 'kernel_name', 'sums': ['task_time(us)'], 'counts': [],
                      'time_range': ('task_start(us)', 'task_stop(us)')},
}


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def decimal_value(value):
    if value is None:
        return None
    text = str(value).strip().replace(',', '')
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def file_meta(path, root):
    try:
        stat = path.lstat()
    except OSError as error:
        return {'relative': str(path.relative_to(root)), 'missing': True, 'error': repr(error)}
    return {'relative': str(path.relative_to(root)), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'mode': oct(stat.st_mode & 0o777),
            'symlink': path.is_symlink()}


def list_files(root):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames
                             if not (pathlib.Path(dirpath) / name).is_symlink())
        for name in sorted(filenames):
            path = pathlib.Path(dirpath) / name
            try:
                stat = path.lstat()
            except OSError as error:
                rows.append({'relative': str(path.relative_to(root)), 'stat_error': repr(error)})
            else:
                rows.append({'relative': str(path.relative_to(root)), 'bytes': stat.st_size,
                             'mtime_ns': stat.st_mtime_ns, 'symlink': path.is_symlink()})
            if len(rows) >= 8192:
                return rows, True
    return rows, False


def signature(paths):
    rows = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            rows.append((str(path), None, None))
        else:
            rows.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


class HashingRaw(io.RawIOBase):
    def __init__(self, raw, digest, limit):
        self.raw = raw
        self.digest = digest
        self.limit = limit
        self.count = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        size = self.raw.readinto(buffer)
        if not size:
            return 0
        self.count += size
        if self.count > self.limit:
            raise RuntimeError('CSV grew beyond per-file cap')
        self.digest.update(memoryview(buffer)[:size])
        return size


def analyze_csv(path, spec, root, remaining, deadline):
    result = {'relative': str(path.relative_to(root)), 'fieldnames': [], 'rows_total': 0,
              'matched_rows': 0, 'groups': [], 'groups_truncated': False,
              'invalid_examples': [],
              'scope': 'Cumulative per-name sums over matched rows; not critical-path (CP) time.',
              'decimal_strings': True, 'empty_or_invalid_not_zero_filled': True}
    meta = file_meta(path, root)
    result['file'] = meta
    if meta.get('missing') or meta.get('symlink'):
        result['error'] = 'missing_or_symlink'
        return result, 0
    if meta['bytes'] > CSV_FILE_CAP:
        result['skipped'] = 'per_file_cap'
        return result, 0
    if meta['bytes'] > remaining:
        result['skipped'] = 'node_cap'
        return result, 0

    before = path.stat()
    digest = hashlib.sha256()
    groups = {}
    invalid = collections.Counter()
    valid = collections.Counter()
    examples = []
    time_range = {}
    step_rows = []
    row_index = 0
    read_complete = False
    text = None
    raw = None
    hr = None
    try:
        raw = path.open('rb', buffering=0)
        hr = HashingRaw(raw, digest, CSV_FILE_CAP)
        buffered = io.BufferedReader(hr, buffer_size=1024 * 1024)
        text = io.TextIOWrapper(buffered, encoding='utf-8-sig', newline='')
        reader = csv.DictReader(text)
        result['fieldnames'] = list(reader.fieldnames or [])
        required = [spec['name']] + list(spec.get('sums', [])) + list(spec.get('counts', []))
        result['missing_columns'] = [name for name in required
                                     if name not in result['fieldnames']]
        for row in reader:
            row_index += 1
            if row_index % 10000 == 0 and time.monotonic() > deadline:
                raise RuntimeError('CSV time budget exceeded')
            result['rows_total'] = row_index
            name = (row.get(spec['name']) or '').strip()
            if not name or not NAME_RE.search(name):
                continue
            if len(name) > 300:
                name = name[:300]
            result['matched_rows'] += 1
            group = groups.get(name)
            if group is None:
                group = {
                    'name': name, 'rows': 0,
                    'sums': {column: None for column in spec.get('sums', [])},
                    'valid': {column: 0 for column in spec.get('sums', [])},
                    'invalid': {column: 0 for column in spec.get('sums', [])},
                    'count_sums': {column: None for column in spec.get('counts', [])},
                    'count_valid': {column: 0 for column in spec.get('counts', [])},
                    'count_invalid': {column: 0 for column in spec.get('counts', [])},
                }
                groups[name] = group
            group['rows'] += 1
            for column in spec.get('sums', []):
                value = decimal_value(row.get(column))
                if value is None:
                    group['invalid'][column] += 1
                    invalid[column] += 1
                    if len(examples) < 8:
                        examples.append({'row': row_index, 'name': name, 'column': column,
                                         'raw': str(row.get(column, ''))[:200]})
                else:
                    group['valid'][column] += 1
                    valid[column] += 1
                    current = group['sums'][column]
                    group['sums'][column] = value if current is None else current + value
            for column in spec.get('counts', []):
                value = decimal_value(row.get(column))
                if value is None:
                    group['count_invalid'][column] += 1
                    invalid[column] += 1
                    if len(examples) < 8:
                        examples.append({'row': row_index, 'name': name, 'column': column,
                                         'raw': str(row.get(column, ''))[:200]})
                else:
                    group['count_valid'][column] += 1
                    valid[column] += 1
                    current = group['count_sums'][column]
                    group['count_sums'][column] = value if current is None else current + value
            start_column, stop_column = spec.get('time_range', (None, None))
            if start_column:
                value = decimal_value(row.get(start_column))
                if value is None:
                    invalid[start_column] += 1
                else:
                    valid[start_column] += 1
                    if 'start_min_us' not in time_range or value < time_range['start_min_us']:
                        time_range['start_min_us'] = value
            if stop_column:
                value = decimal_value(row.get(stop_column))
                if value is None:
                    invalid[stop_column] += 1
                else:
                    valid[stop_column] += 1
                    if 'stop_max_us' not in time_range or value > time_range['stop_max_us']:
                        time_range['stop_max_us'] = value
            if spec.get('step_rows') and len(step_rows) < 64:
                step_rows.append({key: ('' if value is None else str(value))
                                  for key, value in row.items() if key is not None})
        read_complete = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        if text is not None:
            try:
                text.close()
            except Exception:
                pass
        elif raw is not None:
            try:
                raw.close()
            except Exception:
                pass

    try:
        after = path.stat()
    except OSError as error:
        after = None
        result['stat_error'] = repr(error)
    result['bytes_read'] = hr.count if hr is not None else 0
    result['sha256_complete'] = bool(read_complete and after is not None and hr is not None
                                     and hr.count == before.st_size)
    result['sha256'] = digest.hexdigest() if read_complete else None
    result['changed_during_read'] = bool(
        after is None or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns))
    result['invalid_rows_by_column'] = dict(invalid)
    result['valid_rows_by_column'] = dict(valid)
    result['invalid_examples'] = examples
    result['time_range_decimal_us'] = {key: str(value) for key, value in time_range.items()}
    result['step_rows'] = step_rows

    rows = []
    for group in sorted(groups.values(), key=lambda item: (-item['rows'], item['name']))[:128]:
        rows.append({
            'name': group['name'], 'rows': group['rows'],
            'sums': {column: (None if value is None else str(value))
                     for column, value in group['sums'].items()},
            'valid_rows': group['valid'], 'invalid_rows': group['invalid'],
            'reported_count_sums': {column: (None if value is None else str(value))
                                    for column, value in group['count_sums'].items()},
            'reported_count_valid_rows': group['count_valid'],
            'reported_count_invalid_rows': group['count_invalid'],
        })
    result['groups'] = rows
    result['groups_truncated'] = len(groups) > 128
    return result, result['bytes_read']


def stable_signature(paths, deadline):
    started = time.monotonic()
    limit = min(STABILITY_MAX, max(0.0, deadline - started))
    previous = None
    checks = 0
    stable = False
    while True:
        current = signature(paths)
        checks += 1
        if previous is not None and current == previous:
            stable = True
            break
        elapsed = time.monotonic() - started
        if elapsed >= limit:
            break
        previous = current
        time.sleep(min(STABILITY_INTERVAL, max(0.0, limit - elapsed)))
    return {'stable': stable, 'checks': checks,
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'limit_seconds': round(limit, 3), 'paths': len(paths)}


def build_worker(folder, root, csv_remaining, deadline, stability):
    out = folder / 'ASCEND_PROFILER_OUTPUT'
    infos = sorted(list(folder.glob('profiler_info*.json'))
                   + list(out.glob('profiler_info*.json')))[:8]
    csv_paths = [out / name for name in CSV_NAMES]

    info_doc = None
    info_sha = None
    info_bytes = None
    info_error = None
    if infos:
        try:
            root_resolved = root.resolve(strict=True)
            if infos[0].is_symlink():
                info_error = 'symlink_refused'
            elif not infos[0].resolve(strict=True).is_relative_to(root_resolved):
                info_error = 'outside_profile_root'
            elif infos[0].stat().st_size > 1024 * 1024:
                info_error = 'profiler_info_over_1MiB'
            else:
                raw_info = infos[0].read_bytes()
                info_bytes = len(raw_info)
                info_sha = sha256_bytes(raw_info)
                info_doc = json.loads(raw_info)
        except Exception as error:
            info_error = repr(error)
    else:
        info_error = 'missing'

    info_timing = {}
    info_keys = []
    rank = None
    if isinstance(info_doc, dict):
        info_keys = list(info_doc)[:64]
        rank = info_doc.get('rank_id')
        start_info = info_doc.get('start_info')
        end_info = info_doc.get('end_info')
        start_value = start_info.get('start_monotonic') if isinstance(start_info, dict) else None
        end_value = end_info.get('MonotonicTimeEnd') if isinstance(end_info, dict) else None
        info_timing = {'start_monotonic_raw': start_value,
                       'end_MonotonicTimeEnd_raw': end_value,
                       'start_type': type(start_value).__name__,
                       'end_type': type(end_value).__name__}

    info_files = [file_meta(path, root) for path in infos]
    csv_files = {path.name: file_meta(path, root) for path in csv_paths}
    trace_meta = file_meta(out / 'trace_view.json', root)
    identity = FOLDER_RE.match(folder.name)
    worker_name = identity.group('worker_name') if identity else None
    pid = int(identity.group('pid')) if identity else None
    timestamp = identity.group('timestamp') if identity else None
    dp_match = DP_RE.search(folder.name)
    tp_match = TP_RE.search(folder.name)
    derived_dp = int(dp_match.group(1)) if dp_match else None
    derived_tp = int(tp_match.group(1)) if tp_match else None

    csv_results = {}
    worker_complete = bool(infos) and isinstance(info_doc, dict) and stability['stable']
    for path in csv_paths:
        if time.monotonic() > deadline:
            csv_results[path.name] = {'relative': str(path.relative_to(root)),
                                      'skipped': 'time_budget'}
            worker_complete = False
            continue
        analysis, used = analyze_csv(
            path, CSV_SPECS[path.name], root, csv_remaining, deadline)
        csv_results[path.name] = analysis
        csv_remaining -= used
        if (analysis.get('sha256') is None or analysis.get('changed_during_read')
                or analysis.get('error') or analysis.get('skipped')
                or analysis.get('missing_columns')):
            worker_complete = False
    if len(csv_results) != len(CSV_NAMES) or any(
            value.get('missing') for value in csv_files.values()):
        worker_complete = False

    worker = {
        'folder': str(folder.relative_to(root)),
        'worker_name': worker_name,
        'pid': pid,
        'timestamp': timestamp,
        'identity_source': 'actual_profile_folder_name',
        'derived_dp': derived_dp,
        'derived_tp': derived_tp,
        'derived_dp_evidence': folder.name if derived_dp is not None else None,
        'derived_tp_evidence': folder.name if derived_tp is not None else None,
        'native_rank_id': rank,
        'native_rank_id_type': type(rank).__name__,
        'profiler_info_files': info_files,
        'profiler_info': info_doc,
        'profiler_info_bytes': info_bytes,
        'profiler_info_sha256': info_sha,
        'profiler_info_error': info_error,
        'profiler_info_timing_raw': info_timing,
        'profiler_info_top_level_keys': info_keys,
        'required_files': {'profiler_info': info_files, 'csv': csv_files,
                           'trace_view': trace_meta},
        'stability': stability,
        'csv': csv_results,
        'worker_complete': worker_complete,
    }
    return worker, csv_remaining


report = {'project_id': 'glm53', 'framework': FRAMEWORK, 'node': NODE, 'run_id': RUN_ID,
          'mode': 'read_only_evidence', 'model_requests': 0, 'SDK_init_calls': 0,
          'new_profiles': 0, 'remote_writes': 0, 'pass': False}
try:
    root = pathlib.Path(PROFILE_ROOT)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError('profile root missing or symlink: ' + str(root))
    root_resolved = root.resolve(strict=True)
    report['profile_root'] = PROFILE_ROOT
    report['profile_root_realpath'] = str(root_resolved)
    report['worker_folder_rule'] = WORKER_FOLDER_RULE

    all_folders = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.is_symlink():
            continue
        out = path / 'ASCEND_PROFILER_OUTPUT'
        if not out.is_dir():
            continue
        if not path.resolve(strict=True).is_relative_to(root_resolved):
            raise RuntimeError('worker folder escapes profile root: ' + str(path))
        all_folders.append({'path': path, 'has_trace': (out / 'trace_view.json').is_file()})
    report['profile_folders_all'] = [{'folder': str(item['path'].relative_to(root)),
                                      'has_trace': item['has_trace']}
                                     for item in all_folders]

    if WORKER_FOLDER_RULE == 'profile_root_reused_trace_required':
        selected = [item for item in all_folders if item['has_trace']]
        excluded = [{'folder': str(item['path'].relative_to(root)),
                     'reason': 'no_trace_view; precheck verified zero traces before this run'}
                    for item in all_folders if not item['has_trace']]
    else:
        selected = list(all_folders)
        excluded = []
    deadline = time.monotonic() + DEADLINE_SECONDS
    stability_paths = []
    for item in selected:
        out = item['path'] / 'ASCEND_PROFILER_OUTPUT'
        stability_paths.extend(sorted(
            list(item['path'].glob('profiler_info*.json'))
            + list(out.glob('profiler_info*.json')))[:8])
        stability_paths.extend(out / name for name in CSV_NAMES)
    stability = stable_signature(stability_paths, deadline)
    report['stability'] = stability
    workers = []
    csv_bytes_read_total = 0
    csv_remaining = CSV_NODE_CAP
    csv_deadline = min(deadline - TRACE_RESERVE_SECONDS,
                       time.monotonic() + CSV_PHASE_SECONDS)
    for item in selected:
        worker, csv_remaining = build_worker(
            item['path'], root, csv_remaining, csv_deadline, stability)
        workers.append(worker)
        csv_bytes_read_total += sum(
            value.get('bytes_read', 0) for value in worker['csv'].values())

    all_files, all_files_truncated = list_files(root)
    report.update({
        'expected_worker_folders': EXPECTED_FOLDERS,
        'worker_folder_count': len(workers),
        'worker_folder_count_matches': len(workers) == EXPECTED_FOLDERS,
        'excluded_folders': excluded,
        'worker_folders': workers,
        'all_files': all_files,
        'all_files_truncated': all_files_truncated,
        'all_files_count': len(all_files),
    })

    rank_entries = [(worker['native_rank_id'], worker['folder']) for worker in workers
                    if type(worker['native_rank_id']) is int]
    seen = {}
    duplicate_ranks = {}
    for rank, folder in rank_entries:
        if rank in seen:
            duplicate_ranks.setdefault(str(rank), [seen[rank]]).append(folder)
        else:
            seen[rank] = folder
    native_rank_ids_unique = (len(rank_entries) == len(workers)
                              and len(seen) == len(rank_entries))
    preferred = [worker for worker in workers
                 if type(worker['native_rank_id']) is int
                 and worker['native_rank_id'] == PREFERRED_RANK]
    if len(preferred) == 1:
        representative = preferred[0]
        selection_reason = 'unique_preferred_native_rank_id'
    elif workers:
        representative = sorted(workers, key=lambda item: item['folder'])[0]
        selection_reason = 'sorted_first_fallback'
    else:
        representative = None
        selection_reason = 'no_worker_folders'
    report['rank_evidence'] = {
        'native_rank_ids': [{'folder': worker['folder'],
                             'rank_id': worker['native_rank_id'],
                             'type': worker['native_rank_id_type']}
                            for worker in workers],
        'native_rank_ids_unique': native_rank_ids_unique,
        'duplicate_native_rank_ids': duplicate_ranks,
        'preferred_native_rank': PREFERRED_RANK,
        'preferred_rank_candidates': [worker['folder'] for worker in preferred],
        'rank_completion_proof': 'NOT_ESTABLISHED',
        'rank_count_is_not_completion_proof': True,
        'interpretation': 'rank_id is reported verbatim; DP/TP are populated only from '
                          'explicit dp/tp tokens in the actual folder name.',
    }

    trace_result = {'selection': None, 'error': None,
                    'transfer_scope': 'one representative trace_view.json only'}
    if representative is None:
        trace_result['error'] = 'no_worker_folders'
    else:
        relative = representative['folder'] + '/ASCEND_PROFILER_OUTPUT/trace_view.json'
        trace_path = root / relative
        trace_result['selection'] = {
            'folder': representative['folder'],
            'worker_name': representative['worker_name'],
            'pid': representative['pid'],
            'native_rank_id': representative['native_rank_id'],
            'selection_reason': selection_reason,
            'preferred_native_rank': PREFERRED_RANK,
        }
        trace_result['relative'] = relative
        meta = file_meta(trace_path, root)
        trace_result['file'] = meta
        if meta.get('missing') or meta.get('symlink'):
            trace_result['error'] = 'missing_or_symlink'
        elif meta['bytes'] > TRACE_RAW_CAP:
            trace_result['error'] = 'raw_cap_exceeded'
        elif time.monotonic() > deadline:
            trace_result['error'] = 'time_budget'
        else:
            before = trace_path.stat()
            digest = hashlib.sha256()
            compressed = io.BytesIO()
            read_bytes = 0
            try:
                with trace_path.open('rb') as source, gzip.GzipFile(
                        fileobj=compressed, mode='wb', compresslevel=1, mtime=0) as sink:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        read_bytes += len(chunk)
                        if read_bytes > TRACE_RAW_CAP:
                            raise RuntimeError('trace grew beyond raw cap')
                        digest.update(chunk)
                        sink.write(chunk)
                after = trace_path.stat()
                packed = compressed.getvalue()
                trace_result.update({
                    'raw_bytes': read_bytes,
                    'raw_sha256': digest.hexdigest(),
                    'gzip_bytes': len(packed),
                    'gzip_sha256': sha256_bytes(packed),
                    'mtime_ns': after.st_mtime_ns,
                    'mtime_float_repr': repr(after.st_mtime),
                    'stat_before': {'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns},
                    'stat_after': {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns},
                    'stable': ((before.st_size, before.st_mtime_ns)
                               == (after.st_size, after.st_mtime_ns)
                               and read_bytes == before.st_size),
                })
                if not trace_result['stable']:
                    trace_result['error'] = 'changed_during_read'
                elif len(packed) > TRACE_GZIP_CAP:
                    trace_result['error'] = 'gzip_cap_exceeded'
                else:
                    trace_result['gzip_base64'] = base64.b64encode(packed).decode()
            except Exception as error:
                trace_result['error'] = repr(error)
    report['representative'] = trace_result

    exact_files = []
    for worker in workers:
        for group in worker['required_files'].values():
            items = group if isinstance(group, list) else [group]
            exact_files.extend(items)
    required_symlinks = [item['relative'] for item in exact_files
                         if isinstance(item, dict) and item.get('symlink')]
    checks = {
        'worker_folders_expected': len(workers) == EXPECTED_FOLDERS,
        'all_workers_complete': bool(workers) and all(worker['worker_complete']
                                                      for worker in workers),
        'stability_all_workers': bool(workers) and all(
            worker['stability']['stable'] for worker in workers),
        'representative_trace_transferred': bool(
            isinstance(report['representative'], dict)
            and report['representative'].get('gzip_base64')
            and report['representative'].get('error') is None),
        'csv_node_cap_respected': csv_bytes_read_total <= CSV_NODE_CAP,
        'no_required_symlinks': not required_symlinks,
    }
    report['checks'] = checks
    report['required_symlinks'] = required_symlinks
    report['csv_bytes_read_total'] = csv_bytes_read_total
    report['limits'] = {
        'csv_file_cap_bytes': CSV_FILE_CAP,
        'csv_node_cap_bytes': CSV_NODE_CAP,
        'csv_bytes_read_total': csv_bytes_read_total,
        'trace_raw_cap_bytes': TRACE_RAW_CAP,
        'trace_gzip_cap_bytes': TRACE_GZIP_CAP,
        'stability_interval_seconds': STABILITY_INTERVAL,
        'stability_max_seconds': STABILITY_MAX,
        'container_deadline_seconds': DEADLINE_SECONDS,
        'csv_phase_seconds': CSV_PHASE_SECONDS,
        'trace_reserve_seconds': TRACE_RESERVE_SECONDS,
    }
    report['pass'] = all(value is True for value in checks.values())
except BaseException as error:
    report['error'] = repr(error)
print(json.dumps(report, ensure_ascii=False))
'''

HOST_TEMPLATE = r'''import ast, hashlib, json, pathlib, socket, subprocess
assert socket.gethostname() == __HOSTNAME__
NODE = __NODE__
FRAMEWORK = __FRAMEWORK__
CID = __CID__
IMAGE = __IMAGE__
RUN_ID = __RUN_ID__
HOST_PROFILE_ROOT = __HOST_PROFILE_ROOT__
EXPECTED_CONTAINER_ROOT = __EXPECTED_CONTAINER_ROOT__
DERIVE_FROM_MOUNTS = __DERIVE_FROM_MOUNTS__
INNER_TIMEOUT = __INNER_TIMEOUT__
CONTAINER_TEMPLATE = __CONTAINER_TEMPLATE__


def cmd(args, timeout=30):
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


report = {'project_id': 'glm53', 'framework': FRAMEWORK, 'node': NODE,
          'hostname': socket.gethostname(), 'host_pass': False,
          'model_requests': 0, 'SDK_init_calls': 0, 'new_profiles': 0,
          'remote_writes': 0}
try:
    inspect = cmd(['docker', 'inspect', CID, '--format',
                   '{{.Id}} {{.State.Running}} {{.Image}}'])
    if inspect.returncode != 0:
        raise RuntimeError('docker inspect failed: ' + inspect.stderr[-2000:])
    fields = inspect.stdout.strip().split()
    if len(fields) < 3 or fields[0] != CID or fields[1] != 'true' or fields[2] != IMAGE:
        raise RuntimeError('container identity mismatch: ' + inspect.stdout.strip()[:2000])
    image = cmd(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'])
    if image.returncode != 0 or image.stdout.strip() != IMAGE:
        raise RuntimeError('image identity mismatch: ' + (image.stdout + image.stderr)[:2000])
    mounts_result = cmd(['docker', 'inspect', CID, '--format', '{{json .Mounts}}'])
    mount_list = json.loads(mounts_result.stdout) if mounts_result.returncode == 0 else []
    if not isinstance(mount_list, list):
        mount_list = []
    report['container_identity'] = {'inspect': inspect.stdout.strip(),
                                    'image': image.stdout.strip()}
    report['mounts'] = mount_list[:64]
    report['mounts_truncated'] = len(mount_list) > 64

    if DERIVE_FROM_MOUNTS:
        host_root = pathlib.Path(HOST_PROFILE_ROOT)
        if host_root.is_symlink() or not host_root.is_dir():
            raise RuntimeError('host profile root missing or symlink: ' + str(host_root))
        resolved = host_root.resolve(strict=True)
        candidates = []
        for mount in mount_list:
            if mount.get('Type') not in ('bind', 'volume'):
                continue
            source = mount.get('Source')
            destination = mount.get('Destination')
            if (not source or not destination or not str(source).startswith('/')
                    or not str(destination).startswith('/')):
                continue
            try:
                source_path = pathlib.Path(source).resolve(strict=True)
            except OSError:
                continue
            if not source_path.is_dir() or not resolved.is_relative_to(source_path):
                continue
            relative = resolved.relative_to(source_path)
            candidates.append({
                'type': mount.get('Type'), 'source': str(source_path),
                'destination': str(destination),
                'container_root': str(pathlib.Path(destination) / relative),
                'rw': mount.get('RW'),
            })
        if not candidates:
            raise RuntimeError('no docker bind/volume mount contains host profile root')
        candidates.sort(key=lambda item: len(item['source']), reverse=True)
        selected_mount = candidates[0]
        container_root = selected_mount['container_root']
        if container_root != EXPECTED_CONTAINER_ROOT:
            raise RuntimeError('derived container root mismatch: %r != %r'
                               % (container_root, EXPECTED_CONTAINER_ROOT))
        report['container_root_derivation'] = {
            'mode': 'docker_inspect_mounts', 'selected': selected_mount,
            'candidates': candidates, 'frozen_output_dir': EXPECTED_CONTAINER_ROOT,
        }
    else:
        container_root = EXPECTED_CONTAINER_ROOT
        report['container_root_derivation'] = {
            'mode': 'fixed_container_root', 'container_root': container_root,
            'mounts_observed': mount_list[:64],
        }

    container_code = CONTAINER_TEMPLATE.replace(
        '__PROFILE_ROOT_LITERAL__', repr(container_root))
    ast.parse(container_code)
    container_sha = hashlib.sha256(container_code.encode()).hexdigest()
    report['profile_root_host'] = HOST_PROFILE_ROOT
    report['profile_root_container_expected'] = EXPECTED_CONTAINER_ROOT
    report['profile_root_container_derived'] = container_root
    report['container_code_sha256'] = container_sha
    try:
        inner = subprocess.run(['docker', 'exec', '-i', CID, 'python3', '-S', '-'],
                               input=container_code, text=True, capture_output=True,
                               timeout=INNER_TIMEOUT)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('container evidence timeout: ' + repr(error))
    report['container_rc'] = inner.returncode
    report['container_stderr_tail'] = inner.stderr[-4000:]
    lines = [line for line in inner.stdout.splitlines() if line.strip()]
    report['container_stdout_lines'] = len(lines)
    if inner.returncode != 0:
        raise RuntimeError('container evidence rc=%s: %s'
                           % (inner.returncode, inner.stderr[-2000:]))
    if not lines:
        raise RuntimeError('container evidence produced no JSON')
    container_report = json.loads(lines[-1])
    if not isinstance(container_report, dict):
        raise RuntimeError('container evidence JSON is not an object')
    report['container_report'] = container_report
    report['host_pass'] = True
except BaseException as error:
    report['error'] = repr(error)
print(json.dumps(report, ensure_ascii=False))
'''


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def render(source, values):
    pattern = r'__(' + '|'.join(re.escape(key) for key in values) + r')__'
    code = re.sub(pattern, lambda match: repr(values[match.group(1)]), source)
    ast.parse(code)
    return code


def check_code(name, code):
    tree = ast.parse(code)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    forbidden = {'torch', 'vllm', 'sglang', 'memcache_hybrid', 'memfabric_hybrid'}
    bad = sorted(imports & forbidden)
    non_stdlib = sorted(module for module in imports
                        if module not in sys.stdlib_module_names)
    assert not bad, (name, bad)
    assert not non_stdlib, (name, non_stdlib)
    return {'name': name, 'sha256': sha256(code.encode()), 'ast': 'pass',
            'imports': sorted(imports), 'forbidden_imports': bad,
            'non_stdlib_imports': non_stdlib}


def load_inventory():
    path = ROOT / 'scripts' / 'inventory_readonly.py'
    raw = path.read_bytes()
    assert sha256(raw) == INVENTORY_SHA
    spec = importlib.util.spec_from_file_location('l20_inventory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.SSH) >= 3
    module.SSH = list(module.SSH)
    module.SSH[2] = SOCKET
    assert module.SSH[2] == SOCKET
    return module


def load_settings(framework):
    path = ROOT / SETTINGS[framework]['path']
    raw = path.read_bytes()
    assert sha256(raw) == SETTINGS[framework]['sha256']
    return json.loads(raw)


def build_plans():
    base = load_inventory()
    plans = {}
    checks = []
    for framework in ('vllm', 'sglang'):
        settings = load_settings(framework)
        assert settings['project_id'] == 'glm53'
        assert settings['framework'] == framework
        assert settings['diagnostic_only'] is True
        assert settings['performance_certificate_allowed'] is False
        assert settings['model_image'] == EXPECTED_IMAGE[framework]
        precheck_d = {item['node']: item for item in settings['precheck_nodes']
                      if item['role'] == 'D'}
        assert sorted(precheck_d) == D_NODES[framework]
        nodes = {}
        for node in D_NODES[framework]:
            runtime = settings['runtime_containers'][str(node)]
            cid = runtime['container_id']
            image = settings['model_image']
            assert cid == EXPECTED_CIDS[framework][node]
            assert precheck_d[node]['container_id'] == cid
            if framework == 'sglang':
                host_root = (runtime['remote_root'] + '/profile/' + settings['run_id'])
                assert host_root == SGLANG_HOST_ROOT
                container_planned = settings['profile_settings']['profile_body']['output_dir']
                assert container_planned == '/run/glm53/profile/' + settings['run_id']
                derive = True
                rule = 'run_specific_root_all_ascend_output'
                root_note = ('profile root is run-specific and was asserted absent before '
                             'the run; select all dirs containing ASCEND_PROFILER_OUTPUT')
                derivation = ('docker inspect CID --format {{json .Mounts}} at collect time; '
                              'derive bind/volume Source->Destination; assert equals frozen '
                              'profile_body.output_dir; no guessed path')
            else:
                host_root = None
                container_planned = VLLM_CONTAINER_ROOT
                derive = False
                rule = 'profile_root_reused_trace_required'
                root_note = ('profile root is reused torch_profiler_dir; precheck verified '
                             'zero trace_view.json before this run; select dirs containing '
                             'ASCEND_PROFILER_OUTPUT/trace_view.json')
                derivation = ('fixed /run/glm53/profile from the frozen precheck/user target; '
                              'docker mounts are recorded read-only')
            preferred = PREFERRED_RANK[framework][node]
            container_program = render(CONTAINER_TEMPLATE, {
                'NODE': node, 'FRAMEWORK': framework, 'RUN_ID': settings['run_id'],
                'EXPECTED_FOLDERS': EXPECTED_WORKERS, 'PREFERRED_RANK': preferred,
                'WORKER_FOLDER_RULE': rule,
            })
            container_code = container_program.replace(
                '__PROFILE_ROOT_LITERAL__', repr(container_planned))
            host_code = render(HOST_TEMPLATE, {
                'HOSTNAME': base.NODES[node][1], 'NODE': node,
                'FRAMEWORK': framework, 'CID': cid, 'IMAGE': image,
                'RUN_ID': settings['run_id'], 'HOST_PROFILE_ROOT': host_root,
                'EXPECTED_CONTAINER_ROOT': container_planned,
                'DERIVE_FROM_MOUNTS': derive, 'INNER_TIMEOUT': INNER_TIMEOUT,
                'CONTAINER_TEMPLATE': container_program,
            })
            container_check = check_code(
                '%s-node-%s-container' % (framework, node), container_code)
            host_check = check_code('%s-node-%s-host' % (framework, node), host_code)
            checks.extend([container_check, host_check])
            nodes[str(node)] = {
                'node': node,
                'hostname': base.NODES[node][1],
                'container_id': cid,
                'image': image,
                'preferred_native_global_rank': preferred,
                'host_profile_root': host_root,
                'container_profile_root_planned': container_planned,
                'container_root_derivation': derivation,
                'worker_folder_rule': rule,
                'worker_folder_rule_note': root_note,
                'host_code': host_code,
                'container_code': container_code,
                'host_code_sha256': host_check['sha256'],
                'container_code_sha256': container_check['sha256'],
                'rendered_host': ('runs/%s/rendered/%s-node-%s-host.py'
                                  % (PREPARE_RID, framework, node)),
                'rendered_container': ('runs/%s/rendered/%s-node-%s-container.py'
                                       % (PREPARE_RID, framework, node)),
            }
        plans[framework] = {
            'settings_path': SETTINGS[framework]['path'],
            'settings_sha256': SETTINGS[framework]['sha256'],
            'settings': settings,
            'coordinator_run_id': settings['run_id'],
            'coordinator_run_dir': 'runs/' + settings['run_id'],
            'collect_run_id': COLLECT_RUN[framework],
            'nodes': nodes,
        }
    return base, plans, checks


def build_manifest(plans, checks):
    manifest = {
        'project_id': 'glm53',
        'mode': 'prepare_only',
        'prepare_run_id': PREPARE_RID,
        'script': {'path': 'scripts/collect_C32_L20_paired_native_evidence_v1.py',
                   'sha256': sha256(SCRIPT.read_bytes())},
        'inventory': {'path': 'scripts/inventory_readonly.py',
                      'sha256': INVENTORY_SHA},
        'ssh': {'socket': SOCKET, 'max_concurrent': MAX_CONCURRENT_SSH,
                'node_timeout_seconds': NODE_TIMEOUT,
                'mapping_timeout_seconds': MAPPING_TIMEOUT,
                'probe_timeout_seconds': PROBE_TIMEOUT},
        'frameworks': {},
        'coordinator_gate': [
            'controller-rpc-state.json phase=DONE',
            'controller-rpc-state.json done_received=true',
            'controller-rpc-state.json eof_without_done=false',
            'controller-rpc-state.json client_rc=0',
            'controller-done.json state.host_wait_timeout=false',
            'controller-done.json state.client_rc=0',
            'controller-done.json report_sha256 equals rpc report_sha256',
            'diagnostic-result.remote.json SHA equals report_sha256',
            'diagnostic report project_id=glm53, framework, run_id match',
            'active paired lock is absent or released',
        ],
        'rules': {
            'worker_folders': ('16 per D node; folder count is never treated as rank '
                               'completion proof'),
            'rank': ('rank_id is reported verbatim; null/duplicate ranks are not filled; '
                     'DP/TP only from explicit dp/tp tokens in the actual folder name'),
            'csv': ('stream SHA + header + name-matched forward/draft/verify groups; '
                    'Decimal strings; empty/invalid values are not zero-filled; sums are '
                    'cumulative and explicitly not CP'),
            'trace': ('one representative per node; prefer unique native global rank 0/16, '
                      'else sorted-first real folder; raw 192MiB, gzip 32MiB; stat-read-stat '
                      'stable; Mac re-verifies transferred bytes'),
            'read_only': ('no model/HTTP/profile/SDK calls; no remote writes; only '
                          'python3 -S stdlib; no full model logs'),
        },
        'remote_actions': 0,
        'model_requests': 0,
        'SDK_init_calls': 0,
    }
    for framework, plan in plans.items():
        manifest['frameworks'][framework] = {
            'settings_path': plan['settings_path'],
            'settings_sha256': plan['settings_sha256'],
            'coordinator_run_id': plan['coordinator_run_id'],
            'coordinator_run_dir': plan['coordinator_run_dir'],
            'collect_run_id': plan['collect_run_id'],
            'nodes': {
                node: {key: value for key, value in item.items()
                       if key not in ('host_code', 'container_code')}
                for node, item in plan['nodes'].items()
            },
            'host_codes_sha256': {node: item['host_code_sha256']
                                  for node, item in plan['nodes'].items()},
            'container_codes_sha256': {node: item['container_code_sha256']
                                       for node, item in plan['nodes'].items()},
        }
    manifest['ast_checks'] = checks
    return manifest


def review_template(manifest, manifest_sha, plans):
    return {
        'project_id': 'glm53',
        'decision': 'PENDING_MASTER_REVIEW',
        'required_decision': 'APPROVED_READONLY_L20_NATIVE_EVIDENCE_COLLECT',
        'collector_sha256': manifest['script']['sha256'],
        'prepare_manifest_sha256': manifest_sha,
        'settings_sha256': {framework: plan['settings_sha256']
                            for framework, plan in plans.items()},
        'coordinator_runs': {framework: plan['coordinator_run_id']
                             for framework, plan in plans.items()},
        'collect_runs': dict(COLLECT_RUN),
        'host_codes_sha256': {framework: {
            node: item['host_code_sha256'] for node, item in plan['nodes'].items()}
            for framework, plan in plans.items()},
        'container_codes_sha256': {framework: {
            node: item['container_code_sha256'] for node, item in plan['nodes'].items()}
            for framework, plan in plans.items()},
        'max_concurrent_SSH': MAX_CONCURRENT_SSH,
        'authorization_scope': ('read-only existing L20 profile evidence; no model/HTTP/'
                                'profile/SDK calls; no remote writes; no replay'),
        'saved_path': REVIEW_PATH,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare():
    base, plans, checks = build_plans()
    assert all(item['ast'] == 'pass' for item in checks)
    manifest = build_manifest(plans, checks)
    run = ROOT / 'runs' / PREPARE_RID
    run.mkdir(parents=True, exist_ok=False)
    (run / 'execution-once.lock').write_text(str(os.getpid()))
    rendered = run / 'rendered'
    rendered.mkdir()
    for framework, plan in plans.items():
        for node, item in plan['nodes'].items():
            (ROOT / item['rendered_host']).write_text(item['host_code'])
            (ROOT / item['rendered_container']).write_text(item['container_code'])
    manifest_path = run / 'prepare-manifest.json'
    write_json(manifest_path, manifest)
    manifest_sha = sha256(manifest_path.read_bytes())
    write_json(run / 'ast-check.json', {
        'project_id': 'glm53', 'mode': 'prepare_only',
        'pass': all(item['ast'] == 'pass' for item in checks),
        'checks': checks,
    })
    review = review_template(manifest, manifest_sha, plans)
    review_path = run / 'review-template.json'
    write_json(review_path, review)
    (run / 'script-sha256.txt').write_text(
        '%s  scripts/collect_C32_L20_paired_native_evidence_v1.py\n'
        % manifest['script']['sha256'])
    sums = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name != 'sha256sums.txt':
            sums.append('%s  %s' % (sha256(path.read_bytes()),
                                    path.relative_to(run)))
    (run / 'sha256sums.txt').write_text('\n'.join(sums) + '\n')
    result = {
        'project_id': 'glm53',
        'mode': 'prepare_only',
        'prepare_run_id': PREPARE_RID,
        'prepare_dir': str(run.relative_to(ROOT)),
        'script_sha256': manifest['script']['sha256'],
        'prepare_manifest': str(manifest_path.relative_to(ROOT)),
        'prepare_manifest_sha256': manifest_sha,
        'ast_check': 'pass',
        'ast_check_path': str((run / 'ast-check.json').relative_to(ROOT)),
        'rendered': {
            framework: {
                node: {
                    'host': item['rendered_host'],
                    'host_sha256': item['host_code_sha256'],
                    'container': item['rendered_container'],
                    'container_sha256': item['container_code_sha256'],
                }
                for node, item in plan['nodes'].items()
            }
            for framework, plan in plans.items()
        },
        'review_template': str(review_path.relative_to(ROOT)),
        'review_saved_path': REVIEW_PATH,
        'remote_actions': 0,
        'model_requests': 0,
        'SDK_init_calls': 0,
    }
    print(json.dumps(result, indent=2))
    return result


def verify_coordinator(framework, settings):
    run = ROOT / 'runs' / settings['run_id']
    gate = {'project_id': 'glm53', 'framework': framework,
            'run_id': settings['run_id'], 'run_dir': str(run.relative_to(ROOT)),
            'paths': {}, 'checks': {}, 'pass': False}
    names = ['controller-rpc-state.json', 'controller-done.json',
             'diagnostic-result.remote.json']
    paths = {name: run / name for name in names}
    for name, path in paths.items():
        if not path.is_file():
            gate['error'] = 'missing ' + name
            return gate, None
        raw = path.read_bytes()
        gate['paths'][name] = {'bytes': len(raw), 'sha256': sha256(raw)}
    try:
        rpc = json.loads(paths['controller-rpc-state.json'].read_text())
        done = json.loads(paths['controller-done.json'].read_text())
        report_raw = paths['diagnostic-result.remote.json'].read_bytes()
        report = json.loads(report_raw)
    except Exception as error:
        gate['error'] = repr(error)
        return gate, None
    done_state = done.get('state') if isinstance(done.get('state'), dict) else {}
    report_sha = rpc.get('report_sha256')
    checks = {
        'rpc_phase_DONE': rpc.get('phase') == 'DONE',
        'rpc_done_received': rpc.get('done_received') is True,
        'rpc_eof_without_done_false': rpc.get('eof_without_done') is False,
        'rpc_client_rc_0': type(rpc.get('client_rc')) is int and rpc.get('client_rc') == 0,
        'done_host_wait_timeout_false': done_state.get('host_wait_timeout') is False,
        'done_client_rc_0': (type(done_state.get('client_rc')) is int
                             and done_state.get('client_rc') == 0),
        'done_report_sha_matches_rpc': done.get('report_sha256') == report_sha,
        'report_sha_is_hex': bool(isinstance(report_sha, str)
                                  and re.fullmatch(r'[0-9a-f]{64}', report_sha)),
        'report_file_sha_matches': sha256(report_raw) == report_sha,
        'report_project_glm53': report.get('project_id') == 'glm53',
        'report_framework': report.get('framework') == framework,
        'report_run_id': report.get('run_id') == settings['run_id'],
    }
    if 'input_manifest_sha256' in report:
        checks['report_input_manifest'] = (
            report['input_manifest_sha256'] == settings['input_manifest_sha256'])
    if 'cache_namespace' in report:
        checks['report_cache_namespace'] = (
            report['cache_namespace'] == settings['namespace'])
    lock = ROOT / 'runs' / 'C32-L20-paired-active.lock'
    if lock.exists():
        try:
            owner = json.loads(lock.read_text())
        except Exception:
            owner = None
        checks['active_lock_released'] = (isinstance(owner, dict)
                                          and owner.get('released') is True)
    else:
        checks['active_lock_released'] = True
    gate['checks'] = checks
    gate['rpc'] = {'phase': rpc.get('phase'), 'done_received': rpc.get('done_received'),
                   'eof_without_done': rpc.get('eof_without_done'),
                   'client_rc': rpc.get('client_rc'),
                   'cpu_connection_rc': rpc.get('cpu_connection_rc'),
                   'prechecks': rpc.get('prechecks')}
    gate['done'] = {'host_wait_timeout': done_state.get('host_wait_timeout'),
                    'client_rc': done_state.get('client_rc'),
                    'report_sha256': done.get('report_sha256')}
    gate['report'] = {
        'phase': report.get('phase'),
        'diagnostic_only': report.get('diagnostic_only'),
        'performance_certificate_allowed': report.get('performance_certificate_allowed'),
        'count_and_terminal_successes': report.get('count_and_terminal_successes'),
        'prefill_count_successes': report.get('prefill_count_successes'),
        'raw_cache_observation_passes': report.get('raw_cache_observation_passes'),
        'backend_HTTP_attempts': report.get('backend_HTTP_attempts'),
    }
    gate['pass'] = all(value is True for value in checks.values())
    return gate, report


def native_control_association(gate, report):
    if not isinstance(report, dict):
        return None
    control = report.get('profile_control')
    compact = None
    keys = []
    if isinstance(control, dict):
        keys = sorted(control)[:64]
        compact = {key: control.get(key) for key in
                   ['pass_control', 'profile_completion', 'profile_start_attempts',
                    'profile_stop_attempts', 'error']}
    return {
        'diagnostic_report_local': gate['paths']['diagnostic-result.remote.json']
        if 'diagnostic-result.remote.json' in gate.get('paths', {}) else None,
        'diagnostic_report_sha256': gate.get('paths', {}).get(
            'diagnostic-result.remote.json', {}).get('sha256'),
        'profile_control_present': isinstance(control, dict),
        'profile_control_keys': keys,
        'profile_control_compact': compact,
        'note': 'Local existing coordinator artifact; not re-transferred from a node.',
    }


def load_collect_plan(framework, review, manifest):
    plan = manifest['frameworks'][framework]
    assert review['settings_sha256'][framework] == plan['settings_sha256']
    assert review['coordinator_runs'][framework] == plan['coordinator_run_id']
    assert review['collect_runs'][framework] == COLLECT_RUN[framework]
    assert review['host_codes_sha256'][framework] == plan['host_codes_sha256']
    assert review['container_codes_sha256'][framework] == plan['container_codes_sha256']
    nodes = {}
    for node_text, item in plan['nodes'].items():
        for relative in (item['rendered_host'], item['rendered_container']):
            candidate = Path(relative)
            assert not candidate.is_absolute() and '..' not in candidate.parts
        host_path = ROOT / item['rendered_host']
        container_path = ROOT / item['rendered_container']
        host_code = host_path.read_text()
        container_code = container_path.read_text()
        assert sha256(host_code.encode()) == item['host_code_sha256']
        assert sha256(host_code.encode()) == review['host_codes_sha256'][framework][node_text]
        assert sha256(container_code.encode()) == item['container_code_sha256']
        assert (sha256(container_code.encode())
                == review['container_codes_sha256'][framework][node_text])
        check_code('%s-node-%s-host' % (framework, node_text), host_code)
        check_code('%s-node-%s-container' % (framework, node_text), container_code)
        nodes[int(node_text)] = {
            'node': int(node_text),
            'hostname': item['hostname'],
            'container_id': item['container_id'],
            'image': item['image'],
            'host_profile_root': item['host_profile_root'],
            'container_root_planned': item['container_profile_root_planned'],
            'host_code': host_code,
            'host_code_sha256': item['host_code_sha256'],
            'container_code_sha256': item['container_code_sha256'],
        }
    return nodes


def invoke_node(base, node, item):
    pod, hostname = base.NODES[node]
    mapping = base.invoke(
        ['kubectl get pod -n default ' + shlex.quote(pod)
         + " -o jsonpath='{.status.hostIP}'"], timeout=MAPPING_TIMEOUT)
    if mapping['rc'] != 0 or mapping['stdout'].strip() != '172.16.12.' + str(node):
        return {'node': node, 'mapping': mapping,
                'error': 'kubectl hostIP mapping mismatch; probe skipped'}
    command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod,
                          '-c', 'hostctl', '--', 'nsenter', '-t', '1', '-m', '--',
                          'python3', '-S', '-'])
    try:
        wire = base.invoke([command], item['host_code'], timeout=PROBE_TIMEOUT)
    except Exception as error:
        return {'node': node, 'mapping': mapping, 'error': repr(error)}
    return {'node': node, 'mapping': mapping, 'wire': wire}


def save_node(run, node, item, result):
    wire = result.get('wire') if isinstance(result.get('wire'), dict) else {}
    mapping = result.get('mapping') if isinstance(result.get('mapping'), dict) else {}
    stdout = wire.get('stdout', '') or ''
    stderr = wire.get('stderr', '') or ''
    meta = {
        'node': node,
        'mapping_rc': mapping.get('rc'),
        'mapping_stdout': (mapping.get('stdout') or '').strip(),
        'wire_rc': wire.get('rc'),
        'stdout_bytes': len(stdout.encode()),
        'stdout_sha256': sha256(stdout.encode()),
        'stderr_tail': stderr[-4000:],
        'error': result.get('error'),
    }
    node_summary = {
        'node': node, 'pass': False,
        'wire_rc': wire.get('rc'), 'error': result.get('error'),
    }
    report = None
    if not result.get('error') and wire.get('rc') == 0 and stdout.strip():
        lines = [line for line in stdout.splitlines() if line.strip()]
        try:
            report = json.loads(lines[-1])
        except Exception as error:
            meta['parse_error'] = repr(error)
    host_ok = False
    container_ok = False
    if isinstance(report, dict):
        host_ok = bool(report.get('host_pass') is True
                       and report.get('container_rc') == 0
                       and report.get('framework') == item.get('framework', report.get('framework'))
                       and report.get('node') == node
                       and report.get('profile_root_host') == item['host_profile_root']
                       and report.get('profile_root_container_expected')
                       == item['container_root_planned']
                       and report.get('profile_root_container_derived')
                       == item['container_root_planned']
                       and report.get('container_code_sha256')
                       == item['container_code_sha256'])
        container = report.get('container_report')
        trace_ok = False
        if isinstance(container, dict):
            trace = container.get('representative')
            if not isinstance(trace, dict):
                trace = None
            if isinstance(trace, dict):
                encoded = trace.pop('gzip_base64', None)
                if encoded:
                    try:
                        packed = base64.b64decode(encoded, validate=True)
                        assert len(packed) == trace['gzip_bytes']
                        assert sha256(packed) == trace['gzip_sha256']
                        raw = gzip.decompress(packed)
                        assert len(raw) == trace['raw_bytes']
                        assert sha256(raw) == trace['raw_sha256']
                        trace_path = run / ('trace-%s-trace_view.json' % node)
                        trace_path.write_bytes(raw)
                        assert sha256(trace_path.read_bytes()) == trace['raw_sha256']
                        trace['mac_bytes'] = len(raw)
                        trace['mac_sha256'] = sha256(raw)
                        trace['mac_verified'] = True
                        trace_ok = True
                        try:
                            summary_path = str(trace_path.relative_to(ROOT))
                        except ValueError:
                            summary_path = str(trace_path)
                        node_summary['trace'] = {
                            'path': summary_path,
                            'bytes': len(raw), 'sha256': sha256(raw),
                        }
                    except Exception as error:
                        trace['mac_verified'] = False
                        trace['mac_error'] = repr(error)
                else:
                    trace['mac_verified'] = False
            container_ok = bool(
                container.get('pass') is True
                and container.get('worker_folder_count') == EXPECTED_WORKERS
                and container.get('worker_folder_count_matches') is True
                and container.get('profile_root') == item['container_root_planned']
                and container.get('framework') == item.get('framework', container.get('framework'))
                and container.get('node') == node
                and container.get('run_id') == item.get('run_id', container.get('run_id'))
                and trace_ok)
            node_summary.update({
                'worker_folder_count': container.get('worker_folder_count'),
                'expected_worker_folders': container.get('expected_worker_folders'),
                'csv_bytes_read_total': container.get('csv_bytes_read_total'),
                'container_checks': container.get('checks'),
                'container_error': container.get('error'),
            })
        meta['host'] = {
            'host_pass': report.get('host_pass'),
            'framework': report.get('framework'),
            'node': report.get('node'),
            'container_identity': report.get('container_identity'),
            'profile_root_host': report.get('profile_root_host'),
            'profile_root_container_expected': report.get('profile_root_container_expected'),
            'profile_root_container_derived': report.get('profile_root_container_derived'),
            'container_root_derivation': report.get('container_root_derivation'),
            'container_code_sha256': report.get('container_code_sha256'),
            'container_rc': report.get('container_rc'),
            'container_stderr_tail': report.get('container_stderr_tail'),
            'error': report.get('error'),
        }
        write_json(run / ('node-%s-report.json' % node), container if isinstance(
            container, dict) else {'error': 'missing container report'})
    else:
        write_json(run / ('node-%s-report.json' % node),
                   {'error': result.get('error') or meta.get('parse_error')
                    or 'no host JSON', 'wire_stdout_tail': stdout[-4000:]})
    node_summary['host_ok'] = host_ok
    node_summary['container_ok'] = container_ok
    node_summary['pass'] = bool(host_ok and container_ok)
    write_json(run / ('node-%s-wire-meta.json' % node), meta)
    return node_summary


def collect(framework, review_sha):
    assert framework in ('vllm', 'sglang')
    script_sha = sha256(SCRIPT.read_bytes())
    review_path = ROOT / REVIEW_PATH
    if not review_path.is_file():
        raise RuntimeError('master review missing: ' + REVIEW_PATH)
    review_raw = review_path.read_bytes()
    assert review_sha and sha256(review_raw) == review_sha, 'review SHA mismatch'
    review = json.loads(review_raw)
    assert review.get('project_id') == 'glm53'
    assert review.get('decision') == 'APPROVED_READONLY_L20_NATIVE_EVIDENCE_COLLECT'
    assert review.get('collector_sha256') == script_sha
    assert review.get('max_concurrent_SSH') == MAX_CONCURRENT_SSH
    prepare_dir = ROOT / 'runs' / PREPARE_RID
    manifest_path = prepare_dir / 'prepare-manifest.json'
    assert manifest_path.is_file(), 'prepare manifest missing; run --prepare first'
    manifest_raw = manifest_path.read_bytes()
    assert sha256(manifest_raw) == review.get('prepare_manifest_sha256')
    manifest = json.loads(manifest_raw)
    assert manifest['script']['sha256'] == script_sha
    settings = load_settings(framework)
    assert review['settings_sha256'][framework] == SETTINGS[framework]['sha256']
    assert manifest['frameworks'][framework]['settings_sha256'] == SETTINGS[framework]['sha256']
    assert review['coordinator_runs'][framework] == settings['run_id']
    nodes = load_collect_plan(framework, review, manifest)
    assert len(nodes) == 2

    base = load_inventory()
    gate, diagnostic_report = verify_coordinator(framework, settings)
    if not gate['pass']:
        print(json.dumps({'project_id': 'glm53', 'framework': framework,
                          'mode': 'collect_refused', 'coordinator': gate},
                         indent=2))
        raise SystemExit(2)
    assert Path(SOCKET).is_socket(), 'authenticated SSH control socket missing'

    run = ROOT / 'runs' / COLLECT_RUN[framework]
    try:
        run.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RuntimeError('collect run directory already exists; replay refused: '
                           + str(run))
    (run / 'execution-once.lock').write_text(str(os.getpid()))
    write_json(run / 'coordinator-verification.json', gate)
    collection_plan = {
        'project_id': 'glm53', 'framework': framework,
        'collect_run_id': COLLECT_RUN[framework],
        'review_path': REVIEW_PATH, 'review_sha256': review_sha,
        'prepare_manifest_path': str(manifest_path.relative_to(ROOT)),
        'prepare_manifest_sha256': sha256(manifest_raw),
        'settings_path': SETTINGS[framework]['path'],
        'settings_sha256': SETTINGS[framework]['sha256'],
        'coordinator_run_id': settings['run_id'],
        'nodes': {str(node): {key: value for key, value in item.items()
                              if key != 'host_code'}
                  for node, item in nodes.items()},
    }
    write_json(run / 'collect-plan.json', collection_plan)
    start = time.monotonic()
    receipt = {
        'project_id': 'glm53', 'framework': framework,
        'collect_run_id': COLLECT_RUN[framework],
        'collector_sha256': script_sha,
        'review': {'path': REVIEW_PATH, 'sha256': review_sha},
        'prepare_manifest': {'path': str(manifest_path.relative_to(ROOT)),
                             'sha256': sha256(manifest_raw)},
        'settings': {'path': SETTINGS[framework]['path'],
                     'sha256': SETTINGS[framework]['sha256']},
        'coordinator': gate,
        'native_control_association': native_control_association(gate, diagnostic_report),
        'max_concurrent_SSH': MAX_CONCURRENT_SSH,
        'node_timeout_seconds': NODE_TIMEOUT,
        'model_requests': 0, 'SDK_init_calls': 0, 'new_profiles': 0,
        'remote_writes': 0, 'nodes': {}, 'pass': False,
    }
    summaries = {}
    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SSH) as pool:
            futures = {}
            for node, item in nodes.items():
                item = dict(item)
                item['framework'] = framework
                item['run_id'] = settings['run_id']
                futures[pool.submit(invoke_node, base, node, item)] = (node, item)
            for future in as_completed(futures):
                node, item = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {'node': node, 'error': repr(error)}
                try:
                    summary = save_node(run, node, item, result)
                except Exception as error:
                    summary = {'node': node, 'pass': False, 'error': repr(error)}
                    write_json(run / ('node-%s-save-error.json' % node),
                               {'error': repr(error)})
                summaries[node] = summary
                receipt['nodes'][str(node)] = summary
                print(json.dumps({'node': node, 'pass': summary['pass'],
                                  'error': summary.get('error')}), flush=True)
    finally:
        receipt['elapsed_seconds'] = round(time.monotonic() - start, 3)
        receipt['pass'] = bool(gate['pass'] and len(summaries) == 2
                               and all(item['pass'] for item in summaries.values()))
        write_json(run / 'receipt.json', receipt)
    print(json.dumps({
        'project_id': 'glm53', 'framework': framework,
        'collect_run_id': COLLECT_RUN[framework],
        'receipt': str((run / 'receipt.json').relative_to(ROOT)),
        'pass': receipt['pass'],
        'nodes': {node: {'pass': item['pass'],
                         'worker_folder_count': item.get('worker_folder_count'),
                         'trace': item.get('trace')}
                  for node, item in summaries.items()},
    }, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--collect', action='store_true')
    parser.add_argument('--framework', choices=['vllm', 'sglang'])
    parser.add_argument('--review-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        assert args.framework, '--collect requires --framework'
        assert args.review_sha256, '--collect requires --review-sha256'
        collect(args.framework, args.review_sha256)


if __name__ == '__main__':
    main()
