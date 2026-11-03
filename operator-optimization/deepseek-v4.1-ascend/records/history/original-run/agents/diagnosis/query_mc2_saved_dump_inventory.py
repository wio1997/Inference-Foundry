#!/usr/bin/env python3
"""Bounded read-only inventory of two frozen profile directory structures.

Only filenames and whitelisted profiler settings are inspected. No tensor,
request, weight, dump payload, device API, or new profiling is accessed.
"""
import argparse, hashlib, json, os, stat
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--task-root', required=True)
ap.add_argument('--trace-root', required=True)
a = ap.parse_args()
T, R = Path(a.task_root).resolve(strict=True), Path(a.trace_root).resolve(strict=True)
manifest = T / 'evidence/used-raw-source-sha256.json'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if sha(manifest) != 'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a':
    raise RuntimeError('STOP_MANIFEST_MISMATCH')
items = json.loads(manifest.read_text())['records']
rank_prefixes = ('dp0_pp0_tp0_dcp0_ep0_rank0_', 'dp1_pp0_tp0_dcp0_ep8_rank0_')
selected = []
for prefix in rank_prefixes:
    refs = [x for x in items if x['relative_path'].startswith(prefix) and '/PROF_' in x['relative_path'] and '/device_' in x['relative_path'] and '/info.json.' in x['relative_path']]
    if len(refs) != 1:
        raise RuntimeError('STOP_RANK_DEVICE_INVENTORY_NOT_UNIQUE')
    info = refs[0]
    parts = Path(info['relative_path']).parts
    rank, prof, device = parts[:3]
    settings_ref = [x for x in items if x['relative_path'] == f'{rank}/profiler_info_{device.removeprefix("device_")}.json']
    if len(settings_ref) != 1:
        raise RuntimeError('STOP_PROFILER_SETTINGS_NOT_UNIQUE')
    settings_file = (R / settings_ref[0]['relative_path']).resolve(strict=True)
    if not settings_file.is_relative_to(R) or sha(settings_file) != settings_ref[0]['sha256']:
        raise RuntimeError('STOP_PROFILER_SETTINGS_HASH_MISMATCH')
    if settings_file.stat().st_size > 2_000_000:
        raise RuntimeError('STOP_PROFILER_SETTINGS_TOO_LARGE')
    settings = json.loads(settings_file.read_text())
    settings_keys = []
    def metadata_keys(obj, path='', depth=0):
        if depth > 8:
            raise RuntimeError('STOP_SETTINGS_NESTING')
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    continue
                if key.lower() in ('record_shapes', 'with_stack', 'profile_memory', 'data_simplification', 'op_args', 'export_type'):
                    settings_keys.append({'field': key.lower(), 'type': type(value).__name__, 'bool_value': value if type(value) is bool else None})
                if isinstance(value, dict):
                    metadata_keys(value, depth=depth+1)
    metadata_keys(settings)
    fixed_dirs = [('rank_root', R/rank), ('export_root', R/rank/'ASCEND_PROFILER_OUTPUT'), ('device_root', R/rank/prof/device)]
    for name in ('data', 'dump', 'dumps', 'tensor', 'tensors', 'tensor_dump', 'dump_data'):
        fixed_dirs.append(('device_' + name, R/rank/prof/device/name))
    entries = []
    for kind, path in fixed_dirs:
        if not path.exists():
            entries.append({'directory': kind, 'status': 'MISSING'})
            continue
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(R) or not path.is_dir():
            raise RuntimeError('STOP_UNSAFE_DIRECTORY')
        found = []
        with os.scandir(path) as it:
            for i, entry in enumerate(it):
                if i >= 1024:
                    raise RuntimeError('STOP_DIRECTORY_ENTRY_LIMIT')
                name = entry.name.lower()
                tags = [x for x in ('expert_token_nums', 'token_per_expert', 'cumsum', 'active_mask', 'dump', 'tensor') if x in name]
                ext = Path(name).suffix
                ext_category = ext if ext in ('.npy','.npz','.json','.bin','.db','.csv','.pkl','.pt') else 'OTHER_OR_NONE'
                found.append({'name_sha256': hashlib.sha256(entry.name.encode()).hexdigest(), 'entry_kind': 'dir' if entry.is_dir(follow_symlinks=False) else ('file' if entry.is_file(follow_symlinks=False) else 'other'), 'tags': tags, 'extension': ext_category})
        entries.append({'directory': kind, 'status': 'READ', 'entry_count': len(found), 'entries': sorted(found, key=lambda x: x['name_sha256'])})
    selected.append({'rank': rank, 'profiler_info_sha256': settings_ref[0]['sha256'], 'settings': settings_keys, 'directories': entries})
print(json.dumps({'scope':'Only fixed EP0 and EP8 profile roots and seven literal device children; names and whitelisted settings only', 'manifest_sha256':sha(manifest), 'samples':selected, 'limits':['No file payloads, weight, request or dump contents were inspected.', 'Names cannot establish or exclude data inside payloads, other roots/ranks, renamed files or unlisted directories.', 'Filename matches do not prove values or bind any replay.', 'No device calls, runtime model import, or changes to service.'], 'no_device_work':True},ensure_ascii=False,indent=2))
