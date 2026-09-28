#!/usr/bin/env python3
"""Run621 reversible all8 actual-W0 DSpark cache descriptor observer.

Only CPU tensor descriptors and format IDs are read on cohort5 cycles64/65.
No device value copy, new model op, HCCL, or synchronization is introduced.
The inherited Run606 diagnostic hooks retain existing Basis/Product/dispatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import loop081_dispatch_patch_run606 as base

SOURCES = base.SOURCES
ORIGINAL = base.ORIGINAL

HELPER = '''
# EXTREME_LOOP081_RUN621_CACHE_DESCRIPTOR_HELPER
def _p621_descriptor(value):
    import torch_npu
    if not isinstance(value, torch.Tensor):
        return {"status": "not_tensor", "type": type(value).__name__}
    storage = value.untyped_storage()
    return {
        "status": "tensor", "device": str(value.device),
        "dtype": str(value.dtype), "shape": list(value.shape),
        "stride": list(value.stride()),
        "format_id": (int(torch_npu.get_npu_format(value))
                      if value.device.type == "npu" else None),
        "tensor_data_ptr": int(value.data_ptr()),
        "storage_base_ptr": int(storage.data_ptr()),
        "storage_cdata": int(storage._cdata),
        "storage_nbytes": int(storage.nbytes()),
        "storage_offset_elements": int(value.storage_offset()),
    }


def _p621_capture_cache_descriptors(handoff, state):
    model = getattr(getattr(handoff.proposer, "model", None), "model", None)
    layers = getattr(model, "layers", None)
    if layers is None or not hasattr(layers, "items"):
        return {"status": "missing_draft_layers", "cycle": state.cycle_index}
    records = []
    for name, layer in layers.items():
        if str(name) not in ("43", "44", "45"):
            continue
        cache_layer = layer.self_attn.dsa_attn.swa_cache_layer
        cache = getattr(cache_layer, "kv_cache", None)
        while isinstance(cache, (list, tuple)) and len(cache) == 1:
            cache = cache[0]
        records.append({
            "layer": str(name), "cache_layer_prefix": str(cache_layer.prefix),
            "block_size": int(cache_layer.block_size),
            "cache_owner_object_id": id(cache_layer),
            "cache": _p621_descriptor(cache),
        })
    common = handoff.common_attn_metadata
    return {
        "status": "descriptors_only_no_device_values",
        "cycle": int(state.cycle_index),
        "run_ts": getattr(handoff, "_p621_run_ts", None),
        "rank": getattr(handoff, "_p621_rank", None),
        "cohort": getattr(handoff, "_p621_cohort", None),
        "draft_model_owner_object_id": id(model),
        "layers": records,
        "common": {name: _p621_descriptor(getattr(common, name, None)) for name in (
            "query_start_loc", "seq_lens", "slot_mapping", "block_table_tensor")},
        "state": {name: _p621_descriptor(getattr(state, name, None)) for name in (
            "target_slot_mapping", "target_positions", "target_input_ids")},
    }
'''


def one(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise ValueError(f'anchor count {src.count(old)} for {old[:80]!r}')
    return src.replace(old, new, 1)


def patch(key: str, source: str) -> str:
    edited = base.patch(key, source)
    if key == 'runner':
        old = '            _extreme_runtime._p606_armed = bool(_p602_enabled)\n'
        new = old + '''            # EXTREME_LOOP081_RUN621_ARM
            _p621_cohort = len(self._extreme_served_cohorts)
            _p621_handoff = _extreme_runtime.proposer
            _p621_handoff._p621_armed = bool(
                _p602_enabled and _p621_cohort == 5
                and os.getenv("EXTREME_RUN621_DESCRIPTOR") == "1")
            _p621_handoff._p621_rank = int(get_tp_group().rank_in_group)
            _p621_handoff._p621_cohort = _p621_cohort
            _p621_handoff._p621_run_ts = os.getenv("RUN_TS")
            _p621_handoff._p621_desc_records = []
'''
        edited = one(edited, old, new)
        old = '                    os.makedirs(_p602_dir, exist_ok=True)\n'
        new = '''                    # EXTREME_LOOP081_RUN621_FLUSH_AFTER_EXISTING_SYNC
                    if getattr(_extreme_runtime.proposer, "_p621_armed", False):
                        _p602_payload["run621_cache_descriptors"] = (
                            _extreme_runtime.proposer._p621_desc_records)
                    os.makedirs(_p602_dir, exist_ok=True)
'''
        edited = one(edited, old, new)
    elif key == 'dspark':
        old = '        mark("begin")\n'
        new = '''        # EXTREME_LOOP081_RUN621_CAPTURE_CPU_DESCRIPTORS
        if (getattr(self, "_p621_armed", False)
                and state.cycle_index in (64, 65)):
            if len(self._p621_desc_records) >= 2:
                raise RuntimeError("Run621 descriptor record cap")
            self._p621_desc_records.append(
                _p621_capture_cache_descriptors(self, state))
        mark("begin")
'''
        edited = one(edited, old, new)
        edited += HELPER
    return edited


def prepared():
    paths = [p.resolve() for p in SOURCES.values()]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate source path')
    collector = base.product.base.HELPER.read_bytes()
    compile(collector, str(base.product.base.HELPER), 'exec')
    manifest = {'helper_sha256': base.sha(collector), 'sources': {}}
    old, new = {}, {}
    for key, path in SOURCES.items():
        data = path.read_bytes()
        if base.sha(data) != ORIGINAL[key]:
            raise ValueError(f'{key} original SHA drift: {base.sha(data)}')
        edited = patch(key, data.decode()).encode()
        compile(edited, str(path), 'exec')
        old[key], new[key] = data, edited
        manifest['sources'][key] = {
            'path': str(path), 'original_sha256': base.sha(data),
            'patched_sha256': base.sha(edited)}
    return manifest, old, new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--offline-confirmed', action='store_true')
    args = parser.parse_args()
    if args.action != 'check' and not args.offline_confirmed:
        raise ValueError('offline confirmation required')
    if args.action in ('check', 'install'):
        manifest, old, new = prepared()
        if args.action == 'install':
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError('fresh state directory required')
            args.state_dir.mkdir(parents=True)
            for key, data in old.items():
                base.atomic(args.state_dir / f'{key}.orig', data)
            base.atomic(args.state_dir / 'manifest.json',
                        (json.dumps(manifest, indent=2) + '\n').encode())
            try:
                for key, path in SOURCES.items():
                    if path.read_bytes() != old[key]:
                        raise ValueError('install race ' + key)
                    base.atomic(path, new[key])
            except BaseException:
                for key, path in SOURCES.items():
                    if base.sha(path.read_bytes()) == manifest['sources'][key]['patched_sha256']:
                        base.atomic(path, old[key])
                raise
    else:
        if args.state_dir is None:
            raise ValueError('state dir required')
        manifest = json.loads((args.state_dir / 'manifest.json').read_text())
        if set(manifest['sources']) != set(SOURCES):
            raise ValueError('source set drift')
        for key, path in SOURCES.items():
            row = manifest['sources'][key]
            if row['path'] != str(path) or row['original_sha256'] != ORIGINAL[key]:
                raise ValueError('source manifest drift ' + key)
            backup = (args.state_dir / f'{key}.orig').read_bytes()
            if base.sha(backup) != ORIGINAL[key]:
                raise ValueError('backup drift ' + key)
            if base.sha(path.read_bytes()) not in (
                    row['original_sha256'], row['patched_sha256']):
                raise ValueError('installed source drift ' + key)
        for key, path in SOURCES.items():
            if base.sha(path.read_bytes()) != ORIGINAL[key]:
                base.atomic(path, (args.state_dir / f'{key}.orig').read_bytes())
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({'action': args.action, **manifest}, indent=2) + '\n')
    print(json.dumps({'action': args.action, 'files': len(SOURCES)}))


if __name__ == '__main__':
    main()
