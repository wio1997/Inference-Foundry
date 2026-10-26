#!/usr/bin/env python3
"""SHA-pinned reversible two-source Run577 terminal diagnostic patch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASCEND = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = {'runner': ASCEND/'worker/model_runner_v1.py',
           'w4a8': ASCEND/'quantization/methods/w4a8.py'}
ORIGINAL = {'runner':'004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba',
            'w4a8':'1cda64ddb13a88f2fe86ccdb59b6555864148e3994022631915720114d7fb2e5'}
HELPER = ROOT/'scripts/loop080_realweight_service_run577.py'
MARK = '# EXTREME_LOOP080_REALWEIGHT_RUN577'


def sha(data): return hashlib.sha256(data).hexdigest()


def patch(key, source):
    if MARK in source: raise ValueError('already patched')
    if key == 'w4a8':
        anchor = '        moe_comm_method = _EXTRA_CTX.moe_comm_method\n'
        block = (f'        {MARK}\n'
                 '        from scripts import loop080_realweight_service_run577 as _rw577\n'
                 '        _rw577.observe(layer, topk_ids, w1, w2, w1_scale, w2_scale,\n'
                 '                       w1_scale_bias, w2_scale_bias, self.quant_type,\n'
                 '                       _EXTRA_CTX.use_mega_moe)\n')
    else:
        start_anchor = ('        _extreme_served_cohorts = getattr(\n'
                        '            self, "_extreme_served_cohorts", set()\n'
                        '        )\n')
        if source.count(start_anchor) != 1:
            raise ValueError('runner terminal entry anchor drift')
        source = source.replace(start_anchor, start_anchor +
            f'        {MARK}\n'
            '        if os.getenv("EXTREME_RUN577_DIR"):\n'
            '            if "/data/wio/Inference_Foundry" not in sys.path:\n'
            '                sys.path.insert(0, "/data/wio/Inference_Foundry")\n'
            '            from scripts import loop080_realweight_service_run577 as _rw577\n'
            '            _rw577.reject_new_cohort(_extreme_req_ids, _extreme_served_cohorts)\n')
        anchor = '                return None\n            _initial_num_computed = (\n'
        block = (f'                {MARK}\n'
                 '                if os.getenv("EXTREME_RUN577_DIR"):\n'
                 '                    from scripts import loop080_realweight_service_run577 as _rw577\n'
                 '                    _rw577.benchmark(_cohort_index)\n')
    if source.count(anchor) != 1: raise ValueError(f'{key} anchor count {source.count(anchor)}')
    if key == 'runner':
        return source.replace(anchor, block + anchor)
    return source.replace(anchor, block + anchor)


def prepared():
    helper = HELPER.read_bytes(); compile(helper, str(HELPER), 'exec')
    files, old, new = {}, {}, {}
    for key, path in SOURCES.items():
        value = path.read_bytes()
        if sha(value) != ORIGINAL[key]: raise ValueError(f'{key} source SHA drift')
        replacement = patch(key, value.decode()).encode()
        compile(replacement, str(path), 'exec')
        files[key] = dict(path=str(path), original=sha(value), patched=sha(replacement))
        old[key], new[key] = value, replacement
    return dict(helper_sha=sha(helper), files=files), old, new


def atomic(path, payload):
    tmp = path.with_name(path.name+'.run577.tmp')
    if tmp.exists(): raise ValueError('stale temporary file')
    with tmp.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if path.exists(): os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp, path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('check','install','restore'))
    ap.add_argument('--state-dir',type=Path)
    ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--offline-confirmed',action='store_true')
    a=ap.parse_args()
    if a.action!='check' and not a.offline_confirmed: raise ValueError('offline confirmation required')
    if a.action in ('check','install'):
        manifest, old, new=prepared()
        if a.action=='install':
            if a.state_dir is None or a.state_dir.exists(): raise ValueError('fresh state dir required')
            a.state_dir.mkdir(parents=True)
            for key, value in old.items(): atomic(a.state_dir/f'{key}.orig',value)
            atomic(a.state_dir/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
            try:
                for key,path in SOURCES.items():
                    if path.read_bytes()!=old[key]: raise ValueError(f'{key} changed during install')
                    atomic(path,new[key])
            except BaseException:
                for key,path in SOURCES.items():
                    if sha(path.read_bytes())==manifest['files'][key]['patched']:
                        atomic(path,old[key])
                raise
    else:
        if a.state_dir is None: raise ValueError('state dir required')
        manifest=json.loads((a.state_dir/'manifest.json').read_text())
        if set(manifest['files'])!=set(SOURCES): raise ValueError('manifest source set drift')
        if manifest['helper_sha']!=sha(HELPER.read_bytes()): raise ValueError('helper drift')
        for key,row in manifest['files'].items():
            if row['path']!=str(SOURCES[key]) or row['original']!=ORIGINAL[key]:
                raise ValueError('source identity drift')
            if sha(SOURCES[key].read_bytes()) not in (row['original'],row['patched']):
                raise ValueError(f'{key} installed source drift')
            if sha((a.state_dir/f'{key}.orig').read_bytes())!=row['original']:
                raise ValueError(f'{key} backup drift')
        for key,path in SOURCES.items():
            if sha(path.read_bytes())!=ORIGINAL[key]:
                atomic(path,(a.state_dir/f'{key}.orig').read_bytes())
    a.record.parent.mkdir(parents=True,exist_ok=True)
    a.record.write_text(json.dumps(dict(action=a.action,**manifest),indent=2)+'\n')
    print(json.dumps(dict(action=a.action,files=len(manifest['files']))))


if __name__=='__main__': main()
