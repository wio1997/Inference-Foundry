#!/usr/bin/env python3
"""Reversible, SHA-pinned Run582 capture identity patch; check is read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES = {
    'graph': ASC/'compilation/acl_graph.py',
    'dsa': ASC/'attention/context_parallel/dsa_cp.py',
    'linear': ASC/'ops/linear_op.py',
    'model': ASC/'models/deepseek_v4.py',
    'runtime': ROOT/'runtime/extreme_decode.py',
}
ORIGINAL = {
    'graph': '6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6',
    'dsa': '27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e',
    'linear': '599f85b8dd4c13f72aa458e21ae7fa13e18548a645fd03b8a534a6b951e7ee02',
    'model': '11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247',
    'runtime': 'eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499',
}
HELPER = ROOT/'scripts/loop081_slice_capture_run582.py'
MARK = '# EXTREME_LOOP081_RUN582'
IMPORT = 'from scripts import loop081_slice_capture_run582 as _s582\n'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def one(src, anchor, block, *, before=True):
    n=src.count(anchor)
    if n!=1: raise ValueError(f'anchor count {n}: {anchor[:100]!r}')
    return src.replace(anchor, (block+anchor) if before else (anchor+block))

def patch(key, src):
    if MARK in src: raise ValueError('Run582 source already patched')
    if key=='graph':
        anchor='                        output = self.runnable(*args, **kwargs)\n'
        replacement=(f'                        {MARK}\n'
                     f'                        {IMPORT}'
                     '                        _s582.graph_begin(entry, self)\n'
                     '                        try:\n'
                     '                            output = self.runnable(*args, **kwargs)\n'
                     '                        finally:\n'
                     '                            _s582.graph_end()\n')
        if src.count(anchor)!=1: raise ValueError('Graph runnable anchor drift')
        src=src.replace(anchor,replacement)
        return one(src,'        entry.aclgraph.replay()\n',
                   f'        {MARK}\n        {IMPORT}'
                   '        _s582.graph_replay(entry, self)\n',before=False)
    if key=='dsa':
        anchor='                output[...] = self._apply_wo_b(o_proj_input, full_gather_wo_a_enabled)\n'
        if src.count(anchor)!=1: raise ValueError('DSA output anchor drift')
        block=(f'                {MARK}\n'
               f'                {IMPORT}'
               '                _s582.attention_before(layer_name, self, o_proj_input, output,\n'
               '                                       full_gather_wo_a_enabled)\n'
               '                output[...] = _s582.projected(\n'
               '                    layer_name, self._apply_wo_b(o_proj_input, full_gather_wo_a_enabled))\n'
               '                _s582.attention_after(layer_name, output)\n')
        return src.replace(anchor,block)
    if key=='linear':
        anchor='            output = tensor_model_parallel_reduce_scatter(output_parallel, 0)\n'
        if src.count(anchor)!=1: raise ValueError('linear RS anchor drift')
        block=(anchor+f'            {MARK}\n'
               f'            {IMPORT}'
               '            _s582.sequence_rs(self.layer.prefix, x, output_parallel, output,\n'
               '                              flash_comm_v1_enabled, mmrs_fusion)\n')
        return src.replace(anchor,block)
    if key=='runtime':
        anchor='                    target_output = self.target.execute(self.state)\n'
        if src.count(anchor)!=1: raise ValueError('runtime target anchor drift')
        block=(f'                    {MARK}\n'
               f'                    {IMPORT}'
               '                    _s582.target_begin(self)\n'
               '                    try:\n'
               '                        target_output = self.target.execute(self.state)\n'
               '                    finally:\n'
               '                        _s582.target_end()\n')
        return src.replace(anchor,block)
    anchor=('        hidden_states = self.self_attn(**attn_kwargs)\n'
            '        hidden_states = self.hc_post(hidden_states, residual, post, comb)\n')
    if src.count(anchor)!=1: raise ValueError('model HC consumer anchor drift')
    replacement=('        hidden_states = self.self_attn(**attn_kwargs)\n'
                 f'        {MARK}\n'
                 f'        {IMPORT}'
                 '        _s582.hc_consumer(self.layer_idx, hidden_states, residual, post, comb)\n'
                 '        hidden_states = self.hc_post(hidden_states, residual, post, comb)\n')
    return src.replace(anchor,replacement)

def prepared():
    helper=HELPER.read_bytes(); compile(helper,str(HELPER),'exec')
    manifest={};old={};new={}
    for key,path in SOURCES.items():
        data=path.read_bytes()
        if sha(data)!=ORIGINAL[key]: raise ValueError(f'{key} source SHA drift: {sha(data)}')
        patched=patch(key,data.decode()).encode()
        compile(patched,str(path),'exec')
        manifest[key]=dict(path=str(path),original=sha(data),patched=sha(patched))
        old[key],new[key]=data,patched
    return dict(helper_sha=sha(helper),files=manifest),old,new

def atomic(path,payload):
    tmp=path.with_name(path.name+'.run582.tmp')
    if tmp.exists(): raise ValueError('stale Run582 temporary file')
    with tmp.open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    if path.exists(): os.chmod(tmp,stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('check','install','restore'))
    p.add_argument('--state-dir',type=Path)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--offline-confirmed',action='store_true')
    a=p.parse_args()
    if a.action!='check' and not a.offline_confirmed:
        raise ValueError('offline confirmation required')
    if a.action in ('check','install'):
        manifest,old,new=prepared()
        if a.action=='install':
            if a.state_dir is None or a.state_dir.exists():
                raise ValueError('fresh state directory required')
            a.state_dir.mkdir(parents=True)
            for key,data in old.items(): atomic(a.state_dir/f'{key}.orig',data)
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
        if a.state_dir is None: raise ValueError('state directory required')
        manifest=json.loads((a.state_dir/'manifest.json').read_text())
        if set(manifest['files'])!=set(SOURCES): raise ValueError('source set drift')
        helper_drift=manifest['helper_sha']!=sha(HELPER.read_bytes())
        for key,path in SOURCES.items():
            row=manifest['files'][key]
            if row['path']!=str(path) or row['original']!=ORIGINAL[key]:
                raise ValueError(f'{key} source identity mismatch')
            if sha((a.state_dir/f'{key}.orig').read_bytes())!=ORIGINAL[key]:
                raise ValueError(f'{key} backup drift')
            if sha(path.read_bytes()) not in (row['original'],row['patched']):
                raise ValueError(f'{key} installed source drift')
        for key,path in SOURCES.items():
            if sha(path.read_bytes())!=ORIGINAL[key]:
                atomic(path,(a.state_dir/f'{key}.orig').read_bytes())
    a.record.parent.mkdir(parents=True,exist_ok=True)
    a.record.write_text(json.dumps(dict(action=a.action,helper_drift_on_restore=(helper_drift if a.action=='restore' else False),**manifest),indent=2)+'\n')
    print(json.dumps(dict(action=a.action,files=len(manifest['files']))))

if __name__=='__main__': main()
