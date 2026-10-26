#!/usr/bin/env python3
"""SHA-pinned reversible Run597 fixed-work diagnostic observer patch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
ASC=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
SOURCES={
    'serving':ROOT/'runtime/fixed_serving.py',
    'runtime':ROOT/'runtime/extreme_decode.py',
    'dspark':ROOT/'bootstrap/vllm_dspark_handoff.py',
    'draft':ASC/'spec_decode/dspark_proposer.py',
    'runner':ASC/'worker/model_runner_v1.py',
}
ORIGINAL={
    'serving':'137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a',
    'runtime':'eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499',
    'dspark':'fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e',
    'draft':'e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f',
    'runner':'004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba',
}
HELPER=ROOT/'scripts/loop081_basis_capture_run597.py'
MARK='# EXTREME_LOOP081_RUN597'
IMPORT='from scripts import loop081_basis_capture_run597 as _b597\n'

def sha(data):return hashlib.sha256(data).hexdigest()
def one(src,old,new):
    if src.count(old)!=1:raise ValueError(f'anchor count {src.count(old)}: {old[:90]!r}')
    return src.replace(old,new)

def patch(key,src):
    if MARK in src:raise ValueError('Run597 already patched')
    if key=='serving':
        src=one(src,'        self._parked = [False] * self.config.batch_size\n',
            '        self._parked = [False] * self.config.batch_size\n'
            f'        {MARK}\n'
            '        self._run597_enabled = bool(getattr(runtime, "_run597_context", {})'
            '.get("enabled", False))\n'
            '        self._run597_host_events = []\n'
            '        self._run597_cycle_index = -1\n'
            '        self._run597_firstpark_marked = False\n')
        src=one(src,'        self.runtime.proposer.park_completed_slots(slots, positions)\n',
            f'        {MARK}\n'
            '        if self._run597_enabled:\n'
            '            self._run597_host_events.extend(\n'
            '                {"slot": int(slot), "next_cycle": self._run597_cycle_index + 1,\n'
            '                 "anchor": int(pos)}\n'
            '                for slot, pos in zip(slots, positions)\n'
            '            )\n'
            '            if not self._run597_firstpark_marked:\n'
            '                self.runtime.state._run597_next_postpark = True\n'
            '                self._run597_firstpark_marked = True\n'
            '        self.runtime.proposer.park_completed_slots(slots, positions)\n')
        src=one(src,'        cycles = 0\n        for index in range(self.max_cycles):\n',
            f'        {MARK}\n'
            '        draft_history = (torch.empty(\n'
            '            (self.max_cycles,) + tuple(self.runtime.state.draft_tokens.shape),\n'
            '            dtype=self.runtime.state.draft_tokens.dtype, device=device)\n'
            '            if self._run597_enabled else None)\n'
            '        initial_last = (self.runtime.state.last_sampled_tokens.clone()\n'
            '                        if self._run597_enabled else None)\n'
            '        initial_draft = (self.runtime.state.draft_tokens.clone()\n'
            '                         if self._run597_enabled else None)\n'
            '        cycles = 0\n        for index in range(self.max_cycles):\n')
        src=one(src,'            count_history[index].copy_(self.runtime.state.num_sampled)\n',
            '            count_history[index].copy_(self.runtime.state.num_sampled)\n'
            f'            {MARK}\n'
            '            if self._run597_enabled:\n'
            '                draft_history[index].copy_(self.runtime.state.draft_tokens)\n'
            '                self._run597_cycle_index = index\n')
        src=one(src,'        counts_cpu = count_history[:cycles].cpu()\n',
            '        counts_cpu = count_history[:cycles].cpu()\n'
            f'        {MARK}\n'
            '        drafts_cpu = draft_history[:cycles].cpu() if self._run597_enabled else None\n')
        src=one(src,'        dag_dir = os.getenv("EXTREME_RUNTIME_DAG_DIR")\n',
            f'        {MARK}\n'
            '        if self._run597_enabled:\n'
            '            '+IMPORT.strip()+'\n'
            '            _b597.save(self, cycles, initial_last, initial_draft,\n'
            '                       tokens_cpu, counts_cpu, drafts_cpu, generated, staged)\n'
            '        dag_dir = os.getenv("EXTREME_RUNTIME_DAG_DIR")\n')
        return src
    if key=='runtime':
        src=one(src,'            use_scheduled = self._schedule_pending\n',
            '            use_scheduled = self._schedule_pending\n'
            f'            {MARK}\n'
            '            '+IMPORT.strip()+'\n'
            '            _b597.branch(self, use_scheduled)\n')
        return one(src,'            mark("prepare_target")\n',
            '            mark("prepare_target")\n'
            f'            {MARK}\n'
            '            _b597.target_prepared(self)\n')
    if key=='dspark':
        return one(src,'        mark("pack_hidden")\n',
            '        mark("pack_hidden")\n'
            f'        {MARK}\n'
            '        '+IMPORT.strip()+'\n'
            '        _b597.dspark_prepared(\n'
            '            self.proposer, state, common, token_indices, token_indices_to_sample,\n'
            '            num_rejected, target_token_ids, target_positions)\n')
    if key=='draft':
        return one(src,'        return num_query_total, token_indices_to_sample, cad, None\n',
            f'        {MARK}\n'
            '        '+IMPORT.strip()+'\n'
            '        _b597.draft_model_inputs(self, num_query_total,\n'
            '                                 token_indices_to_sample, cad)\n'
            '        return num_query_total, token_indices_to_sample, cad, None\n')
    if key=='runner':
        anchor=('                _extreme_wall_start = time.perf_counter()\n'
                '                _cohort_output = FixedCohortServing(\n')
        return one(src,anchor,
            f'                {MARK}\n'
            '                '+IMPORT.strip()+'\n'
            '                _b597.bind_cohort(\n'
            '                    _extreme_runtime, len(self._extreme_served_cohorts),\n'
            '                    _extreme_req_ids, os.getenv("RUN_TS"))\n'+anchor)
    raise ValueError('unknown source key')

def atomic(path,data):
    tmp=path.with_name(path.name+'.run597.tmp')
    if tmp.exists():raise ValueError('stale temporary file')
    with tmp.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    if path.exists():os.chmod(tmp,stat.S_IMODE(path.stat().st_mode))
    os.replace(tmp,path)

def prepared():
    helper=HELPER.read_bytes();compile(helper,str(HELPER),'exec')
    manifest=dict(helper_sha256=sha(helper),sources={})
    old={};new={}
    for key,path in SOURCES.items():
        data=path.read_bytes()
        if sha(data)!=ORIGINAL[key]:
            raise ValueError(f'{key} original SHA drift: {sha(data)}')
        edited=patch(key,data.decode()).encode()
        compile(edited,str(path),'exec')
        old[key]=data;new[key]=edited
        manifest['sources'][key]=dict(path=str(path),original_sha256=sha(data),
                                       patched_sha256=sha(edited))
    return manifest,old,new

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('check','install','restore'))
    ap.add_argument('--state-dir',type=Path)
    ap.add_argument('--record',required=True,type=Path)
    ap.add_argument('--offline-confirmed',action='store_true')
    args=ap.parse_args()
    if args.action!='check' and not args.offline_confirmed:
        raise ValueError('offline confirmation required')
    if args.action in ('check','install'):
        manifest,old,new=prepared()
        if args.action=='install':
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError('fresh state directory required')
            args.state_dir.mkdir(parents=True)
            for key,data in old.items():atomic(args.state_dir/f'{key}.orig',data)
            atomic(args.state_dir/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
            try:
                for key,path in SOURCES.items():
                    if path.read_bytes()!=old[key]:raise ValueError('install race '+key)
                    atomic(path,new[key])
            except BaseException:
                for key,path in SOURCES.items():
                    if sha(path.read_bytes())==manifest['sources'][key]['patched_sha256']:
                        atomic(path,old[key])
                raise
    else:
        if args.state_dir is None:raise ValueError('state dir required')
        manifest=json.loads((args.state_dir/'manifest.json').read_text())
        if set(manifest['sources'])!=set(SOURCES):raise ValueError('source set drift')
        for key,path in SOURCES.items():
            row=manifest['sources'][key]
            if row['path']!=str(path) or row['original_sha256']!=ORIGINAL[key]:
                raise ValueError('source manifest drift '+key)
            backup=(args.state_dir/f'{key}.orig').read_bytes()
            if sha(backup)!=ORIGINAL[key]:raise ValueError('backup drift '+key)
            if sha(path.read_bytes()) not in (row['original_sha256'],row['patched_sha256']):
                raise ValueError('installed source drift '+key)
        for key,path in SOURCES.items():
            if sha(path.read_bytes())!=ORIGINAL[key]:
                atomic(path,(args.state_dir/f'{key}.orig').read_bytes())
    args.record.parent.mkdir(parents=True,exist_ok=True)
    args.record.write_text(json.dumps(dict(action=args.action,**manifest),indent=2)+'\n')
    print(json.dumps(dict(action=args.action,files=len(SOURCES))))

if __name__=='__main__':main()
