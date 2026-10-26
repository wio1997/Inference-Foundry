#!/usr/bin/env python3
"""CPU-only negative gates for Run578 offline counter admission."""
import copy,csv,glob,json,shutil,tempfile
from pathlib import Path
from loop080_profile_salvage_run578 import (ROOT,analyse_csv,
                                            validate_profiler_identity)

base=ROOT/'evidence/20260928_loop080_bound/run578/live/b/bench/profile/rank0'
source=next(base.glob('**/kernel_details.csv'))
info=json.loads(next(base.glob('**/profiler_info_0.json')).read_text())
meta=json.loads(next(base.glob('**/profiler_metadata.json')).read_text())
validate_profiler_identity(info,meta,0)
negative=0
for change in ('rank','metric','schedule'):
    trial=copy.deepcopy(info)
    if change=='rank':trial['rank_id']=7
    if change=='metric':trial['config']['experimental_config']['_aic_metrics']='ACL_AICORE_PIPE_UTILIZATION'
    if change=='schedule':trial['config']['common_config']['schedule']['active']=1
    try:validate_profiler_identity(trial,meta,0)
    except ValueError:negative+=1
    else:raise AssertionError(f'{change} metadata accepted')
with tempfile.TemporaryDirectory(prefix='run578_salvage_gate_',dir=ROOT/'evidence/20260928_loop080_bound/run578') as td:
    path=Path(td)/'kernel_details.csv'
    with source.open(newline='') as f:
        rows=list(csv.DictReader(f));fields=list(rows[0])
    def check(rows,should_pass):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
        try:analyse_csv(path,0,{})
        except ValueError:
            if should_pass:raise
            return
        if not should_pass:raise AssertionError('corrupted CSV accepted')
    check(rows,True)
    for change in ('task','nan','all_zero','duplicate_replay'):
        trial=copy.deepcopy(rows)
        if change=='task':trial[86]['Task ID']='2'
        elif change=='nan':trial[0]['aic_read_main_memory_datas(KB)']='nan'
        elif change=='all_zero':
            for row in trial:
                for key in fields:
                    if key.endswith('(KB)'):row[key]='0'
        else:trial[86:]=copy.deepcopy(trial[:86])
        check(trial,False)
        negative+=1
print(json.dumps({'status':'cpu_only_salvage_gate_pass','negative_cases':negative}))
