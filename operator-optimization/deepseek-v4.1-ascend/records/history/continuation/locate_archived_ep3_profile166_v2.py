from pathlib import Path
import os,time,socket,json

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
root=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis')
start=time.monotonic();visited=0;found=[];stop='complete_within_declared_analysis_tree'
for parent,dirs,files in os.walk(root,topdown=True,followlinks=False):
    visited+=len(dirs)+len(files)
    dirs[:]=[d for d in dirs if d not in {'.git','__pycache__'}]
    if len(Path(parent).relative_to(root).parts)>12:dirs[:]=[]
    for d in dirs:
        if d.startswith('dp0_pp0_tp3_dcp0_ep3_rank3_2128_'):
            p=Path(parent)/d/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db'
            found.append({'rank_dir':str(Path(parent)/d),'database':str(p),'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None})
    if len(found)>=8:stop='result_limit8';break
    if visited>=200000:stop='entry_limit200000';break
    if time.monotonic()-start>15:stop='time_limit15s';break
print(json.dumps({'hostname':socket.gethostname(),'root':str(root),'scope':'archive analysis tree filename lookup only, depth<=12,entry<=200000,time<=15s','visited_entries':visited,'seconds':time.monotonic()-start,'stop':stop,'found':found,'no_device_work':True},ensure_ascii=False))
