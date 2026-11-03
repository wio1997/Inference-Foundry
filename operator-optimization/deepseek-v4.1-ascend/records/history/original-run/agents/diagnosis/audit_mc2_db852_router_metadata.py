#!/usr/bin/env python3
"""Read only selected historical MC2 metadata, never tensor contents or NPU."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'agents/diagnosis/mc2-db852-extremes-pmu.json'
rows = json.loads(source.read_bytes())['results']
base = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces'
remote = r'''import json, sqlite3, pathlib, sys
base,rank,targets=sys.argv[1:]
p=pathlib.Path(base)/rank
dbs=list((p/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
if len(dbs)!=1: raise RuntimeError('Expected exactly one profiler DB')
c=sqlite3.connect(dbs[0].as_uri()+'?mode=ro',uri=True)
c.execute('pragma query_only=ON')
result=[]
for target in json.loads(targets):
 gid=target['globalTaskId']
 rows=c.execute('select blockNum,mixBlockNum,inputShapes,outputShapes,attrInfo,gridDim,blockDim from COMPUTE_TASK_INFO where globalTaskId=?',(gid,)).fetchall()
 if len(rows)!=1: raise RuntimeError('Expected one compute record')
 r=rows[0]; item=dict(target, blockNum=r[0], mixBlockNum=r[1])
 for name, sid in zip(('inputShapes','outputShapes','attrInfo','gridDim','blockDim'),r[2:]):
  matches=c.execute('select value from STRING_IDS where id=?',(sid,)).fetchall()
  item[name]=matches[0][0] if len(matches)==1 else None
 result.append(item)
names=('tensor','tiling','dump','snapshot','graph')
files=sorted((f.name,f.stat().st_size) for f in p.rglob('*') if f.is_file() and any(s in f.name.lower() for s in names))
tables={t[0]:[v[1] for v in c.execute('pragma table_info("'+t[0]+'")')] for t in c.execute("select name from sqlite_master where type='table'").fetchall()}
print(json.dumps({'db':dbs[0].name,'selected':result,'candidate_filenames_and_bytes':files,'profiler_schema':tables}))
'''
output = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'ranks': {}}
for ep in (0, 3):
    matches = [x for x in rows if x['ep'] == ep]
    if len(matches) != 2 or any(len(x['matches']) != 1 or x['global_id_task_count'] != 1 for x in matches):
        raise ValueError('target identity not unique')
    targets = [{'layer': x['layer'], 'globalTaskId': x['matches'][0][0]} for x in matches]
    cmd = shlex.join(['docker', 'exec', 'wio-vllm-v41-repro169-stage-20261028', 'python3', '-c', remote, base, matches[0]['target']['rank'], json.dumps(targets)])
    p = subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh', cmd], capture_output=True, text=True, timeout=50, check=True)
    output['ranks'][str(ep)] = json.loads(p.stdout)
dest = root / 'agents/diagnosis/mc2-db852-router-metadata.json'
dest.write_text(json.dumps(output, indent=2) + '\n')
print('wrote', dest.name, 'sha256', hashlib.sha256(dest.read_bytes()).hexdigest())
