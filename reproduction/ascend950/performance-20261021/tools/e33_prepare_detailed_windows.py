#!/usr/bin/env python3
"""Retain workload controls; new IDs/namespaces for detailed profiling only."""
from pathlib import Path
import hashlib,json
from e33_prepare_windows import render
ROOT=Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'data/E33-window-manifest.json').read_text())
records=[]
for old in source['records']:
    tag='DI' if '-DI-' in old['run_id'] else 'V'
    suffix=old['run_id'].split('-C32-',1)[1]
    namespace=old['namespace'].replace('e33-','e33-d1-',1)
    rid,text=render(tag,'D1-'+suffix,old['kind'],namespace)
    path=ROOT/'configs/e33'/(rid+'.py')
    if path.exists() and path.read_text()!=text:raise RuntimeError('Refuse replacement '+str(path))
    path.write_text(text)
    records.append({**old,'run_id':rid,'config':str(path.relative_to(ROOT)),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'namespace':namespace,'revision':'D1'})
result={**source,'records':records,'revision':'D1','revision_reason':'Installed runtime disabled op_attr in Text-only export; Level1 Text+Db/record_shapes provides communication and shape coverage. Original NORMAL U/P/U evidence retained, not overwritten.','unchanged':['input token IDs','output count','C32/N64','cache contract','model/scheduler/parallel flags']}
(ROOT/'data/E33-detailed-window-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'configs':len(records),'revision':'D1','workload_control_changed':False}))
