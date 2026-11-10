"""Inspect existing trace schema and a few concrete idle boundaries. CPU only."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,sys

root=Path(sys.argv[1])
analysis=json.load(open(root/'balanced-analysis.json'))
rows=[r for r in analysis['files'] if r['phase']=='steady' and r['ep'] in [0,8]]
out=[]
for row in rows:
    path=Path(row['file']);raw=json.load(path.open());events=raw['traceEvents'] if isinstance(raw,dict) else raw
    processes={e['pid']:e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    hw={p for p,n in processes.items() if n=='Ascend Hardware'};cann={p for p,n in processes.items() if n=='CANN'}
    hardware=[e for e in events if e.get('ph')=='X' and e.get('pid') in hw]
    kernels=[e for e in hardware if e.get('args',{}).get('Task Type') in ['MIX_AIC','AI_CORE','AI_VECTOR','AIV','AIC','MIX_AIV']]
    if not kernels:kernels=[e for e in hardware if 'Task Id' in e.get('args',{}) and e.get('name') not in ['EVENT WAIT','NOTIFY_WAIT_SQE','MODEL_WAIT_COMPLETE']]
    kernels.sort(key=lambda e:float(e['ts']))
    free=sorted([e for e in events if e.get('name')=='Free' and e.get('ph')=='X'],key=lambda e:-float(e['dur']))
    entry={'ep':row['ep'],'file':str(path),'output_files':[p.name for p in path.parent.iterdir()],
           'hardware_names':Counter(e.get('name') for e in hardware).most_common(22),
           'task_types':Counter(str(e.get('args',{}).get('Task Type')) for e in hardware).most_common(),
           'model_ids':Counter(str(e.get('args',{}).get('Model Id')) for e in kernels).most_common(),
           'enqueue_examples':[e for e in events if e.get('cat')=='enqueue'][:2],
           'dequeue_examples':[e for e in events if e.get('cat')=='dequeue'][:2],
           'cann_main_names':Counter(e.get('name') for e in events if e.get('ph')=='X' and e.get('pid') in cann and e.get('tid')==728).most_common(25),
           'graph_api_examples':[e for e in events if e.get('ph')=='X' and e.get('pid') in cann and any(x in e.get('name','').lower() for x in ['modelexecute','captureexecute','modelexecute','modelexec','model_execute','modelexecution','modelricapture'])][:10],
           'gaps':[]}
    for p in [path.parent/'operator_details.csv',path.parent/'api_statistic.csv']:
        if p.exists():
            with p.open() as stream:
                r=csv.DictReader(stream);entry[p.name]={'columns':r.fieldnames,'first':next(r,None)}
    for gap in free[:6]:
        a=float(gap['ts']);b=a+float(gap['dur'])
        before=[e for e in kernels if float(e['ts'])<a];after=[e for e in kernels if float(e['ts'])>=b-1]
        near=[e for e in events if e.get('ph')=='X' and e.get('pid') in cann and float(e.get('ts',0))<b and float(e.get('ts',0))+float(e.get('dur',0))>a]
        near=sorted(near,key=lambda e:-min(float(e['dur']),b-a))[:5]
        entry['gaps'].append({'gap':gap,'previous_kernels':before[-2:],'next_kernels':after[:2],'cann_during':near})
    out.append(entry)
(root/'idle-schema-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('probe_complete',len(out))
