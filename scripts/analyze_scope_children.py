import argparse
import glob
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--raw',required=True)
p.add_argument('--phases',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
trace_path=glob.glob(str(Path(a.raw)/'dp0_pp0_tp0*'/'ASCEND_PROFILER_OUTPUT'/'trace_view.json'))[0]
trace=json.loads(Path(trace_path).read_text())
marks=json.loads(Path(a.phases).read_text())
start=datetime.fromisoformat(marks['warm_start_utc']).timestamp()*1e6
end=datetime.fromisoformat(marks['warm_end_utc']).timestamp()*1e6
chosen={}
for name in ('prepare input','draft_token'):
    scopes=[e for e in trace if e.get('ph')=='X' and e.get('cat')=='cpu_op' and e.get('name')==name and start<=float(e['ts'])<end]
    scopes.sort(key=lambda e:float(e['dur']))
    chosen[name]=scopes[len(scopes)//2]
out={'source':trace_path,'representatives':{}}
for name,scope in chosen.items():
    lo=float(scope['ts']); hi=lo+float(scope['dur']); tid=scope['tid']
    children=[]
    for e in trace:
        if e.get('ph')!='X' or e.get('cat')!='cpu_op' or e.get('tid')!=tid or e is scope:
            continue
        t=float(e['ts']); u=t+float(e['dur'])
        if lo<=t and u<=hi and float(e['dur'])>=100:
            children.append({'name':e['name'],'start_ms':(t-lo)/1000,'duration_ms':float(e['dur'])/1000})
    totals=Counter()
    for e in children: totals[e['name']]+=e['duration_ms']
    out['representatives'][name]={'scope_duration_ms':float(scope['dur'])/1000,'tid':tid,
        'top_child_duration_sums_ms':totals.most_common(30),
        'top_individual_children':sorted(children,key=lambda e:e['duration_ms'],reverse=True)[:30]}
Path(a.out).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['representatives'],indent=2)[:5500])
