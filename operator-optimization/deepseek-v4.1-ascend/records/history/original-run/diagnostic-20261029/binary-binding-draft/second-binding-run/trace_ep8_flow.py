#!/usr/bin/env python3
"""Stream a single exported trace for two observed DB connections and tasks."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run')
job=base/'offline-ep8-text-20261029'
status=json.loads((job/'status.json').read_text())
assert status['state']=='ANALYSE_RETURNED'
trace=list(job.glob('dp1_pp0_tp0_dcp0_ep8_rank0_746_20261029050352206_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json'))
assert len(trace)==1
trace=trace[0]
queue=json.loads((base/'offline-ep8-20261029/ep8-layer14-queue-audit.json').read_text())
targets=[]
for x in queue['results']:
    for op in x['operations']:
        for enq in op['queues']:
            for correlation in enq['correlations']:
                for d in correlation['matching_dequeues']:
                    for a in d['cann_all']:
                        if a['cann_api']['name']=='launch':
                            targets.append({'call_id':x['call_id'],'api':a['cann_api'],'tasks':a['task_rows']})
assert len(targets)==2
keys={str(x['api']['connectionId']<<32):x for x in targets}
by_task={(t['startNs'],t['streamId'],t['taskId']):x['call_id'] for x in targets for t in x['tasks']}
flows={key:[] for key in keys}
flow_counts=Counter()
x_events=[]
dec=json.JSONDecoder(parse_float=Decimal)
before=trace.stat()
count=0
with trace.open(encoding='utf8') as f:
    buf=f.read(1048576);pos=0;started=False;eof=False
    while True:
        while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
        if not started:
            if pos>=len(buf) or buf[pos]!='[':raise RuntimeError('unexpected trace root')
            pos+=1;started=True;continue
        if pos>=len(buf):
            if eof:raise RuntimeError('truncated trace')
            buf=f.read(1048576);pos=0;eof=not buf;continue
        if buf[pos]==']':break
        try:obj,end=dec.raw_decode(buf,pos)
        except json.JSONDecodeError:
            rem=buf[pos:];more=f.read(1048576)
            if not more or len(rem)+len(more)>4194304:raise RuntimeError('truncated or oversized item')
            buf=rem+more;pos=0;continue
        pos=end;count+=1
        if not isinstance(obj,dict) or obj.get('ph') not in ('s','t','f','X'):continue
        try:t=int(Decimal(str(obj['ts']))*1000)
        except (KeyError,InvalidOperation,TypeError,ValueError):continue
        if obj['ph']=='X':
            a=obj.get('args',{});a=a if isinstance(a,dict) else {}
            key=(t,a.get('Physic Stream Id'),a.get('Task Id'))
            if key in by_task:
                x_events.append({'call_id':by_task[key],'start_ns':t,'stream':key[1],'task':key[2],
                                 'name':obj.get('name'),'pid':obj.get('pid'),'tid':obj.get('tid')})
        else:
            ident=obj.get('id')
            if isinstance(ident,bool) or not isinstance(ident,(int,str)) or str(ident) not in keys:continue
            k=str(ident);flow_counts[k]+=1
            if len(flows[k])<50:
                flows[k].append({'phase':obj['ph'],'ts_ns':t,'category':obj.get('cat'),
                                 'pid':obj.get('pid'),'tid':obj.get('tid')})
after=trace.stat()
assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'trace changed'
h=hashlib.sha256()
with trace.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
result={'trace_bytes':before.st_size,'trace_sha256':h.hexdigest(),'items_scanned':count,
        'targets':[{'call_id':x['call_id'],'db_cann_launch':x['api'],'db_tasks':x['tasks'],
                    'flow_id_candidate':int(k),'flow_count':flow_counts[k], 'flow_records':flows[k],
                    'flow_truncated':flow_counts[k]>50} for k,x in keys.items()],
        'device_x_exact_start_stream_task':x_events,
        'limits':['CANN connection shifted left 32 is a candidate flow ID convention, not established universal contract.',
                  'Text and DB exports are independent offline parses of copied same raw; check completeness and precise timestamp conversion.',
                  'Exact record correspondence is not proof of producer/consumer tensor causality or selected binary identity.',
                  'The DB parser reported SQLite I/O error; no completeness claim.']}
(job/'ep8-layer14-flow-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
