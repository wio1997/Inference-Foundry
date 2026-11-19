"""Join actual Ascend CPU step ranges to replay calls and device tasks."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3

STEP = re.compile(r'step\[(?P<mode>[A-Z0-9_]+) bs=(?P<bs>\d+)(?P<fields>[^\]]*)\]')
COUNTER = re.compile(r'([cg]_(?:sq|sqsq|sqsk|sk)|forward_id|toks|c|g)=(\d+)')
AGGREGATES = ('sq', 'sqsq', 'sqsk', 'sk')

def span_data(name):
    match = STEP.fullmatch(name)
    if not match:
        return None
    fields = {}
    for token in match['fields'].split():
        part = COUNTER.fullmatch(token)
        if part is None or part[1] in fields:
            return None
        fields[part[1]] = int(part[2])
    counts = {key: value for key, value in fields.items() if '_' in key and key != 'forward_id'}
    for group in ('c', 'g'):
        present = {f'{group}_{field}' for field in AGGREGATES}.intersection(counts)
        if present and len(present) != len(AGGREGATES):
            return None
    return {'name': name, 'mode': match['mode'], 'batch_size': int(match['bs']),
            'forward_id': fields.get('forward_id'),
            'extend_num_tokens': fields.get('toks'),
            'context_generation_batch_sizes': {k: fields[k] for k in ('c', 'g') if k in fields},
            'actual_query_KV_aggregate_fields': counts}


def sqlite_integer(value):
    # Ascend profiler DBs can store CANN timestamps as decimal strings.
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r'-?\d+', value):
        return int(value)
    raise ValueError(f'Expected a profiler integer, found {value!r}')


def optional_integer(value):
    return None if value is None else sqlite_integer(value)

def analyze(path):
    path = path.resolve(strict=True)
    db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA temp_store=MEMORY')
    try:
        spans = []
        unknown = []
        for begin,end,tid,name in db.execute("""
            SELECT p.startNs,p.endNs,p.globalTid,s.value FROM PYTORCH_API p
            JOIN STRING_IDS s ON p.name=s.id WHERE s.value LIKE 'step[%'
        """):
            row=span_data(name)
            if row is None:
                unknown.append(name)
                continue
            begin, end, tid = map(sqlite_integer, (begin, end, tid))
            assert begin <= end, (begin, end, row)
            spans.append(dict(row, start_ns=begin, end_ns=end, global_tid=tid))
        spans.sort(key=lambda row: (row['start_ns'], row['end_ns'], row['global_tid']))
        by_tid=defaultdict(list)
        for index, row in enumerate(spans):
            row['id'] = index
            by_tid[row['global_tid']].append(row)
        begins={tid:[row['start_ns'] for row in group] for tid,group in by_tid.items()}
        latest_end={}
        for tid,group in by_tid.items():
            horizon=[]
            for row in group:
                horizon.append(max(horizon[-1],row['end_ns']) if horizon else row['end_ns'])
            latest_end[tid]=horizon
        graph_calls=[]
        for begin,end,cid,tid in db.execute("""
            SELECT a.startNs,a.endNs,a.connectionId,a.globalTid FROM CANN_API a
            JOIN STRING_IDS s ON a.name=s.id WHERE s.value='aclmdlRIExecuteAsync'
        """):
            begin, end = map(sqlite_integer, (begin, end))
            cid, tid = map(optional_integer, (cid, tid))
            assert begin <= end, (begin, end, cid)
            options=[]
            if tid in by_tid:
                i=bisect_right(begins[tid],begin)-1
                horizon=latest_end[tid]
                while i>=0 and horizon[i]>=end:
                    row=by_tid[tid][i]
                    if row['start_ns']<=begin<=end<=row['end_ns']:
                        options.append(row['id'])
                    i-=1
            call={'start_ns':begin,'end_ns':end,'connection_id':cid,'global_tid':tid,
                  'enclosing_step_ids':sorted(options),'direct_device_model_counts':{}}
            graph_calls.append(call)
        graph_calls.sort(key=lambda row: (row['start_ns'], row['end_ns']))
        duplicated=Counter(x['connection_id'] for x in graph_calls if x['connection_id'] is not None)
        by_cid={x['connection_id']:x for x in graph_calls
                if x['connection_id'] is not None and duplicated[x['connection_id']]==1}
        device_ids=set()
        task_counts=defaultdict(Counter)
        for cid,device,model,count in db.execute("""
                SELECT connectionId,deviceId,modelId,COUNT(*) FROM TASK GROUP BY connectionId,deviceId,modelId
        """):
            cid, device, model = map(optional_integer, (cid, device, model))
            device_ids.add(device)
            if cid in by_cid:
                task_counts[cid][(device,model)]+=count
        for cid,call in by_cid.items():
            call['direct_device_model_counts']={f'{device}:{model}':n
                for (device,model),n in task_counts[cid].items()}
        aggregates={}
        modes=sorted({v['mode'] for v in spans})
        for mode in modes:
            rows=[r for r in spans if r['mode']==mode]
            audited=[r for r in rows if r['actual_query_KV_aggregate_fields']]
            sums=Counter()
            for row in audited:
                sums.update(row['actual_query_KV_aggregate_fields'])
            ids={x['id'] for x in rows}
            aligned=[v for v in graph_calls if len(v['enclosing_step_ids'])==1 and v['enclosing_step_ids'][0] in ids]
            sourced=[v for v in aligned if v['direct_device_model_counts']]
            actual_query_samples=sum('g_sq' in r['actual_query_KV_aggregate_fields'] or
                                      'c_sq' in r['actual_query_KV_aggregate_fields'] for r in rows)
            aggregates[mode]={'steps':len(rows),'batch_size_histogram':dict(Counter(v['batch_size'] for v in rows)),
                'actual_annotated_query_steps':actual_query_samples,
                'query_KV_fields_sums_only_for_annotated_steps':dict(sums),
                'uniquely_enclosed_graph_calls':len(aligned),'direct_device_join_graph_calls':len(sourced),
                'logical_query_full_coverage':actual_query_samples==len(rows)}
        return {'database_bytes':path.stat().st_size,
            'actual_device_ids':sorted(device for device in device_ids if device is not None),
            'unattributed_device_ids_present':None in device_ids,
            'basic_step_spans':len(spans),'unparsed_step_names':unknown,'all_graph_calls':len(graph_calls),
            'graph_CIDs_duplicate':sum(n>1 for n in duplicated.values()),
            'graph_calls_without_enclosing_step':sum(not row['enclosing_step_ids'] for row in graph_calls),
            'graph_calls_with_multiple_enclosing_steps':sum(len(row['enclosing_step_ids'])>1 for row in graph_calls),
            'by_mode':aggregates,
            'spans':spans,'graph_calls':graph_calls,
            'scope':'Actual CPU ranges and direct device task joins. No query assumption from batch-size-only spans. Time and counts have diagnostic scope.'}
    finally:
        db.close()

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('database',type=Path)
    a=cli.parse_args()
    print(json.dumps(analyze(a.database),indent=2))
