"""Read-only CPU reduction of immutable Run249; never imports torch or sends requests."""
import collections
import gzip
import hashlib
import json
import pathlib
import re
import sqlite3
import sys
import time


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def union(intervals, lo=-float('inf'), hi=float('inf')):
    total = 0.0
    end = lo
    for s, e in sorted(intervals):
        s, e = max(s, lo), min(e, hi)
        if e > max(s, end):
            total += e - max(s, end)
            end = e
    return total


def one(path, output):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    obj = json.loads(data)
    del data
    es = obj['traceEvents'] if isinstance(obj, dict) else obj
    meta = {e['pid']: e['args'].get('name') for e in es
            if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    py = next(p for p, n in meta.items() if n == 'Python')
    hw = next(p for p, n in meta.items() if n == 'Ascend Hardware')
    cann = next(p for p, n in meta.items() if n == 'CANN')
    comm = next(p for p, n in meta.items() if n == 'Communication')
    rank = int(re.search(r'_rank(\d+)_', str(path)).group(1))
    rows = [(i, e, float(e['ts']), float(e.get('dur', 0)))
            for i, e in enumerate(es) if e.get('ph') == 'X']
    cpu = [r for r in rows if r[1]['pid'] == py and r[1]['tid'] == py
           and r[1].get('cat') == 'cpu_op']
    hardware = [r for r in rows if r[1]['pid'] == hw]
    communication = [r for r in rows if r[1]['pid'] == comm and r[1]['name'].startswith('hcom_')]
    embedding = sorted([r for r in cpu if r[1]['name'] == 'aten::embedding'], key=lambda r: r[2])
    scopes = [r for r in cpu if r[1]['name'].startswith('vllm::')]
    sync = [r for r in rows if ('synchron' in r[1]['name'].lower())]
    start = min(r[2] for r in hardware if r[1]['args'].get('Task Type') not in ('PROFILING_ENABLE', 'PROFILING_DISABLE'))
    end = max(r[2] + r[3] for r in hardware if r[1]['args'].get('Task Type') not in ('PROFILING_ENABLE', 'PROFILING_DISABLE'))
    cann_by_conn = collections.defaultdict(list)
    for r in rows:
        if r[1]['pid'] == cann and 'connection_id' in r[1].get('args', {}):
            cann_by_conn[r[1]['args']['connection_id']].append(r)
    def raw(r):
        return {'index': r[0], 'event': r[1]}
    periods = []
    # Ten embeddings must be verified against installed target/MTP sources.
    for k, r in enumerate(embedding[::2]):
        lo = r[2]
        hi = embedding[2*k+2][2] if 2*k+2 < len(embedding) else end
        pp = {'ordinal': k, 'lo_us': lo, 'hi_us': hi, 'extent_us': hi-lo,
              'embedding': raw(r), 'draft_embedding': raw(embedding[2*k+1]) if 2*k+1 < len(embedding) else None}
        groups = collections.defaultdict(list)
        for _, e, s, d in hardware:
            groups[e['args'].get('Task Type', '?')].append((s, s+d))
        pp['hardware_unions_us'] = {name: union(v, lo, hi) for name, v in groups.items()}
        pp['compute_union_us'] = union([(s,s+d) for _,e,s,d in hardware if e['args'].get('Task Type') in ('MIX_AIC','AI_CORE','AI_VECTOR_CORE','MIX_AIV')],lo,hi)
        pp['communication_union_us'] = union([(s,s+d) for _,e,s,d in hardware if e['args'].get('Task Type') == 'COMMUNICATION'],lo,hi)
        pp['main_cpu_op_union_us'] = union([(s,s+d) for _,e,s,d in cpu],lo,hi)
        pp['cpu_scope_totals_us'] = dict(collections.Counter())
        for name in sorted(set(e['name'] for _,e,s,d in scopes)):
            pp['cpu_scope_totals_us'][name] = sum(max(0,min(s+d,hi)-max(s,lo)) for _,e,s,d in scopes if e['name']==name)
        pp['scope_count'] = dict(collections.Counter(e['name'] for _,e,s,d in scopes if lo<=s<hi))
        pp['sync'] = [raw(r) for r in sync if lo<=r[2]<hi and r[3]>1000]
        periods.append(pp)
    # Stream order plus actual CANN connection; this is a bound, not a DAG proof.
    stream_gaps = []
    stream_groups = collections.defaultdict(list)
    for r in hardware:
        stream_groups[r[1]['tid']].append(r)
    for stream, group in stream_groups.items():
        group.sort(key=lambda r:(r[2],r[0]))
        for prev, r in zip(group, group[1:]):
            gap = r[2] - prev[2] - prev[3]
            if gap <= 20:
                continue
            conn = r[1].get('args',{}).get('connection_id')
            launches = cann_by_conn.get(conn, [])
            # A task can have nested ACL/CANN events; preserve all candidates.
            stream_gaps.append({'stream':stream,'gap_us':gap,'previous':raw(prev),'next':raw(r),
                                'cann': [raw(x) for x in launches]})
    count = collections.Counter(e['name'] for _,e,s,d in cpu)
    totals = collections.defaultdict(float)
    for _,e,s,d in cpu: totals[e['name']] += d
    result = {'rank':rank,'raw':{'path':str(path),'bytes':path.stat().st_size,'sha256':sha},
              'metadata':meta,'hardware_extent_us':end-start,'hardware_lo_us':start,'hardware_hi_us':end,
              'embeddings':[raw(r) for r in embedding], 'periods':periods,
              'cpu_top_inclusive': [{'name':n,'count':count[n],'inclusive_us':d} for n,d in sorted(totals.items(),key=lambda x:-x[1])[:30]],
              'main_cpu_op_union_us':union([(s,s+d) for _,e,s,d in cpu],start,end),
              'longest_stream_gaps':sorted(stream_gaps,key=lambda x:-x['gap_us'])[:40],
              'stream_gap_sums_us':{stream:sum(g['gap_us'] for g in stream_gaps if g['stream']==stream) for stream in stream_groups},
              'flow_counts':dict(collections.Counter(e.get('cat','?') for e in es if e.get('ph')=='s')),
              'notes':['Intervals overlap; do not add inclusive sums.', 'Profile ON only; no direct decomposition of unprofiled 265ms/token.', 'Embedding pairing is provisional until model-source validation.', 'No model or service operation.']}
    write(output/f'rank{rank}.json',result)
    with gzip.open(output/f'collectives{rank}.json.gz','wt') as f:
        json.dump([raw(r) for r in communication],f)
    if rank == 0:
        keep = []
        for i,e in enumerate(es):
            if e.get('ph') == 'X' and (e['pid'] in (hw,comm,cann) or (e['pid']==py and (e.get('cat')=='cpu_op' or e['name'].startswith(('Enqueue@','Dequeue@'))))):
                keep.append({'index':i,'event':e})
            elif e.get('ph') in ('s','f'):
                keep.append({'index':i,'event':e})
        with gzip.open(output/'selected_rank0.json.gz','wt') as f:json.dump(keep,f)
        db = path.parent/'ascend_pytorch_profiler_0.db'
        with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as cx:
            schema = cx.execute("select name,sql from sqlite_master where type='table'").fetchall()
        write(output/'database_schema.json',schema)
    return {'rank':rank,'sha256':sha,'periods':len(periods),'cpu_union_us':result['main_cpu_op_union_us']}


if __name__ == '__main__':
    root, dest, ranks = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    dest.mkdir(exist_ok=True,parents=True)
    wanted=set(map(int,ranks.split(','))) if ranks!='all' else set(range(16))
    paths=sorted(root.glob('*_rank*_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    index=[]
    for path in paths:
        rank=int(re.search(r'_rank(\d+)_',str(path)).group(1))
        if rank not in wanted:continue
        started=time.monotonic();row=one(path,dest);row['analysis_wall_s']=time.monotonic()-started
        index.append(row);print(json.dumps(row),flush=True)
    write(dest/'index.json',index)
