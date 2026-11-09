#!/usr/bin/env python3
"""Targeted read-only SQL over existing C8 prefill/output handoffs; no profiling."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import zlib

sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))

def read_database(spec):
    from pathlib import Path
    import sqlite3
    from collections import defaultdict
    p=Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio/')
    stat=p.stat();assert [stat.st_size,stat.st_mtime_ns]==spec['identity']
    c=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON')
    strings=dict(c.execute('SELECT id,value FROM STRING_IDS'))
    wanted=[i for i,s in strings.items() if s in ['aclrtSynchronizeEvent','aclrtSynchronizeStreamWithTimeout']]
    waits=[list(r[:4])+[strings[r[4]]] for r in c.execute('SELECT startNs,endNs,globalTid,connectionId,name FROM CANN_API WHERE name IN ('+','.join('?' for _ in wanted)+') ORDER BY startNs',wanted)]
    out=[]
    for w in spec['windows']:
        q='''SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.connectionId,t.taskType,k.opType
             FROM TASK t LEFT JOIN COMPUTE_TASK_INFO k USING(globalTaskId)
             WHERE t.rowid BETWEEN ? AND ? ORDER BY t.startNs,t.rowid'''
        rows=[[*r[:5],strings[r[5]],strings.get(r[6],'')] for r in c.execute(q,(w['target_last_task'],w['next_target_task']))]
        by_stream=defaultdict(list)
        for row in rows:by_stream[row[3]].append(row)
        pairs=[]
        for stream,items in by_stream.items():
            for a,b in zip(items,items[1:]):
                if a[5]=='MEMCPY_ASYNC' and b[5]=='EVENT RECORD':pairs.append([a,b])
        anchors=[r for r in rows if r[0] in [w['target_last_task'],w['draft_last_task'],w['next_target_task']]]
        selected={r[4] for r in anchors}|{r[4] for pair in pairs for r in pair}
        api={}
        for cid in selected:
            api[cid]=[list(r[:3])+[strings[r[3]]] for r in c.execute('SELECT startNs,endNs,globalTid,name FROM CANN_API WHERE connectionId=?',(cid,))]
        assert all(len(api[cid])==1 for cid in selected)
        clock_min=min(api[r[4]][0][0] for r in anchors)
        clock_max=max(api[r[4]][0][1] for r in anchors)
        nearby=[v for v in waits if clock_min-1000000<=v[0]<=clock_max+2000000]
        out.append(dict(**w,anchors=anchors,memcpy_record_pairs=pairs,CANN_API_by_connection=api,host_waits=nearby))
    c.close();after=p.stat();assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return dict(rank=spec['rank'],database=str(p),identity=spec['identity'],windows=out)

def main():
    from price_vocab_selection_fusion import records,RECEIPT,cached,SPANS,BODY
    waves,_,_,_,_=records();profiles=cached(RECEIPT);spans=json.loads(SPANS.read_bytes());bodies=cached(BODY)['profiles']
    specs=[]
    # Four eager waves plus one fully populated verification graph per rank.
    for rank,p in enumerate(profiles):
        windows=[];chosen_graph=False
        for n,rs in waves.items():
            w=rs[rank]
            if not w.get('scheduled') or n+1 not in waves:continue
            if w['kind']=='G':
                if chosen_graph or w['calls']['target']['stats']['m']!=24:continue
                chosen_graph=True
            target,=[c for c in p['calls'] if c[0]=='target' and c[3][-1][2]==w['calls']['target']['end']]
            draft,=[c for c in p['calls'] if c[0]=='draft' and c[3][-1][2]==w['calls']['draft']['end']]
            nxt=waves[n+1][rank]
            next_task=spans[rank][nxt['ordinal']][0] if nxt['kind']=='G' else bodies[rank]['bodies'][nxt['ordinal']]['HcPre_TASK_ROWS'][0]
            windows.append(dict(wave=n,kind=w['kind'],ordinal=w['ordinal'],scheduled=w['scheduled'],target_M=target[2],B=w['calls']['target']['stats']['b'],target_last_task=target[3][-1][0],draft_last_task=draft[3][-1][0],next_target_task=next_task))
        assert len(windows)==(5 if rank<8 else 4)
        specs.append(dict(rank=rank,database=p['database'],identity=[p['bytes'],p['mtime_ns']],windows=windows))
    source=inspect.getsource(read_database)
    remote='''import json,sys
from concurrent.futures import ThreadPoolExecutor
'''+source+'''\nspecs=json.loads(sys.argv[1])
with ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(read_database,specs))
print(json.dumps(rows,separators=(',',':')))
'''
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-I','-S','-c',remote,json.dumps(specs)],capture_output=True,text=True,timeout=55)
    if r.returncode:
        (OUT/'sampler-output-handoff-read-error.txt').write_text(r.stderr)
        raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);raw=json.dumps(d,separators=(',',':')).encode();p=OUT/'sampler-output-handoff-raw.json.zlib';p.write_bytes(zlib.compress(raw))
    receipt=dict(status='READ_ONLY_EXISTING_PROFILE_EXTRACTION',artifact=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_specs=specs,rank_count=len(d),window_count=sum(len(r['windows']) for r in d),new_NPU_work=False,new_profile_collection=False,offline_SQL_extractions=1,remote_writes=[])
    (OUT/'sampler-output-handoff-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','rank_count','window_count','new_NPU_work','offline_SQL_extractions','remote_writes']}))

if __name__=='__main__':main()
