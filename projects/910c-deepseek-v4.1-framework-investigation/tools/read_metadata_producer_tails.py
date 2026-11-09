#!/usr/bin/env python3
"""Qualify old C8 target metadata completion motifs without collecting a trace."""
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
from price_vocab_selection_fusion import records,cached,RECEIPT,SPANS,BODY

def read_database(spec):
    import json
    import sqlite3
    from pathlib import Path
    p=Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio/')
    before=p.stat();assert [before.st_size,before.st_mtime_ns]==spec['identity']
    c=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON')
    strings=dict(c.execute('SELECT id,value FROM STRING_IDS'))
    query='''SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.connectionId,t.taskType,k.opType
        FROM TASK t LEFT JOIN COMPUTE_TASK_INFO k USING(globalTaskId)
        WHERE t.startNs>=? AND t.startNs<? ORDER BY t.startNs,t.rowid'''
    windows=[]
    for w in spec['windows']:
        rows=[[*r[:5],strings[r[5]],strings.get(r[6],'')] for r in c.execute(query,(w['previous_draft_end'],w['body_end']))]
        meta=[r for r in rows if r[-1] in ('SparseFlashMlaMetadata','VllmAscendQuantLightningIndexerV2Metadata')]
        tail=[];issues=[]
        if [r[-1] for r in meta]!=['SparseFlashMlaMetadata']*3+['VllmAscendQuantLightningIndexerV2Metadata']*2:
            issues.append('target metadata motif is not exactly3SMLA+2QLI in this bounded interval')
        if meta:
            final=meta[-1]
            same=[r for r in rows if r[3]==final[3] and r[1]>=final[1]]
            tail=same[:5]
            if len(tail)<3 or tail[0][0]!=final[0] or tail[1][5]!='MEMCPY_ASYNC' or tail[2][5] not in ('EVENT RECORD','MEM_WRITE_VALUE'):
                issues.append('no exact final metadata -> copy -> ready-record motif')
            else:tail=tail[:3]
        else:issues.append('no metadata operations found')
        windows.append(dict(**w,metadata=meta,tail=tail,issues=issues,qualified=not issues))
    c.close();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return dict(rank=spec['rank'],database=str(p),identity=spec['identity'],windows=windows)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    waves,_,_,_,_=records();profiles=cached(RECEIPT);spans=json.loads(SPANS.read_bytes());bodies=cached(BODY)['profiles']
    prior=OUT/'sampler-output-handoff-receipt.json';selected=json.loads(prior.read_bytes())['source_specs']
    specs=[]
    for old in selected:
        rank=old['rank'];p=profiles[rank];windows=[]
        assert old['database']==p['database'] and old['identity']==[p['bytes'],p['mtime_ns']]
        for w in old['windows']:
            n=w['wave'];row=waves[n][rank]
            if row['kind']=='G':start,end=spans[rank][row['ordinal']][2:]
            else:
                b=bodies[rank]['bodies'][row['ordinal']];start,end=b['start_ns'],b['end_ns']
            assert start==row['start']
            windows.append(dict(wave=n,kind=w['kind'],B=w['B'],scheduled=w['scheduled'],
                previous_draft_end=waves[n-1][rank]['calls']['draft']['end'],body_start=start,body_end=end))
        specs.append(dict(rank=rank,database=p['database'],identity=old['identity'],windows=windows))
    remote='import json,sys\nfrom concurrent.futures import ThreadPoolExecutor\n'+inspect.getsource(read_database)+'''
with ThreadPoolExecutor(max_workers=2) as pool: out=list(pool.map(read_database,json.loads(sys.argv[1])))
print(json.dumps(out,separators=(',',':')))
'''
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-I','-S','-c',remote,json.dumps(specs)],capture_output=True,text=True,timeout=55)
    assert r.returncode==0,r.stderr
    rows=json.loads(r.stdout);raw=OUT/'metadata-producer-tail-raw.json.zlib';raw.write_bytes(zlib.compress(r.stdout.encode()))
    qualified=[dict(rank=r['rank'],**w) for r in rows for w in r['windows'] if w['qualified']]
    rejected=[dict(rank=r['rank'],wave=w['wave'],issues=w['issues']) for r in rows for w in r['windows'] if not w['qualified']]
    q=[]
    for w in qualified:
        ready=w['tail'][-1];q.append(dict(rank=w['rank'],wave=w['wave'],kind=w['kind'],B=w['B'],
            marker_type=ready[5],metadata_stream=ready[3],last_metadata_task=w['metadata'][-1][0],copy_task=w['tail'][1][0],ready_task=ready[0],
            ready_before_body_start_ms=(w['body_start']-ready[2])/1e6,
            ready_before_body_end_ms=(w['body_end']-ready[2])/1e6))
    receipt=dict(status='READ_ONLY_C8_METADATA_TAIL_MOTIF_QUALIFICATION',source_receipt_sha256=digest(prior),
        reader_sha256=digest(__file__),raw_path=str(raw),raw_sha256=digest(raw),source_specs=specs,
        windows=sum(len(r['windows']) for r in rows),qualified=len(q),unqualified=rejected,qualified_windows=q,
        qualification='Unique source-derived3SMLA+2QLI then copy/ready-record motif in each bounded target interval. Event handles and all tensor addresses are absent; not an identity proof or native integration result. Body end is graph replay end or eager last-HcPost end, at or before the proposed end-of-forward join. No new fence host/API overhead measured.',
        new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    path=OUT/'metadata-producer-tail-receipt.json';path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(windows=receipt['windows'],qualified=len(q),unqualified=len(rejected),
        minimum_ready_before_body_end_ms=min((r['ready_before_body_end_ms'] for r in q),default=None),
        maximum_ready_after_body_start_ms=max((-r['ready_before_body_start_ms'] for r in q),default=None),
        marker_types=sorted({r['marker_type'] for r in q}),new_NPU_work=0)))

if __name__=='__main__':main()
