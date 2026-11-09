#!/usr/bin/env python3
"""Bounded immutable SQL of saved C8 aux/main-to-first-draft delivery."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

sys.dont_write_bytecode = True
BASE = Path('/Users/wio/work/branch-fusion-audit-20261104')
OUT = Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records, cached, RECEIPT

REMOTE = r'''
import concurrent.futures,json,pathlib,sqlite3,sys,zlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
def read(spec):
 p=pathlib.Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio')
 before=p.stat();assert [before.st_size,before.st_mtime_ns]==spec['identity']
 db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON')
 strings=dict(db.execute('SELECT id,value FROM STRING_IDS'))
 query='''+"'''"+r'''SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.modelId,k.opType,k.inputShapes,k.outputShapes
 FROM TASK t JOIN COMPUTE_TASK_INFO k USING(globalTaskId)
 WHERE t.rowid BETWEEN ? AND ? AND t.streamId=? AND t.modelId=? ORDER BY t.startNs,t.rowid'''+"'''"+r'''
 result=[]
 for w in spec['windows']:
  record=dict(w)
  try:
   kv=db.execute('SELECT startNs,endNs,streamId,modelId FROM TASK WHERE rowid=?',(w['first_context_task'],)).fetchone()
   pre=db.execute('SELECT startNs,endNs,streamId,modelId FROM TASK WHERE rowid=?',(w['first_HcPre'],)).fetchone()
   assert kv and pre and kv[0]==w['first_context_ns'] and pre[0]==w['first_HcPre_ns'] and kv[2:]==pre[2:]
   rows=[[ *r[:5],*[strings[v] for v in r[5:]]] for r in db.execute(query,(w['first_context_task']-512,w['first_context_task'],kv[2],kv[3]))]
   mm=[r for r in rows if r[5]=='MatMulV2' and r[6].endswith(';640,15360"') and r[2]<=kv[0]]
   if not w['active']:
    assert not mm,'Idle path unexpectedly has main projection'
    record.update(qualified=True,route='idle_no_main',parent_start_ns=None,selected_rows=None)
   else:
    assert len(mm)==1,('main_projection_count',len(mm))
    main=mm[0];n=int(main[6].split(',')[0].strip('"'));assert n==w['context_rows']
    cats=[r for r in rows if r[5]=='ConcatD' and r[7]==f'"{n},15360"' and r[2]<=main[1]]
    assert len(cats)==1,('aux_cat_count',len(cats))
    cat=cats[0];at=rows.index(cat)
    beforecat=rows[max(0,at-6):at]
    indexed=[r[5] for r in beforecat]==['IndexCheck','Index']*3
    selection=beforecat if indexed else []
    norms=[r for r in rows if main[2]<=r[1]<kv[0] and r[5]=='RmsNorm' and r[6]==f'"{n},5120;5120"']
    assert len(norms)==1,('main_norm_count',len(norms))
    norm=norms[0]
    start=selection[0][1] if selection else cat[1]
    assert w['target_head_end_ns']<=start<main[1]<main[2]<=norm[1]<norm[2]<=kv[0]<pre[0]
    record.update(qualified=True,route='indexed' if indexed else 'prefix',selected_rows=n,
      parent_start_ns=start,parent_end_ns=pre[0],main_task=main,aux_cat_task=cat,
      aux_index_tasks=selection,main_norm_task=norm,main_to_norm_elapsed_ns=norm[2]-main[1],
      parent_elapsed_ns=pre[0]-start,selection_to_context_start_ns=kv[0]-start,
      returned_compute_rows=len(rows))
  except Exception as e:record.update(qualified=False,error=type(e).__name__+': '+str(e))
  result.append(record)
 db.close();after=p.stat();assert [before.st_size,before.st_mtime_ns]==[after.st_size,after.st_mtime_ns]
 return dict(rank=spec['rank'],database=spec['database'],identity=spec['identity'],windows=result)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(read,json.load(sys.stdin)))
sys.stdout.buffer.write(zlib.compress(json.dumps(out,separators=(',',':')).encode()))
'''

def main():
    waves,_,_,_,_ = records();profiles=cached(RECEIPT)
    receipt=BASE/'full-review/runs/frozen-draft-context-panels.json'
    manifest=json.loads(receipt.read_bytes());specs=[];pins=[]
    for pin in manifest['rank_files']:
        rank=pin['rank'];p=BASE/pin['path'];raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pin['sha256']
        panels=json.loads(zlib.decompress(raw));profile=profiles[rank]
        assert len(panels)==len(waves)==231
        ws=[]
        for wave,(anchor,head,sections,later) in zip(waves,panels):
            r=waves[wave][rank]
            ws.append(dict(wave=wave,active=bool(r.get('scheduled')),kind=r['kind'],ordinal=r['ordinal'],
                first_context_task=sections[0][1],first_context_ns=sections[0][3],
                first_HcPre=anchor,first_HcPre_ns=later[0][2],context_rows=sections[0][0],
                target_head_end_ns=r['calls'].get('target',{}).get('end',0)))
        specs.append(dict(rank=rank,database=profile['database'],identity=[profile['bytes'],profile['mtime_ns']],windows=ws))
        pins.append(dict(path=str(p),sha256=pin['sha256']))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE],
                     input=json.dumps(specs).encode(),capture_output=True,timeout=240)
    assert r.returncode==0,r.stderr.decode()
    data=json.loads(zlib.decompress(r.stdout));blob=OUT/'context-parent-raw.json.zlib';blob.write_bytes(r.stdout)
    invalid=[dict(rank=p['rank'],**w) for p in data for w in p['windows'] if not w['qualified']]
    d=dict(raw_path=str(blob),raw_sha256=hashlib.sha256(r.stdout).hexdigest(),source_specs=specs,
        reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_pins=pins,
        panel_manifest=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()),
        windows=sum(len(p['windows']) for p in data),qualified=sum(w['qualified'] for p in data for w in p['windows']),
        unqualified=invalid,identity_contract='Reuse prior hashed immutable C8 DB receipt; exact size/mtime and saved task IDs/timestamps checked before/after bounded readonly SQL. Not a new full DB hash or profile capture.',
        new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    (OUT/'context-parent-receipt.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(windows=d['windows'],qualified=d['qualified'],unqualified=len(invalid),examples=invalid[:2],new_NPU_work=0)))

if __name__=='__main__':main()
