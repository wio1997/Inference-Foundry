#!/usr/bin/env python3
"""Read existing C8 first-attention input prefixes; no new model/profile work."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys,zlib
sys.dont_write_bytecode=True
BASE=Path('/Users/wio/work/branch-fusion-audit-20261104');ROOT=Path('audit/execution-search')
sys.path.insert(0,str(BASE/'tools'))
from price_vocab_selection_fusion import records,cached,RECEIPT
from price_prefill_attention_delivery import BODY
from read_frozen_vocab_tasks import SPANS
REMOTE=r'''
import bisect,concurrent.futures,json,pathlib,socket,sqlite3,sys,zlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
def read(spec):
 p=pathlib.Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio');before=p.stat();assert [before.st_size,before.st_mtime_ns]==spec['identity']
 db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON');db.execute('PRAGMA temp_store=MEMORY')
 strings=dict(db.execute('SELECT id,value FROM STRING_IDS'));types=[i for i,v in strings.items() if v in ('HcPre','HcPost','SparseFlashMla')]
 emb_shapes=[i for i,v in strings.items() if v.startswith('"16160,5120;')]
 assert types and emb_shapes,(len(types),len(emb_shapes))
 where='k.opType IN ('+','.join('?' for _ in types)+') OR k.inputShapes IN ('+','.join('?' for _ in emb_shapes)+')'
 query='SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.modelId,k.opType,k.inputShapes,k.outputShapes FROM TASK t JOIN COMPUTE_TASK_INFO k USING(globalTaskId) WHERE '+where+' ORDER BY t.startNs,t.rowid'
 rows=[[*r[:5],*[strings[v] for v in r[5:]]] for r in db.execute(query,types+emb_shapes)];starts=[r[1] for r in rows]
 def between(a,b):return rows[bisect.bisect_left(starts,a):bisect.bisect_left(starts,b)]
 result=[]
 for w in spec['windows']:
  out=dict(w)
  if not w['active']:out.update(qualified=True,route='idle_no_request_credit');result.append(out);continue
  try:
   body=between(w['target_start_ns'],w['target_end_ns']+1)
   pre=[r for r in body if r[5]=='HcPre'];ends=[r for r in body if r[5] in ('SparseFlashMla','HcPost')]
   assert pre and ends,('target_first_boundary',len(pre),len(ends))
   firstpre=pre[0];end=ends[0]
   assert firstpre[1]<end[1]
   external=[r for r in between(w['previous_body_end_ns'],w['target_start_ns']) if r[6].startswith('"16160,5120;')]
   assert external,('external_embedding_missing',w['kind'],w['wave'])
   emb=external[-1];assert emb[2]<=firstpre[1]
   draft_pre=db.execute('SELECT startNs,endNs,streamId,modelId FROM TASK WHERE rowid=?',(w['first_HcPre'],)).fetchone();assert draft_pre and draft_pre[0]==w['first_HcPre_ns']
   draft_rows=between(w['first_HcPre_ns'],w['draft_score_end_ns'])
   de=[r for r in draft_rows if r[5] in ('SparseFlashMla','HcPost')];assert de,('draft_first_boundary',len(draft_rows))
   draft_emb=[r for r in between(w['first_context_ns'],w['first_HcPre_ns']) if r[6].startswith('"16160,5120;')];assert draft_emb,('draft_embedding_missing',w['wave'])
   demb=draft_emb[-1]
   assert end[1]<=w['target_end_ns']<=w['target_head_end_ns']<demb[1]<de[0][1]
   out.update(qualified=True,route='target_and_draft_first_prefix',target_start_task=emb,target_first_HcPre=firstpre,target_prefix_end_task=end,
     draft_start_task=demb,draft_prefix_end_task=de[0],target_prefix_ns=end[1]-emb[1],draft_prefix_ns=de[0][1]-demb[1])
  except Exception as e:out.update(qualified=False,error=type(e).__name__+': '+str(e))
  result.append(out)
 db.close();after=p.stat();assert [before.st_size,before.st_mtime_ns]==[after.st_size,after.st_mtime_ns]
 return dict(rank=spec['rank'],database=spec['database'],identity=spec['identity'],selected_compute_rows=len(rows),embedding_shape_ids=len(emb_shapes),windows=result)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:out=list(p.map(read,json.load(sys.stdin)))
sys.stdout.buffer.write(zlib.compress(json.dumps(out,separators=(',',':')).encode()))
'''
def main():
    assert (ROOT/'embedding-entry-primary.json').exists()
    waves,_,_,_,_=records();ranks=cached(RECEIPT);bodies=cached(BODY)['profiles'];spans=json.loads(Path(SPANS).read_bytes());old=json.loads((ROOT/'context-parent-receipt.json').read_text());specs=[]
    for spec in old['source_specs']:
        rank=spec['rank'];windows=[];previous_end=0
        for w in spec['windows']:
            wave=w['wave'];r=waves[wave][rank]
            if r['kind']=='G':_,_,a,b=spans[rank][r['ordinal']]
            else:body=bodies[rank]['bodies'][r['ordinal']];a=body['start_ns'];b=body['end_ns']
            windows.append(dict(w,target_start_ns=a,target_end_ns=b,previous_body_end_ns=previous_end,draft_score_end_ns=r['calls']['draft']['end']))
            previous_end=b
        specs.append(dict(rank=rank,database=spec['database'],identity=spec['identity'],windows=windows))
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],input=json.dumps(specs).encode(),capture_output=True,timeout=240)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    d=json.loads(zlib.decompress(r.stdout));p=ROOT/'first-static-parent-raw.json.zlib';p.write_bytes(r.stdout)
    invalid=[dict(rank=x['rank'],**w) for x in d for w in x['windows'] if not w['qualified']]
    out=dict(raw_path=str(p),raw_sha256=sha256(r.stdout).hexdigest(),source_specs=specs,windows=sum(len(x['windows']) for x in d),active_windows=sum(w['active'] for x in d for w in x['windows']),qualified=sum(w['qualified'] for x in d for w in x['windows']),unqualified=invalid,
        identity_contract='Reuse existing full-hashed C8 immutable DB receipt; size/mtime before and after, no new profile. Explicit source-derived first SMLA or earlier HCPost boundary; all prefix work/free waits relaxed for an optimistic static-work parent, not all actually deletable.',
        reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],whole_chain_complete=False)
    (ROOT/'first-static-parent-receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(windows=out['windows'],qualified=out['qualified'],unqualified=len(invalid),examples=invalid[:3],new_NPU_work=0)))
if __name__=='__main__':main()
