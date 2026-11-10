#!/usr/bin/env python3
"""Match exact old kernel names to source-bounded C8 windows, including idle ranks."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,zlib
ROOT=Path('audit/execution-search')
REMOTE=r'''
import concurrent.futures,json,pathlib,socket,sqlite3,sys,zlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
names=['copy_and_expand_dflash_and_dspark_inputs_kernel','_engram_req_index_kernel','_write_hash_cache_kernel','_hash_ids_kernel']
def read(spec):
 p=pathlib.Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio');s=p.stat();assert [s.st_size,s.st_mtime_ns]==spec['identity']
 db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON');db.execute('PRAGMA temp_store=MEMORY')
 ids=dict(db.execute('SELECT id,value FROM STRING_IDS WHERE value IN (?,?,?,?)',names));assert len(ids)==4,('missing_exact_names',spec['rank'],ids)
 rows=[[*r[:5],ids[r[5]]] for r in db.execute('SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.modelId,k.name FROM TASK t JOIN COMPUTE_TASK_INFO k USING(globalTaskId) WHERE k.name IN (?,?,?,?) ORDER BY t.startNs,t.rowid',list(ids))]
 out=[]
 for w in spec['windows']:
  expansion=[r for r in rows if r[5]==names[0] and w['target_head_end_ns']<=r[1] and r[2]<=w['first_context_ns']]
  lo=w['target_start_task'][1] if w['active'] else w['target_start_ns'];hi=w['target_end_ns']
  engram=[r for r in rows if r[5]!=names[0] and lo<=r[1] and r[2]<=hi]
  counts={n:sum(r[5]==n for r in engram) for n in names[1:]}
  out.append(dict(wave=w['wave'],active=w['active'],kind=w['kind'],expansion=expansion,engram=engram,engram_counts=counts,qualified_expansion_count=(len(expansion)==1 if w['active'] else len(expansion)<=1),qualification='Expansion lies after current target head and before first draft context. Engram kernels lie after current outer lookup/before target body end; no per-stream cause or device bit claim.'))
 db.close();a=p.stat();assert [s.st_size,s.st_mtime_ns]==[a.st_size,a.st_mtime_ns]
 return dict(rank=spec['rank'],database=str(p),identity=spec['identity'],all_matching_native_tasks=rows,windows=out)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(read,json.load(sys.stdin)))
sys.stdout.buffer.write(zlib.compress(json.dumps(out,separators=(',',':')).encode()))
'''
def main():
    raw=ROOT/'first-static-math-raw.json.zlib';source=json.loads(zlib.decompress(raw.read_bytes()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],input=json.dumps(source).encode(),capture_output=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    out=json.loads(zlib.decompress(r.stdout));dest=ROOT/'input-kernel-windows-raw.json.zlib';dest.write_bytes(r.stdout)
    active=[(x['rank'],w) for x in out for w in x['windows'] if w['active']];invalid=[dict(rank=rank,**w) for rank,w in active if not w['qualified_expansion_count']]
    receipt=dict(status='ORIGINAL_C8_EXACT_NAMED_KERNEL_WINDOWS',raw_path=str(dest),raw_sha256=sha256(r.stdout).hexdigest(),source_native_window_path=str(raw),source_native_window_sha256=sha256(raw.read_bytes()).hexdigest(),ranks=len(out),all_rank_windows=sum(len(x['windows']) for x in out),active_rank_windows=len(active),active_expansion_count_distribution={str(k):sum(len(w['expansion'])==k for _,w in active) for k in sorted({len(w['expansion']) for _,w in active})},invalid=invalid,source_plan_sha256=sha256((ROOT/'input-resource-source-plan.json').read_bytes()).hexdigest(),qualification='Original immutable16rank C8 only; exact compute-name IDs, task bounds, size/mtime. Existing full-hash profile receipts reused; no claim of new full-DB hash or all consumer causality. Source plan was derived before this population/price.',new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],whole_chain_complete=False,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'input-kernel-windows-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['invalid','reader_sha256','qualification']}|dict(invalid_count=len(invalid),invalid_examples=invalid[:1])))
if __name__=='__main__':main()
