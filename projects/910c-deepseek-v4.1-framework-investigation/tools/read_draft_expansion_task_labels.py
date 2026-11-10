#!/usr/bin/env python3
"""Probe only saved native label availability, never start another profile."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,zlib
ROOT=Path('audit/execution-search')
REMOTE=r'''
import json,pathlib,socket,sqlite3,sys
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
out=[]
for spec in json.loads(sys.argv[1]):
 p=pathlib.Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio');s=p.stat();assert [s.st_size,s.st_mtime_ns]==spec['identity']
 db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON');db.execute('PRAGMA temp_store=MEMORY')
 schema={t:list(db.execute('PRAGMA table_info('+t+')')) for t in ['TASK','COMPUTE_TASK_INFO']}
 hits=list(db.execute("SELECT id,value FROM STRING_IDS WHERE value LIKE '%copy_and_expand%' OR value LIKE '%engram_req_index%' OR value LIKE '%hash_ids_kernel%' OR value LIKE '%write_hash_cache%'"));db.close();a=p.stat();assert [s.st_size,s.st_mtime_ns]==[a.st_size,a.st_mtime_ns]
 out.append(dict(rank=spec['rank'],database=str(p),identity=spec['identity'],schema=schema,matching_kernel_labels=hits))
print(json.dumps(out))
'''
def main():
    plan=json.loads((ROOT/'input-resource-source-plan.json').read_text());assert plan['draft_attention_groups']==1
    raw=json.loads(zlib.decompress((ROOT/'first-static-math-raw.json.zlib').read_bytes()));spec=[{k:r[k] for k in ['rank','database','identity']} for r in raw if r['rank'] in [0,8]];assert len(spec)==2
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr)
    rows=json.loads(r.stdout);out=dict(status='EXISTING_NATIVE_LABEL_PROBE_ONLY',profiles=rows,source_plan_sha256=sha256((ROOT/'input-resource-source-plan.json').read_bytes()).hexdigest(),qualification='Two original immutable C8 native DBs only. Missing function labels do not mean missing kernels or zero time; do not resample just to assign a stopped repeat-only recipe.',new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'draft-expansion-native-labels.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(profiles=len(rows),labels=[dict(rank=x['rank'],matches=x['matching_kernel_labels'],compute_columns=[v[1] for v in x['schema']['COMPUTE_TASK_INFO']]) for x in rows],new_profile=0)))
if __name__=='__main__':main()
