"""Bounded readonly re-extraction from all32 already captured and hash-pinned profiler DBs."""
from pathlib import Path
import json
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE,ORIGINAL
from query_saved_raw_profile_v98 import remote

def main():
    assert not(O/'REVIEW-CHECKPOINT-v98.json').exists()
    refs=read(CACHE/'analysis/profile-capabilities-index-v12-v2.json')['databases']
    native=read(CACHE/'analysis/native-execution-identity-v14/index.json')['databases']
    planned=[]
    for i,r in enumerate(refs):
        assert str(r['path']).startswith(str(ORIGINAL)+'/')and r['bytes']<256*1024*1024
        planned.append(dict(database_index=i,path=r['path'],sha256=r['sha256'],bytes=r['bytes'],expected_compute_executions=native[i]['summary']['compute_executions']))
    assert sum(r['bytes']for r in planned)<4*1024**3
    code='''import os,socket,sqlite3,hashlib,json,re
from pathlib import Path
from collections import Counter
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
refs=REFS
rows=[]
def q(x):return '"'+x.replace('"','""')+'"'
for r in refs:
 p=Path(r['path']);before=p.stat();assert before.st_size==r['bytes']
 h=hashlib.sha256()
 with p.open('rb')as f:
  while True:
   block=f.read(1024*1024)
   if not block:break
   h.update(block)
 sha=h.hexdigest();assert sha==r['sha256']
 con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
 tables=[x[0]for x in con.execute("select name from sqlite_master where type='table' order by name")]
 schemas={t:[dict(x)for x in con.execute('pragma table_info('+q(t)+')')]for t in tables}
 columns={t:[x['name']for x in v]for t,v in schemas.items()}
 possible={t:[c for c in cs if re.search('addr|pointer|length|memory|workspace|params|argument|tiling',c,re.I)]for t,cs in columns.items()}
 possible={t:cs for t,cs in possible.items()if cs}
 ops=[dict(x)for x in con.execute("select n.value as operator,count(*)as executions,sum(case when coalesce(a.value,'')!='' then 1 else 0 end)as with_nonempty_attr from TASK t join COMPUTE_TASK_INFO c on t.globalTaskId=c.globalTaskId join STRING_IDS n on c.opType=n.id left join STRING_IDS a on c.attrInfo=a.id group by n.value order by n.value")]
 assert sum(x['executions']for x in ops)==r['expected_compute_executions']
 details={}
 for op in ['DispatchFFNCombine','MemSet','ZerosLike']:
  vals=con.execute("select c.attrInfo,coalesce(a.value,'')as attrs,i.value as input_shapes,o.value as output_shapes,dt.value as input_dtypes,odt.value as output_dtypes,count(*)as executions from TASK t join COMPUTE_TASK_INFO c on t.globalTaskId=c.globalTaskId join STRING_IDS n on c.opType=n.id left join STRING_IDS a on c.attrInfo=a.id left join STRING_IDS i on c.inputShapes=i.id left join STRING_IDS o on c.outputShapes=o.id left join STRING_IDS dt on c.inputDataTypes=dt.id left join STRING_IDS odt on c.outputDataTypes=odt.id where n.value=? group by c.attrInfo,i.value,o.value,dt.value,odt.value order by executions desc",(op,)).fetchall()
  details[op]=[dict(x)for x in vals]
 con.close();after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 rows.append(dict(database_index=r['database_index'],path=str(p),sha256=sha,bytes=before.st_size,stat_unchanged=True,table_names=tables,all_table_columns=columns,
   candidate_address_or_argument_columns=possible,operators=ops,target_operator_regimes=details))
print(json.dumps(dict(hostname=socket.gethostname(),databases=rows,bytes_read_for_identity=sum(x['bytes']for x in rows),SQLite_mode='ro&immutable=1',new_capture=False,no_model_Torch_NPU_service_or_sourcechange=True)))
'''.replace('REFS',repr(planned))
    packet=remote(code,timeout=50);write(O/'raw-operator-evidence-extraction-v98.json',packet)
    assert packet['returncode']==0 and packet['result'],packet
    rows=packet['result']['databases'];tot={}
    for op in ['DispatchFFNCombine','MemSet','ZerosLike']:
        ds=[d for r in rows for d in r['target_operator_regimes'][op]]
        tot[op]=dict(executions=sum(d['executions']for d in ds),with_nonempty_attr=sum(d['executions']for d in ds if d['attrs']))
    print(json.dumps(dict(databases=len(rows),original_hashes_all_match=True,target_operators=tot,bytes_read_for_identity=packet['result']['bytes_read_for_identity'],new_capture=False)))

if __name__=='__main__':main()
