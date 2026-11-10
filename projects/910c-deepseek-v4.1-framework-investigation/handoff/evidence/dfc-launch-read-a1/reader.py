#!/usr/bin/env python3
"""Read archived DFC launch fields; hash/bind existing16rank DBs, no NPU work."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zlib

BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,sqlite3,sys
spec=json.loads(sys.argv[1]);out=[]
for r in spec:
 p=Path(r['database']);assert p.is_file() and Path('/data/tiankuan/wio') in p.parents
 rawhash=sha256(p.read_bytes()).hexdigest();assert rawhash==r['sha256']
 con=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True)
 sql='''+"'''"+'''
 SELECT t.rowid,t.startNs,t.endNs,t.globalTaskId,c.blockNum,c.mixBlockNum,ty.value,gr.value,bd.value,ish.value
 FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId
 JOIN STRING_IDS op ON op.id=c.opType
 LEFT JOIN STRING_IDS ty ON ty.id=c.taskType
 LEFT JOIN STRING_IDS gr ON gr.id=c.gridDim
 LEFT JOIN STRING_IDS bd ON bd.id=c.blockDim
 LEFT JOIN STRING_IDS ish ON ish.id=c.inputShapes
 WHERE op.value='DispatchFFNCombine' AND t.rowid IN (
 '''+"'''"+'''+','.join('?' for x in r['selected_task_rows'])+') ORDER BY t.rowid'
 rows=[list(x) for x in con.execute(sql,r['selected_task_rows'])]
 assert len(rows)==len(r['selected_task_rows'])==160
 con.close();out.append(dict(rank=r['rank'],database=str(p),database_sha256=rawhash,rows=rows))
print(json.dumps(out))
'''

def main():
 p=BASE/'.cache/first-token-v168-protocol-bodies.json.zlib';native=json.loads(zlib.decompress(p.read_bytes()));spec=[];expected={}
 for prof in native['profiles']:
  rank=prof['DP']*8+prof['TP'];tasks=[t for b in prof['bodies'] for t in b['F21_TASK_ROWS']];assert len(tasks)==160
  by={r[0]:r for b in prof['bodies'] for r in b['rows']}
  for t in tasks:expected[rank,t]=by[t]
  spec.append(dict(rank=rank,database=prof['database'],sha256=prof['sha256'],selected_task_rows=tasks))
 arg=json.dumps(spec,separators=(',',':'));assert len(arg)<90000
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,arg],capture_output=True,text=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr)
 data=json.loads(r.stdout)
 for prof in data:
  for v in prof['rows']:
   orig=expected[prof['rank'],v[0]];assert v[1:3]==orig[1:3] and v[3]==orig[6]
 from collections import Counter
 groups=Counter((r[4],r[5],r[6],r[7],r[8]) for prof in data for r in prof['rows'])
 result=dict(status='ARCHIVED_MATCHED2560_EAGER_DFC_LAUNCH_FIELDS',source_cache=dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()),profiles=data,
  columns=['task_row','start_ns','end_ns','globalTaskId','blockNum','mixBlockNum','taskType','gridDim','blockDim','inputShapes'],
  groups=[dict(blockNum=k[0],mixBlockNum=k[1],taskType=k[2],gridDim=k[3],blockDim=k[4],calls=n) for k,n in groups.items()],
  reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[],
  scope='Exact original archived firsttoken eager tasks only, everyDB SHA and taskstart/end/globalid matched. Profile block/mix fields are not device cycle occupancy or a current per-call tiling buffer attestation.')
 Path('audit/execution-search/dfc-archived-launch-fields.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(profiles=len(data),calls=sum(len(x['rows']) for x in data),groups=result['groups'])))

if __name__=='__main__':main()
