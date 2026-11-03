#!/usr/bin/env python3
"""Identify whitelisted public interface names from stage192 fixed host API rows.
Read-only; unknown strings stay hashed. Never emit arguments, paths or raw strings.
"""
import argparse, base64, hashlib, json, re, sqlite3
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--trace-root',required=True); p.add_argument('--prior-b64',required=True); a=p.parse_args()
root=Path(a.trace_root)
db=root/'dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
raw=base64.b64decode(a.prior_b64,validate=True)
if hashlib.sha256(raw).hexdigest()!='fcac062cbd5c2c48e8ef178e186b59d35a854cb563d45297ea1052e413ea3a08' or sha(db)!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('input hash mismatch')
x=json.loads(raw)
if len(x['samples'])!=6 or sum(len(s['rows']) for s in x['samples'])!=54:raise RuntimeError('sample count mismatch')
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
# Full-string identifiers from known public runtime and operator namespaces only.
allowed=re.compile(r'(?:aten::|torch_npu::|npu::|vllm::|_C::|aclnn|aclrt|acl)[A-Za-z0-9_:.]{1,150}\Z')
# Python instrumentation may include arbitrary source paths; never expose them.
rows=[]
for sample in x['samples']:
 table=sample['table']
 if table not in ('CANN_API','PYTORCH_API'):raise RuntimeError('table mismatch')
 lo=sample['interval_start_ns'];tid=sample['globalTid']
 for item in sample['rows']:
  start=lo+item['start_relative_ns']; end=start+item['duration_ns']
  found=c.execute('SELECT s.value AS name FROM '+table+' a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND CAST(a.startNs AS INTEGER)=? AND CAST(a.endNs AS INTEGER)=? AND a.type=? AND a.connectionId IS ? LIMIT 21',(tid,start,end,item['type'],item['connectionId'])).fetchall()
  matched=[str(v['name']) for v in found if hashlib.sha256(str(v['name']).encode()).hexdigest()==item['name_sha256']]
  if len(matched)!=1 or len(found)>20:raise RuntimeError('row not uniquely reidentified')
  n=matched[0]
  rows.append({'opId_local':sample['opId_local'],'table':table,'start_relative_ns':item['start_relative_ns'],'duration_ns':item['duration_ns'],'name_sha256':item['name_sha256'],'name_class':item['name_class'],'public_interface_exact':n if allowed.fullmatch(n) else None,'name_withheld':not bool(allowed.fullmatch(n))})
c.close()
print(json.dumps({'scope':'Only 54 fixed stage192 rows, exact reidentification in SHA-matched DB; whitelist-only output','input_sha256':{'stage192':hashlib.sha256(raw).hexdigest(),'db':sha(db)},'rows':rows,'limitations':['Exact public interface name does not establish tensor/request dependencies.','CPU record coverage does not measure exclusive thread execution or device idle.','Unknown names withheld; None does not mean the record lacks a name.','No unbounded output, device work or target code changes.']},indent=2))
