#!/usr/bin/env python3
"""Read all exported CSV/PMU columns for six fixed EP8 BF16 MatMul tasks."""
import argparse,csv,hashlib,json,sqlite3
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);ap.add_argument('--csv',required=True);a=ap.parse_args()
T=Path(a.task_root);db=Path(a.db).resolve(strict=True);csvpath=Path(a.csv).resolve(strict=True)
src=T/'agents/diagnosis/engram-matmul-task-links.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
if sha(db)!='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58':raise RuntimeError('unexpected DB hash')
if sha(csvpath)!='488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7':raise RuntimeError('unexpected CSV hash')
if sha(src)!='3e6196f3005366117421c78a359dc036e9d79f7299f2b72d239c2401202e76d1':raise RuntimeError('unexpected targets hash')
samples=json.loads(src.read_text())['samples']
if len(samples)!=6:raise RuntimeError('unexpected target count')
targetmap={(str(x['target']['model']),str(x['target']['stream']),x['target']['start_ns']):x for x in samples}
found={}
with csvpath.open(newline='') as f:
 reader=csv.DictReader(f);columns=reader.fieldnames
 for r in reader:
  if r.get('Type')!='MatMulV2' or r.get('Input Shapes')!='"3,6144;25600,6144"':continue
  try:start=int(Decimal(r['Start Time(us)'].strip())*1000)
  except Exception:continue
  key=(r['Model ID'],r['Stream ID'],start)
  if key not in targetmap:continue
  duration=int(Decimal(r['Duration(us)'].strip())*1000)
  if start+duration!=targetmap[key]['target']['end_ns']:raise RuntimeError('duration mismatch')
  if key in found:raise RuntimeError('duplicate CSV task')
  found[key]=r
if len(found)!=6:raise RuntimeError('not all six CSV tasks matched')
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
tables=[dict(x) for x in c.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]
pmu={}
for gid in sorted({x['exact_task']['globalTaskId'] for x in samples}):
 rows=[dict(r) for r in c.execute('''SELECT p.timestampNs,p.name,s.value AS counter_name,p.value FROM TASK_PMU_INFO p
 LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTaskId=? ORDER BY p.timestampNs,p.name LIMIT 5001''',(gid,)).fetchall()]
 if len(rows)>5000:raise RuntimeError('PMU scope exceeds cap')
 info=c.execute('SELECT * FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(gid,)).fetchone()
 pmu[str(gid)]={'rows':rows,'counter_names':sorted({str(x['counter_name']) for x in rows}),
  'compute_task_info':dict(info) if info else None}
out=[]
for key,x in targetmap.items():
 r=found[key];t=x['target'];gid=x['exact_task']['globalTaskId'];nearest=sorted({v['timestampNs'] for v in pmu[str(gid)]['rows']},key=lambda z:abs(z-t['end_ns']))
 out.append({'target':t,'globalTaskId':gid,'csv_all_fields':r,'nearest_pmu_timestamp_ns':nearest[0] if nearest else None,
 'nearest_pmu_delta_ns':nearest[0]-t['end_ns'] if nearest else None})
print(json.dumps({'scope':'EP8 six exact matched CSV rows, all PMU records for their two globalTaskIds; all SQLite table schemas',
 'db_sha256':sha(db),'csv_sha256':sha(csvpath),'target_sha256':sha(src),'csv_columns':columns,'db_table_schemas':tables,
 'samples':out,'all_pmu_by_reused_globalTaskId':pmu,
 'limitations':['Nearest PMU time is a candidate, not a proven replay mapping; globalTaskId is reused.',
 'CSV icache is instruction cache, not L2 or HBM; field semantics require version-matched documentation.',
 'No absence in these exports proves that no raw or binary profiler counters exist.',
 'Cannot infer cache misses, HBM saturation, internal wait or speedup from pipeline ratios alone.']},ensure_ascii=False,indent=2))
c.close()
