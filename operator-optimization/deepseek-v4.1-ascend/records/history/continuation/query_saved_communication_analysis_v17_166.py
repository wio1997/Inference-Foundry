from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
OUT=P/'native-communications-v15/saved-communication-analysis-tables-v17.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
surface=pin(P/'native-execution-identity-v14/saved-CPU-process-surface-coverage-v17.json','247d991e7169d1b12178997be576db398f66b76ca32b32915f8a8a58f448ed92')
idx=pin(P/'native-communications-v15/index.json','20ea4452458001c372c2d2c9b5cd9f4f13d951cfa7c77bb7e5e4bf3faa259016')
comm_by_dir={}
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);comm_by_dir[Path(d['database_path']).parent.parent.name]=d
records=[];ratios=[]
for ref in surface['helper_databases']:
 path=Path(ref['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256'];db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
 tables={name:[dict(r)for r in db.execute('SELECT rowid AS analysis_rowid,* FROM "'+name+'"')]for name in ['StepTraceTime','CommAnalyzerTime','CommAnalyzerBandwidth','CommAnalyzerMatrix']};db.close()
 native=comm_by_dir[path.parent.parent.name];ops=defaultdict(list)
 for op in native['COMM_OP_records']:ops[op['opName_text']].append(op)
 matches=[]
 for row in tables['CommAnalyzerTime']:
  candidates=ops.get(row['hccl_op_name'],[]);v=dict(analysis_rowid=row['analysis_rowid'],exact_native_op_candidates=[o['COMM_OP_rowid']for o in candidates])
  if len(candidates)==1 and isinstance(row['elapse_time'],(float,int))and row['elapse_time']>0:
   op=candidates[0];ratio=(op['endNs']-op['startNs'])/row['elapse_time'];v['native_ns_over_raw_analysis_elapse']=ratio;ratios.append(ratio)
  matches.append(v)
 records.append(dict(database_index=native['database_index'],path=str(path),sha256=ref['sha256'],tables=tables,exact_name_comparison=matches))
def stats(xs):return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None
names=Counter(r['hccl_op_name']for d in records for r in d['tables']['CommAnalyzerTime'])
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(records),table_record_counts={name:sum(len(d['tables'][name])for d in records)for name in ['StepTraceTime','CommAnalyzerTime','CommAnalyzerBandwidth','CommAnalyzerMatrix']},step_rows=sum(len(d['tables']['StepTraceTime'])for d in records),exact_native_name_candidate_hist=dict(Counter(len(m['exact_native_op_candidates'])for d in records for m in d['exact_name_comparison'])),native_ns_over_raw_analysis_elapse=stats(ratios),no_new_source_model_or_device_work=True,overall_complete=False,limits=['Raw analysis values preserved; table headings do not prove units,interval semantics or actual exposed request cost.','Exact name matching retains cardinality; no normalized-name or nearest-time association invented.','StepTraceTime communication/not-overlapped/free categories and HCCL wait/sync/transit fields require their installed exporter definitions before physical interpretation. NativeEPinsideMC2isnotmadevisiblebythisHCCLtable.'])
payload=(json.dumps(dict(summary=summary,databases=records),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,first_database=dict(database_index=records[0]['database_index'],table_samples={k:v[:2]for k,v in records[0]['tables'].items()}),most_common_analysis_names=names.most_common(6),full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
