from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
index=pin(P/'analysis/native-execution-identity-v14/index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
ROOT=P/'analysis/native-communications-v15';ROOT.mkdir(exist_ok=True);OUT=ROOT/'index.json';assert not OUT.exists();refs=[];t0=time.monotonic()
for ref in index['databases']:
 di=ref['database_index'];dest=ROOT/('db-%02d.json'%di)
 if dest.exists():
  b=dest.read_bytes();s=json.loads(b);assert s['database_sha256']==ref['summary'].get('database_sha256',s['database_sha256']);refs.append(dict(database_index=di,summary=s['summary'],path=str(dest),sha256=hashlib.sha256(b).hexdigest()));continue
 d=pin(Path(ref['path']),ref['sha256'])
 db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:int(time.monotonic()-t0>145),100000)
 strings=dict(db.execute('SELECT id,value FROM STRING_IDS'));tables={r[0]for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 enums={t:[dict(r)for r in db.execute('SELECT * FROM '+t)]for t in tables if t.startswith('ENUM_HCCL_')}
 dtype={r['id']:r['name']for r in enums.get('ENUM_HCCL_DATA_TYPE',[])}
 ops=[dict(r)for r in db.execute('SELECT rowid AS COMM_OP_rowid,* FROM COMMUNICATION_OP')]
 schedules=[dict(r)for r in db.execute('SELECT rowid AS COMM_SCHEDULE_rowid,* FROM COMMUNICATION_SCHEDULE_TASK_INFO')]
 for s in schedules:
  for k in ['name','taskType','opType']:s[k+'_text']=strings.get(s.get(k))
 scheduleids={s['globalTaskId']for s in schedules};bypost=defaultdict(list);byop=defaultdict(list)
 for r in db.execute('SELECT rowid AS COMM_TASK_rowid,* FROM COMMUNICATION_TASK_INFO'):byop[r['opId']].append(dict(r))
 for t in d['TASK_records']:bypost[t['connectionId']].append(t)
 host_by_dev=defaultdict(set)
 for c in d['CONNECTION_IDS']:host_by_dev[c['connectionId']].add(c['id'])
 api_by_host=defaultdict(list)
 for a in d['linked_PYTORCH_APIs']:api_by_host[a['connectionId']].append(a)
 for op in ops:
  op['opName_text']=strings.get(op['opName']);op['groupName_text']=strings.get(op['groupName']);op['dataType_text']=dtype.get(op['dataType'])
  ts=byop.get(op['opId'],[]);op['communication_task_rowids']=[r['COMM_TASK_rowid']for r in ts]
  op['internal_record_type_counts']=dict(Counter(strings.get(r['taskType'],str(r['taskType']))for r in ts))
  for col,enum in [('linkType','ENUM_HCCL_LINK_TYPE'),('transportType','ENUM_HCCL_TRANSPORT_TYPE')]:
   emap={r['id']:r['name']for r in enums.get(enum,[])};op[col+'_counts']=dict(Counter(emap.get(r[col],str(r[col]))for r in ts))
  op['src_dst_record_counts']=[dict(src=key[0],dst=key[1],records=n)for key,n in Counter((r['srcRank'],r['dstRank'])for r in ts).items()]
  op['same_native_connection_TASK_rows']=[t['task_record_rowid']for t in bypost.get(op['connectionId'],[])]
  op['same_native_connection_schedule_TASK_rows']=[t['task_record_rowid']for t in bypost.get(op['connectionId'],[])if t['globalTaskId']in scheduleids]
  op['native_PYTORCH_API_candidates']=[a for h in sorted(host_by_dev.get(op['connectionId'],[]))for a in api_by_host.get(h,[])]
  op['graph_control_container_candidates']=[dict(graph_ordinal=gi,model_id=g['model_id'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'])for gi,g in enumerate(d['graph_API_control_bindings'])if g['control_pair_valid']and g['control_pair_start_ns']<=op['startNs']and op['endNs']<=g['control_pair_end_ns']]
 summary=dict(database_index=di,COMM_OP_records=len(ops),COMM_TASK_records=sum(len(v)for v in byop.values()),COMM_SCHEDULE_definitions=len(schedules),op_data_type_rank_hist=[dict(op_name=k[0],dtype=k[1],rankSize=k[2],records=v)for k,v in Counter((o['opName_text'],o['dataType_text'],o['rankSize'])for o in ops).items()],rank_size_counts=dict(Counter(o['rankSize']for o in ops)),groups=sorted({o['groupName_text']for o in ops}),native_API_candidate_count_hist=dict(Counter(len(o['native_PYTORCH_API_candidates'])for o in ops)),graph_container_candidate_count_hist=dict(Counter(len(o['graph_control_container_candidates'])for o in ops)),COMM_TASK_orphan_opIds=sorted(set(byop)-{o['opId']for o in ops}))
 db.close();obj=dict(database_index=di,database_path=d['database_path'],database_sha256=d['database_sha256'],summary=summary,enums=enums,COMM_OP_records=ops,COMM_SCHEDULE_definitions=schedules,COMM_TASK_records_by_opId={str(k):v for k,v in byop.items()})
 b=(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n').encode();dest.write_bytes(b);refs.append(dict(database_index=di,summary=summary,path=str(dest),sha256=hashlib.sha256(b).hexdigest()))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(refs),COMM_OP_records=sum(r['summary']['COMM_OP_records']for r in refs),COMM_TASK_records=sum(r['summary']['COMM_TASK_records']for r in refs),COMM_SCHEDULE_definitions=sum(r['summary']['COMM_SCHEDULE_definitions']for r in refs),rank_size_counts=dict(sum((Counter(r['summary']['rank_size_counts'])for r in refs),Counter())),native_API_candidate_count_hist=dict(sum((Counter(r['summary']['native_API_candidate_count_hist'])for r in refs),Counter())),graph_container_candidate_count_hist=dict(sum((Counter(r['summary']['graph_container_candidate_count_hist'])for r in refs),Counter())),no_target_source_or_device_work=True,whole_modules_closed=0,overall_complete=False,limits=['COMM_OP intervals,communication physical tasks,andTASK compute/schedule records are different layers; no duration sum across them.','Source API candidates use nativeconnection relation; graph associations use explicitcontrolcontainers andretainambiguity. No nearest-time orcross-rankclockjoin.','Raw bandwidth/size fields retained withoutassumingunitsorlinkpeak. COMM_TASK globalTaskId isnotjoinedblindlytoTASK; opId isusedonlyinsideCOMMnamespace.'])
b=(json.dumps(dict(summary=summary,databases=refs),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,first_database=refs[0]['summary'],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
