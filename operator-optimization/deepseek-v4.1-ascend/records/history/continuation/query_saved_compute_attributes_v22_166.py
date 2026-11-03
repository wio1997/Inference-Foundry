from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket,sqlite3
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';OUT=P/'native-PMU-complete-v18/saved-COMPUTE-attribute-field-census-v22.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
records=[];groups=defaultdict(list);status=Counter();counts=Counter()
for ref in ix['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];p=Path(d['database_path']);st=p.stat();assert hashlib.sha256(p.read_bytes()).hexdigest()==d['database_sha256']
 db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');strings=dict(db.execute('SELECT id,value FROM STRING_IDS'));schema=[dict(r)for r in db.execute('PRAGMA table_info(COMPUTE_TASK_INFO)')];db.close()
 executions=Counter(t['globalTaskId']for t in d['TASK_records']);rr=[]
 for c in d['COMPUTE_definitions']:
  raw=c.get('attrInfo');text=strings.get(raw);state='NULL_OR_UNRESOLVED_ID'if text is None else'EMPTY_OR_PLACEHOLDER'if str(text).strip()in['','N/A','[]','{}','None']else'NONEMPTY_STRING_CANDIDATE';n=executions[c['globalTaskId']];status[state]+=n;counts[c['opType_text']]+=n
  row=dict(globalTaskId=c['globalTaskId'],COMPUTE_rowid=c['compute_record_rowid'],op_type=c['opType_text'],executions=n,raw_attrInfo=raw,string_table_value_at_exact_id=text,lookup_state=state);rr.append(row);groups[(c['opType_text'],text,state)].append(dict(database_index=di,globalTaskId=c['globalTaskId'],executions=n))
 assert (p.stat().st_size,p.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns)
 records.append(dict(database_index=di,path=str(p),sha256=d['database_sha256'],source_stat_unchanged=True,COMPUTE_schema=schema,attribute_records=rr))
gg=[dict(op_type=k[0],raw_attribute_string=k[1],lookup_state=k[2],definitions=len(v),executions=sum(r['executions']for r in v),execution_definition_refs=v)for k,v in groups.items()]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(records),compute_executions=sum(counts.values()),compute_types=len(counts),attribute_lookup_execution_states=dict(status),distinct_op_attribute_strings=len(gg),no_new_source_model_device_or_service_work=True,overall_complete=False,limits=['This retrieves the exact STRING_IDS entry addressed by raw COMPUTE.attrInfo;raw IDs/strings retained. Semantic interpretation needs exporter/schema binding,not arbitrary integer-as-FK assumptions.','Attribute text is not tensor/scalar input payload or tiling/binary identity unless the stored value actually contains that information.','Aplaceholder/emptyfield doesnot prove a default scalarvalue orabsence of internal stages.','All original DBsha/stat checks pass;readonlyimmutableSQL,no raw DB mutations.'])
assert summary['compute_executions']==286950
b=(json.dumps(dict(summary=summary,attribute_string_groups=gg,databases=records),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
priority={'ForeachAddList','DispatchFFNCombine','HcPost','HcPre','MoeGatingTopKHash','QuantBatchMatmulV3','DequantSwigluQuant'};selected=[{k:v for k,v in r.items()if k!='execution_definition_refs'}for r in gg if r['op_type']in priority]
print(json.dumps(dict(summary=summary,selected_attribute_groups=selected,other_attribute_group_headers=[dict(op_type=r['op_type'],lookup_state=r['lookup_state'],executions=r['executions'],raw_attribute_length=len(str(r['raw_attribute_string'])))for r in gg if r['op_type']not in priority],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
