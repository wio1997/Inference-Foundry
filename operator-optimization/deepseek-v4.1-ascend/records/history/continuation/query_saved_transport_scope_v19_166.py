from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
OUT=P/'native-communications-v15/transport-and-topology-evidence-scope-v19.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
src=P/'native-communications-v15/saved-communication-analysis-tables-v17.json'
d=pin(src,'511d1d6834ac9211b442cca38a754507323320358ce770a225a6219651da75d3')
ix=pin(P/'native-communications-v15/index.json','20ea4452458001c372c2d2c9b5cd9f4f13d951cfa7c77bb7e5e4bf3faa259016')
rows=[];totals=Counter();transports=Counter();matrix_transports=Counter();native_rank_sizes=Counter();group_ids=set()
for ref,db in zip(ix['databases'],d['databases']):
 native=pin(Path(ref['path']),ref['sha256']);assert native['database_index']==db['database_index']
 ops=native['COMM_OP_records'];names=Counter(r['opName_text']for r in ops)
 native_rank_sizes.update(str(r.get('rankSize'))for r in ops)
 tables=db['tables'];bw=tables['CommAnalyzerBandwidth'];mat=tables['CommAnalyzerMatrix']
 transports.update(str(r['transport_type'])for r in bw);matrix_transports.update(str(r['transport_type'])for r in mat)
 bwn={r['hccl_op_name']for r in bw};matn={r['op_name']for r in mat}
 groups=defaultdict(lambda:dict(matrix_src=set(),matrix_dst=set(),matrix_transports=set(),bandwidth_transports=set()))
 for r in bw:groups[r['group_name']]['bandwidth_transports'].add(r['transport_type'])
 for r in mat:
  g=groups[r['group_name']];g['matrix_src'].add(r['src_rank']);g['matrix_dst'].add(r['dst_rank']);g['matrix_transports'].add(r['transport_type'])
 group_ids.update(groups)
 row=dict(database_index=db['database_index'],native_operations=len(ops),bandwidth_rows=len(bw),matrix_rows=len(mat),bandwidth_distinct_native_names=len(bwn),matrix_distinct_native_names=len(matn),bandwidth_names_without_native_match=sorted(bwn-set(names)),matrix_names_without_native_match=sorted(matn-set(names)),matrix_aggregate_labels=dict(Counter(r['hccl_op_name']for r in mat)),groups={k:{kk:sorted(vv)for kk,vv in v.items()}for k,v in groups.items()})
 rows.append(row);totals.update({k:row[k]for k in ['native_operations','bandwidth_rows','matrix_rows','bandwidth_distinct_native_names','matrix_distinct_native_names']})
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(rows),totals=dict(totals),bandwidth_transport_row_counts=dict(transports),matrix_transport_row_counts=dict(matrix_transports),native_COMM_OP_rankSize_counts=dict(native_rank_sizes),observed_group_names=sorted(group_ids),unmatched_bandwidth_names=sum(len(r['bandwidth_names_without_native_match'])for r in rows),unmatched_matrix_names=sum(len(r['matrix_names_without_native_match'])for r in rows),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),no_new_source_DB_device_or_service_work=True,overall_complete=False,limits=['Transport labels/rank coordinates are saved HCCL analysis evidence,not a complete physical link graph or bandwidth ceiling.','Matrix aggregate labels/top selections must not be counted as all point-to-point operations.','The COMM_OP population is distinct from MC2 internal EP dispatch; rankSize8 HCCL groups do not describe all16 native MC2 EP participants.','No EP16 rank-to-device/physical-link map or same-call MC2 packet volume is invented from TP8 group labels.','No transit/bandwidth unit or additive physical traffic interpretation is introduced.'])
payload=(json.dumps(dict(summary=summary,databases=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,first_database=rows[0],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
