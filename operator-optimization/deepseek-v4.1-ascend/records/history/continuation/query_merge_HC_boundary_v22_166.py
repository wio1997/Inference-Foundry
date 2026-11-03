from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18';OUT=M/'shared-routed-merge-to-HC-boundary-v22.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');ci=pin(C/'index.json','20ea4452458001c372c2d2c9b5cd9f4f13d951cfa7c77bb7e5e4bf3faa259016');gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991');rs=pin(M/'resource-functional-attribution-and-uncertainty-v20-v3.json','c8e2c128d208597aeeb07a2b434506c930397d773b0ab5767a0fadba7e4840c4')
shapes=[r for r in rs['shape_role_context_regimes']if r['signature']['opType_text']=='ForeachAddList'];assert sum(r['summary']['executions']for r in shapes)==4472
bydb=defaultdict(list)
for g in gr['graph_executions']:bydb[g['database_index']].append(g)
rows=[];between_ops=Counter();between_tasks=Counter();comm_gaps=Counter();same_stream=Counter();sigcounts=Counter();gaps=[];overlaps=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc=pin(Path(ci['databases'][di]['path']),ci['databases'][di]['sha256']);tt={t['task_record_rowid']:t for t in d['TASK_records']};defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for g in bydb[di]:
  phase=g['phase_records'][0]['phase']if len(g['phase_records'])==1 else'PHASE_UNBOUND'
  for b in g['blocks']:
   ts=[tt[i]for i in b['task_record_rowids']];merges=[t for t in ts if defs[t['globalTaskId']]['opType_text']=='ForeachAddList'];posts=sorted([t for t in ts if defs[t['globalTaskId']]['opType_text']=='HcPost'],key=lambda t:t['startNs']);assert len(merges)==1 and len(posts)==2
   merge=merges[0];hc=posts[-1];a,z=merge['endNs'],hc['startNs'];gaps.append(z-a);same_stream[str(merge['streamId']==hc['streamId'])]+=1
   compute=[t for t in ts if t['task_record_rowid']not in[merge['task_record_rowid'],hc['task_record_rowid']]and a<=t['startNs']and t['endNs']<=z];between_ops.update(defs[t['globalTaskId']]['opType_text']for t in compute)
   other=[t for t in tt.values()if t['task_record_rowid']not in[merge['task_record_rowid'],hc['task_record_rowid']]and min(z,t['endNs'])>max(a,t['startNs'])];between_tasks.update(t['task_type_name']for t in other)
   comm=[r for r in cc['COMM_OP_records']if min(z,r['endNs'])>max(a,r['startNs'])];comm_gaps.update(str(r['opType_text'])for r in comm)
   rec=dict(database_index=di,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],phase=phase,merge=dict(TASK=merge,definition=defs[merge['globalTaskId']]),FFN_HcPost=dict(TASK=hc,definition=defs[hc['globalTaskId']]),gap_ns=z-a,intervening_compute_keys=[t['task_record_rowid']for t in compute],overlapping_other_TASK_keys=[t['task_record_rowid']for t in other],overlapping_COMM_OP_keys=[r['COMM_OP_rowid']for r in comm]);rows.append(rec)
   sigcounts[json.dumps(dict(merge=defs[merge['globalTaskId']],HC=defs[hc['globalTaskId']]),sort_keys=True)]+=1
def stats(xs):return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),all_merge_executions=4472,graph_merge_HC_pairs=len(rows),graph_same_TASK_stream_counts=dict(same_stream),merge_to_HC_gap_ns=stats(gaps),intervening_compute_op_counts=dict(between_ops),overlapping_other_TASK_type_counts=dict(between_tasks),overlapping_COMM_OP_type_counts=dict(comm_gaps),native_merge_origins=dict(Counter({k:sum(r['summary']['executions']for r in shapes if r['origin']==k)for k in sorted({r['origin']for r in shapes})})),merge_shape_context_regimes=len(shapes),pair_signature_count=len(sigcounts),no_new_source_DB_model_device_or_service_work=True,overall_complete=False,limits=['Graph pairs use exact saved structural40-layer marker blocks,not nearest-time inference or native invocation ordinal pairing.','Profile shapes/dtypes/order do not prove scalaralpha value,tensorpointer identity,allconsumers orloadednativeHCrounding/tiling.','Gap andvisibleinterveningtaskinventory are not causal latency savings.','Native/direct merge shape/context groups remain separate fromthegraph3200pairing;16unboundgraph phases stayunbound.'])
payload=(json.dumps(dict(summary=summary,all_merge_shape_context_regimes=shapes,graph_pairs=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,merge_shape_context_regimes=[dict(origin=r['origin'],context=r['context'],signature=r['signature'],executions=r['summary']['executions'])for r in shapes],first_graph_pairs=rows[:2],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
print(json.dumps(public,ensure_ascii=False))
