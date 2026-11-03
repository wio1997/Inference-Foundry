from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket,re,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
cidx=pin(P/'native-communications-v15/index.json','20ea4452458001c372c2d2c9b5cd9f4f13d951cfa7c77bb7e5e4bf3faa259016')
g=pin(P/'native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
chains=pin(P/'native-execution-identity-v14/native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
contexts=pin(P/'native-execution-identity-v14/fullchain-source-contexts-v15.json','1c003dbca3d72a216c646d209dc54bba38e936db32b0622b2efdf866b2b53ec9')
queue=pin(P/'native-execution-identity-v14/queue-bridge-decoded-connections-v15.json','56dcb3950d574d21663db30712559ccaf63800c74bbfbd8e21d3603d4ccec54c')
OUT=P/'native-communications-v15/communication-and-queue-review-v15.json';assert not OUT.exists()
qreview=[]
for r in queue['records']:
 deq=r['backend_Dequeue_scopes'];groups=r['enclosing_frontend_scopes'];assert len(deq)==len(groups)==1
 enq=groups[0]['Enqueue'];same_process=enq['globalTid']>>32==deq[0]['globalTid']>>32;ordered=enq['endNs']<=deq[0]['startNs']
 source=[s for s in groups[0]['scopes']if s['chain_key']and s['project_frames']];minimum=min(s['endNs']-s['startNs']for s in source);inner=[s for s in source if s['endNs']-s['startNs']==minimum]
 qreview.append(dict(database_index=r['database_index'],task_record_rowid=r['TASK']['task_record_rowid'],op_type=r['definition']['opType_text'],same_process=same_process,enqueue_before_dequeue=ordered,operation_name_matches=enq['name'].removeprefix('Enqueue@')==deq[0]['name'].removeprefix('Dequeue@'),innermost_source_candidates=inner,Enqueue=enq,Dequeue=deq[0],native_launch=r['exact_native_launch'],queue_flow_rows=r['queue_key_matches']))
graphmap={(r['database_index'],r['graph_ordinal']):r for r in g['graph_executions']};oprows=[];groups=defaultdict(list);edgegroups=defaultdict(list)
for ref in cidx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];amap={a['api_rowid']:a for a in chains['databases'][di]['API_records']}
 for op in d['COMM_OP_records']:
  family=op['opName_text'].split('__')[0];gc=op['graph_control_container_candidates'];bk=[];overlaps=[]
  if len(gc)==1:
   graph=graphmap[(di,gc[0]['graph_ordinal'])]
   bk=[b['block_ordinal']for b in graph['blocks']if b['startNs']<=op['startNs']and op['endNs']<=b['endNs']]
   overlaps=[b['block_ordinal']for b in graph['blocks']if b['startNs']<op['endNs']and op['startNs']<b['endNs']]
  apis=op['native_PYTORCH_API_candidates'];ks=sorted({amap[a['api_rowid']]['chain_key']for a in apis if a['api_rowid']in amap})
  row=dict(database_index=di,COMM_OP_rowid=op['COMM_OP_rowid'],opId=op['opId'],name=op['opName_text'],family=family,count=op['count'],dtype=op['dataType_text'],rankSize=op['rankSize'],group_name=op['groupName_text'],startNs=op['startNs'],endNs=op['endNs'],duration_ns=op['endNs']-op['startNs'],graph_candidates=gc,contained_marker_block_ordinals=bk,overlapping_marker_block_ordinals=overlaps,native_API_rowids=[a['api_rowid']for a in apis],source_chain_keys=ks,linkType_counts=op['linkType_counts'],transportType_counts=op['transportType_counts'],internal_record_type_counts=op['internal_record_type_counts'])
  oprows.append(row);groups[(family,op['dataType_text'],op['rankSize'],op['count'])].append(row)
  if gc:edgegroups[(di,gc[0]['graph_ordinal'])].append(row)
graphstats=[];cohort_groups=defaultdict(list)
for (di,gi),ops in edgegroups.items():
 graph=graphmap[(di,gi)];cohort=graph['phase_records'][0].get('schedule_position_cohort_key')if len(graph['phase_records'])==1 else None
 seq=sorted(ops,key=lambda r:(r['startNs'],r['COMM_OP_rowid']))
 signature=[(o['family'],o['dtype'],o['rankSize'],o['count'],o['contained_marker_block_ordinals'])for o in seq]
 stats=dict(database_index=di,graph_ordinal=gi,graph_COMM_OP_records=len(ops),inside_one_marker_block=sum(len(o['contained_marker_block_ordinals'])==1 for o in ops),outside_no_marker_overlap=sum(not o['overlapping_marker_block_ordinals']for o in ops),overlaps_but_not_contained=sum(bool(o['overlapping_marker_block_ordinals'])and not o['contained_marker_block_ordinals']for o in ops),per_block_operation_count_hist=dict(Counter(sum(b['block_ordinal']in o['contained_marker_block_ordinals']for o in ops)for b in graph['blocks'])),family_counts=dict(Counter(o['family']for o in ops)),local_operation_duration_sum_ns=sum(o['duration_ns']for o in ops),recorded_sequence_signature_sha256=hashlib.sha256(json.dumps(signature,separators=(',',':')).encode()).hexdigest(),exact_schedule_cohort=cohort,ep_rank=graph['phase_records'][0]['ep_rank']if len(graph['phase_records'])==1 else None,sequence=signature)
 graphstats.append(stats)
 if cohort:cohort_groups[cohort].append(stats)
cohorts=[dict(key=k,members=len(v),ep_ranks=sorted(s['ep_rank']for s in v),sequence_signature_counts=dict(Counter(s['recorded_sequence_signature_sha256']for s in v)),local_comm_sum_ns=dict(minimum=min(s['local_operation_duration_sum_ns']for s in v),maximum=max(s['local_operation_duration_sum_ns']for s in v)),no_cross_rank_timestamp_subtraction=True)for k,v in cohort_groups.items()]
regimes=[dict(family=k[0],dtype=k[1],rankSize=k[2],count=k[3],operations=len(v),duration_ns=dict(minimum=min(o['duration_ns']for o in v),median=statistics.median(o['duration_ns']for o in v),maximum=max(o['duration_ns']for o in v)),source_chain_keys=sorted({s for o in v for s in o['source_chain_keys']}),source_bound_operations=sum(bool(o['source_chain_keys'])for o in v),graph_operations=sum(bool(o['graph_candidates'])for o in v))for k,v in groups.items()]
missing=[r for r in oprows if not r['native_API_rowids']and not r['graph_candidates']]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),COMM_OP_records=len(oprows),families=dict(Counter(o['family']for o in oprows)),dtype_counts=dict(Counter(o['dtype']for o in oprows)),group_count_regimes=len(regimes),graph_COMM_OP_records=sum(s['graph_COMM_OP_records']for s in graphstats),graph_inside_one_marker_block=sum(s['inside_one_marker_block']for s in graphstats),graph_outside_marker_blocks=sum(s['outside_no_marker_overlap']for s in graphstats),graph_partial_marker_overlaps=sum(s['overlaps_but_not_contained']for s in graphstats),graph_operation_count_hist=dict(Counter(s['graph_COMM_OP_records']for s in graphstats)),cohorts=len(cohorts),cohorts_with_uniform_operation_sequence=sum(len(c['sequence_signature_counts'])==1 for c in cohorts),native_source_chain_bound_operations=sum(bool(o['source_chain_keys'])for o in oprows),unassigned_COMM_OP_records=len(missing),queue_tasks=len(qreview),queue_tasks_unique_innermost_source=sum(len(r['innermost_source_candidates'])==1 for r in qreview),queue_tasks_valid_PID_name_order=sum(r['same_process']and r['enqueue_before_dequeue']and r['operation_name_matches']for r in qreview),no_source_DB_or_device_work=True,overall_complete=False,
 limits=['Communication operations,internal transport tasks,andcompute time remain separate layers. Local sum ofoperationintervaldurations isnotrequestcriticalpathoradditivewithkernels.', 'Within anexactrecordedschedulecohort,identicalorderedoperation/shape/blocksignatures arestructuralsymmetry,notper-tensorroutecontentsorexposurecausality.', 'Queue association explicitlydecodesCONNECTION_IDS foreignkeys andusesuniqueEnqueue/Dequeueflow,PID,name,CPUorderplusnativeCANNlaunch; itisamulti-hopderivedsourceassociation,notrewrittenoriginaldirectnativeflow.'])
payload=(json.dumps(dict(summary=summary,queue_source_review=qreview,communication_operations=oprows,graph_communication_partitions=graphstats,graph_cohort_sequence_checks=cohorts,communication_count_regimes=regimes,unassigned_communication_operations=missing),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
short=lambda f:f.replace('/vllm-workspace/vllm-ascend/vllm_ascend/','a/').replace('/vllm-workspace/vllm/vllm/','v/')
generic=[r for r in contexts['chains']if r['project_frames']and any(s in r['project_frames'][0]for s in ['/ops/linear.py','/ops/layernorm.py','/parameter.py','base_device_communicator.py','/w8a8_dynamic.py','/vocab_parallel_embedding.py'])]
public=dict(summary=summary,queue_sources=[dict(database_index=r['database_index'],task_rowid=r['task_record_rowid'],op=r['op_type'],source=[dict(name=s['name'],chain_key=s['chain_key'],project_frames=[short(f)for f in s['project_frames'][:5]])for s in r['innermost_source_candidates']])for r in qreview],unassigned_communication=missing,generic_source_page=[dict(key=r['chain_key'],executions=r['compute_executions'],ops=r['op_type_counts'],project_frames=[short(f)for f in r['project_frames'][:10]])for r in generic[:12]],cohort_sequence_checks=cohorts,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>31000:public.pop('generic_source_page');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
