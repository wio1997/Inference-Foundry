from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'graph-shared-expert-and-MC2-order-v20.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
d=pin(M/'resource-functional-attribution-and-uncertainty-v20-v3.json','c8e2c128d208597aeeb07a2b434506c930397d773b0ab5767a0fadba7e4840c4')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
graph=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
assignment={(a[0],a[1]):a for a in d['execution_assignments']};bydb=defaultdict(list)
for g in graph['graph_executions']:bydb[g['database_index']].append(g)
rows=[];counts=Counter();stages=defaultdict(Counter);phasecounts=Counter();gaps=defaultdict(list)
for ref in ix['databases']:
 di=ref['database_index'];part=pin(Path(ref['path']),ref['sha256']);tasks={t['task_record_rowid']:t for t in part['TASK_records']};defs={c['globalTaskId']:c for c in part['COMPUTE_definitions']}
 for g in bydb[di]:
  phase=g['phase_records'][0]['phase']if len(g['phase_records'])==1 else'PHASE_UNBOUND'
  for b in g['blocks']:
   tt=[tasks[i]for i in b['task_record_rowids']];mc=[t for t in tt if defs[t['globalTaskId']]['opType_text']=='DispatchFFNCombine'];assert len(mc)==1;mc=mc[0]
   shared=[t for t in tt if assignment[(di,t['task_record_rowid'])][3]==['M13']];assert len(shared)==5
   sr=[]
   for t in sorted(shared,key=lambda t:(t['startNs'],t['endNs'])):
    a=assignment[(di,t['task_record_rowid'])];ri=a[10];stage={45:'INPUT_QUANT',46:'GATE_UP',47:'ACTIVATION_QUANT',48:'DOWN',49:'ROUTED_SHARED_ADD'}[ri]
    pos='BEFORE'if t['endNs']<=mc['startNs']else'AFTER'if t['startNs']>=mc['endNs']else'OVERLAP'
    stages[stage][pos]+=1;stages[stage]['SAME_STREAM'if t['streamId']==mc['streamId']else'DIFFERENT_STREAM']+=1
    sr.append(dict(stage=stage,task_record_rowid=t['task_record_rowid'],streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],relative_to_MC2=pos,signed_start_gap_from_MC2_end_ns=t['startNs']-mc['endNs']))
    if pos=='AFTER':gaps[stage].append(t['startNs']-mc['endNs'])
   counts['blocks']+=1;counts['all_shared_before']+=all(r['relative_to_MC2']=='BEFORE'for r in sr);counts['all_shared_after']+=all(r['relative_to_MC2']=='AFTER'for r in sr);counts['any_overlap']+=any(r['relative_to_MC2']=='OVERLAP'for r in sr);phasecounts[phase]+=1
   rows.append(dict(database_index=di,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],phase=phase,MC2=dict(task_record_rowid=mc['task_record_rowid'],streamId=mc['streamId'],startNs=mc['startNs'],endNs=mc['endNs']),shared=sr))
assert len(rows)==3200
def stats(xs):return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),graph_executions=len(graph['graph_executions']),blocks=len(rows),counts=dict(counts),block_phase_counts=dict(phasecounts),shared_stage_order_and_stream_counts={k:dict(v)for k,v in stages.items()},signed_after_gap_from_MC2_end_ns={k:stats(v)for k,v in gaps.items()},no_new_source_DB_model_device_or_service_work=True,overall_complete=False,limits=['Exact temporal graph marker blocks and recorded native TASK streams/times,not nearest-step matching or tensor-consumer proof.','Different stream IDs do not prove potential concurrent scheduling;resource reservations,graph event/control edges and actual enable flags remain separate.','Shared-stage AFTER gaps include whatever lies between recorded tasks;not removable launch latency or causal wait without dependency evidence.','No observed overlap in this capture does not prove the feature is absent,incorrect,globally disabled or beneficial to change.','Unboundgraph phase stays unbound. No source/device/config mutation or inference run.'])
b=(json.dumps(dict(summary=summary,blocks=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,first_blocks=rows[:2],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
