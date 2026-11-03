from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
M=P/'native-PMU-complete-v18';OUT=M/'C2-exact-graph-schedule-row-domain-v24.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
p=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f');roots={r['root_key']:r for r in p['prepare_roots']}
g=pin(P/'native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
b=pin(M/'current-compressor-state-and-pool-norm-boundaries-v24.json','b5b12c5860c862c3c824bb2f4e3e805b1fd0fdebf01c3a895beed3d5bace13b4')
bygraph=Counter((r['database_index'],r['graph_ordinal'])for r in b['graph_C2_boundaries']);rows=[];unbound=[];hist=Counter();parity=Counter();missing=0
for graph in g['graph_executions']:
 key=(graph['database_index'],graph['graph_ordinal']);assert bygraph[key]==3
 if len(graph['prepare_root_keys'])!=1:
  unbound.append(dict(database_index=key[0],graph_ordinal=key[1],prepare_root_keys=graph['prepare_root_keys']));continue
 root=roots[graph['prepare_root_keys'][0]];meta=root['metadata'];sched=meta['scheduled_tokens'];assert sum(sched.values())==meta['total_scheduled_tokens']
 state=meta['cached_request_state'];computed=dict(zip(state.get('req_ids',[]),state.get('num_computed_tokens',[])))
 computed.update({r['req_id']:r['num_computed_tokens']for r in meta['new_request_state']})
 rr=[];num=0;num_known=True
 for request,n in sched.items():
  s=computed.get(request)
  if isinstance(s,int):cnt=(s+n)//2-s//2;parity[str(s%2)]+=1;method='HOST_START_AND_SCHEDULE_UNDER_CONTIGUOUS_CACHE_CHUNK_CONTRACT'
  elif n%2==0:cnt=n//2;missing+=1;method='EVEN_LENGTH_PARITY_INDEPENDENT_UNDER_CONTIGUOUS_CACHE_CHUNK_CONTRACT'
  else:cnt=None;missing+=1;num_known=False;method='UNKNOWN_START_PARITY'
  if cnt is not None:num+=cnt
  rr.append(dict(request_id=request,scheduled_tokens=n,host_num_computed_tokens=s,conditional_pair_endpoints=cnt,method=method))
 conditional=num if num_known else None
 hist[json.dumps(dict(requests=len(sched),scheduled_rows=sum(sched.values()),per_request_n=sorted(sched.values()),conditional_pair_endpoints=conditional),sort_keys=True)]+=1
 rows.append(dict(database_index=key[0],graph_ordinal=key[1],prepare_root_key=root['root_key'],phase_records=graph['phase_records'],requests=rr,conditional_pair_endpoints_per_C2_source=conditional,C2_sources=3,recorded_physical_rows_per_C2_norm=24))
assert len(rows)==64 and len(unbound)==16 and all(r['conditional_pair_endpoints_per_C2_source']is not None for r in rows)
total=sum(r['conditional_pair_endpoints_per_C2_source']*3 for r in rows);physical=len(rows)*3*24
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),exact_bound_graphs=len(rows),phase_unbound_graphs=len(unbound),C2_source_blocks_bound=len(rows)*3,C2_source_blocks_unbound=len(unbound)*3,graph_schedule_groups=[dict(signature=json.loads(k),graphs=v)for k,v in hist.items()],host_start_parity_counts=dict(parity),requests_without_host_start=missing,
 conditional_completed_pair_endpoint_rows=total,recorded_physical_C2_norm_rows=physical,conditional_pair_endpoint_fraction=total/physical,
 crucial_assumption='Each relevant cache chunk contains the scheduled contiguous positions s..s+n-1 and helper used_len equals n for these exact graphs. Host metadata alone does not bind device used_len,post-correction positions,slot/page payload or skip flags.',
 publication_limit='This is a conditional arithmetic count of completed-pair endpoints,not measured valid published rows. Invalid slots/skip masks can remove endpoints;missing device contract can prevent applying the count.',
 graphs=rows,unbound_graphs=unbound,no_new_source_DB_model_device_or_service_work=True,overall_complete=False)
bb=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(bb)
print(json.dumps(dict(summary={k:v for k,v in result.items()if k not in ['graphs','unbound_graphs']},first_graph=rows[0],full_result=str(OUT),result_sha256=hashlib.sha256(bb).hexdigest()),ensure_ascii=False))
