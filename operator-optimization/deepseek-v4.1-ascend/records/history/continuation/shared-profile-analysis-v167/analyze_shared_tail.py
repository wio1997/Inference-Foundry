"""Actual per-envelope native patterns and an explicitly fixed-service relaxation."""
import collections,concurrent.futures,hashlib,json,sqlite3,statistics
from pathlib import Path
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166-official-db')
PATTERN=['aclnnDynamicQuantV2_DynamicQuant_DynamicQuant',
 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3','DequantSwigluQuant',
 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3',
 'aclnnForeachAddListV2_ForeachAddListV2_ForeachAddList']
ENVS={r['profile']:r['envelopes'] for r in json.loads((H/'NATIVE-ENVELOPES-v166.json').read_text())['profiles']}
def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 s=dict(c.execute('select id,value from STRING_IDS'))
 rows=[dict(r) for r in c.execute('select t.rowid,t.*,x.name,x.inputShapes,x.outputShapes from TASK t join COMPUTE_TASK_INFO x on t.globalTaskId=x.globalTaskId where t.modelId!=4294967295 order by t.startNs,t.endNs,t.rowid')]
 by_stream=collections.defaultdict(list)
 for r in rows:by_stream[(r['modelId'],r['streamId'])].append(r)
 episodes=[];issues=[]
 for key,rs in by_stream.items():
  for i,r in enumerate(rs):
   if s[r['name']]!='DispatchFFNCombine':continue
   tail=rs[i+1:i+6]
   if [s[x['name']] for x in tail]!=PATTERN:
    issues.append({'rowid':r['rowid'],'reason':'PATTERN_MISMATCH','next_names':[s[x['name']] for x in tail]});continue
   match=[e for e in ENVS[db.parent.parent.name] if e['native_model_id']==r['modelId'] and e['device']==r['deviceId'] and e['native_pid']==r['globalPid'] and e['start_ns']<=r['startNs']<=tail[-1]['endNs']<=e['end_ns']]
   if len(match)!=1:issues.append({'rowid':r['rowid'],'reason':'NO_UNIQUE_PARENT_ENVELOPE'});continue
   allrows=[r,*tail];serial=all(a['endNs']<=b['startNs'] for a,b in zip(allrows,allrows[1:]))
   if not serial:issues.append({'rowid':r['rowid'],'reason':'OBSERVED_OVERLAP'});continue
   R=r['endNs']-r['startNs'];S=sum(x['endNs']-x['startNs'] for x in tail[:4]);M=tail[4]['endNs']-tail[4]['startNs'];span=tail[-1]['endNs']-r['startNs']
   # Lower bound for ONLY the declared unchanged-service DAG: R and S may
   # overlap freely,but dependent M cannot start until both chains complete.
   lower=max(R,S)+M;assert lower<=span
   episodes.append({'profile':db.parent.parent.name,'native_model_id':r['modelId'],'stream_id':r['streamId'],
     'parent_launch_rowid':match[0]['launch_rowid'],'rowids':[x['rowid'] for x in allrows],
     'task_ids':[x['taskId'] for x in allrows],'start_ns':r['startNs'],'end_ns':tail[-1]['endNs'],
     'native_names':[s[x['name']] for x in allrows],'input_shapes':[s[x['inputShapes']] for x in allrows],
     'R_ns':R,'S_four_kernel_service_ns':S,'dependent_merge_ns':M,'observed_span_ns':span,
     'fixed_service_relaxation_lower_ns':lower,'overlap_and_gap_allowance_ns':span-lower,
     'no_gap_ideal_overlap_allowance_ns':min(R,S),'source_role':'Native pattern matches existing F21/shared-expert contract;not a per-operand creator/reader proof.'})
 graphs=[]
 for e in ENVS[db.parent.parent.name]:
  es=sorted([x for x in episodes if x['parent_launch_rowid']==e['launch_rowid']],key=lambda x:x['start_ns'])
  disjoint=all(a['end_ns']<=b['start_ns'] for a,b in zip(es,es[1:]));assert disjoint
  allowance=sum(x['overlap_and_gap_allowance_ns'] for x in es)
  # This graph-level lower bound additionally freezes all outside-episode
  # serial segment lengths;it is NOT a hardware/request/global algorithm bound.
  graphs.append({'profile':db.parent.parent.name,'parent_launch_rowid':e['launch_rowid'],'episodes':len(es),
   'graph_duration_ns':e['duration_ns'],'sum_R_ns':sum(x['R_ns'] for x in es),'sum_S_ns':sum(x['S_four_kernel_service_ns'] for x in es),
   'allowance_within_fixed_outside_segments_ns':allowance,
   'fixed_outside_segment_relaxation_lower_ns':e['duration_ns']-allowance,
   'hypothetical_time_ratio_cap':e['duration_ns']/(e['duration_ns']-allowance),
   'kernel_service_only_overlap_allowance_ns':sum(x['no_gap_ideal_overlap_allowance_ns'] for x in es)})
 c.close();return {'profile':db.parent.parent.name,'episodes':episodes,'graphs':graphs,'issues':issues}
if __name__=='__main__':
 dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,dbs))
 eps=[e for r in rows for e in r['episodes']];graphs=[g for r in rows for g in r['graphs']]
 quant=lambda xs:{'min':min(xs),'median':statistics.median(xs),'max':max(xs)}
 out={'status':'ACTUAL_NATIVE_SHARED_TAIL_PATTERNS_WITH_RESTRICTED_SERVICE_DAG_BOUND','profiles':rows,
  'summary':{'profiles':len(rows),'episodes':len(eps),'graphs':len(graphs),'issues':sum(len(r['issues']) for r in rows),
   'episodes_per_graph':dict(collections.Counter(g['episodes'] for g in graphs)),
   'R_us':quant([e['R_ns']/1000 for e in eps]),'S_four_kernel_service_us':quant([e['S_four_kernel_service_ns']/1000 for e in eps]),
   'merge_us':quant([e['dependent_merge_ns']/1000 for e in eps]),
   'graph_hypothetical_ratio_cap':quant([g['hypothetical_time_ratio_cap'] for g in graphs]),
   'graph_shared_service_fraction':quant([g['sum_S_ns']/g['graph_duration_ns'] for g in graphs])},
  'bound_status':'CONDITIONAL_CEILING_IN_UNCHANGED_SERVICE_DAG_ONLY',
  'proof':'For fixed branch services R and S and dependent merge M,every schedule needs at least max(R,S)+M. This is a relaxation lower bound,not a chosen feasible schedule.',
  'domain':['Current native episode kernels retain their measured service durations and output/quantization semantics.',
   'The four-kernel S branch may be released independently of R only if input identity/readiness/alias/epoch/consumer contracts permit.',
   'Dependent merge waits for both;not counted as an independently overlapable branch.',
   'Graph-level expression additionally preserves all outside-episode serial segment lengths.'],
  'limits':['Not an actual reachable speedup,physical hardware ceiling,full fusion bound or request critical-path bound.',
   'R can contain cross-rank waits;changing schedule may change its service duration,outside this restricted model.',
   'Cube/Vector/memory contention can prevent the relaxed overlap;source/reader/numerical legality is not supplied by this pattern.',
   'No per-rank or per-iteration cross-process matching from timestamps or local IDs.'],
  'new_independence_proofs':0,'historical_native_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False}
 (H/'SHARED-TAIL-SERVICE-MODEL-v167.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary']))
