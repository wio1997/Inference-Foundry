import json,hashlib,statistics,collections
from pathlib import Path
h=Path(__file__).resolve().parent
p=h/'SHARED-TAIL-SERVICE-MODEL-v167.json';d=json.loads(p.read_text());eps=[e for a in d['profiles'] for e in a['episodes']];gs=[g for a in d['profiles'] for g in a['graphs']]
shapes=collections.Counter(tuple(e['input_shapes']) for e in eps)
checks={'all_32_profiles':len(d['profiles'])==32,'all_11520_episodes':len(eps)==11520,'all_288_graphs':len(gs)==288,'one_native_shape_tuple':len(shapes)==1,'zero_analysis_issues':sum(len(a['issues']) for a in d['profiles'])==0,'all_episode_relaxation_identities':all(e['fixed_service_relaxation_lower_ns']==max(e['R_ns'],e['S_four_kernel_service_ns'])+e['dependent_merge_ns']<=e['observed_span_ns'] for e in eps),'all_graph_algebra':all(g['fixed_outside_segment_relaxation_lower_ns']==g['graph_duration_ns']-g['allowance_within_fixed_outside_segments_ns']>0 and g['hypothetical_time_ratio_cap']==g['graph_duration_ns']/g['fixed_outside_segment_relaxation_lower_ns'] for g in gs)}
for a in d['profiles']:
 for g in a['graphs']:
  es=sorted([e for e in a['episodes'] if e['parent_launch_rowid']==g['parent_launch_rowid']],key=lambda e:e['start_ns'])
  assert len(es)==40 and all(x['end_ns']<=y['start_ns'] for x,y in zip(es,es[1:]))
  assert sum(e['overlap_and_gap_allowance_ns'] for e in es)==g['allowance_within_fixed_outside_segments_ns']
assert all(checks.values())
out={'status':'PASS_SCOPED_ALGEBRA_AND_DOMAIN_REVIEW','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checks':checks,'shape_tuple':list(next(iter(shapes))),'domain':['All episodes retain original serial dependency order;only within-episode R versus S may overlap.','No earlier release outside an episode,no cross-layer or cross-iteration overlap/fusion.','Measured services R,S,M are fixed,including waits already inside R;changing collective readiness invalidates those service constants.','Original output and quantization semantics retained;merge waits for both branches.','Outside episode serial segments/gaps remain fixed;relaxation optimistically permits zero internal gaps and no resource contention.','Input/alias/last-reader/ownership/numerical legality must hold independently;this analysis does not supply them.'],'proof':'Each episode requires at least max(R,S)+M. Serial composition with fixed outside segments gives Gnew >= Gold - sum(episodeSpan-max(R,S)-M). This derives a lower bound by relaxation,not from an exhibited schedule.','not_claimed':['Physical hardware/theoretical algorithm minimum','Reachable or end-to-end request speedup','Any all-reader independence proof','Any global communicator or cross-rank cohort identity'],'overall_complete':False}
(h/'shared-tail-bound-independent-review.json').write_text(json.dumps(out,indent=2)+'\n')
# Document the creation route exactly as observed: native launch identity works;
# same-thread Python range transport does not.
p=h/'creation-probe-offline-review.json';r=json.loads(p.read_text())['review'];t=r['tables'];s={a['id']:a['value'] for a in t['STRING_IDS']['rows']};apis=t['CANN_API']['rows'];tasks=t['TASK']['rows'];comps={a['globalTaskId']:a for a in t['COMPUTE_TASK_INFO']['rows']};links=[]
for task in tasks:
 if task['modelId']==4294967295 or task['globalTaskId'] not in comps:continue
 match=[a for a in apis if a['connectionId']==task['connectionId']];assert len(match)==1
 api=match[0];marks=[m for m in t['MSTX_EVENTS']['rows'] if m['globalTid']==api['globalTid'] and m['startNs']<=api['startNs']<=api['endNs']<=m['endNs']]
 parents=[a for a in apis if a['globalTid']==api['globalTid'] and a['startNs']<=api['startNs']<=api['endNs']<=a['endNs'] and s[a['name']].startswith('aclnn')]
 links.append({'native_task':task,'native_name':s[comps[task['globalTaskId']]['name']],'exact_connection_api':dict(api,name_text=s[api['name']]),'enclosing_aclnn_apis':[dict(a,name_text=s[a['name']]) for a in parents],'same_thread_MSTX_source_ranges':len(marks)})
assert len(links)==6 and all(len(x['enclosing_aclnn_apis'])==1 for x in links) and all(x['same_thread_MSTX_source_ranges']==0 for x in links)
(h/'creation-native-route-review.json').write_text(json.dumps({'status':'EXACT_NATIVE_LAUNCH_AND_ACLNN_PARENT_AVAILABLE_PYTHON_SOURCE_TRANSPORT_UNPROVEN','database_sha256':r['sha256'],'captured_compute_instances':len(links),'unique_native_connections':sorted({x['native_task']['connectionId'] for x in links}),'links':links,'reason':'Async framework worker thread differs from Python creator MSTX thread. No nearest-time source binding.','next_single_capability_check':'CPU+NPU profile during the same two-op creation,to test official CPU/native correlation.','overall_complete':False},indent=2)+'\n')
print(json.dumps({'checks':checks,'native_connections':3,'captured_compute_instances':6}))
