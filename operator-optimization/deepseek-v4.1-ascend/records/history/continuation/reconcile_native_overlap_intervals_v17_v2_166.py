from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,socket,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
OUT=P/'native-communications-v15/native-interval-overlap-reconciliation-v17.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'native-execution-identity-v14/index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');comm=pin(P/'native-communications-v15/communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c');helper=pin(P/'native-communications-v15/saved-communication-analysis-tables-v17.json','511d1d6834ac9211b442cca38a754507323320358ce770a225a6219651da75d3')
opbydb=defaultdict(list)
for op in comm['communication_operations']:opbydb[op['database_index']].append((op['startNs'],op['endNs']))
hbydb={d['database_index']:d for d in helper['databases']}
def merged(xs):
 out=[]
 for s,e in sorted((s,e)for s,e in xs if e>s):
  if out and s<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],e))
  else:out.append((s,e))
 return out
def size(xs):return sum(e-s for s,e in xs)
rows=[]
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];defs={c['globalTaskId']for c in d['COMPUTE_definitions']};ct=[t for t in d['TASK_records']if t['globalTaskId']in defs];c=merged((t['startNs'],t['endNs'])for t in ct);h=merged(opbydb[di]);u=merged(c+h);inter=size(c)+size(h)-size(u);alltasks=[(t['startNs'],t['endNs'])for t in d['TASK_records']if t['endNs']>t['startNs']];allhull=max(e for s,e in alltasks)-min(s for s,e in alltasks)
 values=dict(compute_union_ns=size(c),communication_union_ns=size(h),compute_communication_intersection_ns=inter,communication_without_compute_ns=size(h)-inter,combined_union_ns=size(u),compute_communication_hull_ns=u[-1][1]-u[0][0],all_positive_TASK_hull_ns=allhull)
 raw=hbydb[di]['tables']['StepTraceTime'];assert len(raw)==1;raw=raw[0]
 comparisons={k:dict(raw_value=raw[k],candidate_unit_multiplier_to_ns=1000,helper_times1000_minus_native_ns=round(raw[k]*1000)-values[v])for k,v in [('computing','compute_union_ns'),('communication','communication_union_ns'),('overlapped','compute_communication_intersection_ns'),('communication_not_overlapped','communication_without_compute_ns'),('stage','compute_communication_hull_ns')]}
 comparisons['stage_all_TASK_hull']=dict(raw_value=raw['stage'],candidate_unit_multiplier_to_ns=1000,helper_times1000_minus_native_ns=round(raw['stage']*1000)-allhull)
 algebra=dict(communication_equals_overlapped_plus_nonoverlapped_raw=raw['communication']-raw['overlapped']-raw['communication_not_overlapped'],stage_minus_compute_nonoverlap_free_raw=raw['stage']-raw['computing']-raw['communication_not_overlapped']-raw['free'])
 rows.append(dict(database_index=di,compute_executions=len(ct),communication_operations=len(opbydb[di]),native_interval_values=values,helper_StepTraceTime_raw=raw,comparisons=comparisons,helper_raw_algebra=algebra,TASK_type_counts=dict(Counter(t['task_type_name']for t in d['TASK_records'])),invalid_compute_intervals=sum(t['endNs']<=t['startNs']for t in ct)))
fields=list(rows[0]['comparisons']);summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(rows),compute_executions=sum(r['compute_executions']for r in rows),communication_operations=sum(r['communication_operations']for r in rows),raw_helper_multiplier_tested=1000,comparison_summary={k:dict(exact_rows=sum(r['comparisons'][k]['helper_times1000_minus_native_ns']==0 for r in rows),minimum_delta_ns=min(r['comparisons'][k]['helper_times1000_minus_native_ns']for r in rows),median_delta_ns=statistics.median(r['comparisons'][k]['helper_times1000_minus_native_ns']for r in rows),maximum_delta_ns=max(r['comparisons'][k]['helper_times1000_minus_native_ns']for r in rows))for k in fields},no_new_source_model_or_device_work=True,overall_complete=False,limits=['This compares recorded native interval unions/intersections with exact helper fields. No source-code metric definition or physical MAC-active time is assumed.','An MC2/composite kernel interval includes internal waits and communication; compute_union is an execution-interval category,not MFU or true arithmetic occupancy.','Communication_without_compute is a local profile interval difference,not request critical-path exposure or removable latency.','CPU preparation,unrecorded work and cross-rank clock relationships are not inferred from the hull or free category. All mismatches remain explicit.'])
payload=(json.dumps(dict(summary=summary,databases=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,first_database=rows[0],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
