from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'resource-attribution-ranking-and-overlap-review-v20.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
d=pin(M/'resource-functional-attribution-and-uncertainty-v20-v3.json','c8e2c128d208597aeeb07a2b434506c930397d773b0ab5767a0fadba7e4840c4')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
aa=d['execution_assignments'];keys={(a[0],a[1])for a in aa};assert len(keys)==len(aa)==286950
assert Counter(a[2]for a in aa)==Counter(DIRECT=154940,GRAPH=132000,QUEUE=10)
assert sum(a[11]for a in aa)==286101
sh=d['shape_role_context_regimes'];assert sum(r['summary']['executions']for r in sh)==286950 and sum(r['summary']['missing_PMU']for r in sh)==849
mods=d['module_resource_buckets'];by={m['module']:m for m in mods};rank=[];within=[]
for m in mods:
 for ratio,b in m['counter_ratio_assignment_bounds'].items():
  if b['certain_ratio']is not None:assert b['minimum_over_optional_membership']-1e-12<=b['certain_ratio']<=b['maximum_over_optional_membership']+1e-12
 q=m['counter_ratio_assignment_bounds']['aic_mac_ratio']
 if q['certain_ratio']is not None:rank.append(dict(module=m['module'],certain_ratio=q['certain_ratio'],minimum=q['minimum_over_optional_membership'],maximum=q['maximum_over_optional_membership'],certain_executions=m['certain']['executions'],PMU_missing=m['certain']['missing_PMU']))
 s=m['certain'];assert s['native_task_duration_sum_ns']>=s['sum_of_per_database_native_interval_union_ns'];within.append(dict(module=m['module'],summed_task_duration_ns=s['native_task_duration_sum_ns'],summed_per_database_union_ns=s['sum_of_per_database_native_interval_union_ns'],overlapping_task_duration_not_additive_ns=s['native_task_duration_sum_ns']-s['sum_of_per_database_native_interval_union_ns']))
rank.sort(key=lambda r:r['certain_ratio']);assert rank[0]['module']=='M12';robust=all(rank[0]['maximum']<r['minimum']for r in rank[1:]);assert robust
missing=[dict(module_candidates=r['module_candidates'],role=r['role'],origin=r['origin'],context=r['context'],op_type=r['signature']['opType_text'],executions=r['summary']['executions'],missing_PMU=r['summary']['missing_PMU'])for r in sh if r['summary']['missing_PMU']]
op12=defaultdict(lambda:dict(executions=0,missing_PMU=0,native_duration_sum_ns=0,MAC_cycles=0,MAC_numerator=0,vec_cycles=0,vec_numerator=0))
for r in sh:
 if r['module_candidates']!=['M12']:continue
 a=op12[r['signature']['opType_text']];s=r['summary'];a['executions']+=s['executions'];a['missing_PMU']+=s['missing_PMU'];a['native_duration_sum_ns']+=s['native_task_duration_sum_ns']
 for prefix,field in [('MAC','aic_mac_ratio'),('vec','aiv_vec_ratio')]:
  b=s['fields'].get(field)
  if b:a[prefix+'_cycles']+=b['cycles'];a[prefix+'_numerator']+=b['weighted_numerator']
for a in op12.values():
 for prefix in ['MAC','vec']:a[prefix+'_reported_ratio']=a[prefix+'_numerator']/a[prefix+'_cycles']if a[prefix+'_cycles']else None
selected=defaultdict(dict)
for a in aa:
 if a[3]in[['M12'],['M13']]:selected[a[0]][a[1]]=a[3][0]
def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(out[-1][1],b)
 return out
def intersect(a,b):
 i=j=total=0
 while i<len(a)and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<=b[j][1]:i+=1
  else:j+=1
 return total
overlaps=[]
for ref in ix['databases']:
 di=ref['database_index'];part=pin(Path(ref['path']),ref['sha256']);ss=defaultdict(list)
 for t in part['TASK_records']:
  mod=selected[di].get(t['task_record_rowid'])
  if mod:ss[mod].append((t['startNs'],t['endNs']))
 a,b=merge(ss['M12']),merge(ss['M13']);ta=sum(y-x for x,y in a);tb=sum(y-x for x,y in b);ti=intersect(a,b);assert 0<=ti<=min(ta,tb)
 overlaps.append(dict(database_index=di,M12_union_ns=ta,M13_union_ns=tb,intersection_ns=ti,M13_without_M12_ns=tb-ti,union_both_ns=ta+tb-ti))
sumover={k:sum(r[k]for r in overlaps)for k in ['M12_union_ns','M13_union_ns','intersection_ns','M13_without_M12_ns','union_both_ns']}
assert sumover['M12_union_ns']==by['M12']['certain']['sum_of_per_database_native_interval_union_ns'] and sumover['M13_union_ns']==by['M13']['certain']['sum_of_per_database_native_interval_union_ns']
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),assignment_checks=dict(unique_execution_keys=len(keys),origin_counts=dict(Counter(a[2]for a in aa)),PMU_groups=286101,shape_regime_execution_conservation=True,missing_PMU_conservation=True,ratio_bounds_contain_certain_ratios=True,task_sum_not_less_than_per_database_union=True),AIC_MAC_counter_ranking=rank,M12_lowest_robust_to_recorded_role_assignment_uncertainty=robust,ranking_scope='11positive-AIC-cycle functional buckets in this saved32DB capture;reported cycle-weighted MAC fraction over observed PMU groups. Not physical/module utilization,hardware efficiency,missing-PMU bound or request critical-path ranking.',missing_PMU_regimes=missing,M12_operator_resources=dict(op12),within_bucket_interval_accounting=within,M12_M13_cross_bucket_native_intervals=overlaps,M12_M13_interval_sum=sumover,MAC_work_model_by_module=[dict(module=m['module'],**m['certain']['shape_MAC_model'])for m in mods],zero_ComputeInfo_attribution_modules=[m['module']for m in mods if not m['certain']['executions']and not m['optional_execution_count']],no_new_source_DB_model_device_or_service_work=True,overall_complete=False,limits=['Overlap is a same-database native interval intersection. It does not prove dependency/hiding/no-cost overlap,CPU exposure,resource independence or removable latency.','M12compound includes internal EP communication/waits that these intervals do not split. Low ratios do not identify the wait cause.','Unattributed-to-ComputeInfo modules may have CPU/graph-control/child work;zero records is not zero execution or zero utilization.','No physical peak or mandatory fused work proven. Shape MAC work models must be rederived for candidate fusion/reuse algorithms.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
