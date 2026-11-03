from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,socket,math,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
OUT=M/'complete-PMU-time-identities-and-caller-resources-v18.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');pmu=pin(M/'index.json','a19fca4b0a52bbfd6c3237fb94b5e36b086000ff4f13c7343116cef704626f9e');generic=pin(E/'generic-helper-full-caller-intake-v18.json','cf8eda94fe8869dd315028cb0546c95695dac8db2b2e0f17fb23c90436f6ffd1')
keymap={tuple(k):r['record_index']for r in generic['records']for k in r['full_execution_keys']};assert len(keymap)==14574
opdata=defaultdict(list);callerdata=defaultdict(list);audits=[];unbound=[];pipe_bad=[];pipe_error=defaultdict(list);ratio_bad=Counter();sums_over_one=Counter();metricnames=Counter();samples=[]
def stat(v):
 if not v:return None
 v=sorted(v);return dict(n=len(v),minimum=v[0],median=statistics.median(v),maximum=v[-1])
def summarize(rows):
 result=dict(executions=len(rows),task_duration_ns=stat([r['duration_ns']for r in rows]),PMU_groups=sum(r['has_PMU']for r in rows),fields={})
 for engine in ['aic','aiv']:
  field=engine+'_total_cycles';ratios=sorted({k for r in rows for k in r['metrics']if k.startswith(engine+'_')and k.endswith('_ratio')})
  for ratio in ratios:
   good=[(r['metrics'].get(field,0),r['metrics'][ratio])for r in rows if isinstance(r['metrics'].get(field),(int,float))and r['metrics'].get(field,0)>0 and isinstance(r['metrics'].get(ratio),(int,float))]
   den=sum(c for c,v in good);result['fields'][ratio]=dict(groups=len(good),reported_cycles=den,cycle_weighted_reported_ratio=sum(c*v for c,v in good)/den if den else None,raw_ratio_stats=stat([v for c,v in good]))
 return result
for ref in idx['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);p=pin(Path(pmu['databases'][di]['path']),pmu['databases'][di]['sha256']);tasks={t['task_record_rowid']:t for t in d['TASK_records']};defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']};bound={}
 for g in p['groups']:
  if len(g['native_TASK_candidates'])!=1:unbound.append(dict(database_index=di,globalTaskId=g['globalTaskId'],timestampNs=g['timestampNs'],candidates=g['native_TASK_candidates']));continue
  rid=g['native_TASK_candidates'][0];assert rid not in bound;assert not g['duplicate_metric_names'];metrics={r['metric_name']:r['value']for r in g['metric_rows']};bound[rid]=metrics;t=tasks[rid];definition=defs[t['globalTaskId']];metricnames.update(metrics)
  audit=dict(database_index=di,task_record_rowid=rid,globalTaskId=t['globalTaskId'],op_type=definition['opType_text'],task_type=t['task_type_name'],blockNum=definition['blockNum'],mixBlockNum=definition['mixBlockNum'],duration_ns=t['endNs']-t['startNs'],engines={})
  for engine,core in [('aic',24),('aiv',48)]:
   cyc=metrics[engine+'_total_cycles'];reported=metrics[engine+'_total_time'];matches=[];pred=[]
   if cyc>0:
    for b in sorted({int(v)for v in [definition['blockNum'],definition['mixBlockNum']]if v and int(v)>0}):
     ns=cyc/(1800*1e6)*1e9/b*math.ceil(b/core);delta=reported-ns;pred.append(dict(block=b,predicted_time_ns=ns,raw_time_minus_prediction=delta))
     if math.isclose(reported,ns,rel_tol=1e-10,abs_tol=0.001):matches.append(b)
   else:matches=['ZERO_CYCLES_ZERO_TIME']if reported==0 else[]
   audit['engines'][engine]=dict(reported_cycles=cyc,reported_time_raw=reported,predictions=pred,matching_block_candidates=matches,reported_time_over_native_duration=reported/audit['duration_ns']if audit['duration_ns']>0 else None)
   ps=0
   for name,v in metrics.items():
    if not name.startswith(engine+'_')or not name.endswith('_ratio'):continue
    if not 0<=v<=1:ratio_bad[name]+=1
    time_name=name[:-6]+'_time'
    if time_name in metrics:
     error=abs(metrics[time_name]-reported*v);pipe_error[name].append(error)
     if not math.isclose(metrics[time_name],reported*v,rel_tol=1e-10,abs_tol=0.001):pipe_bad.append(dict(database_index=di,task_record_rowid=rid,ratio=name,error=error))
     ps+=v
   if ps>1+1e-10:sums_over_one[engine]+=1
  audits.append(audit)
 for rid,t in tasks.items():
  if t['globalTaskId']not in defs:continue
  c=defs[t['globalTaskId']];row=dict(database_index=di,task_record_rowid=rid,op_type=c['opType_text'],duration_ns=t['endNs']-t['startNs'],has_PMU=rid in bound,metrics=bound.get(rid,{}),definition=c)
  opdata[c['opType_text']].append(row)
  if(di,rid)in keymap:callerdata[keymap[(di,rid)]].append(row)
  if len(samples)<4 and rid in bound and c['opType_text']in ['MatMulV2','DispatchFFNCombine','RmsNorm','SparseFlashMla']and c['opType_text']not in {x['op_type']for x in samples}:samples.append(dict(op_type=c['opType_text'],TASK=t,definition=c,metrics=bound[rid]))
op_summary=[dict(op_type=k,**summarize(v))for k,v in sorted(opdata.items(),key=lambda q:str(q[0]))];callers=[]
for r in generic['records']:
 rs=callerdata[r['record_index']];assert len(rs)==r['compute_executions'];shapes=Counter(json.dumps({k:v.get(k)for k in ['inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text','blockNum','mixBlockNum']},sort_keys=True)for row in rs for v in [row['definition']])
 callers.append(dict(record_index=r['record_index'],chain_key=r['chain_key'],summary=summarize(rs),shape_regimes=[dict(definition=json.loads(k),executions=n)for k,n in shapes.items()],execution_keys=[[row['database_index'],row['task_record_rowid']]for row in rs]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),PMU_execution_groups=len(audits),unbound_PMU_groups=len(unbound),native_compute_executions=sum(len(v)for v in opdata.values()),generic_chains=len(callers),generic_compute_executions=sum(c['summary']['executions']for c in callers),metric_name_groups=dict(metricnames),time_formula_test=dict(assumed_captured_frequency_Hz=1800*1e6,core_counts=dict(aic=24,aiv=48),formula='cycles/frequency_Hz*1e9/block*ceil(block/core)',engines={engine:dict(matching_groups=sum(bool(a['engines'][engine]['matching_block_candidates'])for a in audits),nonzero_cycle_groups=sum(a['engines'][engine]['reported_cycles']>0 for a in audits),nonzero_with_no_matching_block=sum(a['engines'][engine]['reported_cycles']>0 and not a['engines'][engine]['matching_block_candidates']for a in audits))for engine in ['aic','aiv']}),pipe_time_equals_reported_total_time_times_ratio=dict(comparisons=sum(len(v)for v in pipe_error.values()),mismatches=len(pipe_bad),maximum_absolute_error_by_ratio={k:max(v)for k,v in pipe_error.items()}),ratio_outside_zero_one_counts=dict(ratio_bad),groups_with_pipe_ratio_sum_over_one=dict(sums_over_one),no_new_source_model_or_device_work=True,overall_complete=False,limits=['Predictions test observed metadata frequency and main/mix block candidates against native reported time; matching does not prove actualPython exporter path or physical peak/MFU.','Reported pipe fractions/times overlap; not an exclusive partition or additive latency/traffic model.','MTE busy cycles are not measured external byte volume or saturated HBM bandwidth. Native MC2 andHCCL kernels can include communication polling/waits.','Native count source/fullchain doesnot establish tensor pointer/consumer identity; graph-only sources andmissingPMU remain separate.'])
payload=(json.dumps(dict(summary=summary,per_execution_time_formula_audit=audits,unbound_groups=unbound,pipe_identity_mismatches=pipe_bad,op_reported_resource_summaries=op_summary,generic_caller_resources=callers,samples=samples),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,first_samples=samples,cache_only_KV_callers=[c for c in callers if c['record_index']in [49,50]],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
