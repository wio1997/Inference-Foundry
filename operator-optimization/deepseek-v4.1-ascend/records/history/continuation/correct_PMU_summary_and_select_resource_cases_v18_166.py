from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
source=M/'complete-PMU-time-identities-and-caller-resources-v18.json';raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='edcfec321a86dc366d2f5e4b69d59035690c4bf4258fecdd3e23227e1a5ad6c1';d=json.loads(raw)
ib=(M/'index.json').read_bytes();assert hashlib.sha256(ib).hexdigest()=='a19fca4b0a52bbfd6c3237fb94b5e36b086000ff4f13c7343116cef704626f9e';idx=json.loads(ib)
OUT=M/'complete-PMU-time-identities-and-caller-resources-v18-v2.json';assert not OUT.exists()
old=d['summary'].pop('metric_name_groups');counts=idx['summary']['metric_name_counts'];assert len(counts)==28 and set(counts.values())=={286101}
d['summary']['metric_name_group_counts']=counts;d['summary']['summary_metadata_correction']='InitialCounter.update(metricsdict) summed values instead of counting metric names; replaced mislabeled field with complete immutable rawcensus counts. Per-execution formula audits,pipe identities andresource summaries unchanged.'
d['summary_correction_provenance']=dict(created_utc=datetime.now(timezone.utc).isoformat(),initial_result=str(source),initial_sha256=hashlib.sha256(raw).hexdigest(),authoritative_metric_count_index=str(M/'index.json'),authoritative_index_sha256=hashlib.sha256(ib).hexdigest(),changed_only_summary_metric_count_field=True)
payload=(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
keys=['aic_mac_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_fixpipe_ratio','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio']
def compact(s):return dict(executions=s['executions'],task_duration_ns=s['task_duration_ns'],PMU_groups=s['PMU_groups'],reported_cycle_weighted_ratios={k:s['fields'][k]['cycle_weighted_reported_ratio']for k in keys if k in s['fields']})
ops=['DispatchFFNCombine','SparseFlashMla','QuantBatchMatmulV3','MatMulV2','MatMulV3','RmsNorm','HcPre','HcPost','hcom_allGather_','hcom_allReduce_','hcom_reduceScatter_']
operator_cases=[dict(op_type=r['op_type'],**compact(r))for r in d['op_reported_resource_summaries']if r['op_type']in ops]
indices=[0,1,2,3,4,11,12,22,23,24,29,34,36,45,49,50,56,60,62]
callers=[dict(record_index=c['record_index'],chain_key=c['chain_key'],summary=compact(c['summary']),shape_regimes=c['shape_regimes'])for c in d['generic_caller_resources']if c['record_index']in indices]
print(json.dumps(dict(summary=d['summary'],resource_operator_cases=operator_cases,resource_caller_cases=callers,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
