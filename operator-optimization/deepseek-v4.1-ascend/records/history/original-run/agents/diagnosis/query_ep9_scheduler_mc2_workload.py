#!/usr/bin/env python3
"""Offline compare two selected EP9 scopes with scheduler totals; no request IDs emitted."""
import collections,hashlib,json,pathlib
from decimal import Decimal
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'; meta=root/'diagnostic-20261029/runtime/metadata/events-752.jsonl'; log=root/'diagnostic-20261029/runtime/logs/service.log'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
hashes={str(p.relative_to(root)):sha(p) for p in (p,meta,log)}
assert hashes[str(p.relative_to(root))]=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashes[str(meta.relative_to(root))]=='9528466dbfcfe95b28a7dd7b98c0f8e69ab405dedcd3d50d5d9976ff8045c65a'
assert hashes[str(log.relative_to(root))]=='3b211068eaa9522c2d246214f885c65d66d0c49b0cbf58037613f561d757d3a7'
with p.open() as f:
 ranks=[x['data'] for line in f if (x:=json.loads(line)).get('kind')=='rank' and x['data'].get('ep')==9]
assert len(ranks)==1;r=ranks[0]
events=[json.loads(line) for line in meta.open()]
def event(call,kind,label):
 a=[x for x in events if x.get('call_id')==call and x.get('kind')==kind and x.get('label')==label]
 assert len(a)==1 and a[0]['pid']==752 and a[0]['tid']==752
 return a[0]
result=[]
for digest_prefix,runner_id,scope_id in [('1b826883','752-5','752-6'),('8c18c293','752-50','752-51')]:
 pb=event(runner_id,'call_begin','NPUModelRunner.execute_model');pe=event(runner_id,'call_end','NPUModelRunner.execute_model');lb=event(scope_id,'call_begin','DeepseekV41DecoderLayer.forward');le=event(scope_id,'call_end','DeepseekV41DecoderLayer.forward')
 assert pb['perf_counter_ns']<=lb['perf_counter_ns']<=le['perf_counter_ns']<=pe['perf_counter_ns'] and lb['layer_index']==0
 sc=pb['schedule'];assert sc['schedule_digest_sha256'].startswith(digest_prefix)
 layers=[v for v in r['layers'] if v['layer_scope_id']==scope_id and v['layer']==0 and v['scheduler_digest']==sc['schedule_digest_sha256']];assert len(layers)==1
 groups=[g for g in layers[0]['operator_groups'] if g['operator_name']=='DispatchFFNCombine'];assert len(groups)==1 and len(groups[0]['tasks'])==1
 t=groups[0]['tasks'][0];row=t['csv'];assert str(t['host_pid'])=='752' and str(t['host_tid'])=='752'
 m=int(row['Input Shapes'].strip('"').split(';')[0].split(',')[0]);sched=sc['scheduled_total'];assert type(sched) is int
 req=sc['requests'];assert isinstance(req,list) and all(type(z['scheduled_tokens']) is int for z in req)
 result.append({'digest_prefix':digest_prefix,'runner_call':runner_id,'layer_call':scope_id,'host_scope_nested':True,'runner_request_count':len(req),'scheduler_recorded_total':sched,'sum_anonymous_scheduled_tokens':sum(z['scheduled_tokens'] for z in req),'mc2_stream':row['Stream ID'],'mc2_task':row['Task ID'],'mc2_device_start_us':t['device_start_us'],'mc2_recorded_duration_us':t['trace_duration_us'],'mc2_input_M':m,'eight_times_M':8*m,'difference_eight_times_M_minus_scheduler_total':8*m-sched,'mc2_output_shapes':row['Output Shapes'],'mc2_output_dtypes':row['Output Data Types']})
terms=['expert_token_nums','expert_tokens','local_token_per_expert','topk_ids','routed_experts','expert_load','num_actual_tokens','x_active_mask','router_output'];lines=log.read_text(errors='replace').splitlines()
counts={t:sum(t in line.lower() for line in lines) for t in terms}
print(json.dumps({'source_sha256':hashes,'results':result,'service_log_exact_term_counts':counts,'service_log_scope':'one captured diagnostic runtime service.log; line counts only','limits':['scheduler totals describe host-side captured scheduling, not actual valid rows, expert distribution, device work or throughput','shape arithmetic and host scope nesting do not establish equal routing, actual padding, cross-peer counts or tensor identity','selected device task association follows existing trace flow; no cross-clock runtime dependency is inferred','service log term searches cannot rule out different naming, other logs or records; no request identifiers or tensor values are emitted']},ensure_ascii=False,indent=2))
