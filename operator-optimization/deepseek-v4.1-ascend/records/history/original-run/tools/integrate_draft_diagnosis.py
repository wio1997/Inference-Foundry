"""Integrate strict and hypothesis-only draft observations into the task."""
from pathlib import Path
from decimal import Decimal
from collections import Counter
import csv,datetime,hashlib,json,statistics
T=Path(__file__).resolve().parents[1]
def read(n):return json.loads((T/n).read_text())
def write(n,v):
 p=T/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(n):return hashlib.sha256((T/n).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
strict=read('eager-draft-mainstream-attribution-v2/summary.json')
sensitivity=read('eager-draft-mainstream-attribution-v5/summary.json')
mc2=read('draft-mc2-analysis/summary.json')
assert strict['draft_call_count']==162 and mc2['valid_calls']==162
all_examined=sum(x['draft_mc2_examined'] for x in strict['rank_audit']);assert all_examined==288
rejected=sum((Counter(x['rejected']) for x in strict['rank_audit']),Counter());assert rejected=={'moe_1_attention_0':126}
rows=list(csv.DictReader((T/'eager-draft-mainstream-attribution-v2/module_metrics.csv').open()))
draft=[r for r in rows if r['kind']=='EAGER_DRAFT_POOLED'];rankset={r['rank'] for r in draft};assert len(rankset)==15
def aggregate(items):
 n=sum((Decimal(x['aic_mac_proxy_numerator']) for x in items),Decimal(0));d=sum((Decimal(x['aic_cycles_denominator']) for x in items),Decimal(0))
 return {'numerator':str(n),'denominator':str(d),'ratio':float(n/d),
         'rank_count':len(items),'calls':sum(int(x['calls']) for x in items),
         'mean_window_us':sum(int(x['window_ns_sum']) for x in items)/sum(int(x['calls']) for x in items)/1000}
groups=[]
for name in sorted({r['module'] for r in rows}):
 rr=[r for r in rows if r['module']==name and r['rank'] in rankset]
 if len(rr)==15:groups.append({'module':name,**aggregate(rr)})
groups.sort(key=lambda x:x['ratio']);assert groups[0]['module']=='draft.decoder_block.pooled'
draftrow=groups[0];target=next(x for x in groups if x['module']=='target.layers.0.forward')
valid=sensitivity['draft_call_count'];validation=sensitivity['draft_attention_validation_counts'];assert sum(validation.values())==valid
sen_draft=next((x for x in sensitivity['ranked_groups'] if x['kind']=='EAGER_DRAFT_POOLED'),None)
sen_target=next(x for x in sensitivity['ranked_groups'] if x['module']=='target.layers.0.forward')
source_path='sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py'
source_cp='sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py'
doc={'status':'PARTIAL_STRICT_AND_SEPARATE_CONDITIONAL_SENSITIVITY',
 'metric':'Reported AIC-cycle-weighted MAC ratio, same main stream only; not whole-NPU utilization/MFU.',
 'strict':{'candidate_mc2_calls':all_examined,'strictly_attributed_calls':strict['draft_call_count'],
           'rank_count':len(rankset),'excluded_calls':all_examined-strict['draft_call_count'],
           'excluded_reason':'No SparseFlashMla in the attention bracket; local query count not directly recorded.',
           'draft_pooled_3block':draftrow,'target_layer0_matching_15ranks':target,
           'ranked_groups':groups,'strict_moe_component':mc2},
 'conditional_sensitivity':{'candidate_mc2_calls':all_examined,'accepted_calls':valid,'unclassified_calls':all_examined-valid,
           'attention_validation_counts':validation,'draft_pooled_3block':sen_draft,'target_layer0_all16ranks':sen_target,
           'empty_query_status':'HYPOTHESIS: source permits zero-local-query without SparseFlashMla and with global KV work; exact per-call local query metadata and stream ownership not directly observed.'},
 'source':{'zero_query_branch':source_path+':692-719','cp_global_query_split':source_cp+':123-150,185-223',
           'source_and_task_attribution':'eager-draft-mainstream-attribution-v2/summary.json',
           'rejected_details':'eager-draft-mainstream-attribution-v2/summary.json',
           'remaining_pattern_examples':'draft-missing-attention-inspection-v3/summary.json'},
 'limitations':['162 strict calls are a selected subset, not a representative sample of all288.',
                'Three draft blocks pooled and some ranks contribute fewer strict calls; 15-rank weighted comparison does not rank individual blocks or the whole framework.',
                'Hypothesized empty-query samples are not promoted to proven attribution; auxiliary-stream events are not counted as draft work.',
                'Profiler observer effect, fixed-rank timing uncertainty, expert assignments and exposed end-to-end time remain unknown.']}
assert len(json.dumps(doc).encode())<32768
write('evidence/draft-module-diagnosis.json',doc)
analysis=read('module-analysis.json')
analysis['status']='PARTIAL'
draftid='draft.decoder_block.pooled'
analysis['modules']=[m for m in analysis['modules'] if m.get('module_id')!=draftid]
analysis['modules'].append({'module_id':draftid,'case_id':'EAGER_DRAFT_STRICT_NONEMPTY_SMLA_15RANK_SUBSET',
 'executed':True,'call_count':draftrow['calls'],'rank_count':draftrow['rank_count'],
 'metrics':[{'name':'CSV_REPORTED_AIC_CYCLE_WEIGHTED_MAC_RATIO_MAIN_STREAM_ONLY','numerator':draftrow['numerator'],
             'denominator':draftrow['denominator'],'value':draftrow['ratio'],'unit':'dimensionless_reported_counter_proxy',
             'evidence_type':'DERIVED_FROM_MEASURED_REPORT','not_whole_device_or_MFU':True}],
 'mean_window_us':draftrow['mean_window_us'],'attribution_status':'SOURCE_INVARIANT_STRICT_SUBSET_ONLY',
 'conditional_sensitivity_path':'evidence/draft-module-diagnosis.json','critical_path_exposed_time':None,
 'source_ids':['DRAFT_STRICT_ATTRIBUTION','DRAFT_MC2_COMPONENT']})
analysis['rankings']=[r for r in analysis['rankings'] if r.get('id') not in ['DRAFT_STRICT_MAINSTREAM_15RANK','DRAFT_CONDITIONAL_MAINSTREAM_16RANK']]
analysis['rankings'].append({'id':'DRAFT_STRICT_MAINSTREAM_15RANK','case_ids':['PROFILE_DRAFT_WINDOWS_STRICT','CAPTURED_TARGET_INTERIOR'],
 'metric_name':'CSV_REPORTED_AIC_CYCLE_WEIGHTED_MAC_RATIO_MAIN_STREAM_ONLY',
 'comparison_contract':'Same 15 ranks and kernel main-stream-only counter aggregation; dissimilar sample counts and target/draft workloads; draft group pools3 layers.',
 'ordered_module_groups':[[g['module']] for g in groups], 'lowest_module_ids':[groups[0]['module']],
 'coverage_statement':'Strict162/288 draft MC2 candidates;15/16 ranks; other modules unranked.',
 'status':'PARTIAL_STRICT_WITH_SELECTION_BIAS','global_full_framework_minimum_known':False})
if sen_draft is not None:
 analysis['rankings'].append({'id':'DRAFT_CONDITIONAL_MAINSTREAM_16RANK','case_ids':['PROFILE_DRAFT_WINDOWS_WITH_EMPTY_QUERY_HYPOTHESIS','CAPTURED_TARGET_INTERIOR'],
  'metric_name':'CSV_REPORTED_AIC_CYCLE_WEIGHTED_MAC_RATIO_MAIN_STREAM_ONLY',
  'comparison_contract':'All16 ranks but some draft calls rely on source-consistent yet unproven zero-local-query classification; module instances pooled; different sample counts.',
  'ordered_module_groups':[[g['module']] for g in sensitivity['ranked_groups']],
  'lowest_module_ids':[sensitivity['ranked_groups'][0]['module']],
  'coverage_statement':f'{valid}/{all_examined} draft candidates; {validation}; not all framework modules.',
  'status':'HYPOTHESIS_SENSITIVITY_ONLY','global_full_framework_minimum_known':False})
analysis['coverage']['missing_module_ids']=[x.replace('Eager3block DSpark','Individual eager DSpark block identities and complete activity') for x in analysis['coverage']['missing_module_ids']]
analysis['coverage']['supplementary_pooled_draft_coverage']={'strict':f'{strict["draft_call_count"]}/{all_examined}, 15ranks',
 'conditional':f'{valid}/{all_examined}, {len(validation)} attribution classes; inferential samples retained separately'}
analysis['performance_comparisons']=[x for x in analysis['performance_comparisons'] if x['object_id']!=draftid]
analysis['performance_comparisons'].append({'object_id':draftid,'object_kind':'draft_blocks_pooled_partial_sample',
 'case_id':'EAGER_DRAFT_STRICT_NONEMPTY_SMLA_15RANK_SUBSET','measured_time_T':str(draftrow['mean_window_us']/1e6),
 'time_unit':'second','effective_work_Q':None,'time_lower_L':None,'throughput_upper_U':None,
 'bound_status':'SYMBOLIC_CEILING','scope_match_status':'UNKNOWN',
 'reason':'Pooled variable shapes, valid tokens, expert routing and all-resource capacities unproven.',
 'utilization_metric_reference':draftrow['ratio'],'unknown_ids':['U_VALID_ROUTING_AND_PADDING','U_HARDWARE_CAPACITY','U_MIXED_RESOURCE_CAPACITY'],
 'candidate_only':False})
write('module-analysis.json',analysis)

records=read('records.json')
newsrc={'DRAFT_STRICT_ATTRIBUTION':'eager-draft-mainstream-attribution-v2/summary.json',
        'DRAFT_MC2_COMPONENT':'draft-mc2-analysis/summary.json',
        'DRAFT_QUERY_PATTERN':'draft-missing-attention-inspection-v3/summary.json'}
for sid,path in newsrc.items():
 if not any(x['id']==sid for x in records['sources']):records['sources'].append({'id':sid,'url_or_path':path,'snapshot_path':path,'sha256':sha(path),'registered_at':now})
if not any(x['id']=='U_DRAFT_EMPTY_QUERY' for x in records['unknowns']):
 records['unknowns'].append({'id':'U_DRAFT_EMPTY_QUERY','statement':'Per-call local query count was not exported; absent SparseFlashMla and a source-consistent zero/concat trace do not prove the empty-query branch or individual draft block identity.'})
records['module_causes']=[c for c in records['module_causes'] if c['id']!='C_DRAFT_POOLED_LOW']
records['module_causes'].append({'id':'C_DRAFT_POOLED_LOW','module_id':draftid,'observed_graph_id':'G1_REFINED_MODULE_GRAPH',
 'affected_cases':['EAGER_DRAFT_STRICT_NONEMPTY_SMLA_15RANK_SUBSET'],
 'observed_symptom':f"Strict pooled draft proxy {100*draftrow['ratio']:.4f}% vs target layer0 {100*target['ratio']:.4f}% for matching15 ranks; strict sample162/288; draft block identities pooled.",
 'mechanism':'Fused MC2 accounts for a large part of observed bracket time and has low MAC counter ratio; cross-rank waiting, expert assignment imbalance, small M, data movement and profiling effects remain competing explanations.',
 'supporting_evidence':['evidence/draft-module-diagnosis.json','draft-mc2-analysis/summary.json','evidence/MC2-known-dependencies.md'],
 'counter_evidence':['Omitted calls may correspond to different empty-query or other patterns; strict selection and unequal call counts bias the aggregate.',
                     'No causal intervention or internal MC2 wait/expert-token event is available.'],
 'missing_evidence':['Per-call local query count and complete context/stream ownership','Per-expert token/work distribution and internal MC2 readiness events',
                     'Individual draft layer identifiers and all-stream attribution','Unprofiled end-to-end exposed time'],
 'status':'HYPOTHESIS_WITH_MEASURED_COMPONENT','confidence_and_reason':'Strict evidence supports descriptive component; relative cause not isolated.',
 'end_to_end_impact':None})
write('records.json',records)
analysis['cause_ids']=list(dict.fromkeys(analysis['cause_ids']+['C_DRAFT_POOLED_LOW']));write('module-analysis.json',analysis)
profile=read('profile-summary.json');profile['supplementary_draft']={'source':'evidence/draft-module-diagnosis.json',
 'strict_draft_ratio':draftrow['ratio'],'strict_target0_matching_rank_ratio':target['ratio'],
 'strict_calls':strict['draft_call_count'],'candidate_calls':all_examined,'strict_ranks':15,
 'conditional_draft_ratio':sen_draft['ratio'] if sen_draft else None,'conditional_calls':valid,
 'global_lowest_known':False,'proxy_not_device_utilization':True}
assert len(json.dumps(profile).encode())<32768;write('profile-summary.json',profile)
index=read('profile-index.json')
for id,path,count in [('DRAFT_STRICT_MAINSTREAM','eager-draft-mainstream-attribution-v2/module_metrics.csv',len(rows)),
                      ('DRAFT_EMPTY_QUERY_SENSITIVITY','eager-draft-mainstream-attribution-v5/module_metrics.csv',None),
                      ('DRAFT_STRICT_MC2','draft-mc2-analysis/strict_draft_calls.csv',162)]:
 if not any(x.get('id')==id for x in index['aggregate_tables']):
  index['aggregate_tables'].append({'id':id,'path_or_table':path,'row_count':count,'sha256':sha(path),
                                   'full_scope_coverage':'Strict partial or explicitly conditional sample only; no auxiliary-stream attribution.'})
write('profile-index.json',index)

report=T/'REPORT.md';s=report.read_text();s+='''

## 后续只读追查：eager 草稿层的低活动样本

此前 40 层排名只覆盖捕获的 target decoder。根据 8 个本地专家、top-3 路由、四个 mHC 标记和一条 SparseFlashMla 组成的约束，从两段前向间隔中严格归属到 162 次 eager 草稿层调用，覆盖 15 个 rank；另 126 次不满足 SparseFlashMla 约束，其中一个 rank 全部未被严格归属。三个草稿 block 暂时只能合并，不知道每次属于哪一个。

用**主流上的同一 MAC 周期代理指标**，这 162 个草稿调用合并约 '''+f"{100*draftrow['ratio']:.4f}%"+'''，对应的相同 15 个 rank 的 target 第 0 层约 '''+f"{100*target['ratio']:.4f}%"+'''。这提示草稿部分也可能是低活动区域，但调用数、输入形状和模块组成不相同，严格样本存在明显选择偏差；不能称它为全框架最低模块，也不能把它与此前包含更多流的 3.6402% 直接混为同一指标。

在这 162 次严格样本里，MC2 的 MAC 代理值约 '''+f"{100*mc2['mc2_mac_counter_proxy']:.4f}%"+'''；MC2 平均约 '''+f"{mc2['mc2_mean_us']:.3f}"+''' 微秒、中位数约 '''+f"{mc2['mc2_median_us']:.3f}"+''' 微秒，占这些模块窗口耗时和的约 '''+f"{100*mc2['sum_mc2_time_over_sum_bracket_time']:.2f}%"+'''。这是采样和，不是端到端关键路径；无法据此区分通信等待、小矩阵、专家负载不均、访存及 profiler 扰动。

源码允许本地 query 数为零时跳过 SparseFlashMla，同时全局 KV 准备仍可能执行。抽样窗口与此一致，因此单独做了条件敏感性统计：'''+str(validation)+'''，共 '''+str(valid)+'''/288 次，'''+('16 rank 草稿合并指标约 '+f"{100*sen_draft['ratio']:.4f}%" if sen_draft else '尚未得到覆盖 16 rank 的组间值')+'''。其中空 query 只是通过窗口形态推断；实际每次 query 数、跨流事件的因果归属未直接记录。剩余缺口和完整分子分母见 `evidence/draft-module-diagnosis.json`。实测不能覆盖的模型模块仍保留 UNKNOWN。
''';report.write_text(s)
state=read('STATE.json');state.update(current_phase='P6',last_update=now,
 module_diagnosis_status='PARTIAL_CAPTURED_TARGET_PLUS_POOLED_DRAFT',
 supplemental_lowest_groups=[{'metric':'MAIN_STREAM_AIC_MAC_PROXY','scope':'15rank strict subset','module':draftid,'value':draftrow['ratio']}],
 next_read_only_action='Continue evidence-driven module coverage and causal analysis; preserve strict versus inferred draft samples and sync records.',overall_work_paused=False)
write('STATE.json',state)
print(json.dumps({'strict_draft_ratio':draftrow['ratio'],'same15rank_target0_ratio':target['ratio'],
 'strict_calls':162,'examined_calls':288,'inferred_sensitivity_counts':validation,
 'conditional_draft_ratio':sen_draft['ratio'] if sen_draft else None,
 'globally_lowest_module_proven':False,'no_target_implementation_modified':True},indent=2))
