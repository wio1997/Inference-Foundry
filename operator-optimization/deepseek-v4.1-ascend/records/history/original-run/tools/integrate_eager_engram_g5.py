# -*- coding: utf-8 -*-
"""Audit data/control graph and integrate existing-profile evidence."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import csv,datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads((T/p).read_text())
def write(p,x):(T/p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
g=read('graphs/source-templates-G5.json');assert not g['whole_model_complete'];ts={x['id']:x for x in g['templates']}
for t in g['templates']:
 ids=[n['id'] for n in t['nodes']];assert len(ids)==len(set(ids));ids=set(ids)
 for x in t['edges']:assert x['producer'] in ids and x['consumer'] in ids
E=ts['ENGRAM'];D=ts['DSPARK_BOUNDARIES'];ep={(x['producer'],x['consumer']):x for x in E['edges']};dp={(x['producer'],x['consumer']):x for x in D['edges']}
for x in [('hash','hash_indices_select'),('hash_indices_select','hash_gather'),('hash_gather','hash_layer_select'),('hash_layer_select','lookup'),('mask_derive','hash'),('mask_derive','gate'),('rows_collective_pad','rows')]:assert x in ep
assert ep[('ready','rows')]['dependency_kind']=='HAPPENS_BEFORE_IF_EVENT' and ep[('mask_ready','gate')]['dependency_kind']=='HAPPENS_BEFORE_IF_EVENT'
for x in [('next_token_ids','seed_copy'),('next_token_ids','input_expand'),('collapse','sp_gather_slice'),('sp_gather_slice','sample_select_pad'),('sample_indices','sample_select_pad'),('sample_select_pad','norm'),('stack_probs','probability_output')]:assert x in dp
for i in range(5):
 for x in [('seed' if i==0 else f'sample_{i-1}',f'embed_{i}'),(f'embed_{i}',f'bias_{i}'),(f'bias_{i}',f'add_{i}'),('logits',f'add_{i}'),(f'add_{i}',f'sample_{i}')]:assert x in dp
r=read('records.json')
for c in r['fusion_candidates']:
 if c.get('candidate_graph_path') and c.get('candidate_graph_sha256'):assert sha(c['candidate_graph_path'])==c['candidate_graph_sha256']
for fid in ['F24','F27','F28','F30']:assert next(c for c in r['fusion_candidates'] if c['id']==fid)['candidate_graph_path'].endswith('-G5.json')
s=read('eager-target-attribution-v2/summary.json');assert s['ranks']==16 and s['all_windows']==2560
assert s['states']=={'ATTENTION_ROLE_CONSISTENT_ORDINAL_LAYER_HYPOTHESIS':1840,'NO_EXPORTED_ATTENTION_UNRESOLVED':720}
assert all(x['ordinal_layer_offsets_consistent_with_roles']==[0] for x in s['rank_audit'])
metrics=list(csv.DictReader((T/'eager-target-attribution-v2/shape-layer-metrics.csv').open()));assert all(abs(float(Fraction(x['aic_mac_cycle_proxy_numerator'])/Fraction(x['aic_cycles_denominator']))-float(x['counter_weighted_mac_ratio']))<1e-12 for x in metrics if x['aic_cycles_denominator']!='0')
case=[x for x in metrics if x['physical_mc2_input_rows']=='834'];assert len(case)==40 and all(int(x['rank_count'])==16 and int(x['calls'])==16 for x in case)
low=min(case,key=lambda x:float(x['counter_weighted_mac_ratio']));assert low['hypothesized_layer']=='0'
e=read('engram-kernel-profile/summary.json');assert len(e['ranks'])==16 and all(x['all_tasks_unique_launch_and_pmu'] for x in e['ranks'])
counts={k:v['duration_us']['n'] for k,v in e['all_rank_event_statistics'].items()};assert sorted(counts.values())==[128,128,128,256]
now=datetime.datetime.now(datetime.timezone.utc).isoformat();o={'timestamp':now,'status':'PROFILE_EVIDENCE_SCOPED_NOT_GLOBAL_COMPLETE','g5_graph_sha256':sha('graphs/source-templates-G5.json'),'eager_windows':2560,'role_consistent_windows':1840,'no_attention_unresolved_windows':720,'candidate_ordinal_layer_offsets_per_rank':'Only shift 0 fits checked attention/compression/indexer roles for each of 16 ranks; inferred role, not explicit layer labels','m834_role_consistent_lowest':low,'engram_named_task_counts':counts,'limits':['Eager windows selected by model/stream and source-marker sequence; layer ID inferred, not explicit.','Indexer detection within attention windows by time is not a proof of direct dependency.','No-attention cases may be zero-query or another path; no metadata proof.','M=834 lowest among the 40 visible layer groups for that physical shape and this proxy only; 16 calls per group.','Engram kernel names and launch IDs cover neither full module nor TP gather/event critical path.','Structural G5 audit checks data consistency, not semantic exhaustion or performance benefit.']}
write('evidence/eager-engram-g5-integration.json',o)
a=read('module-analysis.json');a['supplementary_eager_target']={'summary_path':'eager-target-attribution-v2/summary.json','metrics_path':'eager-target-attribution-v2/shape-layer-metrics.csv','scope':'partial role-consistent same-stream eager target windows; group by physical M','no_exported_attention_unresolved':720,'layer_ids_inferred_not_explicit':True};a['supplementary_engram_kernels']={'summary_path':'engram-kernel-profile/summary.json','scope':'four exact profiler kernel names; not full module'};write('module-analysis.json',a)
p=read('profile-summary.json');p['supplementary_eager_target']=a['supplementary_eager_target'];p['supplementary_engram_kernels']=a['supplementary_engram_kernels'];assert len(json.dumps(p).encode())<32768;write('profile-summary.json',p)
s2=read('STATE.json');s2.update(last_update=now,checkpoint_status='EAGER_ENGRAM_G5_DIAGNOSED_PARTIALLY',module_diagnosis_status='CAPTURED_TARGET_PLUS_EAGER_TARGET_AND_POOLED_DRAFT_PARTIAL',next_read_only_action='Investigate no-attention eager target metadata and Engram collective/event ownership; remaining fusion consumers and physical hardware bounds unknown.');write('STATE.json',s2)
with (T/'REPORT.md').open('a') as f:f.write('''\n\n## eager target 与 Engram 的补充归因\n\n已有 trace 中提取了 16 个 rank、2560 个 eager target MoE 窗口：1840 个有同 stream 的 attention，并与压缩输入及 indexer 层角色一致；720 个没有观察到 attention，具体分支未知。对每个 rank 穷举 40 个层序号偏移，仅偏移 0 与可检查的角色一致。这加强了层序号推测，但不等同于 profiler 显式层标签，也不能由物理 M 推断 prefill。\n\n例如物理 M=834 的 40 个角色一致层组，每组 16 个调用；按相同 stream 的 AIC 周期加权 MAC 计数代理，推测第 0 层约 16.61%，在该组最低。它仅是该形状和采样范围的排名，不是整个框架的最低利用率；窗口平均约 11.19 毫秒也不能直接当成端到端暴露时间。完整数据与 720 个未归属窗口保存在 `eager-target-attribution-v2/`。\n\n另外通过精确任务名及 TASK→CANN launch 关联，识别 16 个 rank 共 640 个 Engram hash/lookup 内核：请求索引、缓存写入、哈希各 128 次，host-UVA 查询 256 次。它们的采样中位耗时分别约 6.059、13.069、127.127、12.160 微秒；采样混合形状及阶段，且可能互相重叠，不能将中位数相加或当作完整 Engram 模块耗时。TP gather、事件等待和用户可见的关键路径仍未归属。详见 `engram-kernel-profile/summary.json`。\n''')
print(json.dumps({'status':'SCOPED_INTEGRATION_PASS','eager_windows':2560,'role_consistent':1840,'role_missing':720,'engram_tasks':sum(counts.values()),'m834_lowest_layer':low['hypothesized_layer'],'m834_proxy':low['counter_weighted_mac_ratio']}))
