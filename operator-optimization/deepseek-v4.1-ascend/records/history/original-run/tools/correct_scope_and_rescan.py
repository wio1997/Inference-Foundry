"""Correct reviewed scope errors and expand finite structural coverage.

Source and G0/G1 remain immutable. Candidate topology is analysis only. A
structural enumeration is not a proof of fusion validity or exhaustiveness.
"""
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import copy,datetime,hashlib,itertools,json,subprocess,sys

T=Path(__file__).resolve().parents[1];K=T.parents[1]
def read(p):return json.loads((T/p).read_text())
def write(p,v):
 f=T/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
saved=T/'evidence/pre-independent-review';assert not saved.exists();saved.mkdir()
for n in ['records.json','fusion-scan.json','module-analysis.json','contract.json','REPORT.md','evidence/IDEAL-MODEL.md','evidence/conditional-performance-comparisons.json','evidence/conditional-nominal-capacity.json']:
 p=saved/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((T/n).read_bytes())

old=read('graphs/source-templates.json');new=copy.deepcopy(old)
new.update(id='SOURCE_TEMPLATES_G2_REVIEW_CORRECTIONS',parent='graphs/source-templates.json',whole_model_complete=False,
           scope='Partial source-boundary templates; external consumers, aliases, dynamic branches and inner kernels still incomplete.')
ts={x['id']:x for x in new['templates']}
def node(t,id,kind,**kw):
 assert all(x['id']!=id for x in t['nodes']);t['nodes'].append({'id':id,'kind':kind,'resource':'UNKNOWN',**kw})
def edge(t,producer,consumer,value,**kw):
 assert producer in {n['id'] for n in t['nodes']} and consumer in {n['id'] for n in t['nodes']}
 t['edges'].append({'producer':producer,'consumer':consumer,'value':value,'dependency_kind':'RAW',**kw})
mhc=ts['MHC_BLOCK_AND_CROSS_LAYER']
edge(mhc,'ffn_post','next_pre','prior HC residual',condition='next-layer Engram does not modify this residual',source='models/deepseek_v41/model.py:919-920,1556-1592')
mhc['branch_note']='Direct residual edge applies only when next-layer Engram does not modify it; separate Engram path and optional pre-Engram aux mean must preserve order and aliasing.'
for tid in ['ATTN_C2_SOURCE','ATTN_C1_SOURCE','ATTN_C1_INDEX_SOURCE']:
 t=ts[tid]
 for name in ['index_cache_history','index_metadata','candidate_state_and_lengths']:
  node(t,name,'EXTERNAL_SHARED_STATE',coverage='PARTIAL_VALUE_AND_ALIAS_DETAIL_UNKNOWN')
  edge(t,name,'qli',name,condition='candidate state used only in enabled source/filter branches; exact valid ranges unknown',source='attention/dsa_v41.py:568-611')
 node(t,'index_output_consumers','EXTERNAL_OUTPUT_OR_PERSISTENT_STATE')
 edge(t,'indices_store','index_output_consumers','stable selected indices and downstream/shared uses',condition='future consumers and lengths not exhaustively expanded')
 if any(n['id']=='ik_store' for n in t['nodes']):
  node(t,'key_state_consumers','EXTERNAL_OUTPUT_OR_PERSISTENT_STATE')
  edge(t,'ik_store','key_state_consumers','persistent key cache writes',condition='future cache consumers and aliasing not exhaustively expanded')
  edge(t,'scale_store','key_state_consumers','persistent index scale writes',condition='future cache consumers and aliasing not exhaustively expanded')
  edge(t,'cmp_store','key_state_consumers','persistent compressed KV writes',condition='future cache consumers and aliasing not exhaustively expanded')
moe=ts['MOE_WRAPPER'];node(moe,'returned_moe_output','EXTERNAL_OUTPUT')
edge(moe,'merge','returned_moe_output','final_hidden_states',source='models/deepseek_v41/model.py:300-326,913-919',condition='merge branch as modeled; exact reductions and output alias require separate review')
kv=ts['DSPARK_SHARED_KV'];node(kv,'persistent_swa_state','EXTERNAL_OUTPUT_OR_PERSISTENT_STATE')
edge(kv,'scatter','persistent_swa_state','context KV cache state',source='models/deepseek_v41/dspark.py:284-309',condition='slot mapping and future readers are retained; exact consumer set unknown')
d=ts['DSPARK_BOUNDARIES'];removed=[e for e in d['edges'] if e['producer']=='main_norm' and e['consumer']=='block0'];assert len(removed)==1
d['edges']=[e for e in d['edges'] if e not in removed]
d['unresolved']='Previous direct main_norm->block0 edge was unsupported and removed. Context projection, KV preparation and graph runner binding remain UNPROVEN; forward embeds tokens and calls each layer.'
write('graphs/source-templates-G2.json',new)

original_index=read('candidates/structural-index.json');assert original_index['count']==928
def enumerate_templates():
 structural=[];seen=set()
 for d in new['templates']:
  nodes={n['id']:n for n in d['nodes']};eligible={i for i,n in nodes.items() if not n['kind'].startswith('EXTERNAL')};adj=defaultdict(set)
  for ed in d['edges']:
   a,b=ed['producer'],ed['consumer']
   if a in eligible and b in eligible:adj[a].add(b);adj[b].add(a)
  sets={frozenset([i]) for i in eligible}
  for size in [2,3,4]:
   sets={s|{v} for s in sets for v in set().union(*(adj[x] for x in s))-s}
   for s in sorted(sets,key=lambda s:tuple(sorted(s))):
    fp=hashlib.sha256(json.dumps([d['id'],sorted(s)]).encode()).hexdigest()
    if fp in seen:continue
    seen.add(fp);structural.append({'id':'G2_STRUCT_'+fp[:16],'template_id':d['id'],'node_ids':sorted(s),
        'category':{2:'PAIR',3:'TRIPLE',4:'MULTI_OPERATOR'}[size],
        'external_inputs':[e for e in d['edges'] if e['consumer'] in s and e['producer'] not in s],
        'external_outputs':[e for e in d['edges'] if e['producer'] in s and e['consumer'] not in s],
        'status':'STRUCTURAL_ONLY','consumer_completeness':'UNKNOWN','equivalence_and_savings_proven':False})
  for src in nodes:
   users=sorted({e['consumer'] for e in d['edges'] if e['producer']==src and e['consumer'] in eligible})
   for a,b in itertools.combinations(users,2):
    fp=hashlib.sha256(json.dumps([d['id'],'SHARED_INPUT',src,a,b]).encode()).hexdigest()
    if fp in seen:continue
    seen.add(fp);structural.append({'id':'G2_STRUCT_'+fp[:16],'template_id':d['id'],'node_ids':[a,b],'shared_input':src,
        'category':'BRANCH_MULTICONSUMER','status':'STRUCTURAL_ONLY','consumer_completeness':'UNKNOWN','equivalence_and_savings_proven':False})
 return structural
items=enumerate_templates();items2=enumerate_templates();assert [x['id'] for x in items]==[x['id'] for x in items2]
write('candidates/structural-index-G2.json',{'scope':'12 revised partial source templates, connected2-4 and shared-input pairs',
      'count':len(items),'categories':dict(Counter(i['category'] for i in items)),
      'second_pass_new':0,'meaning_of_second_pass':'Re-enumeration only; not semantic candidate expansion or proof of exhaustion',
      'items':items})

# Cross-module graph sweep is based on G1 module/composite graph, not on full
# primitive lowering. Exclude only the declared next-invocation recurrence.
g=read('graphs/G1.json');nodes={n['id']:n for n in g['nodes']};edges=[e for e in g['edges'] if e.get('dependency_kind')!='NEXT_INVOCATION_RAW']
adj=defaultdict(set)
for e in edges:
 a,b=e['producer'],e['consumer'];assert a in nodes and b in nodes;adj[a].add(b);adj[b].add(a)
front={frozenset([x]) for x in nodes};cross=[];limit=150000;truncated=False;by_size={}
for size in [2,3,4]:
 nxt=set()
 for s in front:
  for v in set().union(*(adj[x] for x in s))-s:
   nxt.add(s|{v})
   if len(nxt)>limit:truncated=True;break
  if truncated:break
 if truncated:by_size[str(size)]={'status':'TRUNCATED','observed_at_least':len(nxt)};break
 front=nxt;selected=[s for s in front if len({nodes[x].get('module') for x in s})>1]
 by_size[str(size)]={'status':'ENUMERATED','all_connected_sets':len(front),'cross_module_sets':len(selected)}
 for s in sorted(selected,key=lambda s:tuple(sorted(s))):
  fp=hashlib.sha256(json.dumps(sorted(s)).encode()).hexdigest()
  cross.append({'id':'G1_CROSS_'+fp[:16],'nodes':sorted(s),'modules':sorted({str(nodes[x].get('module')) for x in s}),
                'internal_edges':[e for e in edges if e['producer'] in s and e['consumer'] in s],
                'external_inputs':[e for e in edges if e['consumer'] in s and e['producer'] not in s],
                'external_outputs':[e for e in edges if e['producer'] in s and e['consumer'] not in s],
                'status':'STRUCTURAL_ONLY','full_primitive_or_alias_coverage':False,'savings_proven':False})
write('candidates/cross-module-G1.json',{'scope':'G1 known composite graph; undirected connected sets2-4 crossing declared module labels; current branch and inputs incomplete',
      'size_coverage':by_size,'enumeration_cap':limit,'truncated':truncated,'count':len(cross),'items':cross})

records=read('records.json');byid={c['id']:c for c in records['fusion_candidates']}
for c in records['fusion_candidates']:
 c['external_consumers_complete']=False
 c['semantic_equivalence_argument']='Current file is a source-template boundary sketch; graph omits some real outputs/consumers, state aliases or runner bindings. Full equivalence and any materialization saving are UNPROVEN.'
 c['original_template_review_status']='HISTORICAL_PARTIAL_GRAPH_NOT_A_COMPLETE_SEMANTIC_CONTRACT'
for fid in ['F01','F04','F13']:
 c=byid[fid];c.update(status='RULED_OUT_IN_SCOPE',implemented=True,new_fusion_opportunity=False,
   reason='This named primitive is already fused in the frozen implementation; it is not counted as a new standalone fusion opportunity.')
for fid,tid,selection in [('F03','MHC_BLOCK_AND_CROSS_LAYER',['ffn_post','next_pre']),
                          ('F23','MOE_WRAPPER',['shared_down','dispatch_ffn_combine','merge']),
                          ('F29','DSPARK_SHARED_KV',['proj','norm','rope','cat','slots','scatter'])]:
 t=ts[tid];sel=set(selection)
 obj={'id':fid+'_G2','original_template_graph_id':tid,'source_template_path':'graphs/source-templates-G2.json',
      'nodes':[n for n in t['nodes'] if n['id'] in sel],
      'internal_edges':[e for e in t['edges'] if e['producer'] in sel and e['consumer'] in sel],
      'external_inputs':[e for e in t['edges'] if e['consumer'] in sel and e['producer'] not in sel],
      'external_outputs':[e for e in t['edges'] if e['producer'] in sel and e['consumer'] not in sel],
      'consumer_completeness':'UNKNOWN','precision_and_alias_conditions':'Original precision/rounding, every alias, persistent write and consumer remain required; no equivalence or gain proof.',
      'case_conditions':'Direct adjacent-layer residual path only if next Engram does not modify it; aux mean may consume the residual first.' if fid=='F03' else 'Current frozen branch and valid state/shape conditions only.'}
 path='candidates/'+fid+'-G2.json';write(path,obj)
 byid[fid].update(candidate_graph_path=path,candidate_graph_sha256=sha(path),status='CONDITIONAL',
                  original_template_review_status='PARTIALLY_CORRECTED_G2_NOT_COMPLETE',external_consumers_complete=False)
for fid in ['F27','F28','F30']:
 byid[fid].update(status='UNKNOWN',runner_binding='UNKNOWN',new_fusion_opportunity=False,
   semantic_equivalence_argument='Source shows separate token-forward and context KV preparation methods. Direct runner invocation/dataflow is not established by the frozen source snapshot; no equivalence proof.')

url='https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md'
src='sources/primary/msinsight/system_tuning.md'
assert 'aic_total_cycles' in (T/src).read_text() and 'aic_mac_ratio' in (T/src).read_text()
records['sources'].append({'id':'MSINSIGHT_COUNTER_DEFINITIONS','url_or_path':url,'version_commit':'efb28234772093b0cabc3c538e71375c957540b2',
  'git_blob_sha1':'071c0db6801be60be8505b2c3944bc04675615c9','snapshot_path':src,'sha256':sha(src),'registered_at':now,
  'chapter_symbol':'表11 AI Core性能指标字段说明；PipeUtilization collection requirements',
  'supported_claim_ids':['CL_REPORTED_MAC_RATIO_DEFINITION']})
records['claims'].append({'id':'CL_REPORTED_MAC_RATIO_DEFINITION','type':'DOCUMENTED',
 'statement':'Vendor describes aic_total_cycles as AI Core execution cycles and aic_mac_ratio as MAC time / total time. Level1+PipeUtilization enables collection. Per-kernel CSV core aggregation and cross-rank uncertainty still require care.',
 'source_ids':['MSINSIGHT_COUNTER_DEFINITIONS']})
records['reviews'].append({'id':'REVIEW_INDEPENDENT_CORRECTIONS_'+now,'reviewer':'GPT primary controller after independent technical review',
 'status':'PARTIAL_WITH_CORRECTIONS','date':now,'review_type':'SEMANTIC_SCOPE_AND_SOURCE_REVIEW',
 'reason':'Cross-module coverage, F03 residual edge, terminal outputs, historical cache inputs, DSpark runner binding, cross-request sharing and mixed-resource capacity assumptions corrected; no formal proof.'})
records['models'].append({'id':'IDEAL_FULL_SHARING_AND_MIXED_RESOURCE_DOMAIN','scope':'Frozen model semantics; ideal fusion, cross-request reuse, alternative equivalent computation and mixed hardware resources permitted',
 'necessary_work_expressions':'UNKNOWN without proof of irreducible unique work; currently do not substitute current cache-miss counts or current Cube operation counts.',
 'capacity_upper_expressions':'UNKNOWN aggregate applicable resource capacity; numeric 16*P8 is not established for arbitrary mixed resource implementations.',
 'real_to_relaxed_coverage_argument':'Wider domain contains narrow independently evaluated INT8 Cube-only domain; a bound from the narrower domain does not apply automatically.'})
for u in records['unknowns']:
 if u['id']=='U_PRIMITIVE_MAPPING':u['statement']='G1 composite and revised G2 template scans only; missing aliases, all future cache consumers, DSpark runner bindings and complete primitive lowering.'
for uid,statement in [('U_CROSS_REQUEST_REUSE','Current cache-hit counter does not prove all remaining request positions need distinct calculations in the broader allowed sharing domain.'),
 ('U_MIXED_RESOURCE_CAPACITY','INT8 Cube nominal capacity does not bound work done concurrently by BF16, vector, CPU or other resources unless the restricted execution domain or total aggregate capacity is proved.'),
 ('U_RUNNER_BINDING','DSpark combine_hidden_states/precompute_and_store_context_kv and token forward are separate source methods; actual invocation/data dependency in runner not yet mapped.')]:
 if not any(x['id']==uid for x in records['unknowns']):records['unknowns'].append({'id':uid,'statement':statement,'next_entry':'Read existing frozen input/token/runtime and runner source; do not infer from current trace counters alone.'})
for con in records['constraints']:
 if con['id']=='DENSE_INT8_COMPUTE_ONLY':
  con['scope']='Conventional dense multiplication with all counted MACs executed on the stated INT8 Cube pool, no alternate-resource computation; framework case additionally requires independent per-request/per-position target computation without additional reuse.'
  con['review_status']='CONDITIONAL_CEILING_NARROW_IMPLEMENTATION_DOMAIN'
write('records.json',records)

scan=read('fusion-scan.json')
for c in scan['coverage']:
 if c['category']=='CROSS_MODULE':c.update(status='PARTIAL_ORIGINAL_TEMPLATE_ONLY',reason='Original index had no separately identified cross-template enumeration; corrected bounded G1 enumeration is separately indexed.')
 if c['template_id'] in ['MHC_BLOCK_AND_CROSS_LAYER','DSPARK_BOUNDARIES']:
  c['original_graph_review']='INCOMPLETE_OR_UNPROVEN_EDGE_CORRECTED_IN_G2'
scan['corrected_source_templates']='graphs/source-templates-G2.json';scan['corrected_structural_index']='candidates/structural-index-G2.json'
scan['cross_module_composite_index']='candidates/cross-module-G1.json'
scan['conditional_candidate_ids']=[x['id'] for x in records['fusion_candidates'] if x['status']=='CONDITIONAL']
scan['semantic_unknown_candidate_ids']=[x['id'] for x in records['fusion_candidates'] if x['status']=='UNKNOWN']
scan['stop_reason'].update(kind='ANALYSIS_PHASE_ONLY_EVIDENCE_GAPS',
 why_not_no_more_fusions='Finite revised template and module-graph sweeps do not expand full primitives, aliases or dynamic runner state; semantic equivalence and all consumers remain unproven.',
 historical_correction='The original coverage entries overstated cross-module enumeration and omitted/assumed specific edges; retained with explicit correction.',
 overall_task_paused=False)
scan['passes'].extend([{'id':'G2_TEMPLATE_STRUCTURAL_RESCAN','new_structural_items':len(items),'duplicate_items':len(items),
 'same_graph_and_rules':True,'not_semantic_exhaustion':True},
 {'id':'G1_CROSS_MODULE_STRUCTURAL_PASS','new_structural_items':len(cross),'size_coverage':by_size,
  'truncated':truncated,'not_complete_primitive_or_semantic_graph':True}])
scan['uncovered_items'] += ['Every template candidate lacks proven complete alias/consumer coverage',
 'Original DSpark main_norm->block0 edge was unsupported; runner binding UNKNOWN',
 'Cross-module G1 sweep only uses composite graph; not all runtime primitive edges']
write('fusion-scan.json',scan)

cap=read('evidence/conditional-nominal-capacity.json')
cap['conditions'].append('All counted MACs are performed on the modeled INT8 Cube resource pool; alternative BF16, vector, CPU or other resources do not execute any of these MACs unless separately added to a proven joint capacity upper bound.')
write('evidence/conditional-nominal-capacity.json',cap)
comparisons=read('evidence/conditional-performance-comparisons.json');comparisons['capacity']=cap
current_analysis=read('module-analysis.json')
current_comparisons={x['object_id']:x for x in current_analysis['performance_comparisons']}
for c in comparisons['comparisons']:
 current=current_comparisons[c['object_id']]
 c['bound_input_path']=current['bound_input_path'];c['calculation_path']=current['calculation_path']
 c['implementation_domain'] += '; all counted MACs execute on the modeled INT8 Cube pool; no equivalent mixed-resource implementation included'
 c['scope_match_status']='CONDITIONAL_DOMAIN_MEMBERSHIP_NOT_FULLY_ESTABLISHED_FOR_OBSERVED_RUN'
 c['missing_evidence']=['That every counted MAC is executed by the modeled Cube pool; no split to other compute resources',
                        'Official nominal formula applies as a physical capacity ceiling for actual 9382 frequency domain']
 if c['object_id']=='TARGET_SELECTED_INT8_C1':
  c['implementation_domain'] += '; each requested non-cache-hit target position and each of the 8*1023 preceding generated positions is computed independently per request without extra cross-request/position reuse, memoization or alternate exact bypass'
  c['missing_evidence'].append('Current cache hit counts do not prove the irreducible number of distinct request-position computations in a sharing-capable domain')
  c['wide_sharing_domain_status']='INCOMPLETE_NO_NUMERIC_UPPER_BOUND'
  c['measured_over_ceiling_ratio_label']='Conditional arithmetic only; observed baseline membership in this narrower implementation domain not fully established.'
write('evidence/conditional-performance-comparisons.json',comparisons)
analysis=current_analysis;bycmp={x['object_id']:x for x in comparisons['comparisons']}
for i,c in enumerate(analysis['performance_comparisons']):
 if c['object_id'] in bycmp:analysis['performance_comparisons'][i]=copy.deepcopy(bycmp[c['object_id']])
analysis['performance_comparisons'].append({'object_id':'TARGET_FULL_SEMANTICS_SHARING_MIXED_RESOURCE_DOMAIN','object_kind':'framework_relaxed_domain',
 'bound_status':'INCOMPLETE','effective_work_Q':8192,'time_lower_L':None,'throughput_upper_U':None,
 'reason':'Neither number of necessary distinct computations with sharing nor aggregate applicable physical capacity over all allowed execution resources has been proved.',
 'unknown_ids':['U_CROSS_REQUEST_REUSE','U_MIXED_RESOURCE_CAPACITY','U_HARDWARE_CAPACITY'],'candidate_only':False})
analysis['metric_source_id']='MSINSIGHT_COUNTER_DEFINITIONS';write('module-analysis.json',analysis)
contract=read('contract.json');contract['utilization_contract']['vendor_metric_documentation']=url
contract['implementation_domain']['conditional_numeric_domain']='Standard dense INT8 Cube-only execution; framework number further assumes per-request/per-position target calculations and no additional reuse. Membership of the observed end-to-end run has not been fully established.'
contract['implementation_domain']['wide_ideal_domain_numeric_upper']='UNKNOWN: sharing and alternative equivalent compute resources allowed'
write('contract.json',contract)
for c in comparisons['comparisons']:
 p=T/c['bound_input_path'];data=json.loads(p.read_text());data['scope']=c['implementation_domain'];data['conditions_path']='evidence/conditional-nominal-capacity.json'
 data['terms'][0]['evidence']='Under explicitly restricted independent dense position calculation and exclusive INT8 Cube pool as applicable; observed domain membership and nominal hardware capacity are not fully proved.'
 data['broad_ideal_domain_numeric_upper']='UNKNOWN'
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 r=subprocess.run([sys.executable,str(K/'tools/calculate_bounds.py'),str(p)],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr
 result=json.loads(r.stdout);assert Fraction(result['layers'][0]['time_lower_from_known_terms']['exact_fraction'])==Fraction(c['time_lower_L']['exact_fraction'])
 write(c['calculation_path'],result)
write('bounds-input.json',read('bounds/TARGET_SELECTED_INT8_C1.input.json'))

ideal=T/'evidence/IDEAL-MODEL.md';s=ideal.read_text()
s=s.replace('固定当前常规 dense GEMM 算法族，必要 MAC 为','固定当前常规 dense GEMM 算法族，所有所计 MAC 必须由声明的 INT8 Cube 容量池执行，必要 MAC 为')
s=s.replace('此约束域要求标准 dense target 验证且不跳过上述语义工作；','此约束域要求标准 dense target 验证、全部计数 MAC 由声明的 INT8 Cube 池执行，且每个请求、每个未命中位置及生成位置都独立计算，不允许额外跨请求/位置结果复用、记忆或等价计算绕过。当前缓存命中计数只说明当前实现口径，不能证明宽松共享域需要这么多不同计算；观测基线是否严格属于该窄域也未全部核实。')
s=s.replace('是很松的条件计算上界，不是预计能达到的整机极限。','仅是这个窄域、上述物理容量假设下很松的条件计算上界；对于允许共享及不同计算资源共同执行的宽域，没有得到数值上界，也不是预计能达到的整机极限。')
ideal.write_text(s)
report=T/'REPORT.md';s=report.read_text()
s=s.replace('扫描两两、三个、四个相连节点、共享输入分支以及已知复合阶段的拆分重组/跨模块边界，枚举 928 个结构项；','扫描两两、三个、四个相连节点和共享输入分支，枚举原版 928 个结构项；原覆盖表对跨模块扫描的标注不准确，现已用 G1 复合图另做有限跨模块结构枚举；')
s=s.replace('27 个有条件候选，3 个已实现的融合（mHC pre、RMSNormCast、SparseFlashMla），1 个 metadata 路径仍未知。','原版登记 27 个有条件候选，3 个已实现的融合（mHC pre、RMSNormCast、SparseFlashMla），1 个 metadata 路径未知。复审发现部分图缺输出和状态输入，且 DSpark 的部分 runner 绑定未经证明；现在依照修正版记录将相应项标记为 UNKNOWN。')
s=s.replace('候选图在 candidates/，原始检查在 evidence/FUSION-SWEEP.md，覆盖/停止在 fusion-scan.json，重叠关系在 records.json。','候选图在 candidates/，原始检查在 evidence/FUSION-SWEEP.md，覆盖/停止在 fusion-scan.json，重叠关系在 records.json；旧图保留供追溯，不构成消费者完整性或语义等价证明。')
s=s.replace('以下独立 Wq_b 是所有 M 行输出需保留的标准 dense 算法域，M 不等于返回 token 数。','以下独立 Wq_b 是所有 M 行输出需保留、所计 MAC 全由声明的 INT8 Cube 容量池执行的标准 dense 算法域，M 不等于返回 token 数；观测样本属于该窄域仍有待核实。')
s=s.replace('对选定 target INT8 必要工作，用 16 设备容量得到的 **116294.745 输出 token/s** 只是极松的条件计算上界：忽略 attention、BF16/FP32、draft/reject、访存、通信和调度。','对选定 target INT8 工作，用 16 设备容量得到的 **116294.745 输出 token/s** 只是极松的窄域条件计算上界：除容量前提外，要求每个未命中/生成位置对每个请求独立计算、不发生额外跨请求/位置复用，且所有计数 MAC 都由该 INT8 Cube 池执行；观测基线是否完全属于该域尚未证实。这个数忽略 attention、BF16/FP32、draft/reject、访存、通信和调度。')
s=s.replace('本次已保存 7 个独立数值对象的精确输入与计算器原始输出；40 层模块和 27 个未实现候选也有对照记录，缺失项保留符号/null。','本次已保存 7 个独立数值对象的精确输入与计算器原始输出；40 层模块和原版候选也有对照记录，缺失项保留符号/null。允许跨请求共享或用其他计算资源完成等价计算的宽域没有得到数值上界。')
s += '\n## 本轮复审补充\n\n原 MHC 模板遗漏了无 Engram 修改时 `ffn_post → next_pre` 的条件残差边；原 F23 未列出 MoE 结果给后续 hc_post，F29 未列出持久 KV 状态，index source 模板漏了历史索引缓存/元数据。DSpark `main_norm → block0` 直接边没有源码证据，已从修订模板移除并保留 runner 绑定未知。修正模板为 graphs/source-templates-G2.json；F03/F23/F29 的修正版边界图保存在 candidates/，仍未证明所有消费者或语义等价。\n\n昇腾官方 [MindStudio Insight 文档]('+url+') 将 `aic_mac_ratio` 描述为 MAC 耗时占总耗时的比例，将 `aic_total_cycles` 描述为 AI Core 执行周期；原文件已按 commit 和 Git blob 哈希保存。本文按报告字段加权得到的是代理指标，采样窗口、内核归属、每核聚合及未覆盖模块的限制仍然存在。\n'
report.write_text(s)
state=read('STATE.json');state.update(current_phase='P3_P6_REVIEW_CORRECTIONS',fusion_scan_status='PAUSED_WITH_GAPS',
 next_read_only_action='Review bounded corrected cross-module enumeration and inter-forward gap evidence; persist corrected narrower theoretical assumptions and sync.',
 theoretical_bound_status='CONDITIONAL_CEILING',last_update=now,overall_work_paused=False)
write('STATE.json',state)
write('evidence/SEMANTIC-SCOPE-CORRECTIONS.json',{'date':now,'frozen_originals':'evidence/pre-independent-review',
 'findings':['Original template missing direct conditional residual edge', 'Original cross-module coverage label exceeded actual enumeration',
             'Template sink and index cache inputs incomplete','DSpark direct runner edge unsupported',
             'C8 count and nominal INT8 capacity imply a ceiling only in narrower per-position independent Cube-only domain'],
 'corrected_templates':'graphs/source-templates-G2.json','corrected_template_count':len(items),
 'cross_module_composite_count':len(cross),'cross_module_size_coverage':by_size,'cross_module_truncated':truncated,
 'whole_model_exhaustion':False,'wide_ideal_domain_numeric_ceiling':'UNKNOWN','not_formal_verification':True})
print(json.dumps({'G2_structural_count':len(items),'G2_categories':dict(Counter(x['category'] for x in items)),
                  'cross_module_size_coverage':by_size,'cross_module_count':len(cross),'truncated':truncated,
                  'conditional_numeric_scope_corrected':len(comparisons['comparisons']),'broad_domain_numeric_ceiling':'UNKNOWN',
                  'target_implementation_modified':False},indent=2))
