"""Validate frozen-image proposer provenance and update partial source graph."""
from pathlib import Path
from collections import defaultdict,Counter
from decimal import Decimal
import copy,csv,datetime,hashlib,itertools,json,shlex,subprocess

T=Path(__file__).resolve().parents[1]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE='wio-vllm-v41-repro169-20261028';STAGE='wio-vllm-v41-repro169-stage-20261028'
def read(p):return json.loads((T/p).read_text())
def write(p,v):
 f=T/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
paths=['/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
       '/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
       '/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py',
       '/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py',
       '/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py']
inspect=[]
for name in [BASE,STAGE]:
 r=subprocess.run([SSH,shlex.join(['docker','inspect','--format','{{json .}}',name])],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr
 d=json.loads(r.stdout);assert d['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b' and d['State']['Running']
 assert not any(x['Destination'].startswith('/vllm-workspace') for x in d['Mounts'])
 code='from pathlib import Path;import hashlib,json;paths='+repr(paths)+';print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}))'
 r=subprocess.run([SSH,shlex.join(['docker','exec',name,'python3','-c',code])],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr
 digests=json.loads(r.stdout)
 for p in paths:assert digests[p]==sha('sources/'+p.lstrip('/')),(name,p)
 inspect.append({'container':name,'container_id':d['Id'],'image_id':d['Image'],'running':True,
                 'vllm_workspace_overmounts':False,'source_sha256':digests})
write('evidence/dspark-source-provenance.json',{'checked_at':now,'source_file_hashes_match_frozen_baseline_and_cpu_stage':True,'containers':inspect,'no_device_work':True})

source=read('graphs/source-templates-G2.json');g=copy.deepcopy(source)
g.update(id='SOURCE_TEMPLATES_G3_DSPARK_RUNNER',parent='graphs/source-templates-G2.json')
t=next(x for x in g['templates'] if x['id']=='DSPARK_BOUNDARIES')
def node(name,kind):
 assert all(n['id']!=name for n in t['nodes']);t['nodes'].append({'id':name,'kind':kind,'resource':'NPU'})
def edge(a,b,value,source,condition):
 t['edges'].append({'producer':a,'consumer':b,'value':value,'dependency_kind':'RAW','source':source,'condition':condition})
for n,k in [('context_buffer','COPY_OR_VIEW'),('context_slots','EXTERNAL_SLOT_METADATA'),
            ('kv_precompute','CONTEXT_KV_PRECOMPUTE_COMPOSITE'),('kv_state','PERSISTENT_SWA_STATE')]:node(n,k)
cond='DSpark proposer, K>0, non-empty context and valid non-null per-layer context slot mappings; branches may differ.'
edge('main_norm','context_buffer','combined target auxiliary states copied to draft context buffer','spec_decode/llm_base_proposer.py:1062-1081; spec_decode/dspark_proposer.py:255-259',cond)
edge('context_buffer','kv_precompute','draft context hidden states','spec_decode/dflash_proposer.py:300-319; spec_decode/llm_base_proposer.py:1478-1494',cond)
edge('context_slots','kv_precompute','per-layer context slot mapping','spec_decode/dspark_proposer.py:267-313; spec_decode/dflash_proposer.py:300-319',cond)
edge('kv_precompute','kv_state','per-layer context SWA cache writes','models/deepseek_v41/dspark.py:284-309',cond)
for i in range(3):
 edge('kv_state','block'+str(i),'relevant cached context subranges','models/deepseek_v41/dspark.py:248-276,295-309; attention/dsa_v41.py:614-625',cond+' Does not establish a universal all-layers barrier or stream timing.')
t['runner_binding']='SUPPORTED for stated branches by proposer source and matching frozen-image file hashes; some branches, wrapper timing and alias details remain UNKNOWN.'
write('graphs/source-templates-G3.json',g)
nodes={n['id']:n for n in t['nodes']};eligible={k for k,v in nodes.items() if not v['kind'].startswith('EXTERNAL')};adj=defaultdict(set)
for e in t['edges']:
 a,b=e['producer'],e['consumer']
 if a in eligible and b in eligible:adj[a].add(b);adj[b].add(a)
sets={frozenset([k]) for k in eligible};items=[]
for size in [2,3,4]:
 sets={s|{x} for s in sets for x in set().union(*(adj[n] for n in s))-s}
 for s in sorted(sets,key=lambda v:tuple(sorted(v))):
  fp=hashlib.sha256(json.dumps(sorted(s)).encode()).hexdigest()
  items.append({'id':'DSPARK_G3_'+fp[:16],'nodes':sorted(s),'category':{2:'PAIR',3:'TRIPLE',4:'MULTI_OPERATOR'}[size],
                'internal_edges':[e for e in t['edges'] if e['producer'] in s and e['consumer'] in s],
                'external_inputs':[e for e in t['edges'] if e['consumer'] in s and e['producer'] not in s],
                'external_outputs':[e for e in t['edges'] if e['producer'] in s and e['consumer'] not in s],
                'status':'STRUCTURAL_ONLY','consumer_and_alias_completeness':'UNKNOWN','equivalence_proven':False})
for src in nodes:
 users=sorted({e['consumer'] for e in t['edges'] if e['producer']==src and e['consumer'] in eligible})
 for a,b in itertools.combinations(users,2):
  fp=hashlib.sha256(json.dumps([src,a,b]).encode()).hexdigest()
  items.append({'id':'DSPARK_G3_'+fp[:16],'nodes':[a,b],'shared_input':src,'category':'BRANCH_MULTICONSUMER',
                'status':'STRUCTURAL_ONLY','consumer_and_alias_completeness':'UNKNOWN','equivalence_proven':False})
write('candidates/dspark-G3-structural-index.json',{'scope':'DSpark partial known runner/source template; connected2-4 and sibling pairs',
 'count':len(items),'categories':dict(Counter(i['category'] for i in items)),'whole_model_exhaustion':False,'items':items})

records=read('records.json');byid={x['id']:x for x in records['fusion_candidates']}
for fid,selection in [('F27',['aux','main_proj','main_norm']),('F28',['embedding','repeat','block0'])]:
 sel=set(selection);obj={'id':fid+'_G3','source_template_path':'graphs/source-templates-G3.json',
  'nodes':[n for n in t['nodes'] if n['id'] in sel],
  'internal_edges':[e for e in t['edges'] if e['producer'] in sel and e['consumer'] in sel],
  'external_inputs':[e for e in t['edges'] if e['consumer'] in sel and e['producer'] not in sel],
  'external_outputs':[e for e in t['edges'] if e['producer'] in sel and e['consumer'] not in sel],
  'runner_binding':'SUPPORTED_IN_STATED_BRANCHES','consumer_completeness':'UNKNOWN',
  'semantic_equivalence_or_savings_proven':False,
  'condition':cond+' Preserve KV state, complete numerical semantics, all consumers and actual stream order.'}
 p='candidates/'+fid+'-G3.json';write(p,obj)
 byid[fid].update(status='CONDITIONAL',candidate_graph_path=p,candidate_graph_sha256=sha(p),
                  runner_binding='SUPPORTED_IN_STATED_BRANCHES',external_consumers_complete=False,
                  semantic_equivalence_argument='Candidate boundary only. Source confirms context-KV path in stated branches, not a direct main_norm activation to block0; equivalence/savings remain unproved.')
known={x['id'] for x in records['sources']}
for p in paths:
 sid='SRC_DSPARK_'+hashlib.sha256(p.encode()).hexdigest()[:12]
 if sid not in known:
  rel='sources/'+p.lstrip('/');records['sources'].append({'id':sid,'url_or_path':p,'snapshot_path':rel,'sha256':sha(rel),
   'registered_at':now,'version_commit':'8fde6d4e335fe300e2b949aab0e31659cce0811d',
   'chapter_symbol':'Actual source methods; exact provenance cross-checked in dspark-source-provenance.json'})
for u in records['unknowns']:
 if u['id']=='U_RUNNER_BINDING':u['statement']='Proposer -> combine -> context buffer -> per-layer KV write -> draft layer path is source-supported for K>0 and valid slots. Other dynamic paths, exact stream ordering and full alias/consumer coverage remain unknown.'

# Compare local elapsed intervals, using prior layer39 captured provenance where
# possible. The elapsed-time difference cancels fixed per-device clock offsets;
# joining two DPs to the same forward ordinal remains an explicit hypothesis.
raw=list(csv.DictReader((T/'mc2-arrival-analysis-v2/rank_target_calls.csv').open()))
byord=defaultdict(list)
for r in raw:byord[int(r['target_call_ordinal'])].append(r)
gap=read('inter-forward-gap-analysis-v2/summary.json');gaprows=[]
for p in gap['paired']:
 if not p['all_interior']:continue
 o=p['target_forward_ordinal'];prior=byord[o*40-1];assert len(prior)==16
 counts=Counter(r['model_id'] for r in prior)
 gaprows.append({'target_forward_ordinal':o,'dp0_minus_dp1_local_gap_us':p['dp0_local_gap_us_mean']-p['dp1_local_gap_us_mean'],
                 'prior_model_ids':dict(counts),'prior_layer39_validation':'Captured role-validated per rank' if set(counts)=={'49'} else 'Eager target ordinal modulo40 assumption, not independently captured-role validated'})
write('evidence/inter-forward-gap-review.json',{'status':'RANK_LOCAL_DURATION_COMPARISON',
 'matched_call_pairing':'Offset0 hypothesis, independently checked by time-overlap sensitivity; no internal op-ID proof',
 'fixed_clock_offset_effect':'Differences of same-rank elapsed intervals do not depend on a fixed epoch offset; clock rate drift/error and cross-DP matching remain unknown.',
 'interior_pairs':gaprows,'no_exported_task_interval_is_not_proven_idle':True,
 'no_task_family_sum_is_a_causal_or_end_to_end_cost':True,
 'source':'inter-forward-gap-analysis-v2/summary.json',
 'v1_output_correction':'evidence/inter-forward-gap-version-comparison.json'})
cause=next(x for x in records['module_causes'] if x['id']=='C_ARRIVAL_IMBALANCE')
cause['supporting_evidence'].append('evidence/inter-forward-gap-review.json: DP0 prior39MC2-to-next0layer local intervals exceed DP1 by 1.764ms, 0.540ms, and 1.311ms in3paired interior windows; ordinal6 prior eager layer assignment remains assumed.')
cause['missing_evidence'].append('Inter-forward task unions overlap and may include other streams; they do not identify which predecessor caused the elapsed difference.')
write('records.json',records)
scan=read('fusion-scan.json');scan['corrected_source_templates_G3']='graphs/source-templates-G3.json'
scan['dspark_structural_index']='candidates/dspark-G3-structural-index.json'
scan['conditional_candidate_ids']=[x['id'] for x in records['fusion_candidates'] if x['status']=='CONDITIONAL']
scan['semantic_unknown_candidate_ids']=[x['id'] for x in records['fusion_candidates'] if x['status']=='UNKNOWN']
scan['passes'].append({'id':'DSPARK_G3_SOURCE_RUNNER_RESCAN','new_structural_items':len(items),'semantic_exhaustion':False,
 'reason':'Recovered proposer/context-KV source branch; aliases, runner alternatives, compiled operators and all consumers are not fully expanded.'})
write('fusion-scan.json',scan)

# Restore the kernel output contract carried by the original numerical records;
# the earlier review correction had accidentally dropped this auxiliary field.
a=read('module-analysis.json');old=read('evidence/pre-independent-review/module-analysis.json')
oldby={x['object_id']:x for x in old['performance_comparisons']}
for c in a['performance_comparisons']:
 if 'output_contract' in oldby.get(c['object_id'],{}):c['output_contract']=oldby[c['object_id']]['output_contract']
write('module-analysis.json',a)
num=read('evidence/conditional-performance-comparisons.json')
for c in num['comparisons']:
 if 'output_contract' in oldby.get(c['object_id'],{}):c['output_contract']=oldby[c['object_id']]['output_contract']
write('evidence/conditional-performance-comparisons.json',num)
profile=read('profile-summary.json');profile['long_wait_or_idle_groups'].append({'id':'INTER_FORWARD_RANK_LOCAL_INTERVAL',
 'state':'OBSERVED_ASSOCIATION_NOT_CAUSAL','evidence':'evidence/inter-forward-gap-review.json'})
assert len(json.dumps(profile).encode())<32768;write('profile-summary.json',profile)
index=read('profile-index.json');index['aggregate_tables'].append({'id':'INTER_FORWARD_GAPS_V2',
 'path_or_table':'inter-forward-gap-analysis-v2/rank_gaps.csv','row_count':88,'grouping_keys':['rank','target_forward_ordinal'],
 'sha256':sha('inter-forward-gap-analysis-v2/rank_gaps.csv'),'full_scope_coverage':'Captured layer0 entries with an observed previous target MC2; no full model attribution'})
write('profile-index.json',index)
report=T/'REPORT.md';s=report.read_text();s+='''

## 后续只读追查：DSpark 链路和前向间隔

冻结基线容器与 CPU stage 使用相同镜像；5 个关键源码文件在两容器和本地快照中的 SHA-256 一致（见 `evidence/dspark-source-provenance.json`）。对 DSpark 的 K>0、上下文非空且 slot 有效的分支，源码显示 proposer 先将 target 辅助状态经 `combine_hidden_states` 投影归一化，放入上下文缓冲区，再调用 `precompute_and_store_context_kv` 写入各草稿层缓存，随后调用草稿模型。修正版 `graphs/source-templates-G3.json` 将它表示为通过上下文缓冲和持久 KV 状态的依赖，不能简化为投影输出直接作为 block0 激活；也不因此证明跨所有层的整阶段等待不可避免。F27/F28 重新记为条件结构候选，F30 的验证/接收绑定仍未知。

在 3 组内部样本中，按各 rank 自己的时钟计算，从上一 target 第 39 层 MC2 结束到下一 target 第 0 层开始，DP0 比 DP1 分别多约 1.764、0.540 和 1.311 毫秒。前后两端使用同一 rank 的时间差，因此固定时钟偏移不改变这个差值；跨 DP 的同轮配对仍依赖 offset0 假设。其中第二组的上一轮是 eager 路径，第 39 层身份依赖调用序号推断，证据较弱。区间内已导出 kernel 的并集与未覆盖时间不能分别等同于本任务设备忙碌和设备空闲；不同 stream 的通信、计算时间也不能简单相加推断原因。

最初的间隔扫描给可能跨窗的 kernel 设了 50ms 回看范围，后续用所有候选区间的结束时间前缀最大值修正；5 组统计未变化。v2 文件一度误写入本地 v1 目录，现已从未修改的远端副本恢复 v1 并保存两个版本及审计记录。原始 trace 未受影响。
''';report.write_text(s)
state=read('STATE.json');state.update(current_phase='P6',last_update=now,overall_work_paused=False,
 next_read_only_action='Audit revised runner graph and conditional comparison; continue module attribution if existing eager trace can support it, otherwise record explicit coverage limits.')
write('STATE.json',state)
print(json.dumps({'source_hashes_verified_in_both_containers':len(paths),'dspark_partial_structural_items':len(items),
                  'F27_F28_status':'CONDITIONAL_IN_STATED_BRANCHES','gap_pairs':gaprows,
                  'wide_ideal_domain_numeric_upper':'UNKNOWN','no_target_implementation_modified':True},indent=2))
