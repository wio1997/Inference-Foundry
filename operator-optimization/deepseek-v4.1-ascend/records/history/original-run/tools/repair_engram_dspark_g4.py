"""Generate source-scoped G4; no target code or device access."""
from pathlib import Path
from collections import Counter,defaultdict
import copy,datetime,hashlib,itertools,json
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads((T/p).read_text())
def write(p,x):
 f=T/p;f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists(),str(f);f.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
g=copy.deepcopy(read('graphs/source-templates-G3.json'))
g.update(id='SOURCE_TEMPLATES_G4_ENGRAM_DSPARK',parent='graphs/source-templates-G3.json',whole_model_complete=False)
e=next(x for x in g['templates'] if x['id']=='ENGRAM')
N={n['id']:n for n in e['nodes']}
N['tokens']['resource']='UNKNOWN'
N['hash'].update(resource='NPU',kind='STATEFUL_NPU_HASH_COMPOSITE',source='engram/hash_state.py:102-148',condition='num_tokens > 0; optional slot-cache writes')
N['lookup'].update(resource='CONDITIONAL_NPU_OR_CPU',kind='ENGRAM_LOOKUP_BRANCH_COMPOSITE',source='engram/embedding.py:378-405',condition='host UVA branch uses NPU kernel reading mapped host memory; device-table branch NPU; CPU reference fallback exists')
e['source']+='; engram/hash_state.py:90-148; engram/embedding.py:378-405'
e['edges']=[x for x in e['edges'] if (x['producer'],x['consumer']) != ('lookup','ready')]
e['nodes'] += [
 {'id':'rows_collective_pad','kind':'CONDITIONAL_DP_EXCHANGE_TP_GATHER_PAD','resource':'COMM_NPU','condition':'branch on DP/TP and output-buffer configuration'},
 {'id':'mask_derive','kind':'MASK_GENERATION','resource':'NPU'},
 {'id':'mask_ready','kind':'EVENT','resource':'CPU_NPU','condition':'when mask_ready_event is provided'},
 {'id':'hash_runtime_metadata','kind':'EXTERNAL_METADATA','resource':'UNKNOWN','description':'positions, query_start_loc, dead_mask, lookback IDs and lookback masks'},
 {'id':'hash_slot_state','kind':'EXTERNAL_SHARED_STATE','resource':'UNKNOWN','description':'conditional block_table, slot_mapping and cache; cache writes occur only if use_slot_cache'},
 {'id':'mapped_or_device_table','kind':'EXTERNAL_SHARED_STATE','resource':'CONDITIONAL_HOST_MAPPED_OR_NPU_OR_CPU','description':'codes and scales; selected by lookup branch'},
 {'id':'token_mask_rotation','kind':'EXTERNAL_METADATA','resource':'UNKNOWN'},
 {'id':'gate_result','kind':'EXTERNAL_OUTPUT_OR_PERSISTENT_STATE','resource':'UNKNOWN'}]
def E(a,b,value,source,condition='source branch; precise runtime realization not fully observed'):
 return {'producer':a,'consumer':b,'value':value,'dependency_kind':'RAW','source':source,'condition':condition}
e['edges'] += [
 E('lookup','rows_collective_pad','lookup values','models/deepseek_v41/model.py:1205-1226; engram/embedding.py:333-355'),
 E('rows_collective_pad','ready','rows after conditional collective and padding','models/deepseek_v41/model.py:1205-1226'),
 E('tokens','mask_derive','input IDs and dead-mask inputs','models/deepseek_v41/model.py:1163-1177'),
 E('hash_runtime_metadata','mask_derive','valid count and optional token indices','models/deepseek_v41/model.py:1163-1177'),
 E('mask_derive','mask_ready','published active mask','models/deepseek_v41/model.py:1156-1177'),
 E('mask_ready','gate','token mask ready; then shard/slice','models/deepseek_v41/model.py:1565-1591'),
 E('hash_runtime_metadata','hash','positions, query_start_loc, masks and lookback','engram/hash_state.py:102-148'),
 E('hash_slot_state','hash','cache and slot/block mappings','engram/hash_state.py:109-148','when use_slot_cache, hash reads and writes state; alias and cross-invocation dependencies not fully expanded'),
 E('hash','hash_slot_state','conditional token cache update','engram/hash_state.py:111-128','when use_slot_cache; state side effect; not a same-invocation acyclic precedence edge'),
 E('mapped_or_device_table','lookup','codes and scales','engram/embedding.py:378-405'),
 E('token_mask_rotation','gate','mask and rotation','engram/layer.py:35-57'),
 E('gate','gate_result','updated hidden states','engram/layer.py:35-57')]
e['conditions']='Mapped host lookup uses a device kernel when UVA branch selected; separate CPU fallback. Event, communication, aliases and branch consumer completeness remain UNKNOWN.'
d=next(x for x in g['templates'] if x['id']=='DSPARK_BOUNDARIES')
d['source']+='; spec_decode/llm_base_proposer.py:1550-1644; spec_decode/dspark_proposer.py:61-65,108-124'
d['edges']=[x for x in d['edges'] if (x['producer'],x['consumer']) not in {('logits','accept'),('norm','confidence'),('markov','logits'),('markov','confidence')}]
d['nodes']=[x for x in d['nodes'] if x['id'] not in {'markov','accept'}]
next(x for x in d['nodes'] if x['id']=='confidence').update(condition='dynamic_spec is not configured in frozen-serve.sh and default method is None; inactive under that config unless overridden elsewhere',connection='UNKNOWN: scheduler implementation absent in current snapshot; proposer helper exists but direct use is unproved')
d['nodes'] += [{'id':'seed','kind':'EXTERNAL_DRAFT_SEED','resource':'NPU'},
 {'id':'draft_token_output','kind':'EXTERNAL_OUTPUT','resource':'UNKNOWN'},
 {'id':'last_hidden_dynamic_input','kind':'EXTERNAL_OR_UNRESOLVED_VALUE','resource':'UNKNOWN','condition':'dynamic_spec branch only; scheduler binding UNKNOWN'},
 {'id':'dynamic_scheduler','kind':'OPAQUE_CONDITIONAL_SCHEDULER','resource':'UNKNOWN','condition':'only if dynamic_spec enabled; implementation absent from source snapshot'}]
d['edges'].append(E('last_hidden_dynamic_input','dynamic_scheduler','last_hidden_states','spec_decode/llm_base_proposer.py:1611-1617','dynamic_spec != None; exact scheduler implementation UNKNOWN'))
source='spec_decode/llm_base_proposer.py:1577-1602'
for i in range(5):
 for name,kind in [('embed','MARKOV_EMBED'),('bias','MARKOV_BIAS'),('add','LOGITS_BIAS_ADD'),('sample','GREEDY_ARGMAX_OR_PROBABILISTIC_SAMPLE')]:d['nodes'].append({'id':f'{name}_{i}','kind':kind,'resource':'NPU','condition':f'dspark, idx={i}; probabilistic sampling is a conditional branch'})
 prev='seed' if i==0 else f'sample_{i-1}'
 d['edges'] += [E(prev,f'embed_{i}',f'draft token at index {i}',source),E(f'embed_{i}',f'bias_{i}','markov embedding',source),E(f'bias_{i}',f'add_{i}','markov bias',source),E('logits',f'add_{i}',f'raw logits slice {i}',source),E(f'add_{i}',f'sample_{i}','biased logits slice',source),E(f'sample_{i}','draft_token_output',f'sampled token {i+1}','spec_decode/llm_base_proposer.py:1590-1602,1636-1644','output return requires parallel_drafting or K=1; other runner path unresolved')]
d['edges'].append(E('draft_token_output','dynamic_scheduler','draft_token_ids','spec_decode/llm_base_proposer.py:1611-1617','dynamic_spec != None; exact scheduler implementation UNKNOWN'))
d['unresolved']='No raw logits -> accept direct edge. Draft return and downstream rejection binding only in supported branches; confidence scheduler internals unknown; aliases and full output consumers incomplete.'
write('graphs/source-templates-G4.json',g)
all_idx={}
for t in [e,d]:
 nodes={x['id']:x for x in t['nodes']};eligible={k for k,v in nodes.items() if not v['kind'].startswith('EXTERNAL') and k not in {'confidence','dynamic_scheduler'}}
 adj=defaultdict(set)
 for x in t['edges']:
  a,b=x['producer'],x['consumer']
  if a in eligible and b in eligible:adj[a].add(b);adj[b].add(a)
 sets={frozenset([x]) for x in eligible};items=[]
 for size in [2,3,4]:
  sets={s|{x} for s in sets for x in set().union(*(adj[n] for n in s))-s}
  for s in sorted(sets,key=lambda x:tuple(sorted(x))):
   items.append({'nodes':sorted(s),'category':{2:'PAIR',3:'TRIPLE',4:'MULTI_OPERATOR'}[size], 'internal_edges':[z for z in t['edges'] if z['producer'] in s and z['consumer'] in s], 'external_inputs':[z for z in t['edges'] if z['consumer'] in s and z['producer'] not in s], 'external_outputs':[z for z in t['edges'] if z['producer'] in s and z['consumer'] not in s], 'status':'STRUCTURAL_ONLY','branch_feasibility_and_full_consumer_completeness':'UNKNOWN'})
 for src in nodes:
  users=sorted({z['consumer'] for z in t['edges'] if z['producer']==src and z['consumer'] in eligible})
  for a,b in itertools.combinations(users,2):items.append({'nodes':[a,b],'shared_input':src,'category':'BRANCH_MULTICONSUMER','status':'STRUCTURAL_ONLY','branch_feasibility_and_full_consumer_completeness':'UNKNOWN'})
 name='candidates/'+t['id'].lower()+'-G4-structural-index.json'
 write(name,{'scope':t['id']+' source-level template; known branches, not primitive lowering','parent_graph':'graphs/source-templates-G4.json','count':len(items),'categories':dict(Counter(x['category'] for x in items)),'whole_model_exhaustion':False,'items':items})
 all_idx[t['id']]={'path':name,'count':len(items),'categories':dict(Counter(x['category'] for x in items))}
r=read('records.json');byid={x['id']:x for x in r['fusion_candidates']}
for fid,tid,chosen in [('F24','ENGRAM',['hash','lookup','ready','rows']),('F27','DSPARK_BOUNDARIES',['aux','main_proj','main_norm']),('F28','DSPARK_BOUNDARIES',['embedding','repeat','block0']),('F30','DSPARK_BOUNDARIES',['collapse','norm','logits']+[f'{x}_{i}' for i in range(5) for x in ['embed','bias','add','sample']])]:
 t=next(x for x in [e,d] if x['id']==tid);s=set(chosen); assert s <= {n['id'] for n in t['nodes']}
 obj={'id':fid,'graph_id':'SOURCE_TEMPLATES_G4_ENGRAM_DSPARK','template':tid,'nodes':[n for n in t['nodes'] if n['id'] in s], 'internal_edges':[x for x in t['edges'] if x['producer'] in s and x['consumer'] in s], 'external_inputs':[x for x in t['edges'] if x['consumer'] in s and x['producer'] not in s], 'external_outputs':[x for x in t['edges'] if x['producer'] in s and x['consumer'] not in s], 'status':'STRUCTURAL_BOUNDARY_ONLY','semantic_equivalence_or_savings_proven':False,'consumer_and_alias_completeness':'UNKNOWN'}
 path=f'candidates/{fid}-G4.json';write(path,obj)
 x=byid[fid];x.update(candidate_graph_path=path,candidate_graph_sha256=sha(path),other_consumers=obj['external_outputs'],semantic_equivalence_argument='G4 is a partial source boundary; branch conditions, aliases, all consumers, equivalence and savings remain unproved.',external_consumers_complete=False)
 if fid=='F30':x.update(status='UNKNOWN',original_node_ids=chosen,original_node_mapping={n:n for n in chosen})
 if fid=='F24':x['semantic_conditions']+=' G4 corrects execution resource: hash and host-UVA lookup launch NPU kernels; CPU fallback is a different branch.'
(T/'records.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
scan=read('fusion-scan.json');scan['corrected_source_templates_G4']='graphs/source-templates-G4.json';scan['g4_structural_indices']=all_idx;scan['passes'].append({'id':'ENGRAM_DSPARK_G4_SOURCE_RESCAN','structural_items_by_template':{k:v['count'] for k,v in all_idx.items()},'semantic_exhaustion':False,'reason':'Corrected resources, conditional state and sequential DSpark steps; full lowering, alias and downstream consumers remain unknown.'});(T/'fusion-scan.json').write_text(json.dumps(scan,ensure_ascii=False,indent=2)+'\n')
paths=['graphs/source-templates-G3.json','graphs/source-templates-G4.json','evidence/frozen-serve.sh','sources/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/hash_state.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/embedding.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/layer.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/npu.py','sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py']
write('evidence/engram-dspark-g4-review.json',{'timestamp':now,'source_sha256':{x:sha(x) for x in paths},'corrections':['Engram hash executes Triton kernels on NPU; host-UVA path is device kernel accessing host-mapped data, CPU reference is a separate branch.','Engram hash depends on runtime metadata, conditional cache state, and mapped/device table; retain state writes and gate inputs/outputs. Mask readiness is separate and lookup readiness occurs after conditional row exchange/gather/padding.','The five Markov/sample steps are sequential within the same drafting loop; raw logits are not the accept operation.','Dynamic confidence is disabled by the frozen shell config and default method None; configuration overrides and scheduler internals not proven.','G3 remains historical; G4 does not claim all primitive paths, execution branches, consumers or aliasing are complete.'],'structural_indices':all_idx,'no_target_code_modified':True,'no_device_work':True})
s=read('STATE.json');s['last_update']=now;s['checkpoint_status']='G4_SOURCE_CORRECTIONS_ENGRAM_DSPARK';s['next_read_only_action']='Audit corrected G4 conditional branches and continue incomplete whole-framework attribution and hardware ceilings.';(T/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (T/'REPORT.md').open('a') as f:f.write('\n\n## 源码复核：Engram 资源与 DSpark 逐步依赖（G4）\n\nEngram 哈希调用 NPU Triton 内核；主机卸载的 UVA 分支也由 NPU 内核读取映射的主机表，CPU 参考路径是另一分支。此前将 hash 和 lookup 统一标成 CPU 不正确。G4 补上位置、请求边界、lookback、掩码、条件 slot cache、表、gate 输入和输出的边界；缓存 alias 和全部消费者仍未证明。\n\n基线 DSpark 配置 5 个投机 token。源码逐步以 token 生成 Markov embedding/bias，给当步 logits 加 bias 后采样，结果成为下一步输入；G3 的 logits 直接连接 accept 是不成立的简化。冻结启动脚本未设置 dynamic_spec，默认 method=None，confidence 调度分支在该配置下关闭，除非有其他覆盖；scheduler 内部绑定尚未验证。G4 独立保留五步和条件分支，未声称整个运行图或融合等价已完备。结构候选仅是有限图枚举，不等于可实现或有收益。详细证据见 evidence/engram-dspark-g4-review.json。\n')
print(json.dumps({'status':'G4_WRITTEN','structural_indices':all_idx,'timestamp':now},ensure_ascii=False))
