"""Enumerate bounded structural opportunities; semantic validity remains explicit."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,itertools,datetime,shutil
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads((T/p).read_text())
def write(p,d):
 f=T/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
cfg=read('evidence/model-config.json')['text_config'];g=read('graphs/G0.json');g1=json.loads(json.dumps(g));g1['id']='G1_REFINED_MODULE_GRAPH';g1['parent_graph_id']=g['id']
for i in range(40):
 g1['edges'].append({'producer':'embedding' if i==0 else f'target.layers.{i-1}.hc_ffn_pre','consumer':f'target.layers.{i}.hc_attn_pre','value':'initial pre_mix' if i==0 else 'delayed ffn_pre coefficient handoff','dependency_kind':'RAW'})
g1['edges'].append({'producer':'target.layers.39.hc_ffn_pre','consumer':'target.final','value':'final pre_mix','dependency_kind':'RAW'})
g1['refinement_note']='Added delayed mHC coefficient dependencies explicitly; G0 retained verbatim. Composite graph remains incomplete.'
write('graphs/G1.json',g1)

templates=[]
def template(id,source):
 d={'id':id,'source':source,'nodes':[],'edges':[],'runtime_graph':False,'complete_primitive_lowering':False,'precision_rule':'Preserve all existing casts, rounding, reduction order and mutable state unless equivalence is independently established'};templates.append(d);return d
def n(d,id,kind,resource='NPU',**kw):d['nodes'].append({'id':id,'kind':kind,'resource':resource,**kw});return id
def e(d,a,b,value='intermediate',kind='RAW'):d['edges'].append({'producer':a,'consumer':b,'value':value,'dependency_kind':kind})
def chain(d,ids):
 for a,b in zip(ids,ids[1:]):e(d,a,b)

for role in ['SWA','C2_SOURCE','C2_CONSUMER','C1_SOURCE','C1_INDEX_SOURCE','C1_CONSUMER']:
 d=template('ATTN_'+role,'attention/dsa_v41.py:320-715; context_parallel/dsa_v41_cp.py:189-224; models/deepseek_v41/indexer.py:110-295')
 for id,kind in [('x','EXTERNAL'),('q_a','LINEAR_BF16'),('q_norm','ROW_REDUCTION_NORM'),('q_quant','ROW_REDUCTION_INT8_QUANT'),('q_b','INT8_LINEAR'),('q_rope','POINTWISE_ROPE'),('kv','LINEAR_BF16'),('kv_norm','ROW_REDUCTION_NORM'),('kv_rope','POINTWISE_ROPE'),('swa_store','PERSISTENT_SCATTER'),('attention','SPARSE_FLASH_COMPOSITE'),('inverse_rope','POINTWISE_ROPE'),('pad','PAD_OR_VIEW'),('restore_heads','COLLECTIVE'),('o_a','GROUPED_LINEAR'),('o_b','LINEAR_BF16'),('reduce_scatter','COLLECTIVE')]:
  n(d,id,kind,'COMM' if kind=='COLLECTIVE' else 'NPU',row_domain='query_local' if id.startswith('q_') else 'global_or_restored')
 chain(d,['x','q_a','q_norm','q_quant','q_b','q_rope','attention','inverse_rope','pad','restore_heads','o_a','o_b','reduce_scatter']);chain(d,['x','kv','kv_norm','kv_rope','swa_store']);e(d,'swa_store','attention','current cache subranges',kind='RAW_PERSISTENT_SUBRANGE')
 has_index=role in ['C2_SOURCE','C1_SOURCE','C1_INDEX_SOURCE']
 if has_index:
  for id,kind in [('iq','LINEAR'),('iq_rope','POINTWISE_ROPE'),('iq_quant','ROW_REDUCTION_INT8_QUANT'),('iw','LINEAR'),('iw_scale','POINTWISE_CAST_SCALE'),('qli','INDEX_SCORE_TOPK_COMPOSITE'),('fixup','INDEX_FIXUP'),('indices_store','PERSISTENT_INDEX_PUBLICATION')]:n(d,id,kind)
  chain(d,['q_norm','iq','iq_rope','iq_quant','qli','fixup','indices_store']);chain(d,['x','iw','iw_scale','qli']);e(d,'indices_store','attention','selected indices')
 elif role!='SWA':
  n(d,'indices_in','EXTERNAL_SHARED_STATE');e(d,'indices_in','attention','reused selection from source layer')
 if role in ['C2_SOURCE','C1_SOURCE']:
  n(d,'compress_proj','LINEAR_FP32' if role=='C2_SOURCE' else 'LINEAR_BF16');n(d,'compress_norm','ROW_REDUCTION_NORM')
  if role=='C2_SOURCE':
   for id,kind in [('x_cast','POINTWISE_CAST'),('score_proj','LINEAR_FP32'),('ring_pool','STATEFUL_POOL_COMPOSITE')]:n(d,id,kind)
   chain(d,['x','x_cast','compress_proj','ring_pool','compress_norm']);chain(d,['x_cast','score_proj','ring_pool'])
  else:chain(d,['x','compress_proj','compress_norm'])
  for id,kind in [('ik','LINEAR'),('ik_norm','ROW_REDUCTION_NORM'),('ik_rope','POINTWISE_ROPE'),('ik_quant','ROW_REDUCTION_INT8_QUANT'),('ik_store','PERSISTENT_SCATTER'),('scale_cast','POINTWISE_CAST'),('scale_store','PERSISTENT_SCATTER'),('cmp_rope','POINTWISE_ROPE'),('cmp_store','PERSISTENT_SCATTER')]:n(d,id,kind)
  chain(d,['compress_norm','ik','ik_norm','ik_rope','ik_quant','ik_store']);chain(d,['ik_quant','scale_cast','scale_store']);chain(d,['compress_norm','cmp_rope','cmp_store']);e(d,'ik_store','qli','key cache');e(d,'scale_store','qli','key scales');e(d,'cmp_store','attention','compressed KV cache')
 elif role!='SWA':
  n(d,'cache_in','EXTERNAL_SHARED_STATE');e(d,'cache_in','attention','long KV from source layer')

d=template('MHC_BLOCK_AND_CROSS_LAYER','models/deepseek_v41/model.py:844-920,1555-1605')
for id,kind in [('residual','EXTERNAL'),('pre_mix','EXTERNAL_SHARED_STATE'),('attn_pre','HC_PRE_COMPOSITE'),('input_norm','ROW_REDUCTION_NORM'),('all_gather','COLLECTIVE'),('attention','ATTENTION_OPAQUE'),('reduce_scatter','COLLECTIVE'),('attn_post','HC_POST_COMPOSITE'),('ffn_pre','HC_PRE_COMPOSITE'),('norm_cast','ROW_REDUCTION_NORM_CAST'),('moe','MOE_OPAQUE'),('ffn_post','HC_POST_COMPOSITE'),('next_pre','HC_PRE_COMPOSITE'),('aux_mean','HC_MEAN'),('engram','ENGRAM_OPAQUE'),('collapse','HC_FP32_REDUCE_BF16'),('final_norm','ROW_REDUCTION_NORM')]:n(d,id,kind,'COMM' if kind=='COLLECTIVE' else 'NPU')
chain(d,['residual','attn_pre','input_norm','all_gather','attention','reduce_scatter','attn_post','ffn_pre','norm_cast','moe','ffn_post']);e(d,'pre_mix','attn_pre','previous layer coefficient');e(d,'residual','attn_post','residual');e(d,'attn_pre','attn_post','post/comb');e(d,'attn_pre','ffn_pre','attn_pre coefficient');e(d,'attn_post','ffn_post','residual');e(d,'ffn_pre','ffn_post','post/comb');e(d,'ffn_post','aux_mean','residual before optional Engram');e(d,'ffn_post','engram','optional next-layer Engram');e(d,'engram','next_pre','Engram-updated residual');e(d,'ffn_pre','next_pre','delayed coefficient');e(d,'ffn_post','collapse','terminal-layer HC state');e(d,'ffn_pre','collapse','terminal pre_mix');e(d,'collapse','final_norm')
d['branch_note']='Cross-layer direct post→pre when no Engram; aux/collapse are separate conditional consumers, not all operations in one invocation.'

d=template('MOE_WRAPPER','models/deepseek_v41/model.py:105-326; ops/fused_moe/moe_comm_method.py:563-611')
for id,kind in [('x','EXTERNAL'),('x_fp32','EXTERNAL'),('router','LINEAR_FP32_OR_HASH'),('topk','ROUTING_TOPK_COMPOSITE'),('dispatch_ffn_combine','MC2_OPAQUE_EXISTING_FUSION'),('shared_quant','ROW_REDUCTION_INT8_QUANT'),('shared_gate_up','INT8_LINEAR'),('swiglu_quant','SWIGLU_CLAMP_QUANT_COMPOSITE'),('shared_down','INT8_LINEAR'),('merge','POINTWISE_SCALE_ADD')]:n(d,id,kind)
chain(d,['x_fp32','router','topk','dispatch_ffn_combine','merge']);e(d,'x','dispatch_ffn_combine','hidden');chain(d,['x','shared_quant','shared_gate_up','swiglu_quant','shared_down','merge'])
d=template('MC2_INTERNAL_KNOWN_STAGES','csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp:213-234,435-538,539-635,795-970')
for id,kind in [('x','EXTERNAL'),('route_quant','ROUTING_QUANT'),('counts','TOKEN_COUNTS_EXCHANGE'),('dispatch','REMOTE_DATA_COPY'),('gmm1','INT8_GROUPED_LINEAR'),('swiglu','SWIGLU_CLAMP_QUANT_COMPOSITE'),('gmm2','INT8_GROUPED_LINEAR'),('combine','REMOTE_DATA_COPY_REDUCE')]:n(d,id,kind,'COMM' if id in ['counts','dispatch','combine'] else 'NPU',already_one_kernel=True)
chain(d,['x','route_quant','counts','dispatch','gmm1','swiglu','gmm2','combine']);d['dependency_note']='Stage edges denote needed subranges; actual implementation contains per-expert cross-core flags, epilogue granularity and SyncAll. No universal full-stage serialization is inferred.'
d=template('ENGRAM','engram/embedding.py; models/deepseek_v41/engram/layer.py:35-57; model.py:1498-1591')
for id,kind,res in [('tokens','EXTERNAL','CPU'),('hash','STATEFUL_HASH','CPU'),('lookup','MAPPED_HOST_LOOKUP','CPU'),('ready','EVENT','CPU_NPU'),('rows','ROW_VIEW_OR_SHARD','NPU'),('proj','LINEAR','NPU'),('gate','FUSED_GATE_OR_ROTATED_GATE','NPU'),('residual','EXTERNAL','NPU')]:n(d,id,kind,res)
chain(d,['tokens','hash','lookup','ready','rows','proj','gate']);e(d,'residual','gate','HC residual');d['conditions']='Mapped lookup already asynchronous; unrotated fused gate and rotation-dependent fallback distinguished by runtime branch, not assumed.'
d=template('DSPARK_BOUNDARIES','models/deepseek_v41/dspark.py:143-336; model.py:1555-1564')
for id,kind in [('aux','EXTERNAL'),('main_proj','LINEAR'),('main_norm','ROW_REDUCTION_NORM'),('tokens','EXTERNAL'),('embedding','EMBEDDING'),('repeat','HC_REPEAT_AND_INIT'),('block0','DECODER_OPAQUE'),('block1','DECODER_OPAQUE'),('block2','DECODER_OPAQUE'),('collapse','HC_FP32_REDUCE_BF16'),('norm','ROW_REDUCTION_NORM'),('logits','LM_HEAD'),('markov','MARKOV_HEAD_OPAQUE'),('confidence','CONFIDENCE_HEAD_OPAQUE'),('accept','TARGET_ACCEPT_OPAQUE')]:n(d,id,kind)
chain(d,['aux','main_proj','main_norm']);chain(d,['tokens','embedding','repeat','block0','block1','block2','collapse','norm','logits','accept']);e(d,'main_norm','block0','combined context; runner binding not expanded');e(d,'norm','confidence');e(d,'markov','logits','bias');e(d,'markov','confidence','Markov features');d['unresolved']='Proposer/sample/accept and shared-KV runner bindings require source expansion; no invented exact runtime DAG.'
d=template('DSPARK_SHARED_KV','models/deepseek_v41/dspark.py:233-312')
for id,kind in [('context','EXTERNAL'),('proj','LINEAR_BF16'),('norm','ROW_REDUCTION_NORM'),('rope','POINTWISE_ROPE'),('cat','COPY_OR_VIEW'),('slots','SLOT_FIXUP'),('scatter','PERSISTENT_SCATTER')]:n(d,id,kind)
chain(d,['context','proj','norm','rope','cat','scatter']);e(d,'slots','scatter','validated physical slots')
write('graphs/source-templates.json',{'templates':templates,'scope':'Explicit source-known operator boundaries; current composite internals and branch gaps retained','whole_model_complete':False})

all_rows=[];structural=[];seen=set();pass_stats=[]
for round in [1,2]:
 new=duplicates=0
 for d in templates:
  nodes={n['id']:n for n in d['nodes']};eligible={i for i,n0 in nodes.items() if not n0['kind'].startswith('EXTERNAL')};adj=defaultdict(set)
  for ed in d['edges']:
   if ed['producer'] in eligible and ed['consumer'] in eligible:
    adj[ed['producer']].add(ed['consumer']);adj[ed['consumer']].add(ed['producer'])
  sets={frozenset([i]) for i in eligible}
  for size in [2,3,4]:
   extended=set()
   for s in sets:
    for v in set().union(*(adj[x] for x in s))-s:extended.add(s|{v})
   sets=extended
   for s in sorted(sets,key=lambda s:tuple(sorted(s))):
    key=(d['id'],tuple(sorted(s)));fingerprint=hashlib.sha256(json.dumps(key).encode()).hexdigest()
    if fingerprint in seen:duplicates+=1;continue
    seen.add(fingerprint);new+=1
    consumers={x:sorted({ed['consumer'] for ed in d['edges'] if ed['producer']==x and ed['consumer'] not in s}) for x in sorted(s)}
    concerns=['preserve operation order, casts/rounding and every external consumer','tile liveness/capacity and existing overlap not yet proved']
    if any(nodes[x]['resource'] in ['COMM','CPU','CPU_NPU'] for x in s):concerns.append('cross-resource boundary requires real dependency and interface review')
    if any('COMPOSITE' in nodes[x]['kind'] or 'OPAQUE' in nodes[x]['kind'] for x in s):concerns.append('composite internals incompletely expanded; no equivalence or savings assumed')
    if any(nodes[x].get('already_one_kernel') for x in s):concerns.append('already within MC2 kernel; structural combination is not a new fusion gain')
    row={'id':'STRUCT_'+fingerprint[:16],'template_id':d['id'],'node_ids':sorted(s),'category':{2:'PAIR',3:'TRIPLE',4:'MULTI_OPERATOR'}[size],'status':'CONDITIONAL','source':d['source'],'external_consumers':consumers,'concerns':concerns,'canonical_fingerprint':fingerprint,'semantics_review':'STRUCTURAL_ONLY','estimated_savings':None}
    structural.append(row)
  # Shared-input siblings and split-regroup are separate rules, not omitted by connected-subgraph enumeration.
  for src in nodes:
   users=sorted({ed['consumer'] for ed in d['edges'] if ed['producer']==src and ed['consumer'] in eligible})
   for a,b in itertools.combinations(users,2):
    key=(d['id'],'SHARED_INPUT',src,a,b);fp=hashlib.sha256(json.dumps(key).encode()).hexdigest()
    if fp in seen:duplicates+=1;continue
    seen.add(fp);new+=1;structural.append({'id':'STRUCT_'+fp[:16],'template_id':d['id'],'node_ids':[a,b],'shared_input':src,'category':'BRANCH_MULTICONSUMER','status':'CONDITIONAL','source':d['source'],'canonical_fingerprint':fp,'semantics_review':'STRUCTURAL_ONLY','conditions':['Shared source is not proof of identical dtype/row range or interchangeable operation','All consumers, aliases and branch conditions must be preserved'],'estimated_savings':None})
  if round==1:
   for cat in ['PAIR','TRIPLE','MULTI_OPERATOR','SPLIT_REGROUP','CROSS_MODULE','BRANCH_MULTICONSUMER']:
    all_rows.append({'template_id':d['id'],'category':cat,'status':'SCANNED_STRUCTURALLY' if cat!='SPLIT_REGROUP' else 'PARTIAL_KNOWN_COMPOSITE_STAGES','rule_scope':'Connected sets2-4; shared-input sibling pairs; explicit known split stages','source':d['source'],'semantic_proof_complete':False,'unknown':'Opaque internals and unresolved branch/alias conditions remain'})
 pass_stats.append({'id':'STRUCTURAL_PASS_'+str(round),'new_structural_items':new,'duplicate_items':duplicates,'same_declared_graph_and_rules':True,'no_new_items_is_not_semantic_exhaustion':True})
write('candidates/structural-index.json',{'scope':'Finite structural enumeration, not validated semantic opportunities','items':structural,'count':len(structural)})
write('evidence/structural-coverage.json',{'rows':all_rows,'passes':pass_stats,'global_exhaustion':False,'unresolved_semantic_review':True})
scan=read('fusion-scan.json');scan.update(status='SCANNING',graph_versions=[{'id':g1['id'],'parent_graph_id':g['id'],'graph_path':'graphs/G1.json','sha256':hashlib.sha256((T/'graphs/G1.json').read_bytes()).hexdigest(),'kind':'SOURCE_REFINEMENT'},{'id':'SOURCE_BOUNDARY_TEMPLATES','graph_path':'graphs/source-templates.json','kind':'PARTIAL_OPERATOR_EXPANSION'}],passes=pass_stats,coverage=all_rows,stop_reason=None,uncovered_items=['Semantic equivalence review for structurally enumerated sets','Exact current graph instance→runtime correlation','Opaque sampling/proposer/accept and metadata/caches internal paths','Expansion beyond size4 remains outside initial rule limit'])
scan['structural_candidate_index']='candidates/structural-index.json';write('fusion-scan.json',scan)
state=read('STATE.json');state.update(current_phase='P2',fusion_scan_status='SCANNING',next_read_only_action='Review structural patterns against source semantics and map31 prior checks; then record P3 coverage gaps and analyze existing module/task correlations',last_progress={'source_templates':len(templates),'structural_items':len(structural),'passes':pass_stats,'semantic_exhaustion_claimed':False});write('STATE.json',state)
print(json.dumps({'source_templates':len(templates),'known_boundary_nodes':sum(len(d['nodes']) for d in templates),'structural_items':len(structural),'passes':pass_stats,'application_changes':False},indent=2))
