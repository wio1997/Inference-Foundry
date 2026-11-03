from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket,statistics,math
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'resource-functional-attribution-and-uncertainty-v20.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');pmu=pin(M/'index.json','a19fca4b0a52bbfd6c3237fb94b5e36b086000ff4f13c7343116cef704626f9e')
ctx=pin(E/'fullchain-source-contexts-v15.json','1c003dbca3d72a216c646d209dc54bba38e936db32b0622b2efdf866b2b53ec9');intake=pin(E/'resource-role-intake-v20.json','06892785a41ff3fa19c6ca86515c71f4d4d3ee21ee5f8272d98e266488d8a175')
graph=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
short=lambda s:s.replace('/vllm-workspace/vllm-ascend/vllm_ascend/','a/').replace('/vllm-workspace/vllm/vllm/','v/')
chains={r['chain_key']:r for r in ctx['chains']};chain_by_key={tuple(k):ck for ck,keys in ctx['execution_keys_by_chain'].items()for k in keys};assert len(chain_by_key)==154940
leaf_index={r['leaf']:r['leaf_index']for r in intake['leaves']};leaf_rules={r['leaf_index']:r['module_candidates']for r in CONFIG['native_leaf_assignments']};generic=CONFIG['reviewed_generic_chain_roles']
graph_rules={json.dumps(r['signature'],sort_keys=True):r for r in CONFIG['first_graph_regime_assignments']}
ginfo={};gcontexts={};gblocks={}
for g in graph['graph_executions']:
 di=g['database_index'];gi=g['graph_ordinal'];phase=g['phase_records'][0]['phase']if len(g['phase_records'])==1 else'PHASE_UNBOUND';gcontexts[(di,gi)]='GRAPH:'+phase
 for b in g['blocks']:
  gblocks[(di,gi,b['block_ordinal'])]=b
  for rid in b['task_record_rowids']:assert(di,rid)not in ginfo;ginfo[(di,rid)]=(gi,b['block_ordinal'])
 for r in g['outside_marker_tasks']:
  rid=r['task']['task_record_rowid'];assert(di,rid)not in ginfo;ginfo[(di,rid)]=(gi,None)
assert len(ginfo)==132000
def direct_role(ck):
 r=chains[ck];fs=[short(f)for f in r['project_frames']];leaf=fs[0];li=leaf_index[leaf];text='\n'.join(fs);mods=leaf_rules[li];role=leaf
 if ck in generic:
  a=generic[ck];mods=[a['module']];role=a['role']
  if 'EMBEDDING'in role:mods=['M05']
  elif 'ATTENTION_INPUT_NORM'in role:mods=['M07']
 elif mods==['CALLER_RULE']:
  if li==46:
   mods=['M18']if '_store_standard_swa_kv'in text else['M09']if '/indexer.py'in text else['M08']if '_write_compressed_source'in text else['M07']
  elif li==56:
   mods=['M09']if '/indexer.py'in text else['M08']if '_write_compressed_source'in text else['M11']if 'dsa_v41.py(712): forward'in text else['M07']
  elif li==58:mods=['M19']if '/dspark.py'in text else['M21']
  elif li==60:mods=['M14']if 'model.py(1583): forward'in text else['M05']
  elif li in[126,127]:mods=['M21']
  elif li in[128,129]:mods=['M18']if '_project_shared_kv'in text else['M21']
  else:mods=['UNRESOLVED_DIRECT_HELPER']
 if any(': '+name in text for name in ['_dummy_run','dummy_run','execute_dummy_batch']):context='CALLER_DUMMY'
 elif '/dspark.py'in text or '/spec_decode/'in text:context='CALLER_DRAFT_OR_DRAFT_METADATA'
 elif '/models/deepseek_v41/'in text:context='CALLER_TARGET_MODEL_OR_ENGRAM'
 else:context='CALLER_WORKER_METADATA_OR_SAMPLING'
 return mods,role,context,li
role_by_chain={ck:direct_role(ck)for ck in chains}
def graph_role(sig,gi,bi,t,di,smla):
 rule=graph_rules.get(json.dumps(sig,sort_keys=True))
 if rule is None:return ['UNRESOLVED_GRAPH_REGIME'],'UNSEEN_TYPED_GRAPH_REGIME',None
 ri=rule['regime_index'];mods=rule['module_candidates'];role='GRAPH_REGIME_'+str(ri)
 if mods==['GRAPH_RULE']:
  if ri==12:mods=['M05','M21'];role='PREFIX_BOOL_ZERO_INIT_UNRESOLVED_FIELD'
  elif ri==23:mods=['M07','M08']if bi==20 else['M07'];role='SWA_KV_OR_C1_PROJECTION'if bi==20 else'SWA_KV_PROJECTION'
  elif ri==24:mods=['M07','M09']if bi in[2,8,14,20,24,28,32,36]else['M07'];role='Q_OR_INDEXER_INPUT_QUANT'
  elif ri in[25,27]:mods=['M07','M08']if bi in[2,8,14,20]else['M07'];role='SWA_KV_OR_COMPRESSED_NORM_ROPE'
  elif ri==28:mods=['M07','M08']if bi==20 else['M07'];role='SWA_OR_C1_CACHE_PUBLICATION'
  elif ri==29:
   a=smla[(gi,bi)]
   if t['endNs']<=a['startNs']:mods=['M07'];role='QUERY_ROPE_BEFORE_ATTENTION'
   elif t['startNs']>=a['endNs']:mods=['M11'];role='OUTPUT_ROPE_AFTER_ATTENTION'
   else:mods=['M07','M11'];role='ROPE_TEMPORAL_ROLE_UNRESOLVED'
 return mods,role,ri
def stats(xs):return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None
def union(rows):
 end=None;total=0
 for r in sorted(rows,key=lambda r:(r['startNs'],r['endNs'])):
  a,b=r['startNs'],r['endNs']
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total
def shape_mac(c):
 if c['opType_text']not in['MatMulV2','MatMulV3','QuantBatchMatmulV3','TransposeBatchMatMul']:return None
 def dims(s):
  try:return [int(x)for x in s.strip('"').split(';')[0].split(',')]
  except (TypeError,ValueError,AttributeError):return None
 a=dims(c['inputShapes_text']);o=dims(c['outputShapes_text'])
 if not a or not o or len(a)<2 or len(o)<2 or min(a+o)<=0 or math.prod(a[:-1])!=math.prod(o[:-1]):return None
 return math.prod(o)*a[-1]
rows=[];assignments=[];shape_rows=defaultdict(list);unseen=defaultdict(list);direct_counts=Counter();unresolved_direct=Counter()
for ref in idx['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);pd=pin(Path(pmu['databases'][di]['path']),pmu['databases'][di]['sha256']);tasks={t['task_record_rowid']:t for t in d['TASK_records']};defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']};metrics={}
 for p in pd['groups']:
  assert len(p['native_TASK_candidates'])==1 and not p['duplicate_metric_names'];rid=p['native_TASK_candidates'][0];assert rid not in metrics;metrics[rid]={x['metric_name']:x['value']for x in p['metric_rows']}
 smla={}
 for (ddi,rid),(gi,bi)in ginfo.items():
  if ddi==di and bi is not None and defs[tasks[rid]['globalTaskId']]['opType_text']=='SparseFlashMla':assert(gi,bi)not in smla;smla[(gi,bi)]=tasks[rid]
 for rid,t in tasks.items():
  if t['globalTaskId']not in defs:continue
  c=defs[t['globalTaskId']];key=di,rid;op=c['opType_text'];ck=chain_by_key.get(key);gi=bi=ri=li=None
  sig={k:c.get(k)for k in ['opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text','blockNum','mixBlockNum']}
  if ck is not None:
   assert key not in ginfo;mods,role,context,li=role_by_chain[ck];origin='DIRECT';direct_counts[li]+=1
  elif key in ginfo:
   gi,bi=ginfo[key];mods,role,ri=graph_role(sig,gi,bi,t,di,smla);context=gcontexts[(di,gi)];origin='GRAPH'
   if ri is None:unseen[json.dumps(sig,sort_keys=True)].append([di,rid,gi,bi])
  else:
   origin='QUEUE';context='QUEUE_SOURCE_BOUND_V15'
   if op=='ForeachAddList':mods=['M13'];role='QUEUE_DERIVED_ROUTED_SHARED_ADD'
   elif op in['Index','IndexCheck']:mods=['M21'];role='QUEUE_DERIVED_DRAFT_POSITION_INDEX'
   else:mods=['UNRESOLVED_QUEUE'];role='UNRESOLVED_QUEUE'
  source_mods=mods[:]
  if str(op).startswith('hcom_'):mods=['M23'];role='HCCL_NAMED_COMPUTE_TABLE_NODE'
  if any(m.startswith('UNRESOLVED')for m in mods):unresolved_direct[role]+=1
  row=dict(database_index=di,task_record_rowid=rid,startNs=t['startNs'],endNs=t['endNs'],op_type=op,origin=origin,context=context,module_candidates=mods,semantic_source_module_candidates=source_mods,role=role,metrics=metrics.get(rid,{}),shape_MAC_candidate=shape_mac(c),signature=sig)
  rows.append(row);shape_rows[(tuple(mods),role,origin,context,json.dumps(sig,sort_keys=True))].append(row)
  assignments.append([di,rid,origin,mods,source_mods,role,context,li,gi,bi,ri,rid in metrics])
assert len(rows)==286950 and Counter(r['origin']for r in rows)==Counter(DIRECT=154940,GRAPH=132000,QUEUE=10)
assert sum(bool(r['metrics'])for r in rows)==286101
for r in CONFIG['native_leaf_assignments']:
 expected=intake['leaves'][r['leaf_index']]['compute_executions'];assert direct_counts[r['leaf_index']]==expected,(r['leaf_index'],expected,direct_counts[r['leaf_index']])
def summarize(rr):
 perdb=defaultdict(list)
 for r in rr:perdb[r['database_index']].append(r)
 out=dict(executions=len(rr),PMU_groups=sum(bool(r['metrics'])for r in rr),missing_PMU=sum(not r['metrics']for r in rr),origins=dict(Counter(r['origin']for r in rr)),contexts=dict(Counter(r['context']for r in rr)),native_task_duration_ns=stats([r['endNs']-r['startNs']for r in rr]),native_task_duration_sum_ns=sum(r['endNs']-r['startNs']for r in rr),sum_of_per_database_native_interval_union_ns=sum(union(v)for v in perdb.values()),fields={},shape_MAC_model=dict(qualified_executions=sum(r['shape_MAC_candidate']is not None for r in rr),MAC_candidates=sum(r['shape_MAC_candidate']or 0 for r in rr),qualification='Product(output logical shape)*last input dimension,only when row/batch dimensions agree;not executed instruction count,necessary fused work,unconditional lower bound or physical efficiency.'))
 for engine in ['aic','aiv']:
  for ratio in sorted({k for r in rr for k in r['metrics']if k.startswith(engine+'_')and k.endswith('_ratio')}):
   vv=[(r['metrics'][engine+'_total_cycles'],r['metrics'][ratio])for r in rr if r['metrics'].get(engine+'_total_cycles',0)>0 and ratio in r['metrics']];den=sum(c for c,v in vv)
   out['fields'][ratio]=dict(positive_cycle_groups=len(vv),cycles=den,weighted_numerator=sum(c*v for c,v in vv),cycle_weighted_reported_ratio=sum(c*v for c,v in vv)/den if den else None)
 return out
def ratio_bounds(certain,optional,engine,ratio):
 def vals(rr):return [(r['metrics'][engine+'_total_cycles'],r['metrics'][ratio])for r in rr if r['metrics'].get(engine+'_total_cycles',0)>0 and ratio in r['metrics']]
 cc=vals(certain);oo=vals(optional);den=sum(c for c,v in cc);num=sum(c*v for c,v in cc)
 def extreme(reverse):
  ds,ns=den,num;best=ns/ds if ds else None
  for c,v in sorted(oo,key=lambda x:x[1],reverse=reverse):
   ds+=c;ns+=c*v;q=ns/ds
   if best is None or(reverse and q>best)or(not reverse and q<best):best=q
  return best
 return dict(certain_positive_cycle_groups=len(cc),optional_positive_cycle_groups=len(oo),certain_ratio=num/den if den else None,minimum_over_optional_membership=extreme(False),maximum_over_optional_membership=extreme(True),bound_scope='All subsets of ambiguous candidate rows with positive denominator;ignores coupled role-count constraints,so may be loose. MissingPMU and unassigned regimes are outside this counter-only interval.')
modules=[]
for mod in ['M'+str(i).zfill(2)for i in range(26)]:
 certain=[r for r in rows if r['module_candidates']==[mod]];optional=[r for r in rows if len(r['module_candidates'])>1 and mod in r['module_candidates']]
 modules.append(dict(module=mod,certain=summarize(certain),optional_execution_count=len(optional),possible_execution_count=len(certain)+len(optional),counter_ratio_assignment_bounds={ratio:ratio_bounds(certain,optional,engine,ratio)for engine,ratio in [('aic','aic_mac_ratio'),('aiv','aiv_vec_ratio'),('aic','aic_mte2_ratio'),('aiv','aiv_mte2_ratio')]},per_context_certain=[dict(context=k,**summarize([r for r in certain if r['context']==k]))for k in sorted({r['context']for r in certain})]))
shapes=[dict(module_candidates=list(k[0]),role=k[1],origin=k[2],context=k[3],signature=json.loads(k[4]),summary=summarize(rr))for k,rr in shape_rows.items()]
ambiguous=[r for r in rows if len(r['module_candidates'])>1];unassigned=[r for r in rows if any(m.startswith('UNRESOLVED')for m in r['module_candidates'])]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),compute_executions=len(rows),PMU_groups=286101,missing_PMU=849,origin_counts=dict(Counter(r['origin']for r in rows)),certainly_attributed=sum(len(r['module_candidates'])==1 and r not in unassigned for r in rows),ambiguous_execution_count=len(ambiguous),unassigned_execution_count=len(unassigned),unseen_graph_regimes=len(unseen),ambiguous_candidate_sets=dict(Counter('|'.join(r['module_candidates'])for r in ambiguous)),functional_module_buckets=sum(m['certain']['executions']>0 for m in modules),shape_role_context_regimes=len(shapes),no_new_source_DB_model_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,limits=['Disjoint recorded functional resource buckets are not complete hierarchical module execution/cost closure. Target/draft/dummy/capture scope is explicit.','Native durations/unions and reported pipeline cycles are different metrics. Unions across module categories can overlap;do not sum them as request critical path.','Reported PMU ratio fractions are nonexclusive/counter-derived;positive-cycle engine groups are not hardware peak efficiency,MFU,actual bytes or a causal speedup ceiling.','Typed-role ambiguity is retained;counter intervals range over observed optional membership only. No missingPMU or unseen regime is filled with0.','Graph phase names are exact retained metadata,not attested formal iteration labels.16unbound graph executions remain unbound.','No cross-rank timestamps,observer counts or historical169 CPU timings are joined here.','Shape MAC candidates are a conditional work model,not an unconditional lower bound for all algorithms/fused implementations.'])
assert summary['certainly_attributed']+len(ambiguous)+len(unassigned)==len(rows)
result=dict(summary=summary,module_resource_buckets=modules,shape_role_context_regimes=shapes,unseen_graph_regimes=[dict(signature=json.loads(k),execution_keys=v)for k,v in unseen.items()],execution_assignment_columns=['database_index','TASK_rowid','origin','module_candidates','semantic_source_module_candidates','role','context','leaf_index','graph_ordinal','block_ordinal','graph_regime_index','has_PMU'],execution_assignments=assignments,attribution_rules=CONFIG)
payload=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
publicmods=[dict(module=m['module'],certain_executions=m['certain']['executions'],optional_executions=m['optional_execution_count'],missing_PMU=m['certain']['missing_PMU'],native_task_duration_sum_ns=m['certain']['native_task_duration_sum_ns'],sum_per_database_native_union_ns=m['certain']['sum_of_per_database_native_interval_union_ns'],counter_ratio_assignment_bounds=m['counter_ratio_assignment_bounds'],shape_MAC_model=m['certain']['shape_MAC_model'])for m in modules]
print(json.dumps(dict(summary=summary,module_resources=publicmods,unseen_graph_regime_samples=result['unseen_graph_regimes'][:4],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
