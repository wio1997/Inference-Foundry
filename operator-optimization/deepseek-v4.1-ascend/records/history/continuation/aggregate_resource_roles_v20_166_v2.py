import json,zlib,base64
CONFIG=json.loads(zlib.decompress(base64.b64decode('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')))
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
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),compute_executions=len(rows),PMU_groups=286101,missing_PMU=849,origin_counts=dict(Counter(r['origin']for r in rows)),certainly_attributed=sum(len(r['module_candidates'])==1 and not any(m.startswith('UNRESOLVED')for m in r['module_candidates'])for r in rows),ambiguous_execution_count=len(ambiguous),unassigned_execution_count=len(unassigned),unseen_graph_regimes=len(unseen),ambiguous_candidate_sets=dict(Counter('|'.join(r['module_candidates'])for r in ambiguous)),functional_module_buckets=sum(m['certain']['executions']>0 for m in modules),shape_role_context_regimes=len(shapes),no_new_source_DB_model_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,limits=['Disjoint recorded functional resource buckets are not complete hierarchical module execution/cost closure. Target/draft/dummy/capture scope is explicit.','Native durations/unions and reported pipeline cycles are different metrics. Unions across module categories can overlap;do not sum them as request critical path.','Reported PMU ratio fractions are nonexclusive/counter-derived;positive-cycle engine groups are not hardware peak efficiency,MFU,actual bytes or a causal speedup ceiling.','Typed-role ambiguity is retained;counter intervals range over observed optional membership only. No missingPMU or unseen regime is filled with0.','Graph phase names are exact retained metadata,not attested formal iteration labels.16unbound graph executions remain unbound.','No cross-rank timestamps,observer counts or historical169 CPU timings are joined here.','Shape MAC candidates are a conditional work model,not an unconditional lower bound for all algorithms/fused implementations.'])
assert summary['certainly_attributed']+len(ambiguous)+len(unassigned)==len(rows)
result=dict(summary=summary,module_resource_buckets=modules,shape_role_context_regimes=shapes,unseen_graph_regimes=[dict(signature=json.loads(k),execution_keys=v)for k,v in unseen.items()],execution_assignment_columns=['database_index','TASK_rowid','origin','module_candidates','semantic_source_module_candidates','role','context','leaf_index','graph_ordinal','block_ordinal','graph_regime_index','has_PMU'],execution_assignments=assignments,attribution_rules=CONFIG)
payload=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
publicmods=[dict(module=m['module'],certain_executions=m['certain']['executions'],optional_executions=m['optional_execution_count'],missing_PMU=m['certain']['missing_PMU'],native_task_duration_sum_ns=m['certain']['native_task_duration_sum_ns'],sum_per_database_native_union_ns=m['certain']['sum_of_per_database_native_interval_union_ns'],counter_ratio_assignment_bounds=m['counter_ratio_assignment_bounds'],shape_MAC_model=m['certain']['shape_MAC_model'])for m in modules]
print(json.dumps(dict(summary=summary,module_resources=publicmods,unseen_graph_regime_samples=result['unseen_graph_regimes'][:4],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
