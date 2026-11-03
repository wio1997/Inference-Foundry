"""Independent raw-native, port-union and restricted-ceiling audit."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product
from fractions import Fraction
import argparse,hashlib,json,sqlite3,statistics,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
 def ck(k,v):
  assert v,k
  checks.append(k)
 def pin(x):
  b=Path(x['path']).read_bytes();return len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
 for x in read(O/'post-pre-inputs-v147.json')['inputs']:ck('input',pin(x))
 for x in read(O/'post-pre-source-contract-v147.json')['parts']:ck('source',pin(x)and Path(x['path']).read_text().splitlines()[x['first_line']-1:x['last_line']]==x['source'])
 p=read(O/'post-pre-native-frontiers-v147.json');bs=p['bindings'];bydb=defaultdict(list);padb=defaultdict(list);api_only=defaultdict(list);seen=set();roles=Counter()
 for i,x in enumerate(bs):bydb[x['database_index']].append((i,x))
 for x in p['pad_groups']:padb[x['database_index']].append(x)
 for x in p['API_only']:api_only[x['database_index']].append(x)
 for ref in p['databases']:
  di=ref['database_index'];ck('rawpin',pin(ref['raw_database']));ck('cachepin',pin(ref['cached_native']));db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs={(x['id'],x['stackDepth'],x['stack'])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS')};aa={x['api_rowid']:x for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')};ts={x['task_record_rowid']:x for x in db.execute('SELECT rowid AS task_record_rowid,*FROM TASK')};ds={x['globalTaskId']:x for x in db.execute('SELECT rowid AS compute_record_rowid,*FROM COMPUTE_TASK_INFO')};ci=defaultdict(set)
  for x in db.execute('SELECT *FROM CONNECTION_IDS'):ci[x['id']].add(x['connectionId'])
  def api(a):return all(a[k]==aa[a['api_rowid']][k]for k in aa[a['api_rowid']].keys())and a['name_text']==ss[aa[a['api_rowid']]['name']]
  for i,r in bydb[di]:
   a,t,d=r['API'],r['TASK'],r['definition'];raw=ts[t['task_record_rowid']];de=ds[t['globalTaskId']];key=di,t['task_record_rowid'];ck('API'+str(i),api(a));ck('TASK'+str(i),all(t[k]==raw[k]for k in raw.keys())and a['api_rowid']in t['native_linked_API_rowids']and t['connectionId']in ci[a['connectionId']]);ck('Compute'+str(i),all((ss[de[k[:-5]]]if k.endswith('_text')else de[k])==v for k,v in d.items()));ck('frames'+str(i),bool(r['selected_frames'])and all((f['id'],f['stackDepth'],f['stack'])in fs and f['id']==a['callchainId']and f['value']==ss[f['stack']]for f in r['selected_frames']));ck('unique',key not in seen);seen.add(key);roles.update(r['roles'])
  if padb[di]or api_only[di]:
   cache=read(Path(ref['cached_native']['path']));byapi=defaultdict(list)
   for t in cache['TASK_records']:
    if t['globalTaskId']in ds:
     for a in t.get('native_linked_API_rowids',[]):byapi[a].append(t['task_record_rowid'])
   for r in padb[di]:
    a=r['API'];zz=[bs[i]for i in r['sibling_bindings']];ck('pad_exact_API_siblings',api(a)and len({z['API']['api_rowid']for z in zz})==1 and zz[0]['API']==a and set(byapi[a['api_rowid']])=={z['TASK']['task_record_rowid']for z in zz});ck('pad_source',all(any('/ops/fused_moe/prepare_finalize.py('+str(r['source_line'])+'):'in f['value']for f in z['selected_frames'])for z in zz));ck('pad_gap_preserved',not r['MemSet_address_length_workspace_or_destination_role_inferred']and r['exact_shared_API_not_equal_storage_or_zero_cost'])
   for r in api_only[di]:ck('API_only_cache_grade',api(r['API'])and not byapi[r['API']['api_rowid']]and r['role']=='MERGE')
  db.close()
 roles.update(r['role']for r in p['API_only']);ck('14111',len(bs)==len(seen)==p['unique_TASK_bindings']==14111 and sum(p['roles'].values())==p['role_binding_count']==17544 and len(p['API_only'])==1)
 # Duplicate occurrences of same-role/cast roles retain per-call occurrence counts separately.
 for k in roles:ck('roles_not_fabricated',roles[k]<=p['roles'][k])
 old_t={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-sp-eager-native-frontiers-v134.json')['records']};old_m={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-moe-native-frontiers-v141.json')['records']};old_x={(r['database_index'],r['forward_API_rowid'],r['to_block']):r for r in read(O/'hc-moe-native-frontiers-v141.json')['bridges']};old_d={(r['database_index'],r['source_order_ordinal']):r for r in read(O/'hc-draft-native-frontiers-v142.json')['records']};used=set();geo=Counter();flags=Counter();dur=defaultdict(lambda:[[],[],[]]);env=defaultdict(list)
 def use(i):used.add(i);return bs[i]
 def same(i,old):return use(i)['TASK']['task_record_rowid']==old['TASK']['task_record_rowid']and use(i)['API']==old['API']
 for kind in ['target_intra','target_cross','draft_intra','draft_cross']:
  for r in p[kind]:
   a,z=use(r['left']),use(r['right']);m=int(z['definition']['inputShapes_text'].strip('"').split(';')[0].split(',')[0]);ck('Post_Pre_shape',a['definition']['opType_text']=='HcPost'and z['definition']['opType_text']=='HcPre'and a['definition']['outputShapes_text']==f'"1,{m},4,5120"'and z['definition']['inputShapes_text'].startswith(f'"{m},4,5120;24,20480;')and a['definition']['outputDataTypes_text']=='DT_BF16');ck('Post_Pre_CPU_order',a['API']['globalTid']==z['API']['globalTid']and int(a['API']['endNs'])<=int(z['API']['startNs']));lo,hi=a['TASK'],z['TASK'];dur[kind][0].append(lo['endNs']-lo['startNs']);dur[kind][1].append(hi['endNs']-hi['startNs']);dur[kind][2].append(hi['startNs']-lo['endNs'])
   if 'intra'in kind:
    pr=use(r['earlier_premix_and_Post_coeff_provider']);ck('earlier_provider',int(pr['API']['endNs'])<=int(a['API']['startNs'])and pr['API']['globalTid']==a['API']['globalTid']);use(r['norm']);use(r['future_Post'])
   else:use(r['incoming_premix_provider'])
   if kind=='target_intra':
    key=r['database_index'],r['forward_API_rowid'],r['block'];ot,om=old_t[key],old_m[key];ck('intra_old',same(r['left'],ot['stages']['POST'])and same(r['right'],ot['stages']['FFN_PRE'])and same(r['earlier_premix_and_Post_coeff_provider'],ot['stages']['HC'])and same(r['norm'],om['stages']['FFN_NORM'])and same(r['future_Post'],ot['stages']['FFN_POST']))
    for k,i in r['consumers'].items():
     if i is None:ck('same_old_APIonly',om['stages'][k]['TASK']is None)
     else:ck('consumer',same(i,om['stages'][k]))
    pp=int(bs[r['consumers']['MC2']]['definition']['inputShapes_text'].strip('"').split(',')[0]);geo[m,pp]+=1;ck('M_vs_P',bs[r['consumers']['SHARED_QUANT']]['definition']['inputShapes_text']==f'"{m},5120"'and bs[r['consumers']['ROUTER']]['definition']['inputShapes_text']==f'"{m},5120;384,5120"'and bs[r['norm']]['definition']['outputDataTypes_text']=='DT_BF16;FLOAT');tt=[bs[i]['TASK']for i in [r['right'],r['norm'],r['consumers']['ROUTER']]];env[m,pp].append(max(t['endNs']for t in tt)-min(t['startNs']for t in tt))
   elif kind=='target_cross':
    ox=old_x[r['database_index'],r['forward_API_rowid'],r['to_block']];ck('targetcross_old',same(r['left'],ox['previous_FFN_Post'])and same(r['right'],ox['next_HC'])and same(r['incoming_premix_provider'],ox['previous_FFN_Pre'])and r['Engram_positive_APIs']==ox['next_Engram_positive_APIs']and r['aux_allGather']==ox['next_aux_allGather']);flags[bool(r['Engram_positive_APIs']),r['aux_mean']is not None]+=1
    if r['aux_mean']is not None:ck('aux_old',same(r['aux_mean'],ox['next_aux_mean']))
   else:
    od=old_d[r['database_index'],r['source_order_ordinal']];ls=od['layers']
    if kind=='draft_intra':
     s=ls[r['block']]['stages'];ck('draftintra_old',all(same(r[k],s[v])for k,v in [('left','POST'),('right','FFN_PRE'),('earlier_premix_and_Post_coeff_provider','HC'),('norm','FFN_NORM'),('future_Post','FFN_POST')]))
    else:ck('draftcross_old',same(r['left'],ls[r['from_block']]['stages']['FFN_POST'])and same(r['right'],ls[r['to_block']]['stages']['HC'])and same(r['incoming_premix_provider'],ls[r['from_block']]['stages']['FFN_PRE'])and not r['Engram_in_source_draft_constructor'])
 ck('2416_edges',sum(len(p[k])for k in ['target_intra','target_cross','draft_intra','draft_cross'])==p['source_Post_Pre_edges']==2416 and [len(p[k])for k in ['target_intra','target_cross','draft_intra','draft_cross']]==[960,936,312,208]);ck('geo',geo=={(800,816):320,(816,816):320,(834,834):320});ck('guards',flags=={(True,False):48,(False,True):72,(False,False):816})
 regs={r['regime_id']:r for r in read(O/'operator-analysis-coverage-ledger-v146.json')['regimes']};pc=Counter();pkeys=set();padids=set()
 for r in p['pad_groups']:
  pd,ms=use(r['PadV3'])['definition'],use(r['MemSet'])['definition'];reg=regs[r['pending_MemSet_regime']];ck('MemSet_still_unbound',ms['opType_text']=='MemSet'and ms['inputShapes_text']==ms['outputShapes_text']==ms['inputDataTypes_text']==ms['outputDataTypes_text']=='N/A'and reg['executions']==320 and'PENDING'in reg['state']and f'({r["source_line"]}):'in reg['role']);pc[r['role']]+=1;pkeys.add((r['database_index'],r['forward_API_rowid'],r['block']))
  for i in r['sibling_bindings']:padids.add(i);use(i)
  if r['role']=='HIDDEN_PAD':ck('hidden_pad',pd['inputShapes_text']=='"800,5120;4;"'and pd['outputShapes_text']=='"816,5120"'and pd['inputDataTypes_text']=='DT_BF16;INT32;DT_BF16')
  elif r['role']=='LOGITS_PAD':ck('logits_pad',pd['inputShapes_text']=='"800,384;4;"'and pd['outputShapes_text']=='"816,384"'and pd['inputDataTypes_text']=='FLOAT;INT32;FLOAT')
  else:
   ck('mask_pad',pd['inputShapes_text']=='"800;2;"'and pd['outputShapes_text']=='"816"'and pd['inputDataTypes_text']=='INT8;INT32;INT8');cc=[bs[i]['definition']for i in r['sibling_bindings']if bs[i]['definition']['opType_text']=='Cast'];ck('mask_casts',[(x['inputShapes_text'],x['inputDataTypes_text'],x['outputDataTypes_text'])for x in cc]==[('"800"','BOOL','INT8'),('""','BOOL','INT8'),('"816"','INT8','BOOL')])
 ck('pad960_2880',pc=={'HIDDEN_PAD':320,'LOGITS_PAD':320,'MASK_PAD':320}and len(pkeys)==320 and len(padids)==p['new_pad_sibling_TASK_bindings']==2880 and used==set(range(len(bs))))
 obs=read(O/'post-pre-observed-frontiers-v147.json');st=lambda v:dict(count=len(v),min=min(v),median=statistics.median(v),max=max(v))
 for k,v in dur.items():ck('native_stats',obs['task_only_by_edgekind'][k]==dict(left_duration_ns=st(v[0]),right_duration_ns=st(v[1]),recorded_gap_ns=st(v[2])))
 # Independent scalar formula plus unique port partition and complete action domain.
 ip=read(O/'post-pre-interface-union-cases-v147.json');cases=set();over=0
 for x in ip['cases']:
  m,pp,h,e=x['M'],x['P'],x['H'],x['E'];a=set(x['actions']);base=x['baseline_port_bytes'];rm=set(x['removed_ports']);remain=x['remaining_port_bytes'];mh=m*h;me=m*e;expected=mh*(8*('POST_PRE_FORWARD'in a)+4*('COLLAPSE_NORM'in a)+2*('NORM_SHARED_FORWARD'in a)+8*('NORM_ROUTER'in a)+4*(pp>m and'NORM_PAD_DIRECT'in a)+8*(x['aux']and'POST_AUX_FORWARD'in a))+8*me*(pp>m and'ROUTER_PAD_DIRECT'in a);baseline=42*mh+8*me+8*mh*x['aux']+(4*pp*h+8*pp*e if pp>m else 0)
  ck('port_union',x['logical_union_credit']==expected==sum(base[k]for k in rm)and sum(base.values())==x['baseline_selected_port_sum']==baseline and remain=={k:v for k,v in base.items()if k not in rm}and sum(remain.values())+expected==baseline and not x['all_native_runtime_or_reader_gates_proved']);ck('raw_late_residual_retained',remain['X_STORE']==remain['X_FUTURE_POST_LOAD']==8*mh and'X_STORE'not in rm);ck('no_PAD_credit_in_unpadded',pp>m or not({'ZB_PAD_READ','LF_PAD_READ'}&rm));ck('headline_not_bound',x['headline_overcredit']==x['incorrect_standalone_headline_sum']-expected>=0)
  if x['headline_overcredit']>0:over+=1
  key=m,pp,x['aux'],tuple(sorted(a));ck('unique_case',key not in cases);cases.add(key)
 expected_cases={(m,pp,aux,tuple(sorted(a for i,a in enumerate(ip['action_universe'])if mask>>i&1)))for(m,pp),aux,mask in product([(800,816),(816,816),(834,834)],[False,True],range(128))};ck('768complete',cases==expected_cases and len(cases)==ip['count']==768 and over>0 and not ip['physical_gain_or_lowerbound'])
 rc=read(O/'post-pre-symbolic-resource-and-retention-v147.json')
 for x in rc['geometry']:
  m,pp,h,e=x['M'],x['P'],x['H'],x['E'];ck('resource_program',x['classical_next_HC_coefficient_ops']==2*m*(4*h)*24 and x['classical_router_ops']==2*m*h*e and x['classical_shared_INT8_GEMM_ops']==6*m*h*2304 and x['classical_same_mode_coeff_router_union']==960*m*h and x['measured_native_Pre_Norm_Router_TASK_envelope_ns']==st(env[m,pp]));ck('symbolic_not_physical',x['valid_actual_physical_capacity_upper']is None and x['time_lower_bound_ns']is None and x['measured_over_bound_ratio']is None and x['not_all_equivalent_algorithms_bound']and'ACTUAL_PHYSICAL_NOT_ESTABLISHED'in x['theory_state'])
 for x in rc['retained_payload_cases']:ck('retention_scope',x['raw4H_BF16_payload']==8*x['tile_rows']*5120 and x['actual_available_payload_budget']is None and x['fullrow_retention_only_not_universal'])
 w=read(O/'post-pre-round-quant-version-progress-witnesses-v147.json');x=w['Post_BF16_boundary'];bf=lambda y:struct.unpack('>f',struct.pack('>I',(lambda u:(u+32767+((u>>16)&1))&0xffff0000)(struct.unpack('>I',struct.pack('>f',y))[0])))[0];rd=list(map(bf,x['Post_unrounded']));ck('Post_round_witness',rd==x['Post_rounded']and bf(sum(a*b for a,b in zip(rd,x['incoming_premix'])))==x['original_collapse_BF16']==0. and bf(sum(a*b for a,b in zip(x['Post_unrounded'],x['incoming_premix'])))==x['wrong_skip_Post_round_collapse_BF16']!=0.)
 x=w['quant_not_affine_homomorphism'];quant=lambda v:[round(Fraction(a)*127/max(abs(Fraction(b))for b in v))for a in v];ck('quant_collision',quant(x['old0'])==quant(x['old1'])==x['same_old_INT8']and quant(x['Post0'])==x['new_INT8_0']and quant(x['Post1'])==x['new_INT8_1']and x['new_INT8_0']!=x['new_INT8_1']and not x['old_packet_sufficient']);ck('quant_affine_values',x['Post0']==[a+b for a,b in zip(x['old0'],x['same_Post_branch'])]and x['Post1']==[a+b for a,b in zip(x['old1'],x['same_Post_branch'])])
 x=w['aux_Engram_versions'];ck('versions',x['rawX1']==[a+b for a,b in zip(x['rawX0'],x['Engram_update'])]and sum(x['rawX0'])/2==x['aux_of_old']!=x['wrong_aux_of_new']==sum(x['rawX1'])/2 and not x['early_pre_on_X0_legal']);x=w['finite_progress'];edges=x['wait_edges'];ck('abstract_cycle_not_native',all(edges[i][1]==edges[(i+1)%3][0]for i in range(3))and not x['actual_native_or_current_deadlock'])
 comp=read(O/'post-pre-composition-v147.json');ck('sixpairs',comp['new_pairs']==[['F03',x]for x in ['F20','F21','F22','F24','F27','F31']]and not comp['all_runtime_gates_proved'])
 if args.preflight:
  rel=read(O/'candidate-relation-clauses-v146.json')
  for pair in comp['new_pairs']:ck('pending',not any({r['a'],r['b']}==set(pair)for r in rel['clauses']))
  for gid,m in comp['group_extensions']:ck('newgroup',m not in next(g for g in rel['higher_order_groups']if g['id']==gid)['members'])
  man=read(O/'checkpoint-manifest-v146.json')
 else:
  cp=read(O/'REVIEW-CHECKPOINT-v147.json');ck('progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0);rel=read(O/'candidate-relation-clauses-v147.json');ck('192_26',len(rel['clauses'])==rel['clause_count']==192 and len({tuple(sorted((r['a'],r['b'])))for r in rel['clauses']})==192 and len(rel['higher_order_groups'])==26)
  for c in comp['clauses']:
   r=next(r for r in rel['clauses']if r['id']==c['id']);ck('conditional',r['a']==c['a']and r['b']==c['b']and not r['all_runtime_gates_proved']and r['critical_path_gain_us']is None)
  for gid,m in comp['group_extensions']:
   g=next(g for g in rel['higher_order_groups']if g['id']==gid);ck('groupnotclosed',m in g['members']and not g['all_subsets_analyzed']and not g['legal_joint_fusion_proved'])
  cv=read(O/'candidate-relation-coverage-v147.json');nv=read(O/'candidate-pair-navigation-v147.json');ck('403pending',cv['scoped_conditional_pairs']==192 and cv['remaining_pair_relations']==nv['remaining_central_pair_reconciliation']==403);a=read(O/'operator-analysis-coverage-ledger-v146.json');z=read(O/'operator-analysis-coverage-ledger-v147.json');ck('nativeglobalunchanged',all(a[k]==z[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a=read(O/'candidate-profile-ledger-v146.json');z=read(O/'candidate-profile-ledger-v147.json');ck('broadergaps',[r.get('conclusion_changing_gap')for r in a['candidates']]==[r.get('conclusion_changing_gap')for r in z['candidates']]);mm=read(O/'module-phase-branch-matrix-v147.json');ck('358_26',len(mm['modules'])==26 and sum(len(m['branches'])for m in mm['modules'])==358)
  for r in read(O/'candidate-record-index-v147.json')['model_candidates']:ck('master',hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
  ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');man=read(O/'checkpoint-manifest-v147.json')
 for x in man['artifacts']:ck('manifest',pin(x))
 result=dict(status='PASS',checks=len(checks),preflight=args.preflight,unique_TASKs=14111,Post_Pre_edges=2416,pad_groups=960,pad_sibling_TASKs=2880,selected_API_only=1,original_M12_MemSet_pending=960,interface_cases=768,headline_overcredit_cases=over,new_scoped_pairs=6,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'post-pre-preflight-v147.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
