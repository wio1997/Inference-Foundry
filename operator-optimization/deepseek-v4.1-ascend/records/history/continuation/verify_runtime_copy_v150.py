"""Independently rebuild copy control identities from immutable raw SQLite rows."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,sqlite3,math
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
 args=argparse.ArgumentParser();args.add_argument('--preflight',action='store_true');args=args.parse_args();checks=0
 def ck(label,value):
  nonlocal checks
  checks+=1
  assert value,label
 def pin(z):
  p=Path(z['path']);return p.stat().st_size==z['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==z['sha256']
 x=read(O/'runtime-copy-control-origins-v150.json');g=read(O/'runtime-copy-existing-graph-controls-v150.json');e=read(O/'runtime-copy-source-role-and-logical-extents-v150.json');c=read(O/'runtime-copy-field-and-legality-contract-v150.json');w=read(O/'runtime-copy-control-boundary-witnesses-v150.json');old=read(O/'ta<REDACTED_TOKEN>.json');bydb=defaultdict(dict);le={(z['database_index'],z['TASK_rowid']):z for z in e['records']};gg=defaultdict(list)
 for z in read(O/'runtime-copy-inputs-v150.json')['inputs']:ck('input_pin',pin(z))
 for z in x['records']:ck('unique_saved_TASK',(z['database_index'],z['TASK']['TASK_rowid'])not in {(z['database_index'],a)for a in bydb[z['database_index']]});bydb[z['database_index']][z['TASK']['TASK_rowid']]=z
 for z in g['controls']:gg[z['database_index']].append(z)
 oldmap={(r['database_index'],r['TASK']['TASK_rowid']):r for r in old['runtime_copy_records']};tot=Counter();sources=Counter();flows=Counter();directions=Counter();gcount=Counter();over=[];chains_used=set();nraw=0
 def smallest(rows):return [z for z in rows if not any(q['api_rowid']!=z['api_rowid']and z['startNs']<=q['startNs']and q['endNs']<=z['endNs']and(z['startNs'],z['endNs'])!=(q['startNs'],q['endNs'])for q in rows)]
 for spec in x['databases']:
  di=spec['database_index'];d=sqlite3.connect('file:'+spec['raw_database']['path']+'?mode=ro',uri=True);d.row_factory=sqlite3.Row;ss=dict(d.execute('select id,value from STRING_IDS'));ts={z['TASK_rowid']:dict(z)for z in d.execute('select rowid AS TASK_rowid,* from TASK')if ss[z['taskType']]=='MEMCPY_ASYNC'};ck('all_raw_copies',set(ts)==set(bydb[di]));nraw+=len(ts);pa={};qbytid=defaultdict(list);enbyflow=defaultdict(list);named=defaultdict(list);cif=defaultdict(set)
  for z in d.execute('select id,connectionId from CONNECTION_IDS'):cif[z[0]].add(z[1])
  for z in d.execute('select rowid AS api_rowid,* from PYTORCH_API'):
   a=dict(z);a.update(startNs=int(a['startNs']),endNs=int(a['endNs']),name_text=ss.get(a['name']),input_shapes_text=ss.get(a['inputShapes']),input_dtypes_text=ss.get(a['inputDtypes']));pa[a['api_rowid']]=a
   if a['type']==50002 and a['name_text'].startswith('Dequeue@'):qbytid[a['globalTid']].append(a)
   if a['type']==50002 and a['name_text'].startswith('Enqueue@'):
    for f in cif[a['connectionId']]:enbyflow[f].append(a)
   if a['type']==50001:named[a['globalTid'],a['name_text']].append(a)
  ca=defaultdict(list)
  for z in d.execute('select rowid AS CANN_API_rowid,* from CANN_API'):
   a=dict(z);a['name_text']=ss[a['name']];ca[a['connectionId']].append(a)
  chainrows=defaultdict(list)
  for z in d.execute('select id,stack,stackDepth from PYTORCH_CALLCHAINS order by id,stackDepth'):chainrows[z[0]].append((z[2],z[1],ss.get(z[1])))
  def ch(cid):return [z[2]for z in chainrows[cid]]if cid is not None else None
  for ctrl in gg[di]:
   na=ctrl['CANN_API'];ck('graph_CANN_raw',na['cann_api_rowid']in[a['CANN_API_rowid']for a in ca[na['connectionId']]]and any(all(na[k]==a[k]for k in ['startNs','endNs','globalTid','connectionId','name'])for a in ca[na['connectionId']]))
   for key,ty in [('MODEL_EXECUTE','MODEL_EXECUTE'),('MODEL_WAIT_COMPLETE','MODEL_WAIT_COMPLETE')]:
    a=ctrl[key];raw=dict(d.execute('select rowid AS task_record_rowid,* from TASK where rowid=?',(a['task_record_rowid'],)).fetchone());raw['task_type_name']=ss[raw['taskType']];ck('graph_control_raw',raw==a and a['task_type_name']==ty)
  for tid,t in ts.items():
   r=bydb[di][tid];prior=oldmap[di,tid];ck('raw_TASK_and_old_exactCANN',r['TASK']==t==prior['TASK']and r['exact_CANN_calls']==ca[t['connectionId']]==prior['CANN_API_same_connection']and not prior['PYTORCH_API_candidates']and prior['connection_family_IDs']==sorted(fid for fid,vals in cif.items()if t['connectionId']in vals));pp=r['host_control_paths'];possible=[]
   for a in ca[t['connectionId']]:
    dq=[q for q in qbytid[a['globalTid']]if q['startNs']<=a['startNs']and a['endNs']<=q['endNs']]
    for q in dq:
     nm=q['name_text'][8:];ens={v['api_rowid']:v for fl in cif[q['connectionId']]for v in enbyflow[fl]if v['name_text']=='Enqueue@'+nm}
     for en in ens.values():
      src=smallest([v for v in named[en['globalTid'],nm]if v['startNs']<=en['startNs']and en['endNs']<=v['endNs']])
      for a0 in src:possible.append((a,q,en,a0))
   ck('exhaustive_queue_control_paths',len(pp)==len(possible))
   for p,(a,q,en,a0)in zip(pp,possible):
    ck('raw_control_path',p['CANN_API']==a and p['Dequeue']==q and p['Enqueue']==en and p['source_API']==a0 and bool(cif[q['connectionId']]&cif[en['connectionId']])and p['queue_flow_ids']==sorted(cif[q['connectionId']]&cif[en['connectionId']])and a['globalTid']>>32==q['globalTid']>>32==en['globalTid']>>32==a0['globalTid']>>32==t['globalPid'])
    frames=ch(a0['callchainId']);ck('source_chain_complete',frames is not None and all(v is not None for v in frames));key=hashlib.sha256(json.dumps(frames,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();saved=x['canonical_chains'][key];ck('source_canonical_content',p['source_chain_key']==key and p['chain_text_complete']and saved['all_frame_text_resolved']and[z['frame']for z in saved['frames']]==frames);chains_used.add(key)
    cp=[v for v in named[a0['globalTid'],'aten::copy_']if v['startNs']<=a0['startNs']and a0['endNs']<=v['endNs']and ch(v['callchainId'])==frames];inn=smallest(cp);ck('unique_copy_parent',cp==p['same_chain_copy_parent_candidates']and inn==p['innermost_same_chain_copy_parents']and len(inn)==1);par=inn[0];lr=le[di,tid];ck('logical_parent_identity',lr['source_API_rowid']==a0['api_rowid']and lr['copy_parent_API_rowid']==par['api_rowid']and lr['CANN_connection']==a['connectionId']and lr['source_chain_key']==key and not lr['actual_DMA_bytes_direction_addresses_or_lifetimes_proved'])
    shapes=[z.strip()for z in par['input_shapes_text'].split(';')];dtypes=[z.strip()for z in par['input_dtypes_text'].split(';')];ee=[]
    for j,(sh,dt)in enumerate(zip(shapes,dtypes)):
     if not dt or dt=='Scalar':continue
     dims=[int(z)for z in sh.split(',')];b=e['declared_logical_dtype_widths'][dt];ee.append(dict(input_index=j,shape=dims,reported_dtype=dt,declared_element_bytes=b,numel=math.prod(dims),logical_extent_bytes=math.prod(dims)*b))
    ck('logical_math_domain',lr['logical_tensor_inputs']==ee and len(ee)==2);sources[di,a0['api_rowid']]+=1;flows[di,tuple(p['queue_flow_ids'])]+=1;directions[p['direction_source_label']]+=1
    isorder=en['endNs']<=q['startNs'];ck('endpoint_overlap_retained',p['CPU_enqueue_not_after_dequeue']==isorder)
    if not isorder:over.append((di,tid))
   candidates=[]
   for ct in gg[di]:
    a,b=ct['MODEL_EXECUTE'],ct['MODEL_WAIT_COMPLETE']
    if t['globalPid']==a['globalPid']and t['deviceId']==a['deviceId']and t['modelId']==a['modelId']==b['modelId']and a['startNs']<=t['startNs']<=t['endNs']<=b['endNs']:candidates.append(ct)
   ck('exhaustive_graph_control_membership',len(candidates)==len(r['graph_control_candidates']))
   for ct,sv in zip(candidates,r['graph_control_candidates']):ck('graph_control_keys',sv['CANN_connection']==ct['CANN_API']['connectionId']and sv['MODEL_EXECUTE_rowid']==ct['MODEL_EXECUTE']['task_record_rowid']and sv['MODEL_WAIT_COMPLETE_rowid']==ct['MODEL_WAIT_COMPLETE']['task_record_rowid']and sv['model_id']==ct['model_id']and sv['phase_already_bound']==bool(ct['phase_records'])and sv['operand_template_source_ordinal_not_inferred']);gcount[di,ct['CANN_API']['connectionId']]+=1
   ck('disjoint_complete_control_partition',(len(pp),len(candidates))in[(1,0),(0,1)]and not r['source_graph_or_phase_promoted']and r['native_byte_direction_pointer_epoch_not_proved']);tot['HOST'if pp else'GRAPH_ONLY']+=1
  d.close()
 ck('16342_all',nraw==len(x['records'])==len(old['runtime_copy_records'])==c['count']==16342 and tot=={'HOST':12262,'GRAPH_ONLY':4080});ck('source_bijection',len(sources)==len(flows)==len(le)==e['count']==12262 and set(sources.values())=={1}and set(flows.values())=={1}and len(chains_used)==len(x['canonical_chains'])==e['groups']==79)
 ck('source_direction_labels',directions==c['source_direction_label_counts']=={'SOURCE_LABEL_HOST_TO_DEVICE':8744,'SOURCE_PRIMITIVE_DIRECTION_UNRESOLVED':3342,'SOURCE_LABEL_DEVICE_TO_HOST':176});ck('80_graphs51_each',len(gcount)==len(g['controls'])==c['graph_controls']==80 and set(gcount.values())=={51}and c['graph_control_only_records']==4080 and c['new_graph_phase_bindings']==0 and c['graph_phase_unbound_records']==816)
 ck('logical_extent_totals',sum(z['logical_tensor_inputs'][0]['logical_extent_bytes']for z in le.values())==e['logical_destination_extent_sum_bytes']and sum(z['logical_tensor_inputs'][1]['logical_extent_bytes']for z in le.values())==e['logical_source_extent_sum_bytes']);ck('overlap8',len(over)==w['endpoint_overlap_count']==c['endpoint_overlap_control_cases']==8 and set(over)=={(z['database_index'],z['TASK_rowid'])for z in w['explicit_queue_identity_despite_recorded_endpoint_overlap']})
 for z in w['logical_copy_removal']:ck('remove_copy_witness',z['consumer_expected']==z['producer_source']==z['direct_producer_to_destination']and z['remove_copy_but_leave_consumer_on_destination']!=z['consumer_expected']and z['early_direct_producer_old_reader_result']!=z['old_reader_before_original_publication_expected']and not z['actual_native_pointer_or_reader_schedule_attested'])
 z=w['logical_extent_vs_DMA_descriptor'];ck('partition_witness',all(sum(v)==z['logical_extent_bytes']for v in z['possible_descriptor_partitions'])and not z['actual_native_partition_selected']);ck('no_false_native_bound',not c['actual_DMA_direction_bytes_pointer_epoch_or_all_readers_proved']and not c['conditional_resource_bound']['native_required_bytes_or_valid_capacity_uppers_available']and not c['conditional_resource_bound']['logical_input_extent_substituted_for_W_required']and not c['new_device_work']and not c['source_read']and not c['overall_complete'])
 if args.preflight:man=read(O/'checkpoint-manifest-v149.json')
 else:
  ck('checkpoint_progress',read(O/'REVIEW-CHECKPOINT-v150.json')['status']=='PROGRESS'and not read(O/'REVIEW-CHECKPOINT-v150.json')['overall_complete']);a=read(O/'operator-analysis-coverage-ledger-v149.json');b=read(O/'operator-analysis-coverage-ledger-v150.json');ck('all_compute_regimes_unchanged',a['regimes']==b['regimes']and all(a[k]==b[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']));a=read(O/'candidate-profile-ledger-v149.json');b=read(O/'candidate-profile-ledger-v150.json');ck('broad_gaps_preserved',[z.get('conclusion_changing_gap')for z in a['candidates']]==[z.get('conclusion_changing_gap')for z in b['candidates']]);rel=read(O/'candidate-relation-clauses-v150.json');cv=read(O/'candidate-relation-coverage-v150.json');ck('192_26_403',len(rel['clauses'])==192 and len(rel['higher_order_groups'])==26 and cv['remaining_pair_relations']==403);ob=read(O/'coverage-current-obligations-v150.json');ck('G14_retained',len(ob['obligations'])==15 and any(z['id']=='G14'and not z['closed']and'16342'in z['remaining']and'4080'in z['remaining']for z in ob['obligations']));mx=read(O/'module-phase-branch-matrix-v150.json');ck('358_26',len(mx['modules'])==26 and sum(len(z['branches'])for z in mx['modules'])==358)
  for z in read(O/'candidate-record-index-v150.json')['model_candidates']:ck('master_sha',hashlib.sha256(Path(z['master_record']).read_bytes()).hexdigest()==z['master_sha256'])
  ck('active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING');man=read(O/'checkpoint-manifest-v150.json')
 for z in man['artifacts']:ck('manifest',pin(z))
 result=dict(status='PASS',checks=checks,preflight=args.preflight,runtime_copy_records=16342,unique_host_queue_source_controls=12262,canonical_chains=79,existing_graph_control_only_records=4080,graph_controls=80,logical_extent_records=12262,endpoint_overlap_cases=8,new_pair_clauses=0,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 if args.preflight:write(O/'runtime-copy-preflight-v150.json',result)
 print(json.dumps(result))


if __name__=='__main__':main()
