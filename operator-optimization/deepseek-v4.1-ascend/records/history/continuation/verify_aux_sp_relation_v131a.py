"""Verify saved native keys and conditional packet/selection contracts, not speedups."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import argparse,hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from aux_sp_models_v131 import canonical,row_plan,cycle,bf16


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(label,ok):
        assert ok,label
        checks.append(label)
    def pinned(r):
        p=Path(r['path']);b=p.read_bytes();return len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    for p in read(O/'aux-sp-inputs-v131.json')['inputs']:ck('input:'+p['path'],pinned(p))
    for p in read(O/'aux-sp-source-contract-v131.json')['parts']:
        ck('source:'+str((p['path'],p['first_line'])),pinned(p) and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    source=read(O/'aux-sp-source-contract-v131.json')
    ck('SWA_roles',[r['layer']for r in source['checkpoint_roles']]==[37,38,39] and all(not r['is_kv_source'] and not r['is_index_source']for r in source['checkpoint_roles']))
    pop=read(O/'communication-native-operator-population-v66.json')['operations']
    popmap={(r['database_index'],r['COMM_OP_rowid']):r for r in pop}
    prof=read(O/'aux-sp-checkpoint-profile-v131.json');sel=read(O/'aux-sp-selector-profile-v131.json')
    ck('240_192_48_72',len(prof['pairs'])==prof['pair_count']==240 and sum(r['root_bound']for r in prof['pairs'])==prof['root_bound_pairs']==192 and prof['phase_unbound_pairs']==48 and len(prof['eager_aux_operators_retained_unpaired'])==72)
    bydb=defaultdict(list);eager=defaultdict(list);selectors=defaultdict(list)
    for r in prof['pairs']:bydb[r['database_index']].append(r)
    for r in prof['eager_aux_operators_retained_unpaired']:eager[r['database_index']].append(r)
    for r in sel['records']:selectors[r['database_index']].append(r)
    seen_aux=set();seen_sp=set();seen_cat=set();counts=Counter();shapes=defaultdict(Counter)
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    oldcache={r['path']:r for r in read(O/'draft-cache-join-inputs-v113.json')['inputs']}
    for ref in sel['databases']:
        di=ref['database_index'];ck('rawpin:'+str(di),pinned(ref) and ref['sha256']==rawrefs[di]['sha256'])
        c=ref['cached_native'];ck('cachepin:'+str(di),pinned(c) and c['sha256']==oldcache[c['path']]['sha256'])
        db=sqlite3.connect(Path(ref['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};cidmap=defaultdict(set)
        for r in db.execute('SELECT *FROM CONNECTION_IDS'):cidmap[r['id']].add(r['connectionId'])
        def check_comm(g):
            key=(di,g['COMM_OP_rowid']);rr=db.execute('SELECT *FROM COMMUNICATION_OP WHERE rowid=?',(key[1],)).fetchone()
            tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(g['TASK_rowid'],)).fetchone()
            ck('comm_source:'+str(key),g==popmap[key])
            ck('rawcomm:'+str(key),all(rr[k]==g[k]for k in ['opId','count','rankSize','startNs','endNs','connectionId']) and strings[rr['groupName']]==g['group_name'] and rr['dataType']==11 and g['dtype']=='BFP16')
            ck('comm_TASK:'+str(key),all(tt[k]==g[k]for k in ['startNs','endNs','connectionId','streamId']))
        for r in bydb[di]:
            a,s=r['aux_allGather'],r['attention_SP_allGather'];ak=(di,a['COMM_OP_rowid']);sk=(di,s['COMM_OP_rowid'])
            assert ak not in seen_aux and sk not in seen_sp;seen_aux.add(ak);seen_sp.add(sk)
            check_comm(a);check_comm(s);m=r['mean'];t=db.execute('SELECT *FROM TASK WHERE rowid=?',(m['TASK_rowid'],)).fetchone()
            d=db.execute('SELECT *FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
            ck('mean:'+str(ak),strings[d['opType']]==m['operator']=='ReduceMean' and all(t[k]==m[k]for k in ['startNs','endNs','streamId']))
            ck('checkpoint_schema:'+str(ak),a['graph_ordinal']==s['graph_ordinal']==r['graph_ordinal'] and r['checkpoint_before_block']==s['block_ordinal'] in [37,38,39] and a['root_bound']==s['root_bound']==r['root_bound'] and a['semantic_role']=='DSPARK_AUX_RESIDUAL_ALLGATHER' and s['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER')
            ck('geometry_order:'+str(ak),a['count']==s['count']==15360 and a['rankSize']==s['rankSize']==8 and r['sender_rows_from_count']==3 and m['endNs']<=a['startNs']<=a['endNs']<=s['startNs'] and r['same_group_label'] and not r['actual_tensor_alias_group_members_or_ready_events_bound'])
        for g in eager[di]:
            check_comm(g);ck('eager_unpaired:'+str((di,g['COMM_OP_rowid'])),g.get('graph_ordinal') is None and (di,g['COMM_OP_rowid'])not in seen_aux)
        # Independently select both positive source arms from original callchains.
        frames=defaultdict(list)
        for r in db.execute("SELECT c.id,c.stackDepth,s.value FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE s.value LIKE '%/worker/model_runner_v1.py(2069): propose_draft_token_ids' OR s.value LIKE '%/worker/model_runner_v1.py(2089): propose_draft_token_ids'"):
            frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['value']))
        parent_ids={r['api_rowid']for r in selectors[di]}
        independently={r['rowid']for r in db.execute('SELECT rowid,*FROM PYTORCH_API WHERE callchainId IS NOT NULL')if r['callchainId']in frames and strings.get(r['name'])=='aten::cat'}
        ck('all_recorded_cats:'+str(di),parent_ids==independently and len(parent_ids)==ref['API_count'])
        for r in selectors[di]:
            p=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(r['api_rowid'],)).fetchone();c=r['exact_nested_native_API'];a=db.execute('SELECT *FROM PYTORCH_API WHERE rowid=?',(c['api_rowid'],)).fetchone()
            ck('parent:'+str((di,r['api_rowid'])),all(p[k]==r[k]for k in p.keys()) and strings[p['name']]=='aten::cat' and frames[p['callchainId']]==r['matched_frames'])
            ck('nested_child:'+str((di,c['api_rowid'])),all(a[k]==c[k]for k in a.keys()if k in c) and strings[a['name']]=='aclnnCat' and a['globalTid']==p['globalTid'] and int(p['startNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(p['endNs']))
            ck('positive_arm:'+str((di,r['api_rowid'])),any(('(2069)'in f['frame'])==(r['role']=='EARLY_PREFIX_SOURCE_ARM')for f in r['matched_frames']))
            ck('one_native_cat:'+str((di,r['api_rowid'])),len(r['native_tasks'])==1)
            n=r['native_tasks'][0];t=n['TASK'];d=n['definition'];key=(di,t['task_record_rowid']);assert key not in seen_cat;seen_cat.add(key)
            tt=db.execute('SELECT *FROM TASK WHERE rowid=?',(key[1],)).fetchone();dd=db.execute('SELECT *FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(t['globalTaskId'],)).fetchone()
            ck('rawcat_TASK:'+str(key),all(tt[k]==t[k]for k in tt.keys()) and t['connectionId']in cidmap[a['connectionId']] and c['api_rowid']in t['native_linked_API_rowids'])
            ck('rawcat_definition:'+str(key),all(dd[k]==d[k]for k in dd.keys()) and strings[dd['opType']]==d['opType_text']=='ConcatD' and strings[dd['outputShapes']]==d['outputShapes_text'] and strings[dd['inputDataTypes']]==d['inputDataTypes_text']=='DT_BF16;DT_BF16;DT_BF16')
            counts[r['role']]+=1;shapes[r['role']][d['outputShapes_text']]+=1
        db.close()
    ck('exact_populations',(len(seen_aux),len(seen_sp),len(seen_cat))==(240,240,88) and dict(counts)==sel['API_counts_by_source_arm']=={'EARLY_PREFIX_SOURCE_ARM':16,'SAMPLED_INDEX_SOURCE_ARM':72} and sel['native_TASK_types']=={'ConcatD':88})
    ck('prefix_shapes',dict(shapes['EARLY_PREFIX_SOURCE_ARM'])=={'"6400,15360"':8,'"6528,15360"':8})
    ck('indexed_shapes',dict(shapes['SAMPLED_INDEX_SOURCE_ARM'])=={'"12,15360"':16,'"24,15360"':48,'"6668,15360"':8})
    models=read(O/'aux-sp-ordered-selection-models-v131.json');ck('7742_finite_models',len(models['cases'])==models['case_count']==7742)
    for c in models['cases']:
        P,N,I,M=c['P'],c['N'],c['I'],c['M'];n=N//P;norm=[i if i>=0 else N+i for i in I]
        aux=[[tuple((101*k+3*i,101*k-2*i))for i in range(N)]for k in range(3)]
        def f(rows):
            flat=sum((list(x)for x in rows),[]);return (sum((j+1)*v for j,v in enumerate(flat))**2,sum(v*v for v in flat))
        allrows=[f([a[i]for a in aux])for i in range(N)];direct=[f([a[i]for a in aux])for i in norm]
        owned=[sorted({i%n for i in norm if i//n==p})for p in range(P)]
        bank={(p,j):allrows[p*n+j]for p,ids in enumerate(owned)for j in ids}
        plan=c['plan'];ck('ordered_selection:'+str(len(checks)),len(I)==M and norm==canonical(I,N)==plan['canonical_indices'] and owned==plan['unique_per_owner'] and plan['unique_rows']==sum(map(len,owned)) and plan['u_max']==max(map(len,owned)) and plan['fixed_count_rows_over_all_senders']==P*max(map(len,owned)) and [bank[i//n,i%n]for i in norm]==direct==[allrows[i]for i in norm])
    loss=read(O/'aux-sp-owner-balance-and-loss-witnesses-v131.json')
    ck('imbalance_fixed_count',[loss['plans'][k]['u_max']for k in ['balanced','skew','duplicate']]==[2,3,1] and [loss['plans'][k]['unique_rows']for k in ['balanced','skew','duplicate']]==[12,12,1])
    # The toy shows information loss only; it is not an actual trained-state witness.
    def rms_first(v):return v[0]/((sum(x*x for x in v)/len(v)+1e-6)**0.5)
    ck('aux_not_SWA_identity',rms_first([1,1])==rms_first([1,-1]) and [1,1][1]!=[1,-1][1])
    ck('offdiagonal_missing',[[x*w for w in [3,5]]for x in [1,2]]==loss['diagonal_projection']['required_outputs'] and len(loss['diagonal_projection']['diagonal_only'])==2)
    costs=read(O/'aux-sp-packet-work-and-capacity-models-v131.json');ck('five_geometries',[(r['N_physical'],r['M_selected'])for r in costs['geometries']]==[(24,12),(24,24),(6400,6400),(6528,6528),(6672,6668)])
    for r in costs['geometries']+costs['late_selector_examples']:
        N,M,P,H,D=r['N_physical'],r['M_selected'],8,5120,512;n=N//P;A=3*2*n*H;S=3*2*n*D;F=2*M*640
        ck('packet_content:'+str((N,M,r.get('name'))),r['baseline_sender_content_bytes']==2*A+F and r['independent_F32_projected_sender_content_bytes']==r['early_aux_KV_copack_sender_content_bytes']==A+S+F and r['early_copack_extra_payload_saving_beyond_F32']==0 and r['early_copack_op_reference']==4 and r['independent_F32_collective_op_reference']==r['baseline_collective_op_reference']==7)
        ck('pack_IO_and_weights:'+str((N,M,r.get('name'))),r['compatibility_pack_unpack_value_bytes_three_checkpoints']==3*(2*2*n*(H+D)+2*2*N*(H+D)) and r['full_main_weight_bytes']==157286400 and r['current_main_weight_shard_bytes']==19660800 and r['extra_main_weight_per_card_bytes']==137625600)
        ratio=r['owner_main_projection_work_ratio'];ck('owner_feature_content:'+str((N,M,r.get('name'))),Fraction(*ratio)==Fraction(N,M) and r['owner_main_context_carrier_sender_content_bytes']==S+2*n*H and r['owner_main_KV_carrier_sender_content_bytes']==2*S and r['owner_main_context_and_KV_sender_content_bytes']==2*S+2*n*H and r['early_feature_shard_packet_sender_content_bytes']==A+S+2*N*640 and r['early_feature_shard_extra_feature_bytes']==2*(N-M)*640)
        ck('snapshot_not_allocation:'+str((N,M,r.get('name'))),r['feature_shard_FP32_accumulator_bytes']==4*N*640 and r['three_full_aux_snapshot_content_bytes']==6*N*H and r['owner_FP32_accumulator_bytes']==4*n*H and r['three_local_aux_snapshot_content_bytes']==6*n*H and not r['physical_latency_or_wire_or_MFU_established'])
        if 'name'in r:ck('late_cost:'+r['name'],r['late_selected_sender_content_bytes']==S+6*H*loss['plans'][r['name']]['u_max']+F and r['late_selected_op_reference']==5 and r['late_exact_selector_cannot_gate_current_attention_packet'])
    ck('specific_payloads',costs['geometries'][0]['baseline_sender_content_bytes']==199680 and costs['geometries'][0]['independent_F32_projected_sender_content_bytes']==116736 and [r['late_selected_sender_content_bytes']for r in costs['late_selector_examples']]==[86016,116736,55296])
    graph=read(O/'aux-sp-causality-and-variant-graph-v131.json')
    ck('scoped_wait_cycle',not cycle(graph['baseline_or_separate_late_edges']) and cycle(graph['invalid_late_exact_copack_edges']) and not cycle(graph['early_prefix_edges']) and ['I','PACK']in graph['invalid_late_exact_copack_edges'])
    ck('five_conditional_variants',len(graph['variants'])==5 and all(v['unproved_native_gates']for v in graph['variants']) and graph['native_graph_nodes_added']==0 and not graph['new_device_work'])
    numeric=read(O/'aux-sp-numeric-and-snapshot-gates-v131.json');a,b,_=numeric['separate_BF16_partial_counterexample']['contributions']
    ck('BF16_rounding_counterexample',bf16(bf16(a)+bf16(b))==0 and bf16(a+b)==2**-8 and numeric['full_norm_domains']==[5120,512,512,512] and not numeric['current_selected_projection_NaN_ties_or_precision_validated'])
    if args.preflight:
        old=read(O/'candidate-relation-coverage-v130.json');r=next(r for r in old['pairs']if {r['a'],r['b']}=={'F27','F32'});ck('actually_pending',r['scope_state']!='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL')
        manifest=read(O/'checkpoint-manifest-v130.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v131.json');ck('open_progress',cp['status']=='PROGRESS' and not cp['overall_complete'] and cp['whole_modules_closed']==0)
        rel=read(O/'candidate-relation-clauses-v131.json');pairs={tuple(sorted((r['a'],r['b'])))for r in rel['clauses']};ck('138_pairs_26_groups',len(rel['clauses'])==len(pairs)==rel['clause_count']==rel['selected_pair_count']==138 and len(rel['higher_order_groups'])==26)
        r=next(r for r in rel['clauses']if r['id']=='REL131-01');ck('conditional_only',{r['a'],r['b']}=={'F27','F32'} and not r['all_runtime_gates_proved'] and r['critical_path_gain_us']is None)
        g=next(g for g in rel['higher_order_groups']if g['id']=='HG85-VERIFY-CONTEXT-DRAFT');ck('existing_group_open',{'F27','F32'}<=set(g['members']) and not g['all_subsets_analyzed'] and not g['legal_joint_fusion_proved'])
        cov=read(O/'candidate-relation-coverage-v131.json');ck('138_26_457',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(138,26,457) and len(cov['pairs'])==595)
        nav=read(O/'candidate-pair-navigation-v131.json');r=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F27','F32'});ck('navigation',r['conditional_clause_ids']==['REL131-01'] and r['semantic_pair_reconciliation_done_in_stated_scope'] and nav['remaining_central_pair_reconciliation']==457)
        m=read(O/'module-phase-branch-matrix-v131.json');ck('358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v130.json');new=read(O/'operator-analysis-coverage-ledger-v131.json');ck('native_uncertainty_unchanged',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v131.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE' and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v131.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,checkpoint_pairs=240,selector_native_ConcatD=88,ordered_selection_models=7742,cost_geometries=5,new_scoped_pair=1,new_native_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'aux-sp-preflight-v131.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
