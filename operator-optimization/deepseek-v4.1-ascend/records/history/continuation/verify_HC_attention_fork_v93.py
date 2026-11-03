"""Verify exact saved-record joins,resource formulas and deadline/source-domain fences."""
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
from math import ceil,sqrt,prod
import argparse,hashlib,json
from finalize_origin_coverage_v9 import O,R,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(label,ok):assert ok,label;checks.append(label)
    source=read(O/'HC-attention-fork-source-contract-v93.json');p=Path(source['path'])
    check('pinned source',hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']and p.read_text().splitlines()[source['first_line']-1:source['last_line']]==source['source'])
    fronts=read(O/'HC-attention-fork-native-frontiers-v93.json');rows=fronts['frontiers']
    check('3200nativefrontiers2560root640unbound',len(rows)==3200 and sum(r['root_bound']for r in rows)==2560)
    check('strict3120ambiguous80notrelabeled',sum(r['SWA_role_unique']for r in rows)==3120 and {r['block_ordinal']for r in rows if not r['SWA_role_unique']}=={20})
    check('44800previousnativeTASKs',fronts['reused_unique_native_TASKs']==44800)
    keys=[(r['database_index'],r['graph_ordinal'],r['block_ordinal'])for r in rows];check('uniquegraphblockkeys',len(set(keys))==3200)
    old=read(O/'projection-graph-chains-v61.json')['graph_projection_chains'];lookup={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in old}
    hc=read(O/'HC-operator-profile-and-edges-v58.json')['exact_graph_role_edges'];hcmap={(r['kind'],r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in hc}
    for r in rows:
        key=(r['database_index'],r['graph_ordinal'],r['block_ordinal']);a=hcmap[('ATTENTION_PRE_TO_NORM',*key)];b=hcmap[('INTRALAYER_POST_TO_PRE',*key)];c=lookup[key]
        assert r['attention_HcPre_TASK']==a['producer_task_rowid']and r['input_norm_TASK']==a['consumer_task_rowid']
        assert r['attention_HcPost_TASK']==b['producer_task_rowid']and r['FFN_HcPre_TASK']==b['consumer_task_rowid']
        assert r['Q_A_TASK']==c['stages']['Q_A']['task_record_rowid']and r['Q_B_TASK']==c['stages']['Q_B']['task_record_rowid']
        assert r['output_reduce_scatter_TASK']==c['stages']['REDUCE_SCATTER']['task_record_rowid']
        assert r['hypothetical_coefficient_internal_service_or_available_slack_ns']is None and not r['native_records_prove_new_branch_overlap']
        assert prod(r['Q_B_recorded_weight_shape'])==1280*32768 and r['Q_B_side_input_shapes']==[[32768],[3]]
    check('allfrontierslinkedwithoutgap-to-gain',True)
    model=read(O/'HC-attention-shared-resource-models-v93.json');planes=model['planes'];check('typedweightplanevalues',[r['bytes']for r in planes]==[1966080,13107200,5242880,41943040])
    cap=24*128;grant=24*65536
    for r in model['cuts']:
        chosen=[p for p in planes if p['id']in r['planes']];W=sum(p['bytes']for p in chosen)*r['layers_in_window']-r['last_pre4_weight_bytes_omitted_from_guaranteed_work']
        check('onewarmfixedB:'+str((len(chosen),r['layers_in_window'])),r['distinct_uncompressed_weight_bytes']==W and r['joint_L1_fixedB_min_elapsed_cycles']==ceil(F(max(0,W-grant),cap)))
        check('relaxedorientation:'+str((len(chosen),r['layers_in_window'])),r['joint_L1_flexible_orientation_relaxed_min_elapsed_cycles']==ceil(F(max(0,W-2*grant),24*384)))
        assert not r['actual_native_L1_path_or_no_compression_and_weight_identity_proved']
    one3=next(r for r in model['cuts']if len(r['planes'])==3 and r['layers_in_window']==1 and r['deadline'].startswith('FOLLOWING'));one4=next(r for r in model['cuts']if len(r['planes'])==4 and r['layers_in_window']==1 and r['deadline'].startswith('FOLLOWING'))
    check('jointperblock6102and19755',one3['joint_L1_fixedB_min_elapsed_cycles']==6102 and one4['joint_L1_fixedB_min_elapsed_cycles']==19755)
    urgent3=next(r for r in model['cuts']if len(r['planes'])==3 and r['layers_in_window']==1 and r['deadline'].startswith('LAST'));urgent4=next(r for r in model['cuts']if len(r['planes'])==4 and r['layers_in_window']==1 and r['deadline'].startswith('LAST'))
    check('Postdeadlineonly20cols5995and19648',urgent3['joint_L1_fixedB_min_elapsed_cycles']==5995 and urgent4['joint_L1_fixedB_min_elapsed_cycles']==19648)
    for n in [3,4]:
        full=next(r for r in model['cuts']if len(r['planes'])==n and r['layers_in_window']==40 and r['deadline'].startswith('FOLLOWING'))
        urgent=next(r for r in model['cuts']if len(r['planes'])==n and r['layers_in_window']==40 and r['deadline'].startswith('LAST'))
        check('earlier39pre4stillrequired:'+str(n),full['distinct_uncompressed_weight_bytes']-urgent['distinct_uncompressed_weight_bytes']==327680 and urgent['earlier_pre4_column_groups_still_required']==39)
    check('mixedpeaksnotadded',not model['mixed_mode_capacity_simplex_proved']and model['individual_useful_compute_cycle_floors']==dict(HC_semantic_HF32_ops=30.,Q_A_BF16_ops=200.,SWA_global24_BF16_ops=640.,Q_B_INT8_ops=640.))
    columns={r['columns']:r for r in model['independent_column_split_reference']}
    check('urgent20referencepaddingequals24',columns[20]['independently_padded_M16_N16_service_ops']==columns[24]['independently_padded_M16_N16_service_ops'])
    check('split4adds50percentpadreference',columns[20]['independently_padded_M16_N16_service_ops']+columns[4]['independently_padded_M16_N16_service_ops']==columns[24]['independently_padded_M16_N16_service_ops']*F(3,2))
    check('rowownercommunicationandwork_notfree',model['source_full_HC_gather_hypothetical_ring_send_bytes']==860160 and model['current_normalized_H_gather_ring_send_reference_bytes']==215040 and model['moving_HC_coeff_to_all24_rows_would_multiply_useful_work_by']==8)
    # Generic input-information witness: same currentY,distinct valid full4H coefficient.
    h,H,eps=4,5120,1e-20;Xa=[1.,0.,0.,0.];Xb=[1.,1.,0.,0.];oldpre=[1.,0.,0.,0.]
    ya=sum(x*p for x,p in zip(Xa,oldpre));yb=sum(x*p for x,p in zip(Xb,oldpre));ca=Xa[1]/sqrt(sum(x*x for x in Xa)/(h*H)+eps);cb=Xb[1]/sqrt(sum(x*x for x in Xb)/(h*H)+eps)
    check('full4HcoefficientnotgenericallyfromY',ya==yb==1 and ca==0 and cb>0)
    dag_cases=0
    for P in range(11):
        for S in range(11):
            for D in range(1,6):
                for A in range(1,21):
                    for Hpost in range(4):
                        early=max(P+S,D+A)+Hpost;late=max(P,D+A)+S+Hpost
                        assert late-early==min(S,max(D+A-P,0))and late>=early;dag_cases+=1
    check('idealunchangedserviceDAGcases',dag_cases==48400)
    packet=read(O/'HC-attention-composition-clauses-v93.json');check('7newpairs1jointcontract',len(packet['clauses'])==7 and len(packet['higher_order_groups'])==1)
    oldpairs={frozenset([r['a'],r['b']])for r in read(O/'candidate-relation-clauses-v92.json')['clauses']}
    check('no priorpairduplicated',all(frozenset([r['a'],r['b']])not in oldpairs for r in packet['clauses']))
    audit=read(O/'operator-graph-coverage-audit-v93.json');check('partialsourcegraph743/1209',audit['node_rows']==743 and audit['edge_rows']==1209 and audit['current_structural_pair_hints']==45)
    for k,path in audit['graph_catalog'].items():check('graphhash:'+k,hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit['graph_sha256'][k])
    gr=read(O/'operator-graph-G5-v93.json');t=next(t for t in gr['templates']if t['id']=='MHC_ATTENTION_COEFFICIENT_FORK');ids={n['id']for n in t['nodes']}
    check('sourceforkendpoints',len(ids)==19 and len(t['edges'])==22 and all(e['producer']in ids and e['consumer']in ids for e in t['edges']))
    if args.preflight:
        for r in read(O/'checkpoint-manifest-v92.json')['artifacts']:
            path=Path(r['path']);check('pretransition:'+path.name,path.stat().st_size==r['bytes']and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'])
        write(O/'HC-attention-fork-math-validation-v93.json',dict(ideal_equal_service_DAG_cases=dag_cases,
          source_input_information_witness=dict(collapsedY=[ya,yb],coefficient_dot_times_rstd=[ca,cb],arbitrary_valid_weight_witness_not_actual_checkpoint=True),
          resource_cut_cases=8,independent_column_reference_cases=3,new_profile_execution_or_Torch_NPU_test=False,overall_complete=False))
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v93.json');check('progressnotcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        op=read(O/'operator-analysis-coverage-ledger-v93.json');check('profilepopulationunchanged',(op['scoped_conditional_review_regimes'],op['scoped_conditional_review_executions'],op['remaining_regime_reconciliation_count'],op['remaining_execution_reconciliation_count'])==(2073,285598,9,1352))
        check('uncertaintyunchanged',op['unchanged_global_uncertainty']==read(O/'operator-analysis-coverage-ledger-v92.json')['unchanged_global_uncertainty'])
        bm=read(O/'module-phase-branch-matrix-v93.json');check('355records24semantic',sum(len(m['branches'])for m in bm['modules'])==355 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        rc=read(O/'candidate-relation-coverage-v93.json');check('122/25/473',rc['scoped_conditional_pairs']==122 and rc['higher_order_groups']==25 and rc['remaining_pair_relations']==473)
        req=read(O/'whole-scope-requirement-audit-v93.json');reqmap={r['id']:r for r in req['requirements']}
        check('fullaudit9requirementsnotclosed',len(reqmap)==9 and not req['overall_complete']and req['whole_modules_closed']==0)
        check('lowestmoduleMFUnotestablished',reqmap['LOWEST_MODULE_UTILIZATION']['state']=='NOT_ESTABLISHED')
        check('fullaudit355/24',reqmap['ALL_MODULE_CONDITIONAL_BRANCHES']['branch_records']==355 and reqmap['ALL_MODULE_CONDITIONAL_BRANCHES']['semantic_modules']==24)
        check('fullaudit9pending1352notzero',len(req['pending_native_rows'])==9 and sum(r['executions']for r in req['pending_native_rows'])==1352)
        check('summary8deadlinecuts',read(O/'HC-attention-summary-v93.json')['shared_weight_cuts']==8 and len(model['cuts'])==8)
        check('canonicalgraphpointer',read(R/'fusion-scan.json')==read(O/'fusion-scan-proposed-v93.json'))
        for r in read(O/'candidate-record-index-v93.json')['model_candidates']:check('masterindex:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        for r in read(O/'candidate-profile-ledger-v93.json')['candidates']:check('candidateconditional:'+r['id'],not r['candidate_closed']and not r['legal_fusion_proved'])
        for r in read(O/'checkpoint-manifest-v93.json')['artifacts']:
            path=Path(r['path']);check('manifest:'+path.name,path.stat().st_size==r['bytes']and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'])
    result=dict(status='PASS',checks=len(checks),native_frontiers=3200,reused_native_tasks=44800,ideal_DAG_cases=dag_cases,preflight=args.preflight,new_native_server_or_model_work=False,overall_complete=False)
    if args.preflight:write(O/'HC-attention-fork-analysis-preflight-v93.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
