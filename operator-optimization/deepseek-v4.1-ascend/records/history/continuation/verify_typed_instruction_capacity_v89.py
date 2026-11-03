"""Verify source selection,exact populations and conditional numerical capacity models."""
import argparse,hashlib,json
from math import ceil,isclose
from pathlib import Path
from collections import Counter
from finalize_origin_coverage_v9 import O,N,R,read,write
from analyze_state_engram_v62 import dimensions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    checks=[]
    def check(label,condition):
        assert condition,label
        checks.append(label)
    for x in read(O/'typed-capacity-primary-receipts-v89.json')['primary_sources']:
        p=Path(x['path']);check('primary:'+p.name,p.stat().st_size==x['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']and x['http_status']==200)
    c=read(O/'typed-capacity-primary-contract-v89.json');rates=c['cube_A3_2201']
    check('2201 typed cycle capacities',{k:r['ops_2perMAC_per_AIC_cycle']for k,r in rates.items()}==dict(INT4=32768,INT8=16384,FP16=8192,BF16=8192,FLOAT_HF32_OFF=2048,FLOAT_HF32_ON=4096))
    check('do not import3510 FLOATk1',read(O/'Cube-primary-tables-v89.json')[1][9][5]=='1'and rates['FLOAT_HF32_OFF']['cube_k']==4)
    check('FLOAT lossless CCbridge is not silently invented',not c['float_L0C_to_L1_lossless_route_documented_in_this_table']and not c['absence_is_universal_unsupported_proof'])
    vec=read(O/'Vector-primary-tables-v89.json')
    check('A3 rates and reductions use separate rows',['Exp','float','32']in vec[5]and ['Ln','float','32']in vec[5]and ['Sqrt','float','32']in vec[5]and ['Mul','float','64']in vec[5]and ['ReduceRepeat','float','float','64/7']in vec[10]and ['Sort32','float','2']in vec[13])
    base={r['regime_id']:r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']}
    mat=read(O/'A3-matrix-typed-capacity-cases-v89.json');rows=mat['regimes']
    check('118regimes35334executions',len(rows)==118 and sum(r['executions']for r in rows)==35334 and len({r['regime_id']for r in rows})==118)
    check('FLOAT21/5304 extension',len([r for r in rows if r['input_dtype']=='FLOAT'])==21 and sum(r['executions']for r in rows if r['input_dtype']=='FLOAT')==5304)
    prior={r['regime_id']:r for r in read(O/'A3-typed-ingress-regime-models-v80.json')['regimes']}
    check('same97priorBF16INT8regimes',set(prior)=={r['regime_id']for r in rows if r['input_dtype']!='FLOAT'})
    for r in rows:
        b=base[r['regime_id']]
        check('saved native fields:'+r['regime_id'],all(r[k]==b[k]for k in ['signature','module_candidates','role','origin','context','executions']))
        a,weight,*_=dimensions(r['signature']['inputShapes_text']);out=dimensions(r['signature']['outputShapes_text'])[0]
        check('dense geometry:'+r['regime_id'],(r['M'],r['K'],r['N'])==(a[0],a[1],out[1])and r['dense_ops_2perMAC']==2*a[0]*a[1]*out[1])
        for m in r['mode_cases']:
            cap=24*rates[m['mode']]['ops_2perMAC_per_AIC_cycle']
            check('compute and joint max:'+r['regime_id']+m['mode'],isclose(m['useful_dense_MAC_work_cycle_floor'],r['dense_ops_2perMAC']/cap)and isclose(r['joint_with_fixed_B_weight_cut_cycle_floors'][m['mode']],max(r['L1_only_fixed_B_weight_floor_cycles'],r['dense_ops_2perMAC']/cap)))
        if r['regime_id']in prior:
            check('reuse old weight floor:'+r['regime_id'],isclose(r['L1_only_fixed_B_weight_floor_cycles'],prior[r['regime_id']]['L1_only_B_min_elapsed_core_clock_cycles']))
        check('no native mode/time/promotion:'+r['regime_id'],not r['native_mode_or_instruction_lowering_proved']and not r['new_profile_regime_review_promotion']and r['actual_wallclock_floor_us']is None)
    mixed=read(O/'mixed-projection-capacity-repricing-v89.json');g=mixed['models'][0];local=mixed['models'][-1]
    check('N24/M3 useful arithmetic',g['rows']==24 and local['rows']==3 and g['useful_dense_service_cycle_floors']['C2_two_FLOAT_HF32_OFF']==5120 and local['useful_dense_service_cycle_floors']['C2_two_FLOAT_HF32_OFF']==640)
    check('one initial weight capacity grant',g['original_fixed_B_weight_cut_cycles']==ceil((26214400-24*65536)/(24*128))==8022 and local['original_fixed_B_weight_cut_cycles']==8022)
    check('weight max unchanged in declared smallM domain',all(x==8022 for r in [g,local]for x in r['original_fixed_B_combined_max_cycle_floors'].values()))
    check('useful8x is not padded2x or runtime',isclose(g['independent_padded_matrix_service_references']['C2_two_FLOAT_HF32_OFF']/local['independent_padded_matrix_service_references']['C2_two_FLOAT_HF32_OFF'],2)and not mixed['native_N24_to_ownerlocal_M3']['padding_reference_is_actual_speedup'])
    rt=read(O/'router-Vector-instruction-cuts-v89.json');oldrt={r['regime_id']:r for r in read(O/'router-vector-regime-models-v63.json')['regimes']}
    check('router7regimes4472executions',rt['regime_count']==7 and sum(r['executions']for r in rt['regimes'])==4472)
    for r in rt['regimes']:
        b=oldrt[r['regime_id']];check('router saved provenance:'+r['regime_id'],r['signature']==b['signature']and r['executions']==b['executions'])
        check('router heterogeneous max only:'+r['regime_id'],isclose(r['max_individual_direct_instruction_cycle_floor'],max(q['source_elements']/(48*q['documented_elements_per_AIV_cycle'])for q in r['normType2_source_instruction_cuts']))and not r['independent_opcode_cut_sum_used'])
        check('sort lowering remains conditional:'+r['regime_id'],not r['full_sort_Sort32_lowering_case']['native_Sort32_MrgSort_lowering_attested']and not r['full_sort_Sort32_lowering_case']['applies_to_all_exact_TopK_algorithms'])
    hc=read(O/'HC-precision-mode-capacity-cases-v89.json')
    check('sourceHF32fact predates current revaluation',hc['previously_reviewed_source_fact_first_version']==58 and not hc['new_mode_or_HF32_optimization_discovery_claimed'])
    for part in hc['source_parts']:
        p=Path(part['path']);check('pinned HC mode clause:'+str(part['first_line']),hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256']and p.read_text().splitlines()[part['first_line']-1:part['last_line']]==part['source'])
    check('HF32 doesnot imply native active/frequency proof',not c['HF32_contract']['native_active_mode_proved']and not hc['actual_native_binary_mode_frequency_or_speedup_proved'])
    check('public dirty build scope explicit',all(r['observed_footer_commit']and r['uncommitted_changes_footer_present']for r in c['document_build_footers']))
    if not args.preflight:
        cp=read(O/'REVIEW-CHECKPOINT-v89.json');check('checkpointprogressnotcomplete',cp['status']=='PROGRESS'and cp['whole_modules_closed']==0 and not cp['overall_complete'])
        op=read(O/'operator-analysis-coverage-ledger-v89.json');check('no regime populations promoted',(op['scoped_conditional_review_regimes'],op['scoped_conditional_review_executions'],op['remaining_regime_reconciliation_count'],op['remaining_execution_reconciliation_count'])==(2073,285598,9,1352))
        check('global uncertainty unchanged',op['unchanged_global_uncertainty']==read(O/'operator-analysis-coverage-ledger-v88.json')['unchanged_global_uncertainty'])
        rel=read(O/'candidate-relation-coverage-v89.json');check('no spurious new pairs',rel['scoped_conditional_pairs']==115 and rel['remaining_pair_relations']==480 and rel['higher_order_groups']==24)
        bm=read(O/'module-phase-branch-matrix-v89.json');check('351branch24semanticmodules',sum(len(m['branches'])for m in bm['modules'])==351 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        for row in read(O/'candidate-profile-ledger-v89.json')['candidates']:
            check('candidate still conditional:'+row['id'],not row['candidate_closed']and not row['legal_fusion_proved'])
        for row in read(O/'candidate-record-index-v89.json')['model_candidates']:
            check('candidate registry digest:'+row['id'],hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'])
        for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation']:
            d=read(O/(base+'-v89.json'));check('nonclosure:'+base,not d['overall_complete'])
        cv=read(O/'module-theory-capacity-coverage-v89.json');check('26module theory ledger rows',len(cv['modules'])==26 and cv['refined_modules']==10 and not any(m['whole_module_closed']for m in cv['modules']))
        for x in read(O/'checkpoint-manifest-v89.json')['artifacts']:
            p=Path(x['path']);check('manifest:'+p.name,p.stat().st_size==x['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,new_server_model_Torch_NPU_or_native_execution=False,overall_complete=False)
    if args.preflight:write(O/'typed-capacity-analysis-preflight-v89.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
