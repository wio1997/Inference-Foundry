"""Conditional local-datapath cycle lowerbounds and saved-counter falsifiers.

Not hardware MFU,HBM bytes,clockmaximum,or actual native-path attestation.
"""
from collections import Counter,defaultdict
from pathlib import Path
from math import prod
from statistics import median
import json
from finalize_origin_coverage_v9 import O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions,fullkey


def stats(a):
    return dict(n=len(a),minimum=min(a),median=median(a),maximum=max(a))if a else None


def main():
    assert not(O/'A3-typed-ingress-regime-models-v80.json').exists()
    doc=read(O/'A3-datapath-primary-evidence-v80.json');assert doc['captured_frequency_MHz']==1800
    selected=[]
    for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']:
        if r['operator']not in ['MatMulV2','MatMulV3','QuantBatchMatmulV3']:continue
        try:a,b,*_=dimensions(r['signature']['inputShapes_text']);out=dimensions(r['signature']['outputShapes_text'])[0]
        except(ValueError,IndexError):continue
        dtypes=r['signature']['inputDataTypes_text'].split(';')
        if not(len(a)==len(out)==2 and a[0]==out[0]and prod(b)==a[1]*out[1]and dtypes[:2]in[['INT8','INT8'],['DT_BF16','DT_BF16']]):continue
        M,K,N=a[0],a[1],out[1];size=1 if dtypes[1]=='INT8'else 2;W=K*N*size
        C=24;R=64*1024
        Bcy=max(0,W-C*R)/(C*128);AorBcy=max(0,W-2*C*R)/(C*(256+128))
        selected.append(dict(regime_id=r['regime_id'],module_candidates=r['module_candidates'],operator=r['operator'],role=r['role'],origin=r['origin'],context=r['context'],
            executions=r['executions'],signature=r['signature'],M=M,N=N,K=K,weight_dtype=dtypes[1],weight_payload_bytes=W,
            conventional_dense_ops_2_per_MAC=2*M*N*K,source_native_median_us=r['measured_native_task_duration']['median']/1000,
            L1_only_weight_B_warm_initial_residency_bytes=C*R,L1_only_weight_either_input_warm_residency_bytes=2*C*R,
            L1_only_B_min_elapsed_core_clock_cycles=Bcy,L1_only_either_input_min_elapsed_core_clock_cycles=AorBcy,
            fixed_1800MHz_B_time_reference_us=Bcy/1800,fixed_1800MHz_either_time_reference_us=AorBcy/1800,
            fixed_1800MHz_time_is_bound_only_if_frequency_upper_assumption_holds=True,
            actual_native_orientation_L1_ingress_or_no_compression_proved=False,physical_numeric_actual_time_ceiling_proved=False))
    assert len(selected)==97 and sum(r['executions']for r in selected)==30030
    base={r['regime_id']:r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']}
    lookup={fullkey(base[r['regime_id']]['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in selected}
    assignments=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:assignments[a[0]][a[1]]=a
    observations=defaultdict(list);missing=Counter();markov=[];native_seen=set()
    for db in range(32):
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');defs={r['globalTaskId']:r for r in native['COMPUTE_definitions']};ts={r['task_record_rowid']:r for r in native['TASK_records']}
        current={}
        for tid,a in assignments[db].items():
            t=ts[tid];d=defs[t['globalTaskId']];key=fullkey(a[3],d['opType_text'],a[5],a[2],a[6],d)
            if key in lookup:
                model=lookup[key];current[tid]=model;assert(db,tid)not in native_seen;native_seen.add((db,tid))
        pmu=read(CACHE/f'analysis/native-PMU-complete-v18/db-{db:02d}.json');matched=set()
        for g in pmu['groups']:
            tids=g['native_TASK_candidates']
            if len(tids)!=1 or tids[0]not in current:continue
            tid=tids[0];assert tid not in matched;matched.add(tid);r=current[tid];t=ts[tid];d=defs[t['globalTaskId']]
            values={x['metric_name']:x['value']for x in g['metric_rows']};total=values['aic_total_cycles'];ratio=values['aic_mte1_ratio']
            blocks=d['blockNum'];active=min(24,blocks);W=r['weight_payload_bytes']
            # Rawcounterdiagnostic is conditionalonrecordedcycleaggregation;no bandwidthmeasurement.
            floorB=max(0,W-active*65536)/128
            floorAny=max(0,W-active*2*65536)/(256+128)
            pipe_equiv=total*ratio
            ob=dict(database_index=db,TASK_rowid=tid,regime_id=r['regime_id'],blockNum=blocks,active_program_capacity_reference=active,
                aic_total_cycles=total,aic_mte1_ratio=ratio,aic_mte1_counter_equivalent_cycles=pipe_equiv,
                minimum_weight_B_L1_ingress_core_cycles=floorB,minimum_weight_either_input_L1_ingress_core_cycles=floorAny,
                B_floor_over_recorded_MTE1_counter= floorB/pipe_equiv if pipe_equiv>0 else None,
                either_floor_over_recorded_MTE1_counter=floorAny/pipe_equiv if pipe_equiv>0 else None,
                B_domain_exceeds_recorded_MTE1_counter=bool(floorB>pipe_equiv+1e-6),
                either_domain_exceeds_recorded_MTE1_counter=bool(floorAny>pipe_equiv+1e-6),
                native_TASK_duration_ns=t['endNs']-t['startNs'],loaded_native_path_or_MAC_MFU_proved=False)
            observations[r['regime_id']].append(ob)
            if(r['K'],r['N'],r['weight_dtype'])==(256,129280,'DT_BF16'):markov.append(ob)
        for tid in current:
            if tid not in matched:missing[current[tid]['regime_id']]+=1
        if db%8==7:print(json.dumps(dict(scanned_DB=db+1,native_selected=len(native_seen))),flush=True)
    assert len(native_seen)==30030 and len(markov)==520
    for r in selected:
        obs=observations[r['regime_id']];assert len(obs)+missing[r['regime_id']]==r['executions']
        r.update(native_PMU_records=len(obs),missing_PMU_records=missing[r['regime_id']],
            native_aic_total_cycles=stats([x['aic_total_cycles']for x in obs]),native_MTE1_ratio=stats([x['aic_mte1_ratio']for x in obs]),
            B_model_exceeds_recorded_MTE1_counter=sum(x['B_domain_exceeds_recorded_MTE1_counter']for x in obs),
            either_model_exceeds_recorded_MTE1_counter=sum(x['either_domain_exceeds_recorded_MTE1_counter']for x in obs))
    allobs=[x for xs in observations.values()for x in xs]
    write(O/'A3-typed-ingress-regime-models-v80.json',dict(regimes=selected,regime_count=97,native_executions=30030,
        native_PMU_records=len(allobs),missing_PMU_records=sum(missing.values()),
        input_dtype_regime_counts=dict(Counter(r['weight_dtype']for r in selected)),
        counter_budget_diagnostics=dict(B_model_exceeds_recorded_MTE1_counter=sum(x['B_domain_exceeds_recorded_MTE1_counter']for x in allobs),
            either_model_exceeds_recorded_MTE1_counter=sum(x['either_domain_exceeds_recorded_MTE1_counter']for x in allobs)),
        declared_domain='ConventionaldenseCubealgorithm usinguncompressed mandatoryweightvalues throughL1→L0;24AIC max,documented2201path capacities. FixedB or eitherA/Borientationsseparate. InitwarmL0storagegrant iscapacityupper,notassumedempty.',
        scope_limits=['GM→L0directpaths on2201bypassL1constraint;nativeinputorientation/pathnotattested. Ifordinarynewimplementation usesdifferentalgorithm/precision/precompute/compression,thisboundmustberebuilt.',
            'Captured1800MHzisnotprovenmaximum;positivecyclefloors arearchitecture-domainmodels,1800MHzmicrosecondvalues conditionalreferences.',
            'PMUMTE1 equivalentcycles aretotalcycles*reportedratio. Budgetexceedance falsifies a conjunction ofmodel/counterpremises;notproofwhichpremise/pathfailed andnotmeasuredbytes/bandwidth/MFU.',
            'Noper-kernelboundadditionasrequestlatency/fusedsubgraphbound. Modulecandidatecountsoverlap;noactuallowestmoduleMFUrankclaimed.'],
        global_profile_ambiguity_or_missing_PMU_rewritten=False,actual_physical_numeric_time_ceiling_proved=False,overall_complete=False))
    W=256*129280*2;R=24*65536
    write(O/'Markov-warm-ingress-frontier-v80.json',dict(records=markov,executions=520,regimes=len({r['regime_id']for r in markov}),
        five_step_recompute_domain=dict(weight_payload_bytes=W,steps=5,L0B_capacity_grant_bytes=R,
            B_orientation_required_weight_load_bytes=5*max(0,W-R),B_orientation_min_elapsed_cycles=5*max(0,W-R)/(24*128),
            B_orientation_fixed1800MHz_time_reference_us=5*max(0,W-R)/(24*128*1800),
            meaning='L2warmweights donotremoveL1→L0servicewhen eachsteprecomputesfullW2andnextIDdependsonglobalargmax. Sumnecessaryloadworkononecapacitycut;notaddindependentresourcebounds.',
            excludes='Memo/precomputedtransitiontable/certifiedpruning/differentalgorithm/directGM→L0/unknownnativeorientation;notanallalgorithmquerylowerbound.'),
        B_floor_over_MTE1_counter=stats([x['B_floor_over_recorded_MTE1_counter']for x in markov if x['B_floor_over_recorded_MTE1_counter']is not None]),
        either_floor_over_MTE1_counter=stats([x['either_floor_over_recorded_MTE1_counter']for x in markov if x['either_floor_over_recorded_MTE1_counter']is not None]),
        B_budget_exceedances=sum(x['B_domain_exceeds_recorded_MTE1_counter']for x in markov),either_budget_exceedances=sum(x['either_domain_exceeds_recorded_MTE1_counter']for x in markov),
        effective_load_throughput_or_memory_bottleneck_proved=False,native_program_recompute_steps_not_newly_executed=True,overall_complete=False))
    planes=[dict(name='Q_B',bytes=1280*32768),dict(name='sharedGateUp',bytes=5120*4608),dict(name='sharedDown',bytes=2304*5120)]
    total=sum(r['bytes']for r in planes)
    write(O/'A3-joint-ingress-cut-proof-v80.json',dict(planes=planes,total_distinct_uncompressed_weight_bytes=total,
        domain='OnewindowrequiringallthreeconventionalINT8GEMMplanes throughL1→L0B;distinctvalue-storage namespace/no crossplanecompression/reuse,24AIC andatmostoneglobalinitialL0Bcapacitygrant.',
        one_initial_L0B_grant_bytes=R,joint_min_elapsed_cycles=(total-R)/(24*128),
        independent_per_kernel_grants_sum_cycles=sum(max(0,p['bytes']-R)/(24*128)for p in planes),
        fixed1800MHz_joint_time_reference_us=(total-R)/(24*128*1800),
        rule='Accumulateprovennecessaryworkononesharedcapacitycutonce;combinewithotherresourcesbymax. InitialRcannotbegiftedonceperoperandwhenprovingajointdistinctplanewindow.',
        observed_per_layer_aliases_or_all_transformed_graphs_attested=False,universal_algorithm_lower_bound=False,critical_path_gain=None,overall_complete=False))
    print(dict(selected_regimes=97,executions=30030,PMU=len(allobs),missing=sum(missing.values()),Markov=520,
        B_budget_exceedances=sum(x['B_domain_exceeds_recorded_MTE1_counter']for x in allobs),either_exceedances=sum(x['either_domain_exceeds_recorded_MTE1_counter']for x in allobs),
        status='CONDITIONAL_LOCAL_PATH_AND_COUNTER_ANALYSIS_READY'))


if __name__=='__main__':main()
