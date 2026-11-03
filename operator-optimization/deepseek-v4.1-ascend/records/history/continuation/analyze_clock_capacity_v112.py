"""Bind clock-field semantics to typed operator cuts; no accelerator execution."""
from pathlib import Path
from fractions import Fraction
from html.parser import HTMLParser
from math import ceil
import hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write

class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)

def record(p):
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    assert not (O/'clock-capacity-inputs-v112.json').exists()
    source=read(O/'clock-source-index-v112.json')
    header=Path(next(r['local_path']for r in source['files']if r['status']=='READ'))
    docs=read(O/'clock-primary-receipts-v112.json')['documents']
    inputs=[O/'REVIEW-CHECKPOINT-v111.json',O/'coverage-current-obligations-v111.json',
        O/'typed-capacity-primary-contract-v89.json',O/'index-projection-work-and-resource-models-v106.json',
        O/'router-Vector-instruction-cuts-v89.json',O/'complete-PMU-resource-contract-v18.json',
        O/'clock-source-index-v112.json',O/'clock-primary-receipts-v112.json',header,
        N/'clock-source166-v112.json',N/'clock-sensors166-v112.json',N/'clock-common166-v112.json']
    inputs += [Path(r['path'])for r in docs]
    write(O/'clock-capacity-inputs-v112.json',dict(inputs=[record(p)for p in inputs],
        operator_questions=['Does the installed MAX-labelled enum attest a physical clock upper bound?',
            'Can the saved 1800MHz report metadata be used for Cube andVector physical ceilings?',
            'Which existing projection-cut comparisons are independent of clock conversion?'],
        no_new_model_tensor_capture_replay_benchmark_or_configuration_change=True,overall_complete=False))
    ls=header.read_text().splitlines()
    sections=[]
    for first,last in [(130,140),(177,183),(923,930),(1741,1743),(1774,1777),(2140,2149)]:
        sections.append(dict(first_line=first,last_line=last,source=ls[first-1:last]))
    documents=[]
    for d in docs:
        h=Text();h.feed(Path(d['path']).read_text());t=' '.join(' '.join(h.parts).split())
        documents.append(dict(**d,normalized_text_sha256=hashlib.sha256(t.encode()).hexdigest()))
        name=Path(d['path']).name
        if name=='dcmia3_049.html':
            assert '支持2、6、7、9' in t and 'AI Core额定频率' in t and 'TDP' in t
        if name=='dcmia3_037.html':assert '额定频率' in t and 'cur_freq' in t
        if name=='dcmia3_039.html':assert 'AICPU的最大运行频率' in t and 'max_freq' in t
        if name=='dcmia3_050.html':assert 'bandwith_util_rate' in t and 'memory_size' in t
    write(O/'clock-field-primary-contract-v112.json',dict(installed_header=record(header),sections=sections,
        official_A3_2610_documents=documents,
        fields=[dict(interface='dcmi_get_device_aicore_info',field='freq',meaning='AI Core rated MHz',hard_upper_bound_attested=False),
            dict(interface='dcmi_get_device_aicore_info',field='cur_freq',meaning='AI Core current MHz',hard_upper_bound_attested=False),
            dict(interface='dcmi_get_device_frequency',field='input_type9/DCMI_FREQ_AICORE_MAX',meaning='Installed enum name MAX; official A3 API describes rated sustained-in-TDP frequency, not a certified supremum over allowed runs.',hard_upper_bound_attested=False),
            dict(interface='dcmi_get_device_frequency',field='input_type12/DCMI_FREQ_VECTORCORE_CURRENT',meaning='Enumerated Vector current frequency; A3 document lists supported types2,6,7,9 only. Target type12 support was not tested.',target_support_attested=False),
            dict(interface='dcmi_get_device_aicpu_info',field='max_freq',meaning='AICPU maximum MHz; different engine, not Cube/Vector upper clock.'),
            dict(interface='dcmi_get_device_hbm_info',field='freq,memory_size,bandwith_util_rate',meaning='Memory clock,capacity and utilization fields do not specify peak byte/s or cache/duplex/packet resource topology.')],
        exact_installed_header_ABI=True,public_document_same_driver_release_not_attested=True,
        common_AIC_AIV_physical_clock_envelope_attested=False,whole_hardware_or_operator_ceiling_closed=False,overall_complete=False))
    sensor=read(N/'clock-sensors166-v112.json')['result'];common=read(N/'clock-common166-v112.json')['result']
    queries=source['commands']+[sensor,common]
    write(O/'clock-query-capability-v112.json',dict(host='bm-jn-zs-zone1-910C-64G-10-105',card=0,chip=0,
        queries=[{k:q[k]for k in ['argv','returncode','stdout','stderr']}for q in queries],
        outcomes=dict(freq_selector='REJECTED_RETURN215',help='SUPPORTED',sensors='FOUR_TEMPERATURES_ONLY',common_with_card_chip='REJECTED_RETURN215'),
        new_current_rated_or_maximum_clock_values_recovered=0,unsupported_general_API_not_inferred=True,
        admin_tool_library_loading_not_excluded=True,manual_DCMI_initialization_or_calls=False,
        no_model_or_application_tensor_capture_replay_benchmark_or_config_change=True,
        SSH_connection_closed=not(N/'ssh166-clock-v112.sock').exists(),overall_complete=False))
    old=read(O/'complete-PMU-resource-contract-v18.json')
    freq=old['native_consistency']['time_formula_test']
    write(O/'clock-report-frequency-reconciliation-v112.json',dict(prior_answer='complete-PMU-resource-contract-v18.json',
        reused_not_rerun=dict(PMU_groups=freq['engines']['aic']['matching_groups'],
            observed_report_frequency_Hz=freq['assumed_captured_frequency_Hz'],
            report_formula=freq['formula'],core_counts=freq['core_counts']),
        new_interpretation_gate='A3 nominal/current/max-engine field semantics do not turn report conversion frequency into an upper envelope of physical clocks.',
        report_frequency_not_relabelled_as_hardware_fmax=True,actual_Task_duration_not_replaced_by_report_pipe_times=True,
        prior_cycle_unit_identity_is_not_new_work=True,new_raw_PMU_census_or_exporter_search=False,
        missing_frequency_or_HBM_bytes_not_filled_by_zero=True,overall_complete=False))
    write(O/'operator-time-conversion-certificate-v112.json',dict(status='SYMBOLIC_CEILING',
        work_capacity_rule=dict(necessary_work='W_e lower bound for one declared operation/representation/numeric/read-set domain',
            capacity='n_e(t)*r_e(mode,t)*f_e(t); must be a valid capacity upper envelope on the same service/resource scope',
            condition='If n_e<=N_e, r_e<=R_e and f_e<=F_e for all relevant times, W_e<=N_e*R_e*F_e*T',
            result='T >= W_e/(N_e*R_e*F_e). If a proved same-domain cycle cut is C_e, T >= C_e/F_e.',
            missing='No physical F_AIC, F_AIV or U_HBM has been established here.'),
        multiple_resources='T>=max(C_AIC/F_AIC,C_AIV/F_AIV,B_lower/U_same_tier,...). Add work only on a proved shared resource;sum delays only with required serial critical-path proof.',
        same_clock_case='max(C_AIC,C_AIV)/F is valid if F is a documented common upper bound for both domains;physical equality of their instantaneous clocks is stronger and not needed for this weaker common-envelope bound.',
        unequal_clock_case='Per-engine conversion precedes max;raw cycles from different engines are not directly comparable without a clock-envelope relation.',
        variable_clock_case='A full counter-to-physical-cycle identity and matching clock time integral could support a trace-specific bound;one current sample,nominal MHz or report scalar does not supply that identity/integral.',
        granularity='Fractional aggregate work/rate relaxations remain valid;whole instruction/grant rounding may strengthen cuts only in the declared discrete service model.',
        clockfree_comparison='Within one stated AIC service domain, ordering of same-clock cycle cuts is unchanged by any common positive clock scale;it is not measured bottleneck attribution.',
        MFU='A comparable all-module physical denominator additionally needs necessary actual work,precision/mode/engine scope,clock capacity and matching elapsed scope. Missing bounds cannot produce a lowest-MFU ranking.',
        no_wallclock_numeric_floor_no_speedup_no_all_algorithm_claim=True,overall_complete=False))
    projections=read(O/'index-projection-work-and-resource-models-v106.json');rows=[]
    for r in projections['models']:
        m=r['M'];q=Fraction(r['IndexQ_useful_compute_cycle_cut']);w=Fraction(r['IndexWeight_useful_compute_cycle_cut']);ing=Fraction(r['combined_fixedB_weight_ingress_cycle_cut'])
        assert q==Fraction(80*m,3)and w==Fraction(5*m,3)
        rows.append(dict(id='INDEX_Q_WEIGHT_M'+str(m),engine='AIC',M=m,
            cuts=dict(INT8_Q=str(q),BF16_Weight=str(w),shared_fixedB_weight_ingress=str(ing)),
            max_relaxed_cycle_cut=str(max(q,w,ing)),discrete_cycle_reference=ceil(max(q,w,ing)),
            strongest_of_these_cuts='INT8_Q'if q>ing else 'shared_fixedB_weight_ingress',
            physical_time='T>=C/F_AIC only after a valid F_AIC upper bound and original path/numeric/domain controls',
            original_cycle_result_reused_not_new=True))
    for r in projections['upstream_panel_models']:
        c=Fraction(r['BF16_compute_cycle_cut']);ing=Fraction(r['fixedB_weight_ingress_cycle_cut'])
        rows.append(dict(id='QA_SWA_INDEX_WEIGHT'+('_C1'if r['C1_source_branch']else ''),engine='AIC',
            query_rows=r['query_rows'],global_KV_rows=r['global_KV_rows'],
            cuts=dict(BF16_compute=str(c),shared_fixedB_weight_ingress=str(ing)),
            max_relaxed_cycle_cut=str(max(c,ing)),discrete_cycle_reference=ceil(max(c,ing)),
            strongest_of_these_cuts='shared_fixedB_weight_ingress',original_cycle_result_reused_not_new=True,
            physical_time='T>=C/F_AIC only in the independently normalized ragged panel domain of v106/v99.'))
    for r in read(O/'router-Vector-instruction-cuts-v89.json')['regimes']:
        rows.append(dict(id='ROUTER_'+r['regime_id'],engine='AIV',M=r['M'],
            max_individual_direct_instruction_cycle_floor=str(Fraction(str(r['max_individual_direct_instruction_cycle_floor']))),
            physical_time='T>=C/F_AIV in pinned normType2 instruction-lowering domain;high-level TopK/reduction algorithm alternatives are not bounded by this specific instruction list.',
            original_cycle_result_reused_not_new=True,sort32_not_universal_TopK_bound=True))
    write(O/'operator-clock-gated-cases-v112.json',dict(cases=rows,projection_cases=7,router_cases=7,
        fixedB_decode_dense_crossover=dict(Q_compute='80*M/3',Weight_compute='5*M/3',shared_ingress=1302,
            rational_M_threshold=str(Fraction(1302*3,80)),largest_integer_M_ingress_strictly_stronger=48,
            first_integer_M_Q_compute_stronger=49,
            scope='Only ordering of these existing full ordinary-product/fixedB L1→L0B/uncompressed-weight/no-cached-output lower cuts,24AIC andone initialgrant. No measured bottleneck/optimal fusion/other orientation claim.'),
        prior_cycle_math_reused=True,new_physical_clock_bound=False,new_native_regime_promotions=0,
        no_cross_engine_MFU_ranking=True,overall_complete=False))
    witnesses=[dict(id='NOMINAL_NOT_ENVELOPE',hypothetical=True,C=1000,nominal_MHz=100,allowed_actual_MHz=200,
            nominal_conversion_us=10,possible_service_us=5,lesson='Absent an upper-envelope premise,T>=C/f_nom does not follow. No such boost is asserted on166.'),
        dict(id='RAW_ENGINE_CYCLE_RANK_REVERSES',hypothetical=True,C_AIC=1000,C_AIV=800,
            F_AIC_MHz=2000,F_AIV_MHz=1000,AIC_service_us='1/2',AIV_service_us='4/5',
            lesson='Smaller raw Vector cycle cut may correspond to a larger time cut;do per-engine conversions beforemax.'),
        dict(id='HBM_FREQ_NOT_BYTE_CAPACITY',hypothetical=True,same_frequency=True,
            differing_unbound_fields=['aggregate_bus_width','effective_data_rate_factor','read_write_sharing','protocol_efficiency','cache/storage_tier'],
            lesson='A clock field plus utilization percent does not specify a unique byte/s capacity upper bound.')]
    write(O/'clock-invalid-inference-witnesses-v112.json',dict(witnesses=witnesses,
        algebra_only_not_device_experiments=True,actual_frequency_values_not_inferred=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',operator_clock_cases=len(rows),new_frequency_values=0,
        native_promotions=0,pending_Pad_probe_unexecuted=True,overall_complete=False)))

if __name__=='__main__':main()
