"""Check exact hardware evidence,clock inference gates and operator cut algebra."""
from pathlib import Path
from fractions import Fraction
from math import ceil
import argparse,hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write
from analyze_clock_capacity_v112 import Text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def ck(name,value):assert value,name;checks.append(name)
    def pinned(row):
        b=Path(row['path']).read_bytes();return len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
    for row in read(O/'clock-capacity-inputs-v112.json')['inputs']:ck('input:'+row['path'],pinned(row))
    src=read(O/'clock-field-primary-contract-v112.json');h=Path(src['installed_header']['path']).read_text();ls=h.splitlines()
    ck('installed_header_sha',pinned(src['installed_header']))
    for s in src['sections']:ck('source_section:'+str(s['first_line']),ls[s['first_line']-1:s['last_line']]==s['source'])
    ck('MAX_symbol_is_nine','DCMI_FREQ_AICORE_MAX = 9'in h)
    ck('current_Vector_symbol_is_twelve','DCMI_FREQ_VECTORCORE_CURRENT = 12'in h)
    for d in src['official_A3_2610_documents']:
        ck('primary_doc:'+d['path'],pinned(d));p=Text();p.feed(Path(d['path']).read_text());text=' '.join(' '.join(p.parts).split())
        ck('normalized_doc:'+d['path'],hashlib.sha256(text.encode()).hexdigest()==d['normalized_text_sha256'])
        if d['path'].endswith('dcmia3_049.html'):
            ck('A3_support_not_all_enum','支持2、6、7、9'in text and '12'in text and 'Vector Core当前频率'in text)
            ck('A3_nine_is_rated_not_proved_max','9：AI Core额定频率'in text and 'TDP'in text)
    ck('no_common_physical_clock_proof',src['common_AIC_AIV_physical_clock_envelope_attested']is False)
    capability=read(O/'clock-query-capability-v112.json');queries=capability['queries']
    ck('bounded_four_queries',len(queries)==4)
    ck('only_info_queries',all(q['argv'][1]=='info' for q in queries))
    ck('freq_rejected',queries[0]['returncode']==215 and 'Error parameter of -t'in queries[0]['stdout'])
    ck('help_success',queries[1]['returncode']==0)
    ck('sensor_only_temperature',queries[2]['returncode']==0 and len(queries[2]['stdout'].splitlines())==4 and all('Temperature'in s for s in queries[2]['stdout'].splitlines()))
    ck('common_rejected_scope',queries[3]['returncode']==215 and 'Error parameter'in queries[3]['stdout'])
    ck('no_frequencies_inferred',capability['new_current_rated_or_maximum_clock_values_recovered']==0)
    ck('no_direct_dcmi_query',capability['manual_DCMI_initialization_or_calls']is False)
    ck('ssh_closed',capability['SSH_connection_closed']and not(N/'ssh166-clock-v112.sock').exists())
    prior=read(O/'clock-report-frequency-reconciliation-v112.json')['reused_not_rerun']
    ck('v18_reused_not_rediscovered',prior['PMU_groups']==286101 and prior['observed_report_frequency_Hz']==1800000000)
    casefile=read(O/'operator-clock-gated-cases-v112.json');cases=casefile['cases']
    ck('14_cases_7_each',len(cases)==14 and sum(c['engine']=='AIC'for c in cases)==7 and sum(c['engine']=='AIV'for c in cases)==7)
    for c in cases:
        if c['id'].startswith('INDEX_Q_WEIGHT_M'):
            m=c['M'];cuts={k:Fraction(v)for k,v in c['cuts'].items()}
            ck('typed_Q:'+str(m),cuts['INT8_Q']==Fraction(2*m*1280*4096,24*16384))
            ck('typed_Weight:'+str(m),cuts['BF16_Weight']==Fraction(2*m*5120*32,24*8192))
            ck('shared_fixedB_grant:'+str(m),cuts['shared_fixedB_weight_ingress']==ceil(Fraction(1280*4096+2*5120*32-24*65536,24*128)))
            ck('max_not_sum:'+str(m),Fraction(c['max_relaxed_cycle_cut'])==max(cuts.values())and c['discrete_cycle_reference']==ceil(max(cuts.values())))
        if 'SWA_INDEX_WEIGHT'in c['id']:
            ck('ragged_panel_weight_cut:'+c['id'],Fraction(c['cuts']['shared_fixedB_weight_ingress'])>Fraction(c['cuts']['BF16_compute']))
    cross=casefile['fixedB_decode_dense_crossover']
    ck('threshold_exact',Fraction(cross['rational_M_threshold'])==Fraction(1953,40))
    ck('integer_boundary_48_49',Fraction(80*48,3)<1302<Fraction(80*49,3))
    # Exhaustively verify the finite advertised integer interval,not actual device behavior.
    ck('crossover_M1_to_2048',all((Fraction(80*m,3)<1302)==(m<=48)for m in range(1,2049)))
    # Upper envelopes compose by max;one common upper envelope is sufficient even if clocks differ.
    fixtures=[(a,b,fa,fb)for a in [1,17,1000]for b in [2,29,800]for fa in [1000,2000]for fb in [700,1500]]
    ck('per_engine_before_max',all(max(Fraction(a,fa),Fraction(b,fb))>=Fraction(max(a,b),max(fa,fb))for a,b,fa,fb in fixtures))
    ck('cross_engine_raw_rank_counterexample',1000>800 and Fraction(1000,2000)<Fraction(800,1000))
    ck('nominal_not_envelope_counterexample',Fraction(1000,200)<Fraction(1000,100))
    ck('hypothetical_not_hardware',all(r['hypothetical']for r in read(O/'clock-invalid-inference-witnesses-v112.json')['witnesses']))
    cert=read(O/'operator-time-conversion-certificate-v112.json')
    ck('symbolic_no_numeric_ceiling',cert['status']=='SYMBOLIC_CEILING'and cert['no_wallclock_numeric_floor_no_speedup_no_all_algorithm_claim'])
    ck('Pad_probe_remains_prepared',read(O/'native-pad-probe-plan-v111.json')['description']['status']=='PREPARED_NOT_EXECUTED')
    if a.preflight:manifest=read(O/'checkpoint-manifest-v111.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v112.json');ck('progress_not_complete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        bm=read(O/'module-phase-branch-matrix-v112.json');ck('353_26',sum(len(m['branches'])for m in bm['modules'])==353 and len(bm['modules'])==26)
        cov=read(O/'candidate-relation-coverage-v112.json');ck('132_25_463',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(132,25,463))
        old=read(O/'operator-analysis-coverage-ledger-v111.json');new=read(O/'operator-analysis-coverage-ledger-v112.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        ck('14_obligations_remain_open',len(read(O/'coverage-current-obligations-v112.json')['obligations'])==14 and all(not g['closed']for g in read(O/'coverage-current-obligations-v112.json')['obligations']))
        for r in read(O/'candidate-record-index-v112.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_active_loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v112.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,operator_clock_cases=14,new_frequency_upper_bounds=0,
        hypothetical_clock_fixtures=len(fixtures),original_model_promotions=0,new_device_compute=False,overall_complete=False)
    if a.preflight:write(O/'clock-capacity-preflight-v112.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
