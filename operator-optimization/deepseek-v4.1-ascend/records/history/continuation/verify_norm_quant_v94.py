"""Independent numeric/value-budget checks and authoritative-state guards for v94."""
from fractions import Fraction as F
from pathlib import Path
from math import floor,log2,ceil
import argparse,hashlib,json
from finalize_origin_coverage_v9 import O,R,read,write

def bf_oracle(x):
    # Independently round an exact positive float value to a normal BF16 grid.
    a=F(x);e=floor(log2(x));step=F(2)**(e-7)
    return float(round(a/step)*step)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(label,ok):assert ok,label;checks.append(label)
    source=read(O/'norm-quant-operator-source-contract-v94.json')
    for p in source['parts']:
        path=Path(p['path']);lines=path.read_text().splitlines();check('sourcesha:'+path.name,hashlib.sha256(path.read_bytes()).hexdigest()==p['sha256'])
        for seg in p['parts']:check('lines:'+path.name+':'+str(seg['first_line']),lines[seg['first_line']-1:seg['last_line']]==seg['source'])
    check('noinstalledsubstitutionclaim',not source['loaded_native_binary_attested'])
    nf=read(O/'norm-quant-native-frontiers-v94.json');fs=nf['frontiers'];check('3200/2560/640profilefrontiers',len(fs)==3200 and nf['root_bound']==2560 and nf['index_query_frontiers']==640 and nf['root_index_query_frontiers']==512)
    check('14080previousTASKs',len({(r['database_index'],t['task_record_rowid'])for r in fs for t in r['stages'].values()})==nf['reused_unique_native_TASKs']==14080)
    check('22/9566observedNQregimes',len(nf['observed_regimes'])==22 and sum(r['executions']for r in nf['observed_regimes'])==9566)
    old=read(O/'projection-graph-chains-v61.json');lookup={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in old['graph_projection_chains']}
    for r in fs:
        prior=lookup[(r['database_index'],r['graph_ordinal'],r['block_ordinal'])]
        for key in ['Q_A','Q_NORM','Q_QUANT','Q_B']:assert r['stages'][key]['task_record_rowid']==prior['stages'][key]['task_record_rowid']
        assert not r['native_pointer_value_identity_proved']and not r['new_fused_precision_or_resource_path_attested']
    check('sourceidentitynotinferredfromshape',True)
    pm=read(O/'norm-quant-precision-and-row-models-v94.json');w=pm['BF16_rounding_witness']
    check('independentBF16witnessround',list(map(bf_oracle,w['FP32_normalized_prefix']))==w['BF16_normalized_prefix']==[34.,11.375])
    check('INT8code42vs43',w['raw_codes']==w['FP32_direct_codes']==[127,43]and w['BF16_then_quant_codes']==[127,42])
    ratios=[127*F(v)/max(map(F,w['BF16_normalized_prefix']))for v in w['BF16_normalized_prefix']]
    check('rationalRNEwitness',list(map(round,ratios))==w['BF16_then_quant_codes'])
    cases=0
    for a in range(1,24):
        for b in range(-a,a+1):
            for c in [F(1,7),F(1,2),F(2),F(17,3)]:
                z=[F(a),F(b)];cz=[c*v for v in z];assert [round(127*v/max(map(abs,z)))for v in z]==[round(127*v/max(map(abs,cz)))for v in cz];cases+=1
    check('2300exactrealidentitycases',cases==pm['exact_real_rational_cases']==2300)
    row=pm['row_frontier_witness'];check('sameprefixdifferentfinalcode',row['final_codes_at_first_coordinate']==[127,64])
    m=read(O/'norm-quant-interface-and-resource-models-v94.json');check('48deduplicatedcostcases',len(m['models'])==m['interface_case_count']==48)
    seen=set()
    for r in m['models']:
        M,K,k=r['M'],r['K'],r['quantized_consumers'];v=r['variant'];key=(M,K,k,v,r['normalized_qr_store_retained']);assert key not in seen;seen.add(key)
        baseline=(2+4*k)*M*K+8*k*M
        if v=='SHARED_QUANT':out=(5+k)*M*K+4*(1+k)*M
        elif v=='NORM_FUSED_SHARED_QUANT':out=(1+k)*M*K+4*(1+k)*M+2*M*K*r['normalized_qr_store_retained']
        else:out=2*M*K+8*M+2*M*K*r['normalized_qr_store_retained']
        assert r['baseline_value_bytes']==baseline and r['transformed_value_bytes']==out and r['conditional_removed_GM_interface_value_bytes']==baseline-out
        assert r['INT8_VC_GM_bridge_value_bytes']>=2*M*K and r['HBM_saving_or_speedup']is None
    check('alllogicalbudgetformulasandVCbridgeretained',True)
    example=[r for r in m['models']if r['M']==3 and r['quantized_consumers']==2 and not r['normalized_qr_store_retained']]
    check('26892/30744conditionalbudget',[r['conditional_removed_GM_interface_value_bytes']for r in example]==[26892,30744])
    for r in m['shared_weight_cuts']:
        check('sharedweightcut:'+str(r['planes']),r['L1_fixedB_min_cycles']==ceil(F(r['weight_bytes']-24*65536,24*128))and r['INT8_useful_compute_min_cycles']==720)
    check('14848/19115sharedL1cyclecuts',[r['L1_fixedB_min_cycles']for r in m['shared_weight_cuts']]==[14848,19115])
    if args.preflight:
        for row in read(O/'checkpoint-manifest-v93.json')['artifacts']:
            p=Path(row['path']);check('pretransition:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v94.json');check('progressnotcompletion',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v94.json');check('prior122/25/473notinflated',cov['scoped_conditional_pairs']==122 and cov['higher_order_groups']==25 and cov['remaining_pair_relations']==473)
        b=read(O/'module-phase-branch-matrix-v94.json');check('356records24semantic',sum(len(m['branches'])for m in b['modules'])==356 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in b['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v94.json');oldop=read(O/'operator-analysis-coverage-ledger-v93.json')
        for field in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+field,op[field]==oldop[field])
        req=read(O/'whole-scope-requirement-audit-v94.json');check('wholegoalstillincomplete',not req['overall_complete']and req['whole_modules_closed']==0 and len(req['requirements'])==9)
        check('lowestMFUstillnotestablished',next(r for r in req['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for row in read(O/'candidate-record-index-v94.json')['model_candidates']:check('mastersha:'+row['id'],hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'])
        for row in read(O/'checkpoint-manifest-v94.json')['artifacts']:
            p=Path(row['path']);check('manifest:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,native_frontiers=3200,reused_TASKs=14080,exact_real_cases=cases,
      interface_cases=48,new_native_model_or_Torch_work=False,overall_complete=False)
    if args.preflight:write(O/'norm-quant-analysis-preflight-v94.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
