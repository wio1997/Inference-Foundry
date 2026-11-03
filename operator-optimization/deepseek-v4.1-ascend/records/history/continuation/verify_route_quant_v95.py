"""Check source binding, partition/reuse math and successor coverage invariants."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math
from finalize_origin_coverage_v9 import O,R,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def check(label,ok):assert ok,label;checks.append(label)
    src=read(O/'route-quant-operator-source-contract-v95.json')
    for r in src['parts']:
        p=Path(r['path']);lines=p.read_text().splitlines();check('sourcesha:'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
        for s in r['segments']:check('lines:'+p.name+':'+str(s['first_line']),lines[s['first_line']-1:s['last_line']]==s['source'])
    for r in src['frozen_parent_matches_original166_stage_files']:
        check('original166parent:'+Path(r['path']).name,hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['original166_stage_sha256'])
    check('noloadedELFclaim',not src['loaded_native_binary_attested'])
    m=read(O/'route-quant-tiling-and-work-models-v95.json');check('5sourcecases4472oldcalls',len(m['cases'])==5 and sum(r['saved_executions']for r in m['cases'])==4472)
    expected={(3,6):(21000,18),(3,3):(21000,9),(2,3):(21000,6),(816,6):(11000,836),(834,6):(11000,834)}
    for r in m['cases']:
        check('case:'+str((r['M'],r['topK'])),(r['expected_source_key'],r['quant_row_evaluations'])==expected[(r['M'],r['topK'])])
        assert r['source_AIV_budget']==40 and r['column_loops']==1 and not r['native_tiling_or_instruction_execution_attested']
        for world in range(1,769):
            localE=24 if r['topK']==6 else 8
            e=((localE*world+1)*4+31)//32*32
            full=((r['M']*r['topK']+31)//32*32)*36+e+13*5120+64<196352
            assert (21000 if full else 11000)==r['expected_source_key']
    check('keysstableacrossallowedworldsize1to768',True)
    check('862040vs799560sourceconditionalrowcounts',m['source_conditioned_total_quant_row_evaluations']==862040 and m['one_quant_per_input_row_reference']==799560)
    check('48partitionsarenotmonotoneinwork',next(r for r in m['cases']if r['M']==816)['alternate_48_assignment_partition_quant_rows']==816 and next(r for r in m['cases']if r['M']==834)['alternate_48_assignment_partition_quant_rows']==858)
    partition_cases=0
    for M in range(1,129):
        for k in range(1,9):
            total=M*k
            for A in range(1,49):
                p=(total+A-1)//A;intervals=[(lo,min(lo+p,total))for lo in range(0,total,p)]
                evals=sum(len(set(a//k for a in range(lo,hi)))for lo,hi in intervals)
                split= M+sum(lo%k!=0 for lo,_ in intervals[1:])
                assert evals==split and M<=evals<=M*k
                assert [a for lo,hi in intervals for a in range(lo,hi)]==list(range(total))
                partition_cases+=1
    check('49152partitioncasescoverassignmentsanddupformula',partition_cases==49152)
    # Any deterministic row-only function commutes with assignment copies. Exercise maps with sentinel drops.
    reuse_cases=0
    for M in range(1,25):
        for k in range(1,7):
            rows=[(r*7+1,r*3-2)for r in range(M)]
            def F(row):return (tuple(x*x-3*x for x in row),sum(abs(x)for x in row))
            for seed in range(13):
                route_order=sorted(range(M*k),key=lambda j:((j*7919+seed*97)%104729,j))
                maps={a:dest for dest,a in enumerate(route_order)if(a+seed)%7!=0}
                baseline={maps[a]:F(rows[a//k])for a in range(M*k)if a in maps}
                cached=[F(row)for row in rows];shared={maps[a]:cached[a//k]for a in range(M*k)if a in maps}
                assert baseline==shared;reuse_cases+=1
    check('1872rowfunctioncopycommutationmodels',reuse_cases==1872)
    q=read(O/'route-quant-reuse-interface-models-v95.json')['current_graph_reference']
    ceiling=read(O/'route-quant-reuse-interface-models-v95.json')['conditional_ceiling']
    check('representationwriteboundnotpaddeduniversal',ceiling['mandatory_expanded_value_write_bytes']==18*5124 and ceiling['B_upper_GM_publication']is None and ceiling['not_HBM_or_network_or_all_algorithms_bound'])
    check('GM338520246432261804',q['baseline_value_bytes']==338520 and q['common_value_bytes']==246432 and q['dual_value_bytes']==261804)
    check('common92088dual76716',q['conditional_common_input_byte_reduction']==92088 and q['conditional_dual_input_byte_reduction']==76716)
    check('Hplus32copiesremain',q['baseline_components']['routing_expanded_record_writes']==q['common_same_quantizer_components']['routing_expanded_record_writes']==18*5152)
    nf=read(O/'route-quant-native-frontiers-v95.json');fs=nf['frontiers']
    check('3200frontiers2560root6400oldTASK',len(fs)==3200 and nf['root_bound']==2560 and nf['reused_unique_native_TASKs']==6400)
    check('exactoldTASKset',len({(r['database_index'],r[f])for r in fs for f in ['MC2_TASK','shared_DQ_TASK']})==6400)
    check('outer24/48notinner40budget',set(r['recorded_MC2_blockNum']for r in fs)=={24} and set(r['recorded_MC2_mixBlockNum']for r in fs)=={48})
    for r in fs:assert r['same_recorded_stream']and not r['native_internal_key_or_same_quantizer_or_pointer_identity_proved']and not r['hypothetical_shared_producer_before_MC2_was_measured']
    check('nohypotheticaloverlapclaimed',True)
    if args.preflight:
        for r in read(O/'checkpoint-manifest-v94.json')['artifacts']:
            p=Path(r['path']);check('pretransition:'+p.name,p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v95.json');check('goalnotcompleted',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rc=read(O/'candidate-relation-coverage-v95.json');check('122/25/473unchanged',rc['scoped_conditional_pairs']==122 and rc['higher_order_groups']==25 and rc['remaining_pair_relations']==473)
        b=read(O/'module-phase-branch-matrix-v95.json');check('357records24semantic',sum(len(m['branches'])for m in b['modules'])==357 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in b['modules'])==24)
        a=read(O/'operator-analysis-coverage-ledger-v95.json');prior=read(O/'operator-analysis-coverage-ledger-v94.json')
        for f in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeunchanged:'+f,a[f]==prior[f])
        req=read(O/'whole-scope-requirement-audit-v95.json');check('fullscope9stillopen',len(req['requirements'])==9 and not req['overall_complete']and req['whole_modules_closed']==0)
        check('lowestMFUnotestablished',next(r for r in req['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for r in read(O/'candidate-record-index-v95.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        for r in read(O/'checkpoint-manifest-v95.json')['artifacts']:
            p=Path(r['path']);check('manifest:'+p.name,p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,partition_models=partition_cases,row_function_copy_models=reuse_cases,
      no_native_or_model_Torch_jobs=True,overall_complete=False)
    if args.preflight:write(O/'route-quant-analysis-preflight-v95.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
