"""Check exact saved TASK identities, ragged algebra and declared-domain resource/release models."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import argparse,hashlib,itertools,json,math
from finalize_origin_coverage_v9 import N,O,R,read,write
from mirror_saved_profile_core_v53 import CACHE

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    for x in read(O/'projection-panel-inputs-v99.json')['inputs']:
        p=Path(x['path']);check('input:'+p.name,p.stat().st_size==x['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'])
    for x in read(O/'projection-panel-source-contract-v99.json')['parts']:
        p=Path(x['path']);ls=p.read_text().splitlines();check('source:'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'])
        for s in x['segments']:check('segment:'+str(s['first_line']),ls[s['first_line']-1:s['last_line']]==s['source'])
    f=read(O/'projection-panel-native-frontiers-v99.json');check('3200frontiers',len(f['frontiers'])==3200 and f['reused_TASKs']==19200)
    check('phase/rolespreserved',(f['root_bound'],f['phase_unbound'],f['unique_SWA_roles'],f['ambiguous_source20_roles'])==(2560,640,3120,80))
    bydb=defaultdict(list)
    for r in f['frontiers']:bydb[r['database_index']].append(r)
    unique=set()
    for db,rs in bydb.items():
        raw=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');tasks={t['task_record_rowid']:t for t in raw['TASK_records']};defs={d['globalTaskId']:d for d in raw['COMPUTE_definitions']}
        for r in rs:
            for kind,v in r['stages'].items():
                t=tasks[v['task_record_rowid']];d=defs[t['globalTaskId']];unique.add((db,v['task_record_rowid']))
                assert(t['startNs'],t['endNs'],t['streamId'])==(v['startNs'],v['endNs'],v['streamId'])
                assert d['opType_text']==v['op']
                if kind=='Q_A':assert d['inputShapes_text']=='"3,5120;1280,5120"'and d['outputDataTypes_text']=='DT_BF16'
                if kind=='SWA_KV':assert d['inputShapes_text']=='"24,5120;512,5120"'and d['inputDataTypes_text']=='DT_BF16;DT_BF16'
            s=r['stages']
            for field,lhs,rhs in [('SWA_with_Qnorm_ns','SWA_KV','Q_NORM'),('SWA_with_Qquant_ns','SWA_KV','Q_QUANT'),('QB_with_KVnorm_ns','Q_B','KV_NORM')]:
                expected=max(0,min(s[lhs]['endNs'],s[rhs]['endNs'])-max(s[lhs]['startNs'],s[rhs]['startNs']));assert expected==r[field]>0
            assert r['Qnorm_starts_before_KVends']and r['Qnorm_finishes_before_KVends']and r['QA_end_before_KVend']
        check('nativeall6stages:'+str(db),True)
    check('unique19200',len(unique)==19200)
    summary=f['summaries'];check('strictroot2496',summary['root_unique']['frontiers']==2496)
    old=read(O/'F33-profile-order-contract-v27.json')['order_audit']['observed_overlap_ns']
    for nk,ok in [('SWA_with_Qnorm_ns','SWA_projection_with_Q_norm'),('QB_with_KVnorm_ns','SWA_norm_with_Q_B')]:
        check('v27layer20crosscheck:'+nk,summary['layer20_source_stream']['overlaps'][nk]['duration_ns']['median']==old[ok]['median'])
    rows=read(O/'projection-panel-work-and-resource-models-v99.json')['models'];check('tworesourcecases',len(rows)==2)
    for r,expected in zip(rows,[(165150720,440401920,18350080,64512,5462),(290979840,566231040,23592960,113664,7168)]):
        check('exacttypedresourcecase:'+str(r['C1_source_branch']),(r['useful_classical_ops'],r['global_rectangular_classical_ops'],r['distinct_weight_value_bytes'],r['AIC_AIV_logical_bridge_total_value_bytes'],r['L1_path_fixedB_conditional_min_cycles'])==expected)
        ops=sum(2*g['M']*g['N']*5120 for g in r['groups']);weights=sum(g['N']*5120*2 for g in r['groups'])
        check('oneweightunion:'+str(r['C1_source_branch']),ops==r['useful_classical_ops']and weights==r['distinct_weight_value_bytes']and sum(g['id']=='SWA'for g in r['groups'])==1)
        check('notonewarmgrantperplane:'+str(r['C1_source_branch']),r['L1_path_fixedB_conditional_min_cycles']==math.ceil(max(0,weights-24*65536)/(24*128)))
        check('separatecomputeandmemorycut:'+str(r['C1_source_branch']),Fraction(r['useful_BF16_arithmetic_conditional_min_cycles'])==Fraction(ops,24*8192)and r['actual_HBM_saving_or_speedup']is None)
    check('incrementalinputnotdoublecounted',rows[1]['conditional_input_read_difference_bytes']==245760+30720 and rows[1]['incremental_QA_input_read_difference_after_existing_SWA_C1_join_bytes']==30720)
    # Raggedoutputmapping preservesroworderandbranchlinearweights;testsmallintegerdomainexactly,notnativeparity.
    cases=0
    for G in range(3,8):
        for J in itertools.permutations(range(G),3):
            H=3;X=[[i+2*j-3 for j in range(H)]for i in range(G)];W=[[2*i-j+1 for j in range(H)]for i in range(7)]
            full=[[sum(x*w for x,w in zip(row,col))for col in W]for row in X]
            QA=[[sum(X[i][k]*W[n][k]for k in range(H))for n in range(3)]for i in J]
            assert QA==[[full[i][n]for n in range(3)]for i in J]
            assert [[sum(X[i][k]*W[n][k]for k in range(H))for n in range(3,7)]for i in range(G)]==[row[3:]for row in full]
            cases+=1
    check('raggedintegerproofcases',cases==420)
    p=read(O/'projection-panel-reduction-and-precision-v99.json')['K_panel_counterexample'];check('Kpartialnormcounterexample',p['completed_plus_square_sum']==4 and p['completed_minus_square_sum']==0 and p['per_K_partial_square_sums']==[1,1])
    # Independentlyderive completiontimesfromreleases;QA0..a;KVafterQA;QBafterQA-normandKV.
    counts=dict(equal=0,better=0,worse=0);DAGcases=0
    for qa,kv,qn,qb,kn in itertools.product([0,1,2,4,8,16,32,64],repeat=5):
        qn_end=qa+qn;kv_end=qa+kv;join=max(qn_end,kv_end);old=max(join+qb,join+kn)
        together=qa+kv;keep=together+qn+max(qb,kn);early=max(together+qn+qb,together+kn)
        panel=max(join+qb,kv_end+kn)
        assert keep-old==min(kv,qn)and panel<=old and panel<=early
        counts['better'if early<old else'worse'if early>old else'equal']+=1;DAGcases+=1
    dag=read(O/'projection-panel-release-DAG-models-v99.json');check('32768DAGcases',DAGcases==dag['cases']==32768 and counts==dag['whole_output_early_KVnorm_comparison'])
    check('bothsignsnotuniversalbenefit',all(counts[x]>0 for x in counts)and not dag['actual_speedup_or_achievable_schedule_proved'])
    for name,w in dag['witnesses'].items():check('positivewitness:'+name,all(v>0 for v in w['durations'].values()))
    packet=read(O/'projection-panel-composition-clauses-v99.json');prior=read(O/'candidate-relation-clauses-v98.json');pairs={frozenset([r['a'],r['b']])for r in prior['clauses']}
    check('4previouslyblankcentralpairs',len(packet['clauses'])==4 and all(frozenset([r['a'],r['b']])not in pairs for r in packet['clauses']))
    modes=read(O/'projection-panel-mode-atom-audit-v99.json');mode={r['candidate']:r for r in modes['records']}
    check('C2C1distinctmodeatoms',mode['F18']['replacement_norm_atoms']==['compress.C2.norm512']and mode['F33']['replacement_norm_atoms']==['compress.C1.norm512']and mode['F19']['replacement_norm_atoms']==['compress.C1.norm512','compress.C2.norm512'])
    for cid,row in mode.items():
        p=Path(row['master_before_path'])if a.preflight else O/(cid+'-master-before-v99.json')
        check('originalcandidatemode:'+cid,hashlib.sha256(p.read_bytes()).hexdigest()==row['master_before_sha256'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v98.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v99.json');check('progressnotwholeclosure',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rel=read(O/'candidate-relation-coverage-v99.json');check('128/25/467',(rel['scoped_conditional_pairs'],rel['higher_order_groups'],rel['remaining_pair_relations'])==(128,25,467))
        clauses=read(O/'candidate-relation-clauses-v99.json');g=next(x for x in clauses['higher_order_groups']if x['id']=='HG75-Q-KV-JOIN');check('existinggrouprefined',{'F05','F06','F07','F08','F13','F18','F33'}==set(g['members'])and not g['legal_joint_fusion_proved'])
        fp=read(O/'candidate-operator-footprints-v99.json');fpi={r['id']:r for r in fp['candidates']}
        check('genericnormatomremoved',all('compress.latent.norm512'not in r['operator_atom_union']for r in fp['candidates']))
        check('F18F33normintersectionempty',not(set(fpi['F18']['operator_atom_union'])&set(fpi['F33']['operator_atom_union'])))
        check('F19bothbranchintersections',set(fpi['F19']['operator_atom_union'])&set(fpi['F18']['operator_atom_union'])=={'compress.C2.norm512'}and'compress.C1.norm512'in set(fpi['F19']['operator_atom_union'])&set(fpi['F33']['operator_atom_union']))
        for row in read(O/'candidate-pair-navigation-v99.json')['pairs']:
            assert row['shared_operator_atom_union']==sorted(set(fpi[row['a']]['operator_atom_union'])&set(fpi[row['b']]['operator_atom_union']))
        check('595navigationintersectionsrecomputed',True)
        bm=read(O/'module-phase-branch-matrix-v99.json');check('359/24unchanged',sum(len(m['branches'])for m in bm['modules'])==359 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v99.json');old=read(O/'operator-analysis-coverage-ledger-v98.json')
        for key in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativecoverage:'+key,op[key]==old[key])
        audit=read(O/'whole-scope-requirement-audit-v99.json');check('9requirementsopen',len(audit['requirements'])==9 and not audit['overall_complete'])
        check('lowestmoduleMFUstillunknown',next(r for r in audit['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for x in read(O/'candidate-record-index-v99.json')['model_candidates']:check('master:'+x['id'],hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v99.json')
    for x in manifest['artifacts']:
        p=Path(x['path']);check('manifest:'+p.name,p.stat().st_size==x['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'])
    result=dict(status='PASS',preflight=a.preflight,checks=len(checks),saved_frontiers=3200,reused_TASKs=19200,ragged_integer_models=cases,DAG_models=DAGcases,new_device_work=False,overall_complete=False)
    if a.preflight:write(O/'projection-panel-analysis-preflight-v99.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
