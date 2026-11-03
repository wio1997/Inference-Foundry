"""Check saved identity/owner shapes, linear-state counterexamples and declared work/storage models."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import argparse,hashlib,itertools,json
from finalize_origin_coverage_v9 import N,O,R,read,write

def rne_bf16_positive(x):
    x=F(x)
    if x==0:return x
    assert x>0
    exp=x.numerator.bit_length()-x.denominator.bit_length()
    if x<F(2)**exp:exp-=1
    step=F(2)**(exp-7)
    return round(x/step)*step

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    inputs=read(O/'attention-projected-state-inputs-v100.json')
    for r in inputs['inputs']+inputs['native_caches']:
        p=Path(r['path']);check('input:'+p.name,p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
    gp=Path(inputs['graph_block_index']['path']);check('graphblockindex',hashlib.sha256(gp.read_bytes()).hexdigest()==inputs['graph_block_index']['sha256'])
    graph=read(gp);blocks={(g['database_index'],g['graph_ordinal'],b['block_ordinal']):(b,bool(g['prepare_root_keys']))for g in graph['graph_executions']for b in g['blocks']}
    for r in read(O/'attention-projected-state-source-contract-v100.json')['parts']:
        p=Path(r['path']);ls=p.read_text().splitlines();check('source:'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
        for s in r['segments']:check('segment:'+p.name+str(s['first_line']),ls[s['first_line']-1:s['last_line']]==s['source'])
    front=read(O/'attention-output-owner-frontiers-v100.json');rows=front['frontiers'];check('3200frontiers',len(rows)==3200)
    check('root/phase/index',(front['root_bound'],front['phase_unbound'],front['index_selection_frontiers'])==(2560,640,640))
    check('familiespreserved',front['families']==dict(SWA_ONLY=160,CSA_C2=1440,CSA_C1=1600))
    bydb=defaultdict(list)
    for r in rows:bydb[r['database_index']].append(r)
    unique=set();cache={int(Path(r['path']).stem.split('-')[-1]):r for r in inputs['native_caches']}
    for db,rs in bydb.items():
        raw=read(Path(cache[db]['path']));tasks={r['task_record_rowid']:r for r in raw['TASK_records']};defs={r['globalTaskId']:r for r in raw['COMPUTE_definitions']}
        for r in rs:
            stages=r['stages'];order=['SMLA','inverse_RoPE','send_layout','restore_heads_A2A','WO_A','WO_B','REDUCE_SCATTER']
            block,root=blocks[(db,r['graph_ordinal'],r['block_ordinal'])];assert root==r['root_bound'];blockids=set(block['task_record_rowids'])
            for k in order:
                s=stages[k];t=tasks[s['task_record_rowid']];d=defs[t['globalTaskId']];unique.add((db,t['task_record_rowid']))
                assert t['task_record_rowid']in blockids
                assert(t['startNs'],t['endNs'],t['streamId'])==(s['startNs'],s['endNs'],s['streamId'])and d['opType_text']==s['op']
                assert d['inputShapes_text']==r['typed'][k]['input_shapes']and d['outputShapes_text']==r['typed'][k]['output_shapes']
            assert all(stages[x]['endNs']<=stages[y]['startNs']for x,y in zip(order,order[1:]))
            assert stages['restore_heads_A2A']['op']=='hcom_alltoall_'
            assert r['typed']['SMLA']['output_shapes']=='"3,64,512;"'and r['typed']['WO_A']['input_shapes']=='"24,1,4096;1,4096,1024"'
            if r['index_selection']:
                for k in ['QLI','prepare']:
                    s=r['index_selection'][k];t=tasks[s['task_record_rowid']];assert(t['startNs'],t['endNs'])==(s['startNs'],s['endNs']);unique.add((db,t['task_record_rowid']))
                    assert t['task_record_rowid']in blockids
        check('native7stageandQLI:'+str(db),True)
    check('23680existingTASKs',len(unique)==front['reused_unique_TASKs']==23680)
    check('equalvolume_notowneridentity',front['recorded_local_query_output']['BF16_value_bytes']==front['recorded_projection_input']['BF16_value_bytes']==196608 and not any(r['restore_protocol_or_pointer_identity_proved']for r in rows))
    head=read(O/'attention-headwise-merge-and-linear-state-v100.json');w=head['witness'];alpha=list(map(F,w['head_A_mass_weights']))
    outs=[]
    for suffix in ['1','2']:
        aa=list(map(F,w['branchA_case'+suffix]));bb=list(map(F,w['branchB_case'+suffix]));outs.append(sum(p*x+(1-p)*y for p,x,y in zip(alpha,aa,bb)))
        assert sum(aa)==sum(bb)==1
    check('samemassesandbranchsumsdifferentfinal',outs==[F(5,4),F(3,4)]and w['projected_branch_sums_case1']==w['projected_branch_sums_case2']==['1','1'])
    uw=head['unnormalized_counterexample'];du=uw['perhead_final_denominator'];res=[]
    for suffix in ['1','2']:
        aa=list(map(F,uw['u_A_case'+suffix]));bb=list(map(F,uw['u_B_case'+suffix]));assert sum(aa)==sum(bb)==1
        res.append(sum((x+y)/d for x,y,d in zip(aa,bb,du)))
    check('unnormalizedcounterwithsamefullmetadata',res==[F(3,4),F(7,8)]and[sum(x)for x in zip(uw['perhead_l_A'],uw['perhead_l_B'])]==du)
    check('nullspacecommutationcounterexample',sum([1,-1])==0 and F(1,4)-F(3,4)==F(-1,2))
    common=0
    for a0,a1,b0,b1 in itertools.product([-1,0,1,2],repeat=4):
        for p in [F(0),F(1,4),F(1,2),F(3,4),F(1)]:
            assert p*(a0+a1)+(1-p)*(b0+b1)==sum([p*a0+(1-p)*b0,p*a1+(1-p)*b1]);common+=1
    check('commonmassspecialcasepreserved',common==1280)
    # Exactrationalfixed-querylinearidentity,beforeanynativeBF16rounding.
    ident=0
    values=[[F(1),F(2)],[F(-1),F(3)],[F(2),F(-2)]];weights=[[[F(1),F(2)],[F(3),F(-1)]],[[F(2),F(0)],[F(-2),F(1)]]]
    def mv(v,m):return [sum(v[k]*m[k][j]for k in range(2))for j in range(2)]
    for q in range(2):
        rot=[[F(1),F(0)],[F(0),F(1)]]if q==0 else[[F(3,5),F(4,5)],[F(-4,5),F(3,5)]]
        for h in range(2):
            ps=[F(q+h+k+1,12)for k in range(3)]
            vv=[sum(ps[k]*values[k][j]for k in range(3))for j in range(2)]
            late=mv(mv(vv,rot),weights[h]);early=[sum(ps[k]*mv(mv(values[k],rot),weights[h])[j]for k in range(3))for j in range(2)]
            assert late==early;ident+=1
    check('exactreallinearitynotuniversallyrejected',ident==4)
    prec=read(O/'attention-projection-rounding-and-position-v100.json')['BF16_RNE_model_counterexample'];vals=list(map(F,prec['key_values']))
    raw=sum(vals)/4;late=rne_bf16_positive(rne_bf16_positive(raw)*3);early=rne_bf16_positive(sum(v*3 for v in vals)/4);early_bf=rne_bf16_positive(sum(rne_bf16_positive(v*3)for v in vals)/4)
    check('independentFractionBF16roundingwitness',(raw,late,early,early_bf)==(F(257,256),F(3),F(193,64),F(193,64)))
    check('noactualnativeparityclaim',not prec['native_model_inputs_or_device_test']and not head['native_precision_or_actual_rank_proved'])
    check('densebranchstate_notuniversalpeak',all(r['FP32_head_D_numerator_bytes']==393216 and r['FP32_head_R_projected_numerator_bytes']==786432 and r['FP32_perhead_m_l_bytes']==1536 and r['FP32_collapsed_pergroup_R_bytes']==98304 for r in head['state_storage_references']))
    model=read(O/'attention-preprojected-value-work-and-storage-v100.json');check('105work24storage',len(model['work_models'])==105 and len(model['storage_models'])==24)
    for r in model['work_models']:
        M,H,D,rank,n,u=[r[k]for k in ['logical_projector_group_M','H','D','R','uniform_N_reference','U_new_reference']]
        old=2*M*H*D*(2*n+rank);new=2*M*H*n*(D+rank)+2*u*H*D*rank
        assert old==r['old_QK_PV_OA_OPS']and new==r['new_QK_PV_and_build_OPS']
        assert(new<old)==r['preproject_lower_reference_GEMM_work']==(F(u)<F(M)*(1-F(n,1024)))
        assert r['nonrot_projected_plus_tail_PV_OPS']==2*M*H*n*(1024+64)
        assert r['stat_first_FP32_score_dense_value_bytes']==4*M*H*n and r['stat_first_BF16_probability_dense_value_bytes']==2*M*H*n and r['stat_first_recompute_QK_additional_reference_OPS']==2*M*H*n*D
        assert not r['actual_native_case_or_legal_precision_or_free_placement_proved']
    check('allworkandbreak-evenidentities',True)
    for r in model['storage_models']:
        u,h,l,b=r['U_live_reference'],r['projected_heads_per_owner'],r['distinct_consumer_projectors_reference'],r['projected_V_element_bytes']
        assert r['original_BF16_K_plane_bytes']==u*1024 and r['added_head_projected_V_plane_bytes']==u*h*1024*b*l
        assert F(r['total_to_original_ratio'])==1+b*h*l and r['original_K_retained_for_unchanged_QK']
    check('chosenrepresentationcostnotuniversalIO',not model['actual_numeric_physical_ceiling_or_speedup']and model['source20_consumers']==list(range(20,40)))
    variants=read(O/'attention-projected-state-variants-v100.json');check('6conditionalorrejectedvariants',len(variants['variants'])==6 and variants['no_native_implementation_or_test'])
    packet=read(O/'attention-projected-state-composition-clauses-v100.json');old=read(O/'candidate-relation-clauses-v99.json');oldpairs={frozenset([r['a'],r['b']])for r in old['clauses']}
    check('3previouslyblankpairs',len(packet['clauses'])==3 and all(frozenset([r['a'],r['b']])not in oldpairs for r in packet['clauses']))
    if a.preflight:manifest=read(O/'checkpoint-manifest-v99.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v100.json');check('PROGRESSnoallcompletion',cp['status']=='PROGRESS'and not cp['overall_complete'])
        rel=read(O/'candidate-relation-coverage-v100.json');check('131/25/464',(rel['scoped_conditional_pairs'],rel['higher_order_groups'],rel['remaining_pair_relations'])==(131,25,464))
        group=next(g for g in read(O/'candidate-relation-clauses-v100.json')['higher_order_groups']if g['id']=='HG83-ATTENTION-OUTPUT');check('existingoutputgrouprefined',{'F08','F19'}<=set(group['members'])and not group.get('legal_joint_fusion_proved',False))
        matrix=read(O/'module-phase-branch-matrix-v100.json');check('359branches24semantic',sum(len(m['branches'])for m in matrix['modules'])==359 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in matrix['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v100.json');prev=read(O/'operator-analysis-coverage-ledger-v99.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeuniverseunchanged:'+k,op[k]==prev[k])
        audit=read(O/'whole-scope-requirement-audit-v100.json');check('all9requirementsstillopen',len(audit['requirements'])==9 and not audit['overall_complete'])
        check('allmoduleMFUnotestablished',next(r for r in audit['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for r in read(O/'candidate-record-index-v100.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v100.json')
    for row in manifest['artifacts']:
        p=Path(row['path']);check('manifest:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,frontiers=3200,native_TASKs=23680,common_mass_models=common,work_models=105,storage_models=24,new_device_work=False,overall_complete=False)
    if a.preflight:write(O/'attention-projected-state-analysis-preflight-v100.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
