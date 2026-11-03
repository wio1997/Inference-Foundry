"""Independent V1 count orientation, margin bounds and kept-row partition checks."""
from pathlib import Path
from fractions import Fraction
import argparse,gzip,hashlib,json,random
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'combine-v1-inputs-v119.json')['inputs']:ck('input:'+r['path'],pinned(r))
    source=read(O/'combine-v1-source-contract-v119.json');texts={}
    for r in source['parts']:
        ck('source:'+r['path'],pinned(r));s=Path(r['path']).read_text();texts[r['path']]=s
        ck('clause:'+r['path']+str(r['first_line']),s.splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    row=next(s for p,s in texts.items()if p.endswith('block_epilogue_pertoken_row.hpp'))
    kernel=next(s for p,s in texts.items()if p.endswith('dispatch_ffn_combine_kernel.hpp'))
    ck('peer_owned_return','dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum'in kernel)
    ck('AIV_index','get_block_idx() + get_subblockid() * get_block_num()'in kernel)
    ck('one_row_epilogue','uint32_t tileLoops = blockM;'in row and 'loopIdx * blockN'in row)
    ck('count_Cast_RINT_scale','RoundMode::CAST_NONE, blockN'in row and 'RoundMode::CAST_RINT, blockN'in row and 'gmPerTokenScale(loopIdx)'in row)
    ck('not_V2repeat128','repeat = 128'not in row)
    ck('private_layout81920',source['private_UB_layout_bytes']==2*5120*(2+2+4)==81920)
    later=read(O/'combine-v1-later-profile-scope-v119.json')
    ck('later960_sourcepred_not_internal',sum(c['executions']for c in later['cohorts'])==960 and later['recorded_launch_shape']==dict(blockNum=24,mixBlockNum=48)and later['no_V1_internal_phase_time_or_speedup'])
    observed=read(O/'combine-v1-observer-peer-loads-v119.json');ck('640_separate_observer',len(observed['rows'])==640 and observed['summary']['formal_cohorts_reused']==87 and observed['no_later_profile_time_or_phase_join'])
    ep=N/'MC2-operator-source-v56/saved-observer'
    ds=read(ep/'EP-count-distribution.json');routes=read(ep/'EP-routes.json')
    rawbytes=(N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz').read_bytes()
    ck('raw_original_SHA',hashlib.sha256(rawbytes).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029')
    raw=[json.loads(line)for line in gzip.decompress(rawbytes).splitlines()];rawgroups={}
    for item in raw:rawgroups.setdefault((item['role'],item['layer'],item['ordinal']),{})[item['ep_rank']]=item
    ck('4736_records_296_complete_groups',len(raw)==4736 and len(rawgroups)==296 and all(len(g)==16 for g in rawgroups.values()))
    ck('raw_saved_C65536_no_overflow',all(r['capacity_max_output_size']==65536 and not r['receiver_count_sum_above_capacity_argument']for r in raw))
    key=lambda r:(r['role'],r['layer'],r['ordinal'],r['phase'])
    d={key(r):r for r in ds['cohorts']};q={key(r):r for r in routes['cohorts']}
    for i,r in enumerate(observed['rows']):
        k=(r['role'],r['layer'],r['observer_ordinal'],'formal_C8_measure');receiver=r['receiver_EP_slot'];c=d[k]
        peers=[x[receiver]for x in q[k]['sender_receiver_assignment_matrix']]
        E=c['E_local'];experts=c['receiver_expert_rows_by_logical_id'][receiver*E:receiver*E+E]
        rg=rawgroups[(r['role'],r['layer'],r['observer_ordinal'])]
        joint=[[int(rg[s]['sender_expert_histogram'].get(str(receiver*E+e),0))for s in range(16)]for e in range(E)]
        ck('raw_joint_recovered:'+str(i),joint==r['expert_peer_return_rows']and list(map(sum,zip(*joint)))==peers and list(map(sum,joint))==experts)
        ck('matrix_orientation:'+str(i),peers==r['peer_owned_return_rows']and experts==r['expert_return_rows']and sum(peers)==sum(experts)==r['R'])
        ck('sourceguard:'+str(i),c['M_by_rank'][receiver]*c['K']>4096 and r['R']<65536)
        new=sum(-(-x//48)for x in experts)
        ck('quanta:'+str(i),r['unit_row_balanced48_perexpert_feasible_quanta']==new and Fraction(r['abstract_lower_ratio'])==Fraction(r['unit_row_expert_barrier_old_quanta_lower'],new))
        exact=sum(map(max,joint));ck('exact_old_joint_quanta:'+str(i),r['unit_row_expert_barrier_old_exact_quanta']==exact and Fraction(r['abstract_exact_row_quanta_ratio'])==Fraction(exact,new))
        ck('not_bytesaving:'+str(i),r['per_peer_BF16_return_payload_bytes']==[x*10240 for x in peers]and r['current_source_C2_read_and_peerwrite_logical_bytes']==sum(peers)*20480)
    ck('owners6_to10',observed['summary']['positive_peer_owners']['minimum']==6 and observed['summary']['positive_peer_owners']['maximum']==10)
    proof=read(O/'combine-v1-row-redistribution-proof-v119.json');ck('840rowrefs',proof['cases']==840 and proof['logical_not_compiled_extent'])
    ck('15360_actual_expert_partitions',proof['actual_observer_expert_partition_cases']==640*24 and proof['actual_logical_segments']==347667)
    ck('raw_joint_not_claimed_for_later_profile',observed['perexpert_peer_joint_counts_recovered_in_observer']and observed['later_hostprofile_joint_counts_still_unbound'])
    ck('read_connection_closed',not(N/'ssh166-v1-v119.sock').exists())
    witness=proof['missing_joint_count_witness'];A=witness['expert_by_peer_A'];B=witness['expert_by_peer_B']
    ck('same_margins_different_L',[sum(x)for x in A]==[sum(x)for x in B]==[4,4]and list(map(sum,zip(*A)))==list(map(sum,zip(*B)))==[4,4]and sum(map(max,A))==8 and sum(map(max,B))==4)
    # Independent row-to-worker inverse, using disjoint worker boundaries and explicit kept row identities.
    rng=random.Random(9119);partition_cases=0;joint_cases=0
    for pp in [1,2,3,16]:
        for ee in [1,2,24]:
            for trial in range(25):
                joint=[[rng.randrange(8)for _ in range(pp)]for _ in range(ee)]
                n=[sum(x)for x in joint];s=list(map(sum,zip(*joint)));R=sum(n)
                old=sum(map(max,joint));lb=max(max(s),sum(-(-x//pp)for x in n));ub=min(R,sum(min(x,max(s))for x in n))
                ck('margins:'+str(joint_cases),lb<=old<=ub);joint_cases+=1
    for pp in [1,2,3,16]:
        for ww in [1,3,16,48]:
            for trial in range(30):
                counts=[rng.randrange(15)for _ in range(pp)];C=rng.randrange(sum(counts)+1)
                # Old raw-prefix clipping, represented by individual valid source rows.
                raw=[]
                for p,n in enumerate(counts):raw.extend((p,j)for j in range(n))
                kept=raw[:C];R=len(kept);assigned=[]
                for w in range(ww):assigned.append(kept[w*R//ww:(w+1)*R//ww])
                ck('partition:'+str(partition_cases),[x for part in assigned for x in part]==kept and max(map(len,assigned))-min(map(len,assigned))<=1)
                partition_cases+=1
    ck('independent_case_counts',joint_cases==300 and partition_cases==480)
    gates=read(O/'combine-v1-composition-and-theory-gates-v119.json')
    ck('no_promotions_device',gates['new_native_promotions']==0 and not gates['new_device_work']and gates['no_new_pair_group_or_candidate_root'])
    ck('row_copy_count_not_invented','originalRrowDMAs/per-roweventcountstayunchanged'in gates['variants'][0]['extra_cost'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v118.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v119.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        mat=read(O/'module-phase-branch-matrix-v119.json');ck('356_26_unchanged',len(mat['modules'])==26 and sum(len(m['branches'])for m in mat['modules'])==356)
        m12=next(m for m in mat['modules']if m['id']=='M12');br=next(r for r in m12['branches']if r['id']=='M12-OP-21')
        ck('existing_V1row_refined',len(m12['branches'])==21 and 'V1_peer_row_review_v119'in br)
        cov=read(O/'candidate-relation-coverage-v119.json');ck('134_26_461',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(134,26,461))
        old=read(O/'operator-analysis-coverage-ledger-v118.json');new=read(O/'operator-analysis-coverage-ledger-v119.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        gs={g['id']:g for g in read(O/'coverage-current-obligations-v119.json')['obligations']}
        ck('coverage_text356_26_461','356 semantic'in gs['G00']['remaining']and '/26 groups leave461'in gs['G10']['remaining'])
        for r in read(O/'candidate-record-index-v119.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_active_loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v119.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,observer_V1_source_predicted_rank_records=640,
        later_profile_V1_source_predicted_executions=960,independent_margin_fixtures=joint_cases,
        independent_partition_fixtures=partition_cases,logical_row_reference_cases=840,
        actual_native_branch_or_phase_speedup_proved=False,new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'combine-v1-preflight-v119.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
