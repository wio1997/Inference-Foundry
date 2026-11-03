"""Independent MC2guard,subcore/peer partition,maskcounterexample andcoverage checks."""
from pathlib import Path
from math import ceil
from fractions import Fraction
import argparse,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'combine-inputs-v116.json')['inputs']:ck('input:'+r['path'],pinned(r))
    src=read(O/'combine-source-contract-v116.json');texts={}
    ck('frozen_parent_matches_selectedsource',src['source_tree_parent_hashes_match_v56']and all(pinned(p)for p in src['source_tree_parent_controls']))
    for p in src['parts']:
        ck('source:'+p['path'],pinned(p));text=Path(p['path']).read_text();texts[p['path']]=text
        ck('section:'+str(p['first_line'])+p['path'],text.splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    kernel=next(t for p,t in texts.items()if p.endswith('dispatch_ffn_combine_kernel.hpp'))
    ep=next(t for p,t in texts.items()if p.endswith('block_epilogue_pertoken_v2.hpp'))
    cast=next(t for p,t in texts.items()if p.endswith('dav_c220/kernel_operator_vec_vconv_impl.h'))
    forward=next(t for p,t in texts.items()if p.endswith('basic_api/kernel_operator_vec_vconv_intf_impl.h'))
    ck('sourceguard','params.problemShape.m() * params.topK <= 4096'in kernel)
    ck('two_distinct_return_calls','CombineV1(params, blockEpilogue2)'in kernel and 'CombineV2(params, blockEpilogue3)'in kernel)
    ck('source16split','int32_t m0 = 16'in kernel and 'aiv_m_rows = m_rows / 2'in kernel)
    ck('falseCast_and_repeat128',ep.count('Cast<')==2 and 'float, ElementC, false'in ep and 'ElementD, float, false'in ep and 'int32_t repeat = 128'in ep)
    ck('SDK2201mapping','__NPU_ARCH__ == 2201'in forward and 'dav_c220/kernel_operator_vec_vconv_impl.h'in forward)
    ck('boolforwarded','CastImpl<DstPrimType, SrcPrimType, isSetMask>'in forward)
    block=cast[cast.index('// Cast::Level 0 - mask count mode'):cast.index('template <typename T, bool halfBlock>')]
    ck('falsebypassesSetMask','if constexpr (isSetMask)'in block and 'CastIntrinsicsImpl(dst, src, roundMode, repeatTime, repeatParams)'in block)
    ck('types_and_rounds','vconv_f162f32('in cast and 'vconv_f322bf16r('in cast)
    branch=read(O/'combine-profile-branch-map-v116.json');ck('3512_960',branch['source_predicted_counts']=={'V2':3512,'V1':960}and sum(r['executions']for r in branch['cohorts'])==4472)
    for r in branch['cohorts']:ck('guard:'+str(r['M'])+':'+str(r['topK']),r['source_predicted_branch']==('V2'if r['M']*r['topK']<=4096 else'V1'))
    ck('not_actualreceiverrows_or_internalbranch',not branch['internal_branch_observed_directly']and branch['actual_received_expert_rows_or_masks_unrecorded'])
    refs=read(O/'combine-row-and-peer-interval-proof-v116.json');ck('514_7565_refs',refs['row_partition_cases']==514 and refs['peer_segment_reference_cases']==7565)
    # Independent chunk-index formula assigns the finalpartial tosubcore1.
    for m in range(1,129):
        q=(m+15)//16;split=q//2
        first=[(16*j,min(16,m-16*j))for j in range(split)]
        second=[(16*j,min(16,m-16*j))for j in range(split,q)]
        ck('rowcover:'+str(m),sum(n for _,n in first+second)==m and all(n==16 for _,n in first)and all(1<=n<=16 for _,n in second))
        if m<=16:ck('one_subcore_only:'+str(m),not first and len(second)==1)
    # Dynamicprogramming enumerates every positiveexpertload partition in a finite domain.
    models=read(O/'combine-mask-gated-work-models-v116.json');bounds={(r['R'],r['positive_experts']):(r['S_min'],r['S_max'])for r in models['fragmentation_bounds']}
    lo={(0,0):0};hi={(0,0):0}
    for z in range(1,7):
        for total in range(z,65):
            choices=[(lo[(total-x,z-1)]+ceil(x/16),hi[(total-x,z-1)]+ceil(x/16))for x in range(1,total+1)if(total-x,z-1)in lo]
            lo[(total,z)]=min(x for x,_ in choices);hi[(total,z)]=max(y for _,y in choices)
            ck('fragmentation_bound:'+str((total,z)),bounds[(total,z)]==(lo[(total,z)],hi[(total,z)]))
    for i,e in enumerate(models['examples']):
        R=sum(e['receiver_expert_rows']);S=sum((x+15)//16 for x in e['receiver_expert_rows'])
        ck('work:'+str(i),e['R']==R and e['S']==S and e['conditional_full_Normal_cast_elements_per_direction']==20*8192*S and e['active_materialized_value_conversions_per_direction']==5120*R and Fraction(e['conditional_padding_work_ratio'])==Fraction(32*S,R))
    ck('Normalonly_notphysical','OnlyifexternalhardwaremodeisNormal'in models['FULL_NORMAL_case']and 'Actualsavedmaskstateunknown'in models['FULL_NORMAL_case'])
    ck('no_nomaskchange_candidate','no maskmutation'in models['candidate'])
    # A counter mask incorrectlyleftbyCast changes each subsequent per-rowfalseMuls extent.
    Ncol=256;normal=[1.0]*(3*Ncol);bad=[1.0]*(5*Ncol)
    for row,scale in enumerate([2,3,4]):
        for j in range(Ncol):normal[row*Ncol+j]*=scale
        for j in range(3*Ncol):bad[row*Ncol+j]*=scale
    ck('Countermode_leak_counterexample',[normal[i*Ncol]for i in range(3)]==[2,3,4]and [bad[i*Ncol]for i in range(3)]==[2,6,24])
    # PointwiseCast prefixdependencies,not floatingpoint/nativeprecision validation.
    for rows in range(1,17):
        active=rows*256;full=set(range(128*64));trim=set(range((4*rows)*64))
        ck('activeprefix:'+str(rows),trim==set(range(active))and trim<=full and max(trim)<4096)
    ub=models['UB_compaction'];ck('privateUBonly',ub['current_two_bank_layout_bytes']==131328 and ub['conditional_active_only_two_bank_layout_bytes']==65664 and 'notglobalGMworkspace'in ub['benefit_scope'])
    ck('connection_closed',not(N/'ssh166-combine-v116.sock').exists())
    if a.preflight:manifest=read(O/'checkpoint-manifest-v115.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v116.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        bm=read(O/'module-phase-branch-matrix-v116.json');ck('355_26',sum(len(m['branches'])for m in bm['modules'])==355 and len(bm['modules'])==26)
        new=next(m for m in bm['modules']if m['id']=='M12');ck('M12_two_explicit_return_branches',len(new['branches'])==21 and {r['id']for r in new['branches']}>= {'M12-OP-20','M12-OP-21'})
        cov=read(O/'candidate-relation-coverage-v116.json');ck('pairs_unchanged',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(134,26,461))
        obligations=read(O/'coverage-current-obligations-v116.json');G={g['id']:g for g in obligations['obligations']}
        ck('current_text355_26_461','355 semantic'in G['G00']['remaining']and '/26 groups leave461'in G['G10']['remaining'])
        old=read(O/'operator-analysis-coverage-ledger-v115.json');new=read(O/'operator-analysis-coverage-ledger-v116.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v116.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_active_loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v116.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,row_partition_cases=514,peer_segment_cases=7565,
        independent_fragmentation_cases=len(bounds),new_device_compute=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'combine-preflight-v116.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
