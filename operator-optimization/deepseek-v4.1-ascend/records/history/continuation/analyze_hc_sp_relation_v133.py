"""HC fork/SP projection composition across ranks, guarded by actual source roles."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write
from hc_sp_models_v133 import schedules,values


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v133.json').exists()
    old=read(O/'HC-operator-numerical-and-dependency-contract-v59.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';p=root/'models/deepseek_v41/model.py'
    parts=[dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])for a,b in [(887,920),(1551,1597)]]
    parts += [dict(**pin(x['path']),first_line=x['first_line'],last_line=x['last_line'],source=x['source'])for x in old['parts']if (x['path'].endswith('hc_pre_m_k_split_core.h')and x['first_line']==396)or(x['path'].endswith('hc_pre_base.h')and x['first_line']==504)or(x['path'].endswith('torch_binding.cpp')and x['first_line']==1541)]
    write(O/'hc-sp-source-and-branch-contract-v133.json',dict(parts=parts,prior_answers=['v58/v59externalpremixY/coefffork','v93coeff/Postdeadlineandsharedresources','v132firstinit/SP-CPqueryownership'],
        facts=[
            'All8capturedPre regimeshaveFLOATexternalpremixinputs. CurrentYusesoldpremixandcurrentroundedHCX;currentpre/post/combusesfull4HoriginalXwithHF32/FLOAT/RMS/Sinkhorn,notnormalizedYorKV.',
            'SourcehasPreMixconditionalProcessY(preMixLocal:mixes01ReduceLocal). Withoutoldpremix,currentpre4/rstd/projectiongateY;post/combcanstillbeaseparatefuturedeadline. CurrentNonebranchreachabilitynotattestedby8positiveexternalregimes.',
            'Currentpost/combneededbyactualattentionPostrowreads;currentpre4neededbyfollowingFFNPreusingupdatedPostresidual. PreviousFFNPrepre4feedsnextattentionY;do notaliasoldpremix/currentprebeforeYretirement.',
            'Graph3120crosslayerboundariescontain160Engramwrites(block1/14),240auxreadercheckpoints(block37/38/39),2720withoutinterveningcomputeinselectedschema. Emptyrecordsdonotcloseallobservers.',
            'EngramchangesHCXbeforeY/currentcoeff/currentPostresidual. Waitonlyrequiredchangedrows/ranges,nouniversalwholeEngrambarrier. AuxdoesnotchangeXbutretainsitsreaderandbanklifetime.',
            'HcPreY_BF16→full5120RMS_Z_BF16→ownerKV/QAprojectionkeepsbothrounds. Fullrowreduction/readinessandscratch/recompute/queuecostsarepriced. AllY/Z/rstd/rawQA/C1/C2/Indexerreadersmustbeclosedperrole.',
            'SplitHCY+ownerprojectedcarrierforkswithcoeff,thenjoinsatPost. Newcarrierreadinessdependsonallactualrowowners/CPquery/rawreaders/communicatormembers;sourcegrouporderandcountsarenotactualcrossrankepisodealignment.',
            'EarlyY/carrriercompletiondoesnotretireX:currentcoeff,Postresidual,aux/Engram/queuedobserversandbankepochsremain. NewNativepartialpublication/handoffrequired;oldkernelcompletion/launchorderalonecannotexposeintermediateY.',
            '312draftfrontiersarepositive264KV/48Nonezeros. KeepnoQ-globalKVanddummyseparate;onlysource-proveduniformdeadinputdomaincanomitcarrier. Target80SWA/C1roleambiguitiesand640phase-unboundlayersretainpriorgrade.'],
        all_native_parameters_publication_or_observer_contracts_bound=False,new_device_work=False,overall_complete=False))
    cases=schedules();counts=Counter();witnesses={}
    for r in cases:
        B,S,P,C=r['baseline'],r['split_only'],r['projection_only'],r['combined'];assert S<=B and C<=P
        delta=(B-C)-((B-S)+(B-P));tag='ADDITIVE'if delta==0 else'APPARENT_SUPERADDITIVE'if delta>0 else'OVERLAPPING_CREDITS';counts[tag]+=1;witnesses.setdefault(tag,r)
    write(O/'hc-sp-rank-fork-schedule-models-v133.json',dict(cases=cases,case_count=len(cases),credit_classes=dict(counts),witnesses=witnesses,
        symbols='C_p=localcoeffready;Y_p=localexternalY+normready;J_p=localownerprojectionservice;Gh/Gk=chosenfullcarriercompletionservice;Ar/Ak=allremainingneededquery/attention/outputpathservice. Zeroorigin,atomicstages,unlimitednoncontendingresources/equalfixedservice. No actual service isidentified.',
        formulas=dict(baseline='max_p(max(Cp,Yp))+Gh+Ar',split_only='max(max_p Cp,max_p Yp+Gh+Ar)',projection_only='max_p(max(Cp,Yp)+Jp)+Gk+Ak',combined='max(max_p Cp,max_p(Yp+Jp)+Gk+Ak)'),
        scope='Selectedfull-carrier/commonoutputjoinmodelonly;row/tile-demandalternativecanstartbeforefullcarrierandmustuseitsowngraph. TheseareconditionalDAGmakespans,notphysicalminimum/throughputbounds. AddPost/launch/packing/servicechangesinonegraph.',
        contention='ActualC/Y/JshareVector/Cube/memory/queues;newCprime/Yprime/Jprimecanchange. Neitherunlimitedoverlapnorserialsumisassumedphysical. Newcollectivemaybe slowerdespite512width.',
        hardware_optimum_or_measured_gain_proved=False,new_device_work=False,overall_complete=False))
    Ms=sorted({r['M']for r in read(O/'HC-operator-profile-and-edges-v58.json')['regimes']if r['operator']=='HcPre'})
    write(O/'hc-sp-interface-work-and-resource-cut-v133.json',dict(cases=[values(m)for m in Ms],
        scope='Current/statednewrepresentationvaluecontents. IfYandZarebothprivateandallreadersclosed,fusingY/RMS/ownerprojectioncanavoid4MH+4MHinterfaces;notuniversalHBMtrafficorallocatedVRAM.',
        gates='PreserveHCYBF16round,wholeH5120RMSandZBF16round;QAlocal/raworC1/C2/Indexerotherplanesremainunlessclosed. Onchipcapacity/layout/recompute/visibilityandextraepochsareunbound. KVGatheroutputplaneisstillrequiredbynewcollectiveABI.',
        retained='Currentfull4Hcoefficientprojection/HF32/FLOAT/cast/Gram/RMS/sigmoid/SinkhornworkandXPostresidualstay;Ynormfusiondoesnotdeletecurrentcoefficientworkorcastworkspace.',
        symbolic_capacity='ForonephysicalcutanddeclaredalgorithmD,ifjointbeta(lambda)isavalidrateUPPERanddeduplicatedWjarenecessaryLOWERSinD,T>=sup_nonnegative_lambda(sum(lambda_j*Wj)/beta(lambda)). Maxwithothercuts/realdependencybounds. ActualmixedHF32/BF16/VectorbetaandnecessaryHBMnotestablished.',
        theory_status='SYMBOLIC_CEILING',actual_MFU_lowest_or_numeric_hardware_time_bound=False,new_device_work=False,overall_complete=False))
    write(O/'hc-sp-reader-generation-and-row-witnesses-v133.json',dict(
        input_retirement=dict(X_readers_last_service_units=dict(Y=2,coefficient=8,Post_residual=10,aux=12),carrier_ready=3,earliest_all_named_readers_retired=12,meaning='Carriercompletion3doesnotauthorizeXbankreuse3. Unitsareillustrative,notmeasured;allotherreadersmustalsoclose.'),
        old_vs_current_premix=dict(X=[1,2,3,4],old_premix=[1,0,0,0],current_pre=[0.25]*4,Y_old=1,Y_wrong_current=2.5,actual_trained_values_recovered=False),
        Engram_input_generation=dict(before_X_channel0=1,after_X_channel0=2,old_premix=[1,0,0,0],Y_before=1,Y_after=2,actual_Engram_delta_proved=False),
        row_demand_ready=dict(KV_row_ready=[2,9],only_row0_demand_ready=2,both_rows_demand_ready=9,meaning='Afinerchosenrow-demandgraphmaystartrow0consumerat2;do notforcewholecarrier9withoutthatfull-carrierdomain.'),
        branch_DAGs=dict(external=[['OLD_PREMIX','Y'],['X_UPDATED','Y'],['X_UPDATED','COEFF'],['COEFF','CURRENT_PRE'],['COEFF','POST_COMB'],['Y','NORM'],['NORM','OWNER_KV'],['OWNER_KV','ATTN_OUT'],['ATTN_OUT','POST'],['POST_COMB','POST'],['X_UPDATED','POST'],['POST','NEXT_FFN_PRE'],['CURRENT_PRE','NEXT_FFN_PRE']],
            no_external=[['X_UPDATED','COEFF_SHARED'],['COEFF_SHARED','CURRENT_PRE'],['CURRENT_PRE','Y'],['COEFF_SHARED','POST_COMB'],['Y','NORM'],['NORM','OWNER_KV'],['OWNER_KV','ATTN_OUT'],['ATTN_OUT','POST'],['POST_COMB','POST']]),
        generation='X_UPDATEDcomesaftertheactualEngramrowwriteifpresent. AuxobserverneedsXbutdoesnotmutateit;preserveitsreaderwithoutinventingaY-after-auxcomputeedge. PreviousFFNPrepre4andpreviousPostXarerolesfromsourceflow,nothistoricalpointerbindings.',
        new_device_work=False,overall_complete=False))
    all_sp=sum(r['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER'for r in read(O/'communication-native-operator-population-v66.json')['operations']);assert all_sp==4472
    write(O/'hc-sp-composition-scope-v133.json',dict(new_pairs=[['F01','F32'],['F02','F32'],['F03','F32']],new_groups=0,
        all_saved_attention_SP_calls=all_sp,newly_joined_target_graph_frontiers=3200,newly_joined_draft_frontiers=312,other_eager_target_SP_not_newly_joined=960,
        large_M_content_cases='M800/816/834reuseexistingv58regimes,notnewHC-SPjoins. Targeteager960remainoutsidecurrentfrontierjoin.',
        existing_group_to_extend='HG93-HC-ATTENTION-FORK-DEADLINE',new_member='F32',existing_split_not_new_discovery=True,
        distinctions=['F01-F32multirankcarrierforkandcoefficientPostdeadline.','F02-F32roundedY/fullRMS/ownerprojectioninterfaceandallrawreaders.','F03-F32previousPost/oldpremix/EngramupdatedX/auxreaderrowpublication.'],
        one_HC_baseline_and_interfaces_counted_once=True,all_runtime_all_subsets_or_native_fusion_closed=False,new_native_nodes=0,new_device_work=False,overall_complete=False))
    names=['init-sp-postcheck-result-v132p.json','candidate-relation-clauses-v132.json','candidate-pair-navigation-v132.json','HC-attention-fork-native-frontiers-v93.json','HC-attention-deadline-and-precision-models-v93.json','HC-attention-fork-variants-v93.json','HC-attention-shared-resource-models-v93.json',
        'HC-operator-profile-and-edges-v58.json','HC-operator-numerical-and-dependency-contract-v59.json','hc-sp-native-frontiers-v133.json','context-sp-profile-frontiers-v130.json','init-sp-native-frontiers-v132.json','communication-native-operator-population-v66.json']
    inputs=[O/n for n in names]+[Path(x['path'])for x in parts]
    write(O/'hc-sp-inputs-v133.json',dict(inputs=[pin(p)for p in sorted(set(inputs))],questions=['Whichpremix/currentcoeff/previousPostgenerationsandSPcarrierrolesarepositivelyrecorded?',
        'HowdoesearlyYcombinewithproject-before-SPacrossrankreadinesswithoutaddingindependentsavings?',
        'WhichEngramwrites/auxreaders/fullnorm/rounds/banklifetimeslimitPost→Y→ownerprojection?'],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_SP_FORK_INTERFACE_AND_POST_GENERATIONS_REVIEWED',schedule_cases=len(cases),credit_classes=dict(counts),work_shapes=Ms,new_scoped_pairs=3,new_device_work=False)))


if __name__=='__main__':main()
