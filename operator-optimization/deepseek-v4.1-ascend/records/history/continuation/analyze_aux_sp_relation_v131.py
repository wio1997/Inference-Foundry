"""F27-F32 checkpoint packet causality and main-projection representation cuts."""
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write
from aux_sp_models_v131 import packets,row_plan,row_selection_cases,cycle,bf16


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def part(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])


def main():
    assert not (O/'REVIEW-CHECKPOINT-v131.json').exists()
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';mo=root/'models/deepseek_v41/model.py';ds=root/'models/deepseek_v41/dspark.py'
    wo=root/'worker/model_runner_v1.py';pr=root/'spec_decode/llm_base_proposer.py'
    checkpoint=read(O/'aux-sp-checkpoint-profile-v131.json');selector=read(O/'aux-sp-selector-profile-v131.json')
    assert checkpoint['pair_count']==240 and selector['API_counts_by_source_arm']=={'EARLY_PREFIX_SOURCE_ARM':16,'SAMPLED_INDEX_SOURCE_ARM':72}and selector['native_TASK_types']=={'ConcatD':88}
    topology=read(O/'projection-source-contract.json');roles=[r for r in topology['roles']if r['layer']in [37,38,39]]
    assert len(roles)==3 and all(not r['is_kv_source']and not r['is_index_source']for r in roles)
    write(O/'aux-sp-source-contract-v131.json',dict(parts=[part(mo,1554,1566),part(mo,887,903),part(ds,180,202),part(ds,279,309),
        part(wo,2061,2093),part(pr,1040,1076)],checkpoint_roles=roles,
        facts=[
            'Auxmean isproducedbefore selectedtargetblocks37/38/39 andgatheredseparately. CurrentSPgather input is mHCweightedcollapse plusinputnorm,not thatmean. F32projectedSWA512cannotreplacefullaux5120 byshape.',
            'Thepaired240capturedcheckpoint records use matchingcount/dtype/P8/group-label butdifferentCOMM/TASK keys. Keep192root/48phase-unbound and72eagerauxunpaired;no per-tensor/group-memberidentity isclaimed.',
            '16positiveTorchcat sourcesatworker2069 usenum_scheduled_tokensprefix;72at2089 useh[token_indices]. ExactnestedaclnnCat→CID→TASKbinds88nativeConcatD. Actualindexvalues/constancy areabsent.',
            'For theNoSpec sourcearm theprefixlength comesfromscheduledinput;forindexed arms prepare_inputs/prepare_inputs_padded maydependoncurrentaccepted/rejectedstate. EqualM/N doesnotproveearlyknownoridentityI.',
            'Auxiliary andSPvalueplanes mayco-packonlywithactualrank/row/parameter/capture/readiness contracts. CurrentattentionneedsitsKVpacketbeforetargetsample;thepostsampleexactImustnotgate thatsamepacket.',
            'Ifall3auxrows stayatcompatibleowners andfullmainweightsareavailable,anearlyrowfunction canproducecontextorpre-RoPEKV candidates beforeI;restoringIafterwardpreservesorder/multiplicity onlywithrow-independent/numerical/error/observer proofs.',
            'Currentmain_proj iscolumnparallel15360→640perrankthenfull5120norm. Ownerfullrow requiresfullmainW orpricedinput/weight exchange. GatheringonlydiagonalA_pW_p blocks isinsufficient.',
            'ContinuingoriginalFP32accumulation acrosscheckpoints isnot3separatelyBF16roundedGEMMs. Nativeaccumulatorresume/order/parity isNOT_ESTABLISHED. Full5120norm and3separate512norms remain.'],
        loaded_group_parameters_native_parity_or_actual_I_attested=False,new_device_work=False,overall_complete=False))
    layouts=row_selection_cases()
    write(O/'aux-sp-ordered-selection-models-v131.json',dict(cases=layouts,case_count=len(layouts),
        identity='S_I f(A0,A1,A2)=f(S_I A0,S_I A1,S_I A2) for a fixed deterministic rowfunction;duplicates/order/validnegativecanonicalindices retained.',
        scope='P1..4,n1..3,M0..4,finiteboundary/duplicateindices. Testfunction isintegernonlinear,notnativeGEMM/RMSNorm.',
        numerical_condition='Actualrowfunction mustpreserveoriginal15360contraction/BF16/full5120norm/threeKVrounds/nativeMpolicy;CP/group/position/RoPE followsoriginalselectedI.',new_device_work=False,overall_complete=False))
    balanced=[0,3,6,9,12,15,18,21,1,4,7,10];skew=list(range(12));dup=[0]*12
    plans={k:row_plan(v,24,8)for k,v in [('balanced',balanced),('skew',skew),('duplicate',dup)]}
    assert [plans[k]['u_max']for k in ['balanced','skew','duplicate']]==[2,3,1]
    write(O/'aux-sp-owner-balance-and-loss-witnesses-v131.json',dict(indices=dict(balanced=balanced,skew=skew,duplicate=dup),plans=plans,
        same_M_or_unique_total_not_fixed_count_transport_saving=True,
        aux_mean_not_recoverable_from_projected_SWA=dict(toy_H=2,toy_SWA_width=1,
            both_HC_planes_A=[1,1],both_HC_planes_B=[1,-1],fixed_positive_mix=[0.5,0.5],
            shared_first_feature_after_full_RMS=True,SWA_projection=[1,0],aux_second_feature=[1,-1],
            meaning='Anunconditionalsubstitutionisnotanalgebraicidentity. Notproofthatthesestatesoccurwithcurrenttrainedparameters.'),
        diagonal_projection=dict(row_inputs=[1,2],output_weight_shards=[3,5],required_outputs=[[3,5],[6,10]],diagonal_only=[3,10],missing_offdiagonal=True),
        actual_I_or_trained_state_or_native_flags_recovered=False,new_device_work=False,overall_complete=False))
    geometries=[(24,12),(24,24),(6400,6400),(6528,6528),(6672,6668)]
    costs=[packets(N,M)for N,M in geometries]
    late=[dict(name=k,**packets(24,12,u_max=plans[k]['u_max']))for k in plans]
    write(O/'aux-sp-packet-work-and-capacity-models-v131.json',dict(geometries=costs,late_selector_examples=late,
        physical_pairing='N/Mexamples reuseexistingv34source-compatiblegeometries,notnewinvocation/tensor bindings. N isphysicaluniformgatherrows;actualsemanticextent/CProwmap/negativeindexrange mustbebound.',
        packet_scope='Sumofdeclaredper-sender BF16AllGather inputcontentsacross3checkpoints+mainfeature;notwire,HCCLalgorithm,HBM,hostlaunches orlatency.',
        early_copack='AUX5120+SWA512 eachcheckpoint:2ops→1undernewpacketABI. PayloadsameasindependentF32projectedSWA+retainedaux;mainfeatureAGremains.',
        late_exact='CurrentSWApacketscannotwaitonpostsampleI. Late3auxbatch isseparate;fixedcounteconomydependsmax uniqueownedrows,notjustM.',
        owner_early='FullmainW157286400B versus19660800Bshard;extra137625600B/card. ComputeNphysicalrows insteadMselected:mainworkratioN/M. Carrycontext5120,threepre-RoPEKV512orbothonlywithallobserverclosure.',
        feature_accumulator='KeepscurrentWshards butallneededauxmuststillarrive;FP32N640snapshot andN640featureAGcanprecomputeNbeforeI. No nativeaccumulatorresume/parityproof;nofreeoffdiagonalblocks.',
        feature_packet_layout='Feature shard contributesN*640values perrank;currentSWAcontributes(N/8)*512values perrank. Theyhave different rowdomains andneedtaggedflatpack plusrole-awareunpack. Featureoutputcannotsharethepacketdeliveringitsmissingthirdauxinput;onlyalatercurrentSWApacketafterthatinputarrivescanbeconsidered.',
        buffer_rule='Snapshotcontentsarenotactualallocatedbytes/savedVRAM. Includeweights,oldaux/context/capture/readers,newpack/accumulatorqueues andinteriterationretirement inonebudget.',
        new_native_API_or_numeric_support_established=False,new_device_work=False,overall_complete=False))
    normal=[('CP','PACK'),('PACK','ATTN'),('ATTN','SAMPLE'),('SAMPLE','I'),('I','LATEAUX'),('LATEAUX','MAIN')]
    blocked=normal+[('I','PACK')]
    early=normal+[('EARLY_PREFIX','PACK')]
    assert not cycle(normal)and cycle(blocked)and not cycle(early)
    write(O/'aux-sp-causality-and-variant-graph-v131.json',dict(
        baseline_or_separate_late_edges=normal,invalid_late_exact_copack_edges=blocked,early_prefix_edges=early,
        late_exact_has_cycle=True,early_prefix_has_cycle=False,
        scope='Chosenpacketisrequiredbycurrentattention. Ifanalternativeeliminatesthatpacketrole,thegraphmustberebuilt;thisisnotanall-algorithmimpossibilityclaim.',
        variants=[
            dict(id='EARLY-AUX-SWA',conditions='FullauxorprovedearlyJ;actualgroup/rowidentity,compatiblelayout andallready/readers. RetainedmainfeatureAG.',unproved_native_gates=True),
            dict(id='LATE-EXACT-AUX',conditions='WaitforcurrentIaftertargetsample,thenuseexistingorderedownerplanner/newlateauxABI. CurrentSWApacketstaysseparate.',unproved_native_gates=True),
            dict(id='EARLY-OWNER-FUTURE',conditions='All3localauxrows/fullWmain/3KVweights availablebeforelastcheckpointpacket;computeNprivatecandidates,restoreIafterward,keeporiginalpostselectRoPE/cache/error/alias semantics.',unproved_native_gates=True),
            dict(id='EARLY-FEATURE-ACCUM',conditions='Globalauxinputstillcommunicated;continueoriginalFP32recurrence/order;preservecastonce andfull5120norm. Nativeprimitiveandparityunproved.',unproved_native_gates=True),
            dict(id='NO-DRAFT-AUX',conditions='K0knownbeforetargetforward,group-uniform,allauxobserversdead andstartup/capture/error effects preserved. CurrentSWAstillhasitsowndemand.',unproved_native_gates=True)],
        readiness='Preserveelement/rangeRAWandbankretirement. No newwholeoperatorbarrier orcriticalpathgain inferred.',
        native_graph_nodes_added=0,new_pair_clauses=1,new_higher_order_groups=0,new_device_work=False,overall_complete=False))
    a=1+2**-8;b=-1.0
    assert bf16(bf16(a)+bf16(b))==0.0 and bf16(a+b)==2**-8
    write(O/'aux-sp-numeric-and-snapshot-gates-v131.json',dict(
        separate_BF16_partial_counterexample=dict(contributions=[a,b,0.0],separately_rounded_then_added=0.0,full_sum_then_BF16=2**-8),
        FP32_resume='Onlyanexactcontinuationoforiginalaccumulationrecurrencecanremovebetween-checkpointBF16rounds. ThefiniteexampledoesnotproveaCANNprimitive,originalMMAorder,NaN/parityorearlyevaluationerrorsemantics.',
        full_norm_domains=[5120,512,512,512],
        current_selected_projection_NaN_ties_or_precision_validated=False,
        checkpoint_snapshots='Oldauxmustremainuntilalloriginalreadersandnewpartialupdatesretire;earlyaccumulatorsorfutureplanes canreplaceonlyall-consumer-closedcontent,notrawpointer/shapeequivalence.',
        early_extra_rows='EarlyphysicalNmayincludepaddingorrowsnotselectedbyI. Preserveerror/exception/dtype/nativeMbehavior;do notcounttheN/Mworkratio asaninvocationbindingorclaimnewnumericparity.',
        new_device_work=False,overall_complete=False))
    inputs=[O/'context-sp-postcheck-result-v130p.json',O/'candidate-relation-clauses-v130.json',O/'candidate-pair-navigation-v130.json',
        O/'aux-sp-checkpoint-profile-v131.json',O/'aux-sp-selector-profile-v131.json',O/'communication-native-operator-population-v66.json',
        O/'index-context-native-chains-v64.json',O/'projection-source-contract.json',O/'F27-F29-row-feature-and-KV-reorganization-v34.json',
        O/'M18-partition-bank-RoPE-and-slot-contract-v34.json',O/'context-sp-source-contract-v130.json',O/'host-launch-original-API-coverage-v126.json',
        O/'aux-sp-selector-parent-only-v131a.json',N/'analyze_aux_sp_profile_parent_only_v131a.py',mo,ds,wo,pr]
    write(O/'aux-sp-inputs-v131.json',dict(inputs=[pin(p)for p in inputs],questions=[
        'DoesreplacingcurrentSPinputbyprojectedKVeliminatethecheckpointauxvalueplane?',
        'Whichinput-selectorarmsarepositivelyrecorded,andwhencanaux/SWAshareanearlypacketwithoutawaitcycle?',
        'Howdocallcount,packing,ownerimbalance,fullweightreplication,N/Mwork andnorm/numeric/snapshotgatescoupleacrossF27andF32?'],
        profile_first=True,no_framework_survey_or_remote_model_NPU_work=True,overall_complete=False))
    print(json.dumps(dict(status='F27_F32_CHECKPOINT_PACKET_RELATION_RECONCILED',graph_pairs=240,selector_ConcatD=88,row_selection_cases=len(layouts),
        cost_geometries=5,new_scoped_pair=1,new_device_work=False)))


if __name__=='__main__':main()
