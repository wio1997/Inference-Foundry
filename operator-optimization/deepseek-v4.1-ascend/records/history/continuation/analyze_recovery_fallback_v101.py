"""Operator-only fallback probability/recovery dataflow, backend-specific recurrence and row-plan costs."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
from copy import deepcopy
from bisect import bisect_right,bisect_left
from itertools import product
import ast,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from reconcile_operator_graph_coverage_v88 import connected_sets

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return struct.unpack('>f',struct.pack('>f',x))[0]
def map_reference(cu,T):
    starts=[0]+cu[:-1]
    return [next((i for i,(a,b)in enumerate(zip(starts,cu))if a<=t<b),0)for t in range(T)]
def map_upper(cu,T):
    return [j if(j:=bisect_right(cu,t))<len(cu)else 0 for t in range(T)]
def map_intervals(cu,T):
    out=[0]*T;start=0
    for i,end in enumerate(cu):
        for t in range(start,min(end,T)):out[t]=i
        start=end
    return out
def block_states(ps,qs,us,backend):
    pi=F(1);u=F(1);pis=[];last=-1
    for i,(p,q,draw)in enumerate(zip(ps,qs,us)):
        ratio=p/q
        pi=min(pi*ratio,F(1))if backend=='TRITON_RECURSIVE_CAP'else pi*min(ratio,F(1))
        u*=draw;pis.append(pi)
        if pi>=u:last=i
    return pis,last
def compose(after,before):
    a,b=after;c,d=before
    return a*c,min(a*d,b)

def main():
    assert not(O/'REVIEW-CHECKPOINT-v101.json').exists()
    root=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend'
    py=root/'sample/rejection_sampler.py';tr=root/'ops/triton/reject_sample.py'
    assert sha(py)=='bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda'
    assert sha(tr)=='dced4b94b74e31e9105d2f7c3bf16b0a096fb6c74983647ce7d056111807f2c3'
    src=[]
    upstream=W/'frozen-source-snapshot/vllm/vllm/v1/sample/rejection_sampler.py'
    assert sha(upstream)=='0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750'
    for p,ranges in [(py,[(435,449),(554,683),(729,795),(897,1055),(1104,1187),(1190,1382)]),(tr,[(344,424),(526,610)]),(upstream,[(616,668)])]:
        ls=p.read_text().splitlines();src.append(dict(path=str(p),sha256=sha(p),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in ranges]))
    exact=json.loads(read(N/'M17-exact-profile166-v33.json')['stdout'])['summary']
    assert exact['matched_counts']['sample_recovered_tokens']==exact['matched_counts']['rejection_random_sample']==0
    assert len(read(O/'heads-sampling-native-chains-v65.json')['greedy_rejections'])==72
    write(O/'recovery-fallback-source-and-applicability-v101.json',dict(parts=src,
      exact_saved_name_audit=exact['matched_counts'],observed_greedy_chains=72,
      profile_absence_scope='Saved32DBexactnamesurface;notprooffallback/impossibleunderallotherbranches. Analysishereisconditionalstaticbranches,nopromotedruntimeexecution.',
      facts=[
        'DefaultPyTorchrecoveryreturnsforT==0,constructsT×BintervalmaskandINT32cast/argmax/any/whereownerplan,thenvirtualizableprivateP/residual,Qgather,twoQsafeplanes,quotient,exceptionguardandrowargmax.',
        'Q==0orisinfQgetsfinalscore-1e10inPyTorchfallback;NaNQisnotcoveredbythatguard. NGRAMclonesPandexcludesdraftID!=PLACEHOLDER;probabilisticresidual=max(P-D,0). Finite/negative-ID/dtype/normalizationdomainsarenotuniversallyvalidatedbythehelper.',
        'AscendTritonrecoverydividesbyQdirectlywithSUB_BLOCK4096/tl.argmax/strictgreatertilemerge. Itisnottheupstreamreciprocal-multiply/FP64GumbelkernelandnotthesameexceptioncontractasPyTorchfallback.',
        'OrdinaryPyTorchverificationusesfirstfailedvalidposition;blockPyTorchverificationuseslastlegalpositionfromcumprod(clamp(Pd/Qd,max1)). Tritonblockupdatespi=min(pi*Pd/Qd,1)sequentially. Thesearenotgenerallythe sameformula.',
        'PyTorchverificationreceivesoriginalFLOAT64uniforms;AscendTritoncallerexplicitlyconvertsuniformstoFLOAT32. Arithmetic/comparisonpathsremainbackend-specific.',
        'Optionalblockwiserecoveryhasresidual=max(p_prefix_before_position*P-D,0);thewrapperrootcallomitsuse_block_verifyanditsdefaultisFalse. enable_block_verifydoesnotautomaticallyselectthisrecoveryhelper.',
        'Qdefaultandper-seededtempdraws/zerodraftsemanticswerealreadyv33. ThisturnkeepsalloriginalRNGeffects;decision-then-lazyrecoveryalsopredatesv101.',
        'VerificationfallbackhasnonemptyB×Sgathersbeforefinalvalid/greedymasksandnoanalogousT==0earlyguard. Anyzero-draftspecializationmustdeclarevalidindex/error/eager-readbehavior;notturnsourceerrorsintonewresults.'
      ],new_target_source_files=0,new_public_documents=0,new_server_or_Torch_NPU_work=False,no_framework_research=True,overall_complete=False))
    prior=['M17-probability-consumer-and-RNG-contract-v33.json','F35-probability-summary-and-lazy-recovery-algebra-v33.json',
      'M17-source-and-conditional-recovery-graphs-v33.json','heads-sampling-native-chains-v65.json','sampling-filter-race-source-contract-v91.json',
      'operator-graph-coverage-audit-v93.json','operator-graph-node-coverage-v93.json','operator-graph-edge-coverage-v93.json','operator-graph-candidate-bindings-v93.json',
      'whole-scope-requirement-audit-v100.json']
    write(O/'recovery-fallback-inputs-v101.json',dict(inputs=[dict(path=str(O/n),bytes=(O/n).stat().st_size,sha256=sha(O/n))for n in prior]+[dict(path=str(N/'M17-exact-profile166-v33.json'),bytes=(N/'M17-exact-profile166-v33.json').stat().st_size,sha256=sha(N/'M17-exact-profile166-v33.json'))],
      prior_complete_local_answers=['v33lazyrecoverysameQ/lastacceptvsfirstreject/RNGeffects','v90greedyfallbackfullwrite','v91randomP/Qraceandmixedgreedymasks'],
      new_scope='NamedPyTorchrandom/recoverybodyprimitives,backendnonidentity,right-biasedownerplanandrecurrence-scanprecisiondomain.',overall_complete=False))

    ps=[F(1,5),F(4,5),F(2,5)];qs=[F(2,5)]*3;us=[F(9,10),F(4,5),F(4,5)]
    tp,tl=block_states(ps,qs,us,'TRITON_RECURSIVE_CAP');fp,fl=block_states(ps,qs,us,'PYTORCH_RATIO_CAP')
    assert(tl,fl)==(2,-1)
    uniform=0.5+2**-26;assert F(1,2)<F(uniform)and f32(uniform)==0.5
    p=[F(3,5),F(3,10),F(1,10)];d=[F(2,5),F(1,10),F(1,2)]
    ordinary=[max(x-y,0)for x,y in zip(p,d)];weighted=[max(F(1,2)*x-y,0)for x,y in zip(p,d)]
    argmax=lambda xs:max(range(len(xs)),key=lambda i:xs[i])
    assert(argmax(ordinary),argmax(weighted))==(0,1)
    write(O/'recovery-backend-nonidentity-v101.json',dict(
      block_clip_counterexample=dict(target_selected=list(map(str,ps)),draft_selected=list(map(str,qs)),uniforms=list(map(str,us)),
        Triton_recursive_pi=list(map(str,tp)),PyTorch_ratio_capped_pi=list(map(str,fp)),Triton_last_accepted=tl,PyTorch_last_accepted=fl,
        max_spec_len_reference=3,entropy=False,synthetic=False,all_inputs_finite_positive=True,
        conclusion='Tritonallaccepts+bonus;PyTorchhasnolegalpositionandusesrecoveryrow0. Exactrealdifference,notsmallroundofforactualC8execution.'),
      uniform_dtype_counterexample=dict(ratio=0.5,uniform_FLOAT64=uniform,uniform_cast_FLOAT32=f32(uniform),PyTorch_accept=False,Triton_accept=True,
        scope='Ordinaryratio>=uniformwithentropy/syntheticoff;possiblefixedoperandcounterexample,notobservedRNGdraw.'),
      optional_block_recovery_counterexample=dict(prefix_pi='1/2',target=list(map(str,p)),draft=list(map(str,d)),same_Q=[1,1,1],
        ordinary_residual=list(map(str,ordinary)),weighted_block_residual=list(map(str,weighted)),ordinary_argmax=0,weighted_argmax=1,
        current_root_does_not_pass_use_block_verify=True),
      Q_guard_contract=dict(PyTorch='QzeroorInf→-1e10;NaNfalls through;preservetorch.argmaxsemantics',Triton='directP_or_residual/Q;tailQ=-Inf;actualtlNaN/tie/sourceprecisiongatesremain'),
      no_automatic_backend_interchange_or_bugfix=True,actual_flags_or_native_parity_proved=False,overall_complete=False))

    count=0;tokens=0
    for B in range(1,6):
        for ns in product(range(4),repeat=B):
            cu=[];s=0
            for n in ns:s+=n;cu.append(s)
            for tail in [-1,0,1,3]:
                T=max(0,s+tail);assert map_reference(cu,T)==map_upper(cu,T)==map_intervals(cu,T);count+=1;tokens+=T
    badcu=[2,1,3];assert map_reference(badcu,2)!=map_upper(badcu,2)
    plans=[]
    for B,S in product([2,4,8,32],[1,5]):
        T=B*S;plans.append(dict(B=B,S=S,T_capacity_reference=T,source_interval_comparisons=2*T*B,
          dense_BOOL_plane_elements_reference=3*T*B,INT32_cast_plane_elements_reference=T*B,owner_map_INT64_elements=T,
          upper_bound_comparison_upper_reference=T*math.ceil(math.log2(B+1)),
          count_grid_is_source_work_not_necessary_bound=True,actual_cu_values_unknown=True))
    write(O/'recovery-owner-plan-and-state-reuse-v101.json',dict(exhaustive_cases=count,checked_token_positions=tokens,models=plans,
      exact_identity='ForB>=1andnonnegative nondecreasingcu,token t>=0mapsfirstintervalcu_start[i]<=t<cu[i],else0. Thisisupper_bound(cu,t),withBresultmappedto0. Repeatedcu/zero-draftrequestsrequireRIGHTbias.',
      interval_fill_alternative='Withthesamevalidcu,disjointrequestintervalfillwritesowner/positioninO(T+B)scalarwork;preservetailt>=cu[-1]→request0. Fusingper-request/positionrecoverycanusegridownerinsteadofmaterializinga map. GenericpublichelperTtailcannotbeskippedwithoutT==cu[-1]oralltailreaderproof;launch/mask/vectorizationcostsremain.',
      zero_draft_witness=dict(cu=[0,2],token=0,source_owner=1,left_biased_owner=0,right_biased_owner=1),
      boundary_and_tail_witness=dict(cu=[1,1,3,3],source_owners_for_T5=map_reference([1,1,3,3],5),tail_maps_to_request0=True),
      nonmonotone_counterexample=dict(cu=badcu,T=2,interval_reference=map_reference(badcu,2),binary_search_not_legal=map_upper(badcu,2)),
      reuse='Anexistingtoken→requestmapcanbesharedonlyforsamecu/row/stride/dtype/capacity/generation. Attentionseqlens/targettokenmapsarenotautomaticallydraftcu. Alternatehelperbranchesdonotexecuteallmapsatonce;nodoubleCSEcredit.',
      selected_recovery='v33selectedRmapalreadyidentifiesrequestandposition,soanewselected-onlyrecoveryconsumerneednotconstructallT×Bmap. OldPyTorchverificationeagerlygathersallRecoveredfirst;changeconsumerreadsetorprovidevalidunusedslots,preserveerrors/epochs.',
      optional_static_S='Blockwiserecoveryusesmax(n).item()thenprefixloop. AnewABIcanuseknownS>=max(n)andidentityratiosoninactivepositions,withvalidT/cu/posdomain;additionalpaddingworkanderror/zero-draftguardsmustbepreserved. Notcurrentrootactivated.',
      native_implementation_or_runtime_plan_identity_proved=False,overall_complete=False))

    # Exact-real associative representation of capped multiplication;nativeFP32canreassociateunderflowsdifferently.
    assoc=0
    for rs in product([F(0),F(1,4),F(1),F(2),F(4)],repeat=3):
        fs=[(r,F(1))for r in rs]
        assert compose(fs[2],compose(fs[1],fs[0]))==compose(compose(fs[2],fs[1]),fs[0]);assoc+=1
    native=[];x=1.0
    for p0,q0 in [(2**-100,1.0),(2**-100,1.0),(1.0,2**-100)]:x=min(f32(f32(x*p0)/q0),1.0);native.append(x)
    reassociated=f32(f32(2**-100)*f32(f32(2**-100)/f32(2**-100)))
    assert native[-1]==0 and reassociated==2**-100
    write(O/'recovery-prefix-scan-and-demand-v101.json',dict(
      real_monoid='Forfinite r>=0,f_r(x)=min(r*x,1). Generalf_(a,b)(x)=min(a*x,b). Composition(after,before)=(a2*a1,min(a2*b1,b2))isassociativeoverexactreals. (1,1)isidentityonx∈[0,1]. ThismodelsTritonrecursion,notcumprod(clamp(eachratio,1)).',
      associativity_cases=assoc,FP32_reassociation_counterexample=dict(P=[2**-100,2**-100,1.0],D=[1.0,1.0,2**-100],sequential_pi=native,
        reassociated_last_pi=reassociated,uniforms=[2**-10,2**-50,2**-50],sequential_last_accepted=-1,reassociated_last_accepted=2,
        operation_model='IEEEbinary32RNEaftereachmul/div,noFMA;noactualTriton/nativeexecution.'),
      span='ThepairscanhasO(S)combinework/O(logS)exact-realspaninaparallelmodel. Itisnotadrop-inFP32optimization;exactcompositionofroundedstepfunctionsisadifferentrepresentation/cost,notruledout. Ssourceiterationsarenotanall-algorithmhardwarelowerbound.',
      entropy_demand='OrdinaryfallbackcomputesentropyforwholeB×Sgatheredrowsbeforevalid/non-greedyprefixdecision. AnewconsumercanlimitprivateentropytoactivevalidvisitedrowswhilekeepingallRNGeffectsandactualraw-or-processedentropy-source. Blocklastlegalgenerallyneedsallnpositions,notfirstfailedposition.',
      lazy_tradeoff='SavingV-wideprivateentropy/residualworkcanadddecisionstagesandreduceparallelism. SourceFP32epsilonlog/reduction/thresholdandFLOAT64/FLOAT32uniformdifferencesremain;notactualcriticalspeedup.',
      output_write_set='Ordinary:acceptedprefix+onefirst-recoveryorbonus;block:prefixthroughlastlegal+nextrecoveryorbonus. PreserveoldgreedyrowsandunwrittenINT32suffix;fullwritefusionrequiresanewwritercontract,notdeletingFillalone.',
      overall_complete=False))

    costs=[]
    for B,S,V,ng in product([2,4,8,32],[1,5],[129280],[False,True]):
        T=B*S
        for K in sorted({0,B,T}):
            for fuse in [False,True]:
                # SixdeclaredprivateFLOATplanes:clone/residual,Qgather,Qsafe1,Qsafe2,ratio,guardedratio.
                base_interfaces=6*8*T*V;remaining_interfaces=0 if fuse else 6*8*K*V
                costs.append(dict(B=B,S=S,T_capacity_reference=T,V=V,NO_DRAFT_PROBS=ng,selected_recovery_rows_reference=K,private_plane_fusion=fuse,
                  standalone_six_FLOAT_private_plane_write_read_reference_bytes=base_interfaces,
                  selected_graph_private_plane_write_read_reference_bytes=remaining_interfaces,
                  conditional_removed_private_interfaces_bytes=base_interfaces-remaining_interfaces,
                  P_and_D_and_Q_consumer_load_value_reference_bytes=(8 if ng else 12)*K*V,
                  original_Q_BxV_FLOAT_RNG_bank_bytes=4*B*V,original_all_uniform_and_seeded_Q_draws_preserved=True,
                  INT32_selected_ID_store_reference_bytes=4*K,
                  private_planes_are_not_a_live_peak_sum_or_actual_HBM=True,new_selection_launch_map_sync_and_precision_cost_unpriced=True,
                  all_other_P_raw_logprob_entropy_readers_not_removed=True))
    write(O/'recovery-fallback-work-and-interface-models-v101.json',dict(models=costs,model_count=len(costs),
      selected_materialization_baseline='DeclaredstandaloneFLOAT32P/D/QandINT32IDreference:sixprivateplanes,eachonewrite+oneread. Actualcompiler/operatorfusioncouldalreadyremoveplanes;notcapturedHBM,allocationpeakorallalgorithmlowerbound.',
      fusion_contract='Computeexcludedcolumnormax(P-D,0),sameguardedFLOAT32quotientandfinitefirstargmaxintiles;keeporiginalP/D/Qimmutable. Carry(score,globalID)withsourcefinitefirsttie. Masknewpaddedlanesto-inf:sourcePyTorchhasexactV,allinvalidQscore-1e10mustnotloseagainstzero-paddedlanes.',
      padded_tail_counterexample=dict(V=3,source_scores=[-1e10]*3,source_argmax=0,wrong_zero_padded_argmax=3,correct_minus_inf_padded_argmax=0),
      normalization_or_log_race_or_reciprocal_substitution_not_proved=True,
      theory='Foropaqueindependentfinitecandidate-scorecomparisonmodelwithnocachedwinner/boundsandKfullrows,needKVscoreobservationsandK(V-1)comparisondecisionsinworstcase. ThisisnotamandatoryGM/HBMreadorDivcountforfused/structured/probability-constrainedalgorithms. Physicaltimeboundrequiresmatchingcapacityuppersandactualnecessarywork;LOG/EXP/DIVnotpricedbyBF16GEMMpeak.',
      theory_status='SYMBOLIC_CEILING',physical_capacity_uppers_unknown=['scoreevaluation','comparison/reduction','GMread/write','DIV/LOG/EXPmatchedmode'],
      actual_runtime_flag_or_native_speedup_or_numeric_physical_ceiling_proved=False,overall_complete=False))

    def template(tid,nodes,edges):
        return dict(id=tid,nodes=[dict(id=i,kind=k,resource='DECLARED_OPERATOR_OR_STATE',candidate_associations=['F31','F35']if i in ['cu','owner_map','row_epoch']else['F35'])for i,k in nodes],
          edges=[dict(producer=a,consumer=b,value=v,dependency_kind='SOURCE_DATA_CONTROL_OR_STATE',condition='Seev101sourceguardinventory;notallarmscoexecute',source='recovery-fallback-source-and-applicability-v101.json')for a,b,v in edges],
          complete=False,native_runtime_execution_attested=False)
    ts=[]
    ts.append(template('RECOVERY_PYTORCH_OWNER_MAP',[(i,k)for i,k in [('cu','DRAFT_CUMULATIVE_COUNTS'),('T','RECOVERY_OUTPUT_CAPACITY'),('empty_guard','T_ZERO_EARLY_RETURN'),('starts','CAT_ZERO_CU_PREFIX'),('tokens','ARANGE_T_INT64'),('bounds','T_BY_B_INTERVAL_COMPARE_AND'),('as_int','BOOL_TO_INT32'),('first','ARGMAX_FIRST_MATCH_INT64'),('any','ANY_MATCH'),('owner_map','WHERE_MATCH_ELSE_REQ0'),('row_epoch','MAP_GENERATION_LIFETIME')]],
      [('T','empty_guard','T==0'),('cu','starts','prefix'),('T','tokens','T'),('starts','bounds','lower'),('cu','bounds','upper'),('tokens','bounds','t'),('bounds','as_int','cast'),('as_int','first','firsttrue'),('bounds','any','match'),('first','owner_map','candidate'),('any','owner_map','fallback0'),('owner_map','row_epoch','publish')]))
    ts.append(template('RECOVERY_PYTORCH_VALUES',[(i,k)for i,k in [('P','TARGET_P_FLOAT'),('D','OPTIONAL_DRAFT_D'),('d','DRAFT_IDS'),('branch','IS_NGRAM'),('cu','DRAFT_CU'),('owner_map','TOKEN_TO_REQUEST'),('clone','P_CLONE'),('exclude','EXCLUDE_DRAFT_COLUMN'),('residual','MAX_P_MINUS_D_ZERO'),('Q','UNCHANGED_REQUEST_Q_RNG_BANK'),('qgather','Q_OWNER_GATHER'),('qsafe0','WHERE_Q_ZERO_EPS'),('qsafeInf','WHERE_Q_INF_EPS'),('ratio','P_OR_RESIDUAL_DIV_QSAFE'),('guard','BAD_Q_TO_NEG1E10'),('argmax','FULL_VOCAB_ARGMAX'),('out','RECOVERED_IDS_WRITE')]],
      [('P','clone','ngram'),('d','exclude','valid!=-1'),('clone','exclude','private'),('branch','exclude','ngram'),('P','residual','prob'),('D','residual','prob'),('branch','residual','probabilistic'),('cu','owner_map','rowplan'),('owner_map','qgather','request'),('Q','qgather','sameQ'),('qgather','qsafe0','zero'),('qsafe0','qsafeInf','safeprior'),('qgather','qsafeInf','isinf'),('exclude','ratio','ngram'),('residual','ratio','probabilistic'),('qsafeInf','ratio','denominator'),('qgather','guard','exceptionmask'),('ratio','guard','finiteelse'),('guard','argmax','scores'),('argmax','out','IDs')]))
    ts.append(template('VERIFY_PYTORCH_ORDINARY',[(i,k)for i,k in [('cu','DRAFT_CU'),('grid','B_BY_S_CLAMPED_TOKEN_INDICES'),('selected','SELECTED_P_D_AND_SAFE_IDS'),('u','ORIGINAL_FLOAT64_UNIFORMS'),('recover','EAGER_ALL_RECOVERED_GATHER'),('entropy','OPTIONAL_FULL_VOCAB_EPS_ENTROPY'),('accept','RATIO_OR_SYNTHETIC_COMPARE'),('valid','VALID_AND_PLACEHOLDER_MASK'),('first','FIRST_FAILED_VALID_POSITION'),('writes','PREFIX_RECOVERY_KEEP_OLD_ROWS'),('bonus','BONUS_WHERE_NO_REJECTION'),('out','SHARED_OUTPUT_BANK')]],
      [('cu','grid','ragged'),('grid','selected','p/d'),('grid','u','gather'),('grid','recover','eagerread'),('grid','entropy','optionalgather'),('entropy','accept','threshold'),('selected','accept','ratio'),('u','accept','draw'),('accept','valid','placeholder/valid'),('valid','first','firstfail'),('first','writes','readset'),('recover','writes','chosenID'),('selected','writes','acceptedIDs'),('first','bonus','no reject'),('cu','bonus','n'),('writes','out','partialwrite'),('bonus','out','slotn')]))
    ts.append(template('VERIFY_PYTORCH_BLOCK',[(i,k)for i,k in [('cu','DRAFT_CU'),('grid','B_BY_S_TOKEN_PLAN'),('selected','SELECTED_P_D'),('ratio','P_DIV_D'),('cap_each','CLAMP_EACH_RATIO_MAX1'),('pi','CUMPROD_CAPPED_RATIOS'),('u','UNIFORM_CUMPROD'),('entropy','OPTIONAL_EPS_ENTROPY_THRESHOLD'),('legal','VALID_NONPLACEHOLDER_LASTLEGAL_TEST'),('last','FLIP_ARGMAX_LAST_ACCEPT'),('recover','EAGER_RECOVERED_GATHER'),('writes','PREFIX_PLUS_NEXT_RECOVERY'),('bonus','BONUS_IF_LAST_PLUS1_GE_N'),('out','SHARED_OUTPUT_BANK')]],
      [('cu','grid','indices'),('grid','selected','p/d'),('selected','ratio','quotient'),('ratio','cap_each','localcap'),('cap_each','pi','prefixproduct'),('grid','u','uniformgather'),('grid','entropy','optionalfullV'),('pi','legal','test'),('u','legal','threshold'),('entropy','legal','threshold'),('grid','legal','valid'),('legal','last','lasttrue'),('grid','recover','allgather'),('last','writes','prefix'),('recover','writes','nextID'),('last','bonus','no recovery'),('cu','bonus','n'),('writes','out','partial'),('bonus','out','slotn')]))
    ts.append(template('RECOVERY_OPTIONAL_BLOCKWISE',[(i,k)for i,k in [('cu','DRAFT_CU'),('owner_map','TOKEN_TO_REQUEST'),('pos','TOKEN_POSITION_IN_REQUEST'),('maxn','MAX_N_ITEM_HOST_READ'),('ratio','SELECTED_P_OVER_D_GUARDED'),('ratio_grid','B_BY_MAX_N_ONES_AND_SCATTER'),('prefix','SEQUENTIAL_CAPPED_PREFIX'),('prior_pi','PREFIX_BEFORE_CURRENT_POSITION'),('P','TARGET_P'),('D','DRAFT_D'),('exclude','NGRAM_CLONE_EXCLUDE'),('residual','MAX_PI_P_MINUS_D_ZERO_OR_PI_P_EXCLUDED'),('Q','ORIGINAL_REQUEST_Q'),('qgather','Q_OWNER_GATHER'),('guarded_ratio','BAD_Q_NEG1E10_ELSE_RESIDUAL_DIV_SAFE'),('argmax','ROW_ARGMAX_STORE')]],
      [('cu','owner_map','rowplan'),('cu','maxn','counts'),('cu','pos','start'),('owner_map','pos','request'),('ratio','ratio_grid','selectedratio'),('maxn','ratio_grid','capacity'),('ratio_grid','prefix','orderedloop'),('prefix','prior_pi','position'),('pos','prior_pi','index'),('owner_map','prior_pi','row'),('P','exclude','ngram'),('P','residual','probabilistic'),('D','residual','probabilistic'),('prior_pi','residual','preposition'),('exclude','residual','ngram'),('owner_map','qgather','row'),('Q','qgather','sameQ'),('residual','guarded_ratio','numerator'),('qgather','guarded_ratio','guard/denominator'),('guarded_ratio','argmax','output')]))
    graph=dict(id='SAMPLER_RECOVERY_PYTORCH_G8_V101',templates=ts,scope='Fiveoperator-levelfallbacktemplatesandconditionalalternativehelper. Currentrootfallbackexecutionnotattested;Triton/backenddifferencesexplicit. Notwhole sampler/frameworkgraph.',complete=False,overall_complete=False)
    write(O/'operator-graph-G8-v101.json',graph)
    nn=[];ee=[];connected={}
    for t in ts:
        ids={n['id']:n for n in t['nodes']};sets=connected_sets(ids,t['edges']);nc=defaultdict(Counter);ec=defaultdict(Counter);connected[t['id']]={str(k):len(v)for k,v in sets.items()}
        for k,subsets in sets.items():
            for ss in subsets:
                for nid in ss:nc[nid][k]+=1
                for i,e in enumerate(t['edges']):
                    if e['producer']in ss and e['consumer']in ss:ec[i][k]+=1
        nn += [dict(graph='G8',template=t['id'],node_id=n['id'],kind=n['kind'],module='M17',connected_set_counts=dict(nc[n['id']]),candidate_source_node_associations=n['candidate_associations'],structural_scope='ALL_CONNECTED_SETS_OF2_TO4_WITHIN_DECLARED_FALLBACK_TEMPLATE',semantic_coverage='SOURCE_SCOPED_BRANCH_CONTRACT_NOT_NATIVE_COMPLETION',native_lowering_and_all_runtime_variants_closed=False)for n in t['nodes']]
        ee += [dict(graph='G8',template=t['id'],edge_index=i,edge=e,connected_set_counts=dict(ec[i]),producer_candidate_associations=ids[e['producer']]['candidate_associations'],consumer_candidate_associations=ids[e['consumer']]['candidate_associations'],semantic_coverage='CONDITIONAL_SOURCE_INTERFACE_REVIEWED',dependencies_are_not_all_same_invocation_DAG_edges=True)for i,e in enumerate(t['edges'])]
    for name,rows in [('node',nn),('edge',ee)]:
        d=deepcopy(read(O/f'operator-graph-{name}-coverage-v93.json'));d['rows']+=rows;d.update(parent=f'operator-graph-{name}-coverage-v93.json',row_count=len(d['rows']),overall_complete=False);write(O/f'operator-graph-{name}-coverage-v101.json',d)
    bind=deepcopy(read(O/'operator-graph-candidate-bindings-v93.json'))
    for cid in ['F31','F35']:
        next(c for c in bind['candidates']if c['id']==cid)['G8_recovery_fallback_v101']=dict(templates=[t['id']for t in ts],all_variants_mapped=False,source_reader_not_native_producer_identity=True)
    hint=next((h for h in bind['structural_pair_hints']if{h['a'],h['b']}=={'F31','F35'}),None)
    if hint is None:
        hint=dict(a='F31',b='F35',locations=[],existing_conditional_clause_ids=['REL73-26'],structure_does_not_prove_same_call_or_joint_legality=True);bind['structural_pair_hints'].append(hint)
    hint['locations'] += [dict(graph='G8',template=r['template'],edge_index=r['edge_index'],kind='DECLARED_DRAFT_OWNER_PLAN_READER')for r in ee if'F31'in r['producer_candidate_associations']and'F35'in r['consumer_candidate_associations']]
    bind.update(parent='operator-graph-candidate-bindings-v93.json',overall_complete=False);write(O/'operator-graph-candidate-bindings-v101.json',bind)
    old=read(O/'operator-graph-coverage-audit-v93.json');catalog=old['graph_catalog']|{'G8':str(O/'operator-graph-G8-v101.json')}
    write(O/'operator-graph-coverage-audit-v101.json',dict(parent='operator-graph-coverage-audit-v93.json',graph_catalog=catalog,graph_sha256={k:sha(Path(v))for k,v in catalog.items()},
      new_source_node_rows=len(nn),new_source_edge_rows=len(ee),node_rows=old['node_rows']+len(nn),edge_rows=old['edge_rows']+len(ee),
      current_structural_pair_hints=len(bind['structural_pair_hints']),G8_templates=5,G8_connected_sets=connected,
      source_rows_not_unique_native_operators_or_new_executions=True,whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(map_cases=count,assoc_cases=assoc,interface_models=len(costs),G8_nodes=len(nn),G8_edges=len(ee),all_node_rows=old['node_rows']+len(nn),all_edge_rows=old['edge_rows']+len(ee),pair_hints=len(bind['structural_pair_hints']),new_pairs=0,new_native_promotions=0,overall_complete=False)))

if __name__=='__main__':main()
