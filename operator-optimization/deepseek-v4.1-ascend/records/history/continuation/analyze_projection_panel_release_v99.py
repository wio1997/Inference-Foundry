"""Saved projection/norm frontiers, ragged panel domains and conditional resource/release cuts."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
from math import ceil
import hashlib,itertools,json
from finalize_origin_coverage_v9 import N,O,R,W,read,write
from analyze_HC_attention_fork_v93 import stats

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def intersect(a,b):return max(0,min(a['endNs'],b['endNs'])-max(a['startNs'],b['startNs']))
def schedule(a,b,c,d,e):
    return dict(original_stage_join=a+max(b,c)+max(d,e),
      joint_projection_whole_output_keep_stage_join=a+b+c+max(d,e),
      joint_projection_whole_output_release_KVnorm_early=a+b+max(c+d,e),
      QA_panel_release_and_KVnorm_early=a+max(max(b,c)+d,b+e))

def main():
    assert not(O/'REVIEW-CHECKPOINT-v99.json').exists()
    inputs=['projection-graph-chains-v61.json','norm-quant-native-frontiers-v94.json','F33-profile-order-contract-v27.json',
      'F33-consumer-and-numerical-contract-v27.json','F33-projection-algebra-and-layout-v27.json','A3-datapath-primary-evidence-v80.json',
      'typed-capacity-primary-contract-v89.json','projection-fusion-variants-and-interactions-v61.json','HC-attention-shared-resource-models-v93.json',
      'norm-quant-interface-and-resource-models-v94.json','norm-quant-precision-and-row-models-v94.json','whole-scope-requirement-audit-v98.json']
    write(O/'projection-panel-inputs-v99.json',dict(inputs=[dict(path=str(O/n),bytes=(O/n).stat().st_size,sha256=sha(O/n))for n in inputs],
      scope='Existingcapture-derivedrecords andpreviousscopedsource/architecturecontracts;newnumericmodelsremainconditional,noactiveNPUorserverwork.',overall_complete=False))
    dsa=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/attention/dsa_v41.py'
    compressor=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py'
    assert sha(dsa)=='3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd'
    sources=[]
    for p,segs in [(dsa,[(350,384),(440,503),(509,568)]),(compressor,[(15,25),(55,71)])]:
        ls=p.read_text().splitlines();sources.append(dict(path=str(p),sha256=sha(p),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in segs]))
    write(O/'projection-panel-source-contract-v99.json',dict(parts=sources,profile_question='HowcanjointQA/SWA/C1projectionscomposewithdifferentRMSgroupswithoutinventingrowworkorwhole-outputbarriers?',
      facts=[
        'Currentmultistreamsource launchesQA,thenKV onauxwhileQnorm/quant run,thenQB overlapsKVnorm/RoPE;part3eventcurrentlygatesKVnormaswell. Actualtimingsarenotsameasallsemanticdependencies.',
        'Queryhiddenandcachehiddenhaveindependentactual-rowclipping/metadata;share_quant requiresexplicitshare_inputs/type/communicationcompatibility. CapturedM3QA andM24KV do notestablishsameinputrowdomain.',
        'C1usesdistinctBF16linearand512RMS;C2usesFLOATWKV/Wgateandpool_projected→BF16norm. F18originalcandidateisC2pool→norm;F33isC1jointprojection. Genericlatent.norm512navigationatomdoesnotestablishsharedinvocation.',
        'C1IndexKeyreadspre-longRoPElatent;longKVrotateslatent inplace andbothpublicationsremain. Aconcatenatedproducerdoesnotdeletefanout/WAR/ownershipcontracts.',
        'Qnorm1280,SWA512andC1latent512arestillseparateparameter/epsilon/reductiondomains;preserveGEMM→BF16andnorm→BF16rounding.'
      ],cached_source_files=2,new_target_source_files=0,no_framework_research=True,no_server_or_device_work=True,overall_complete=False))
    priorfps=read(O/'candidate-operator-footprints-v98.json');idx={r['id']:r for r in read(O/'candidate-record-index-v98.json')['model_candidates']}
    mappings={'F18':['compress.C2.norm512'],'F19':['compress.C1.norm512','compress.C2.norm512'],'F33':['compress.C1.norm512']};audits=[]
    for cid,newatoms in mappings.items():
        old=next(r for r in priorfps['candidates']if r['id']==cid);p=R/'candidates'/(cid+'.json');master=read(p)
        assert sha(p)==idx[cid]['master_sha256']and 'compress.latent.norm512'in old['operator_atom_union']
        if cid=='F18':assert master['original_template_graph_id']=='ATTN_C2_SOURCE'and set(master['original_node_mapping'])=={'ring_pool','compress_norm'}
        audits.append(dict(candidate=cid,master_before_sha256=sha(p),master_before_path=str(p),previous_generic_atom='compress.latent.norm512',replacement_norm_atoms=newatoms,
          old_union_atoms=old['operator_atom_union'],original_template=master.get('original_template_graph_id'),original_nodes=master.get('original_node_mapping'),
          scope='F18C2only;F33C1only;F19hasC1/C2forkvariantswithsharedpostbranchkey/longKVcode. Mode-specificatomsarestillinvocation/row/layer/generationqualified;notphysicalaliasproof.'))
    write(O/'projection-panel-mode-atom-audit-v99.json',dict(records=audits,
      correction='F18/F33genericnormintersectionisafalsecurrent-workidentity. C1andC2canbothoccuratdifferentlayersofthemodel,butarenotthe samenorminvocation. F19retainsbothqualifiedmodevariants.',
      no_new_native_attribution_or_removal_claim=True,overall_complete=False))

    base=read(O/'projection-graph-chains-v61.json');q=read(O/'norm-quant-native-frontiers-v94.json');qidx={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in q['frontiers']}
    fronts=[];used=set()
    for r in base['graph_projection_chains']:
        key=(r['database_index'],r['graph_ordinal'],r['block_ordinal']);s=r['stages'];qr=qidx[key]
        assert r['root_bound']==qr['root_bound']
        for k in ['Q_A','Q_NORM','Q_QUANT','Q_B']:
            assert s[k]['task_record_rowid']==qr['stages'][k]['task_record_rowid']
            assert (s[k]['startNs'],s[k]['endNs'])==(qr['stages'][k]['startNs'],qr['stages'][k]['endNs'])
        stages={k:s[k]for k in ['Q_A','Q_NORM','Q_QUANT','Q_B','SWA_KV','KV_NORM']}
        for v in stages.values():used.add((r['database_index'],v['task_record_rowid']))
        fronts.append(dict(database_index=key[0],graph_ordinal=key[1],block_ordinal=key[2],root_bound=r['root_bound'],stages=stages,
          SWA_role_unique=r['KV_BF16_candidates']==1,KV_BF16_candidates=r['KV_BF16_candidates'],
          SWA_with_Qnorm_ns=intersect(s['SWA_KV'],s['Q_NORM']),SWA_with_Qquant_ns=intersect(s['SWA_KV'],s['Q_QUANT']),
          QB_with_KVnorm_ns=intersect(s['Q_B'],s['KV_NORM']),
          Qnorm_starts_before_KVends=s['Q_NORM']['startNs']<s['SWA_KV']['endNs'],
          Qnorm_finishes_before_KVends=s['Q_NORM']['endNs']<=s['SWA_KV']['endNs'],
          QA_end_before_KVend=s['Q_A']['endNs']<s['SWA_KV']['endNs'],
          metadata='Source/stream-derivedSWAroleatlayer20stillambiguousphysicaltensoridentity;noassignmentpromotion.'))
    assert len(fronts)==3200 and len(used)==19200
    summaries={}
    for name,sel in [('all',fronts),('root_unique',[r for r in fronts if r['root_bound']and r['SWA_role_unique']]),('layer20_source_stream',[r for r in fronts if not r['SWA_role_unique']])]:
        summaries[name]=dict(frontiers=len(sel),overlaps={k:dict(duration_ns=stats([r[k]for r in sel]),positive=sum(r[k]>0 for r in sel))for k in ['SWA_with_Qnorm_ns','SWA_with_Qquant_ns','QB_with_KVnorm_ns']},
          Qnorm_before_KVend=sum(r['Qnorm_starts_before_KVends']for r in sel),Qnorm_done_before_KVend=sum(r['Qnorm_finishes_before_KVends']for r in sel))
    write(O/'projection-panel-native-frontiers-v99.json',dict(frontiers=fronts,frontier_count=3200,root_bound=2560,phase_unbound=640,unique_SWA_roles=3120,ambiguous_source20_roles=80,reused_TASKs=19200,
      summaries=summaries,prior_C1_order_evidence='F33-profile-order-contract-v27.json',new_native_attribution_or_regime_promotions=0,
      observed_overlap_not_recoverable_latency_or_native_fused_schedule=True,overall_complete=False))

    models=[];H=5120;warm=24*65536;B=24*128;P=24*8192
    for c1 in [False,True]:
        groups=[dict(id='QA',M=3,N=1280),dict(id='SWA',M=24,N=512)]+([dict(id='C1',M=24,N=512)]if c1 else[])
        scalar=sum(g['M']*g['N']for g in groups);weights=sum(H*g['N']*2 for g in groups);totalN=sum(g['N']for g in groups)
        ops=2*H*scalar;rect=2*24*H*totalN;independent_padded=2*H*sum(ceil(g['M']/16)*16*ceil(g['N']/16)*16 for g in groups)
        rect_pad=2*32*H*totalN;old_input=2*H*sum(g['M']for g in groups);joint_input=2*H*24
        models.append(dict(C1_source_branch=c1,groups=groups,H=H,
          ragged_output_scalars=scalar,rectangular_global_output_scalars=24*totalN,
          useful_classical_ops=ops,global_rectangular_classical_ops=rect,global_rectangular_work_ratio=str(Fraction(rect,ops)),
          independent_M16_N16_product_reference_ops=independent_padded,global_M32_rectangular_reference_ops=rect_pad,
          pad_ratio_is_not_actual_issue_or_universal_lower_bound=True,
          distinct_weight_value_bytes=weights,SWA_weight_counted_once=True,
          independent_input_read_value_bytes=old_input,one_global_hidden_read_reference_bytes=joint_input,
          conditional_input_read_difference_bytes=old_input-joint_input,
          incremental_QA_input_read_difference_after_existing_SWA_C1_join_bytes=2*3*H,
          AIC_to_AIV_BF16_GM_write_value_bytes=2*scalar,AIC_to_AIV_BF16_GM_read_value_bytes=2*scalar,
          AIC_AIV_logical_bridge_total_value_bytes=4*scalar,
          separate_AIC_AIV_bridge_still_present_in_one_kernel=True,
          one_initial_L0B_grant_bytes=warm,L1_to_L0B_fixed_orientation_rate_upper_bytes_per_cycle=B,
          L1_path_fixedB_conditional_min_cycles=ceil(max(0,weights-warm)/B),
          useful_BF16_arithmetic_conditional_min_cycles=str(Fraction(ops,P)),
          combined_matrix_and_fixedB_cut_cycles=max(Fraction(ops,P),Fraction(max(0,weights-warm),B)).__ceil__(),
          shared_rate_bound_rule='MaxcomputeandL1_Bcuts. GMwrite/readneedvalidseparateorprovensharedresourcecapacityupper;logicaltotalcannotbechargedtounprovedHBMbandwidth.',
          domain='24AICreference;classicaldenseBF16namedproducts,mandatorydistinctuncompressedweightplanes,noprune/dedup/compressedweightreuse,fixedB L1→L0path,onlyoneinitialL0Bresidentgrant,noimportedcomputedpartials. RaggedJmap/queryhidden==globalhidden[J],sameweight/round/rowgenerationrequired. Notallalgorithms,nativepathorwallclockproof.',
          row_ready_contract='PerQArowall1280complete-roundedoutputsneededforitsRMS;perSWA/C1row512respectively. Noall-rows/all-branchesbarrierisnecessarysolelyfromRMS. Keepnormparametersandfanoutseparate.',
          actual_HBM_saving_or_speedup=None))
    write(O/'projection-panel-work-and-resource-models-v99.json',dict(models=models,
      prior_naive_QA_KV_8over3_domain='Alreadyv61;thisturnaddsC1raggedtriple/oneunionandRMSreleasecost,notrediscovery.',
      union_rule='F06andF33sharetheSWAproducer. Product/weights/bridgeunioncountsSWAonce. Inputreadcredits30720+245760canonlybecomposedagainstoneexplicitthree-producerbaseline;donotaddpairwhole-costboundsorchargeSWA twice.',
      existing_norm_quant_models='norm-quant-interface-and-resource-models-v94.json;QA→normbridgeisoutsideitsqr→quantbudget. INT8Vector→CubeGMbridgealreadyretainedthere;nosamedoublecredit.',
      actual_native_path_or_numeric_physical_ceiling_proved=False,overall_complete=False))

    # Counterexample: equalper-K-panelnorms do not determine the completed projection's row norm.
    p=[1,0];plus=[1,0];minus=[-1,0]
    witness=dict(p=p,q_plus=plus,q_minus=minus,per_K_partial_square_sums=[sum(x*x for x in p),sum(x*x for x in plus)],
      same_partial_square_sums_for_minus=True,completed_plus_square_sum=sum((x+y)**2 for x,y in zip(p,plus)),completed_minus_square_sum=sum((x+y)**2 for x,y in zip(p,minus)))
    write(O/'projection-panel-reduction-and-precision-v99.json',dict(K_panel_counterexample=witness,
      K_domain='sumsq(sum_kpart z_kpart)=sum_part sumsq(z_part)+2sum_cross dot(z_part_i,z_part_j). PartialKnorms alone discardcross terms. Notaproofallalgebraicnormmethodsareimpossible.',
      N_domain='AftercompleteKandBF16roundforeachoutputlane,sum_N z_j^2=sum_Ntile partial_sum(z_j^2) inexactarithmetic;preserveoriginalFLOATreduction/rsqrt/epsilon/tolerancedomainbeforenativeparityclaim.',
      per_group_stats='QA1280,SWA512,C1512eachhasownstatistic. Joint1792/2304norm orone1024SWA/C1norm changesfunction;priorv27independent512counterexampleapplies.',
      second_dependency='NormalizedBF16qr then DynamicQuantamax/scale remainsasecondrowdependency. v94BF16[42,43]counterexampleblocksunqualifiedRMSscalarcancellation/FP32directquant. Reusingroundedqrquantrequiresallreaders/quantizersidentical.',
      row_buffer_or_recompute='Alegalepilogue mayretaincompletedroundedrowtiles orreread/recompute afterstats;theircapacity/GM/compute costs mustbeincluded. Storagechoicecannotdeleteaninformationdependencyforfree.',
      float_rounding_and_alias_all_reader_proof=False,overall_complete=False))

    counter=Counter();witnesses={};count=0
    for a,b,c,d,e in itertools.product([0,1,2,4,8,16,32,64],repeat=5):
        s=schedule(a,b,c,d,e);old=s['original_stage_join'];keep=s['joint_projection_whole_output_keep_stage_join'];early=s['joint_projection_whole_output_release_KVnorm_early'];panel=s['QA_panel_release_and_KVnorm_early']
        assert keep-old==min(b,c)and panel<=old and panel<=early
        relation='worse'if early>old else'better'if early<old else'equal';counter[relation]+=1;count+=1
        if all(x>0 for x in [a,b,c,d,e]):witnesses.setdefault(relation,dict(durations=dict(QA=a,KV=b,Qnorm_quant=c,QB=d,KVnorm_rope=e),completion=s))
    write(O/'projection-panel-release-DAG-models-v99.json',dict(cases=count,whole_output_early_KVnorm_comparison=dict(counter),witnesses=witnesses,
      definitions=dict(a='QA',b='KV',c='Qnorm+quant',d='QB',e='KVnorm+RoPE/publication'),
      formulas=dict(original='a+max(b,c)+max(d,e)',joint_keep_stage_join='a+b+c+max(d,e)',joint_early_KVnorm='a+b+max(c+d,e)',panel_early_KVnorm='a+max(max(b,c)+d,b+e)'),
      domain='Idealunchangedservicetimes/singleCubeQA→KV→QB;independentorunlimitedVectorresource relaxation,allinputsreadyatstart,nowire/aliascost. Notmeasuredfusedtimesorvalidcapacitylowerbound. C1serviceanditsreadersareseparateadditionalwork.',
      conclusions=['Keepingoldstagejoinwithwhole-projectionoutputbarrierlosesmin(b,c)inthismodel.',
        'ReleasingKVnormearliercanmakewhole-outputjoinbetterorevenworse;fusionisnotuniversallyfasterorslower.',
        'PublishingcompletedQArowpanelsrestoresitsshorterdependencywithoutrequiringallKV/C1rowsready;thisisacandidateinterface,notanimplementednativehandoff.'],
      actual_speedup_or_achievable_schedule_proved=False,overall_complete=False))
    pairs=[('F05','F06','QA_PRODUCER_AND_NORMALIZATION_PANEL_READY','Ragged/localQAproduceroutputsremainBF16andfull1280-rowRMS/quantdependencies;fulljointproducerbarriercanloseexistingoverlap.'),
      ('F06','F08','SWA_PRODUCER_SHARED_WITH_KV_COMPOSITE','CountoneSWAprojection/bridgewithitsindependent512norm/RoPE/publication;allowseparaterowpanelreleasewithoutinventingsameQA/KVrows.'),
      ('F06','F33','RAGGED_QA_SWA_C1_PRODUCER_UNION','InC1mode,threeoutputgroupsandoneSWAproducer;localQA3/globalSWA24/globalC124withindependentnorms,weightsandreaders. Oneunioninput/weight/resourcebaseline.'),
      ('F18','F33','C1_C2_MODE_DISTINCT_NORM_NO_GENERIC_ATOM_CSE','F18isC2pool→norm,F33C1jointprojection→norm. Genericlatent.norm512didnotproveonecurrentoperator. Splitmode-qualifiedatoms;preserveC2pool/FLOATandC1directBF16branches. Bothmodescanoccuratdifferentlayers;notwhole-modelmutualexclusion.')]
    evidence=['projection-panel-source-contract-v99.json','projection-panel-native-frontiers-v99.json','projection-panel-work-and-resource-models-v99.json','projection-panel-reduction-and-precision-v99.json','projection-panel-release-DAG-models-v99.json','projection-panel-mode-atom-audit-v99.json']
    write(O/'projection-panel-composition-clauses-v99.json',dict(clauses=[dict(id=f'REL99-{i:02d}',a=a,b=b,relation=kind,operator_claim=claim,evidence=evidence,
      applicable_domain='Typedrow/weight/generation/source-modeandprecisioncontracts;actualCANN9.2nativehandoff/alias/physicalcapacitynotclosed.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None)for i,(a,b,kind,claim)in enumerate(pairs,1)],
      extend_group='HG75-Q-KV-JOIN',new_members=['F05','F18','F33'],new_groups=0,new_main_candidates=0,new_native_promotions=0,overall_complete=False))
    print(json.dumps(dict(frontiers=len(fronts),TASKs=len(used),resource_models=len(models),DAG_cases=count,DAG_relation_counts=dict(counter),new_pairs=4,new_groups=0,overall_complete=False)))

if __name__=='__main__':main()
