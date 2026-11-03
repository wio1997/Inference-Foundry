from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib,json,re
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl
from finalize_F33_consumers_v27 import graph_check


def receipt(label,embedded=False):
    r=read(N/(label+'.json'));assert r['returncode']==0,label
    d=json.loads(r['stdout'])
    if embedded:
        b=(json.dumps(d['result'],ensure_ascii=False,separators=(',',':'))+'\n').encode()
        assert hashlib.sha256(b).hexdigest()==d['result_sha256'],label
    return d


def main():
    assert not(O/'REVIEW-CHECKPOINT-v28.json').exists(),'refuse duplicate finalization'
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    front=read(O/'whole-coverage-frontier-audit-v28.json')
    assert front['summary']['missing_module_candidate_links']==83
    assert front['summary']['phase_matrix_modules_with_missing_candidate_links']==24
    assert front['summary']['phase_branch_records']==210
    p=receipt('sampler-profile166-v28');s=receipt('sampler-source166-v28-v2',True)
    oldraw=receipt('rejection-raw-consumer166-v28',True);raw=receipt('rejection-raw-consumer166-v28-v2',True)
    assert read(N/'sampler-source166-v28.json')['returncode']==1
    ps=p['summary'];assert(ps['selected_native_APIs'],ps['selected_TASKs'],ps['selected_fullchains'])==(840,1144,14)
    assert ps['ambiguous_contexts']==[] and ps['context_API_counts']=={'TARGET_SAMPLER_CALLER':840}
    opcounts={r['opType']:r['TASKs']for r in ps['context_TASK_op_counts']}
    assert opcounts=={'Cast':480,'ArgMaxV2':160,'IndexCheck':144,'Index':144,'TensorMove':72,'Fill':72,'rejection_greedy_sample_triton':72}
    defs=raw['result']['entropy_guard_definitions']
    assert len(defs)==1 and 'enable_entropy_verify'in defs[0]['expression']
    uses=raw['result']['argument_uses']
    assert [u['line']for u in uses]==[461,462]
    assert uses[0]['ancestor_guards'][0]['branch']=='test'
    assert oldraw['result']['argument_uses'][0]['ancestor_guards'][0]['branch']=='orelse'
    refs=[dict(label=label,receipt=str(N/(label+'.json')),full_result=obj['full_result'],sha256=obj['result_sha256'])
          for label,obj in [('sampler-profile166-v28',p),('sampler-source166-v28-v2',s),('rejection-raw-consumer166-v28-v2',raw)]]
    write(O/'M17-sampler-profile-contract-v28.json',dict(generated_utc=stamp,evidence_grade='MEASURED_NATIVE_FULLCALLER',
        summary=ps,chain_headers=p['chain_headers'],evidence_refs=refs,
        scope='Selected actualsampler/rejectionfileancestry,all32 savedDBparts. Doesnotreplace vocabularyproducer/CPUmetadata/proposercontexts.',
        branch_value_limits='Argmax/rejectionlaunch positivelyproves not-all-random branch andHAS_TRITON for those calls. It doesnot prove all_greedy=True;mixedsampling alsoexecutes greedyargmax.',
        whole_module_closed=False,overall_complete=False))
    source_contract=dict(generated_utc=stamp,evidence_grades=['MEASURED','DOCUMENTED','DERIVED','UNKNOWN'],
        source_files=[dict(path=f['path'],sha256=f['sha256'])for f in s['result']['files']],source_receipts=refs[1:],
        reviewed_facts=[
            'Worker_sample:2977-2994 dispatches normal Sampler when spec_decode_metadata is None;otherwise obtains draft_probs and calls rejection_sampler. It updates async token IDs first and respects lmhead-TP row trimming.',
            'Rejectionforward:192 selects bonus rows;:193-203 forces max_num_logprobs=-1 and overrides mode to processed_logits or raw_logits. Usernotrequestinglogprobs doesnot alone mean thisinternal tensor wasnevergenerated.',
            'Base Sampler.forward:87-94 captures requested raw logits/logprobs beforeprocessors;:97 makes an FP32 workingtensor;:99 appliesprocessors;:103 samples. Optionalspecific-token/topk/full-output consumers andprocessed-logit modes remain.',
            'In this rejectionforward subtree,bonus_sampler_output.logprobs_tensors.logprobs is consumed at:247 only inside original sampling_metadata.max_num_logprobs is not None. The bonusIDs are independently live.',
            'Rejectionforward:209 selects target rows,:211 promotes toFP32,:217 clones when notis_processed_logprobs_mode to preserve raw values beforeinplaceprocessors/constraints. Exactnative217 clone ispositivelyrecorded72times.',
            'Target raw logits also flow through ori_target_logits into rejection_sample. Its onlydata load is:462 softmax under using_entropy_verify and ori_target_logits is notNone;using_entropy_verify is Ascend rejection_sampler_config.enable_entropy_verify at:436. No-final-logprobs alone doesnot make rawtarget dead.',
            'Base sample:261 computes greedy argmax whenever notall_random. It returns early at:262-272 onlywhen all_greedy;otherwise continues temperature,argmax-invariant processors,top-k/p andmixing. Greedy nativecalls cannot establish all_greedy bythemselves.',
            'Native rejectionlaunch is guardedbynotall_random andHAS_TRITON;wrapper443 has a special uniformspec_len1/is_greedyNone path,whileactualgeneric456pathisobserved. Preserve raggedcu_num_draft_tokens,per-rowis_greedy,syntheticRNG/rates andprobability alternatives.',
            'Sampler.forward:110 promotesIDs tolong foroptional index consumers,then:140 returnsint32;PyTorchargmax alreadyreturnsint64 whileCANNArgMaxV2 TASK outputsint32 anditslowering addsacast. Removingcastsrequiresretainingactualconsumers/ABI.',
            'Rejectionoutput:467-472 allocatesint32[B,max_spec_len+1] andfillsPLACEHOLDER beforethekernel. Fusinginitializationrequireswritingeveryobservableoutputslot,not leavingtailuninitialized.'],
        answered_local_questions=['Actualnative samplingbranches/typed staging','Forcedbonusrawscoregeneration anditsdirectgetter','Targetrawclone/rawmode andentropyconsumer','Greedy-vs-mixed guard distinction','OptionalID-index/logprob andplaceholderoutput consumers'],
        unresolved=['Actualper-call originalmax_num_logprobs/logprob_token_ids,all_greedy/all_random,processorconstraints andentropy flag values are notfully attested bydevicecalls.',
            'Non-greedybody of305-line rejectionhelperwasnotfullyread;itremainsaseparate explicitbranchgap.',
            'Historicalrowindex/alias/storage/producerreadiness andnumericalNaN/tie policy fornewreductions.',
            'SamplingCPUcost,currentcriticalpath andfutureconsumer generation/lifetime;noimplementationorNPUexperiment.'],
        source_quality='Firstwholemethodexcerptstoppedat260-lineguard. v2usesnamedmethods plusnativecontexts/ancestorconditions forlonghelper. Rawconsumer v1mislabelledloadinIf.testasorelse;v2correctedtest/body againstsourcewithoutchangingfileSHA.',
        stop_rule='Do not expand whole sampler/repository to strengthen alreadyconditionaldataflow. Nextprofilefrontier is orphanF04 dual-outputFFNnorm/cast roles,or a specificallydecisive unresolvedsamplingflag.',
        implementation=False,whole_module_closed=False,overall_complete=False)
    write(O/'M17-sampling-source-and-liveness-v28.json',source_contract)
    # Source-refined analysis graph. Flags select consumers; no new executable implementation.
    nodes=['vocab_logits','bonus_select','bonus_raw_scores','bonus_working_FP32','bonus_processors','bonus_argmax','bonus_ID64','bonus_ID32',
           'target_select','target_raw_FP32','target_clone','target_processors_constraints','target_argmax','entropy_probs','output_fill','rejection','final_logprobs','next_iteration']
    edges=[('vocab_logits','bonus_select','RAW'),('bonus_select','bonus_raw_scores','FORCED_RAW_OR_PROCESSED_LOGIT_RETURN'),
           ('bonus_select','bonus_working_FP32','RAW'),('bonus_working_FP32','bonus_processors','INPLACE_EFFECT'),('bonus_processors','bonus_argmax','RAW'),
           ('bonus_argmax','bonus_ID64','ABI_CONVERSION'),('bonus_ID64','bonus_ID32','ABI_CONVERSION'),('bonus_ID32','rejection','RAW'),
           ('vocab_logits','target_select','RAW'),('target_select','target_raw_FP32','RAW'),('target_raw_FP32','target_clone','RAW_RAW_MODE'),
           ('target_clone','target_processors_constraints','INPLACE_EFFECT'),('target_processors_constraints','target_argmax','RAW'),
           ('target_argmax','rejection','RAW'),('output_fill','rejection','OBSERVABLE_PLACEHOLDER_INIT'),
           ('target_raw_FP32','entropy_probs','CONDITIONAL_ENTROPY_VERIFY'),('entropy_probs','rejection','CONDITIONAL_PROBABILITY_PATH'),
           ('bonus_raw_scores','final_logprobs','CONDITIONAL_ORIGINAL_MAX_LOGPROBS'),('target_raw_FP32','final_logprobs','CONDITIONAL_RAW_LOGPROBS'),
           ('bonus_ID64','final_logprobs','CONDITIONAL_INDEX_CONSUMER'),('rejection','final_logprobs','CONDITIONAL_SELECTED_IDS'),('rejection','next_iteration','ID_AND_ACCEPTANCE_STATE')]
    graph=dict(nodes=[dict(id=n)for n in nodes],edges=[dict(a=a,b=b,kind=k)for a,b,k in edges]);graph_check(graph)
    variants=[
        dict(id='F35-A',status='CONDITIONAL_LOCAL_DATAFLOW_OPPORTUNITY',change='Generate/return fullbonus raw/processedscoretensor onlywhen itscallerhasaliveconsumer.',
            requires='Originalmax_num_logprobs=None,nootherrequestedscoreconsumer/errorobservable;preservebonusprocessors,predict_bonus_token behavior andselectedID. Rejectioncaller currentlyforcesinternal-1evenwhenitsfinalgetterisnotused.',
            excludes='Do not turnoffbonus token sampling orrequestedlogprobs;HTTPomissionisnotruntimeflagattestation. Inprocessedmode returnedscoremayaliasliveworkinglogits,soavoidingreturnaloneisnotamatrix-copy saving.'),
        dict(id='F35-B',status='CONDITIONAL',change='Avoid targetraw-preservation clone when nofinalraw-scoreconsumer andentropy verificationdoesnotreadoldvalues.',
            requires='Noaliveoriginalrawlogprobconsumer,enable_entropy_verify=False orseparatelypreservedentropyinput;workingFP32processors/constraintsremainexact.',
            excludes='No-final-logprobs aloneis insufficient. Do notchange entropy/synthetic/probabilitysemantics.'),
        dict(id='F35-C',status='CONDITIONAL_FUSION_AND_REORGANIZATION',change='Fuse requiredrowselection,allowedpromotion andgreedyreduction withoutmaterializingeachintermediatematrix.',
            requires='Exactbonus/targetrowmapsincludingduplicates/raggedcases;allprocessors/masks retainFP32arithmetic andorder;anyscoreconsumers retained. BF16-onlyreductiononlyinaprovedorder-preservingdomainwithsameNaN/tie/indexrules.',
            excludes='Do notblindlydropfloatconversionacrosspenalties,scaling,rawlogprobs orentropyconsumers.'),
        dict(id='F35-D',status='CONDITIONAL',change='Retainint32IDs end-to-endwherelong-onlyindexconsumersareinactive;combineplaceholderinitializationwithrejectiononlyifeveryobservedslotiswritten.',
            requires='VocabIDsfitint32,allindex/ABIconsumerspreserved;exactfirstrejection,bonusplacement,raggedlengths,per-rowgreedymask,PLACEHOLDER andacceptanceupdates.',
            excludes='Do notleaveunusedtailuninitialized orskiprequiredint64 indexing.'),
        dict(id='F35-E',status='RULED_OUT_AS_UNCONDITIONAL_DELETION',change='Removeallrawscores/clones becauserequesttemperatureis0 orlogprobsomitted.',reason='Internalbonusoverride andoptionalentropy/processor/logprobconsumers distinguishthischange.')]
    f35=dict(id='F35',name='目标采样的按需raw logits、行选择/转换/argmax与拒绝输出重组',
        modules=['M16','M17','M21','M24'],profile_review_state='840_NATIVE_APIS_1144_TASKS_AND_DIRECT_SCORE_CONSUMERS_BOUND;CONDITIONAL_UNPRICED',
        original_node_mapping=dict(bonus='rejection_sampler.forward192-204;Sampler.forward87-150',target='rejection_sampler.forward209-249',greedy='AscendSampler84;rejection_sample510/513',raw_entropy='rejection_sample436/461-464'),
        candidate_graph=graph,graph_scope='Partialsource-refinedraw-mode branchgraph;notpointer/event-complete,notanadoptedjointgraphforallvariants.',variants=variants,
        profile_evidence=ps,source_contract='M17-sampling-source-and-liveness-v28.json',evidence_root=str(O),
        evidence_files=['M17-sampler-profile-contract-v28.json','M17-sampling-source-and-liveness-v28.json','F35-sampling-condition-ledger-v28.json'],
        relation_to_existing=[dict(candidate='F30',relation='overlaps',scope='F30draftMarkov/vocabownershipmayreorganizescores;F35targetbonus/rejectionstaging isdistinctbutjointownershipsavingsnotadditive.'),
                              dict(candidate='F31',relation='interacts',scope='Raggedspecmetadata,outputIDs,acceptedcounts andnextstep publicationmustremainconsistent.')],
        all_consumers_closed=False,legal_fusion_proved=False,candidate_closed=False,critical_path_gain_us=None,
        critical_path_gain_status='UNKNOWN_NOT_SUM_OF_TASK_TIMES_OR_LOGICAL_ARRAY_BYTES',implementation=False,new_performance_result=False,overall_complete=False)
    conditions=[
        ('native_population','MEASURED','840nativeAPIs/1144TASKs/14fullchains/7typedops,allactualtargetsamplercallers;not invocationcount.'),
        ('non_spec_vs_spec','DOCUMENTED_NATIVE_CALLERS','16API/site normalpath and72API/site rejectionpathrecorded;no cross-site same-call identity inferredfromequalcounts.'),
        ('greedy_guard','DOCUMENTED_POSITIVE_GUARD','greedyargmaxmeansnotall_random;all_greedy/mixed distinction remainsopen.'),
        ('bonus_raw_override','DOCUMENTED_AND_MEASURED','Internalmax_num_logprobs=-1 forcesbonusraw/mode output;native rawfloat94 sitesrecorded.'),
        ('bonus_raw_liveness','DERIVED_CONDITIONAL','Directcallergetsfullbonuslogits onlywithoriginalmax_num_logprobs!=None;originalflagandotherconsumersmustbeexplicit.'),
        ('target_raw_clone','DOCUMENTED_AND_MEASURED','72nativeclone217 prove raw-preservationbranch forrecordedcalls;not zero-copyabsence.'),
        ('entropy','DOCUMENTED_CONCLUSION_CHANGING_CONSUMER','ori_target_logits.softmax consumesrawscoreunderenable_entropy_verify;no-final-logprobs alonecannotjustifyclone removal.'),
        ('processors','RETAINED_UNKNOWN_FLAGS','KeepFP32processors/penalties/masks/constraints,predict_bonus_token differences andordering.'),
        ('score_output_modes','DOCUMENTED_RETAINED','Raw/processedlogits/logprobs,full/topk/specific-token branches remain;requestedoutputs cannotdrop.'),
        ('row_domains','MEASURED_PARTIAL','12/24/14inputrows map to2/4bonusrows and10/20targetrows. Preserveactualindices;14->4+10 warnsraggednotuniformB*5.'),
        ('dtype_and_ties','CONDITIONAL','BF16->FP32 isvalue-preservingforfinitevaluesbutFP32processorresults,NaN/tie/reduction/indexcontracts remain.'),
        ('IDs','DOCUMENTED_CONDITIONAL','Longintermediate servesoptionalindexconsumers;finalINT32ABI andallliveindicesretained.'),
        ('placeholder','DOCUMENTED_RETAINED','Fill[B,6] andgenericrejectionkernels observed;fusingfillrequiresallsentinel/tailwrites.'),
        ('probability_synthetic','EXPLICIT_UNREVIEWED_OR_RETAINED','Random/mixed/draft_probs/entropy/block/syntheticpathsnotremoved;longnon-greedybodynotfullyaudited.'),
        ('cross_iteration','PARTIAL','OutputtokenIDs/acceptedcountsfeednextstep;rollback,rowremapping andbuffer/eventgenerations remain.'),
        ('full_vocab_producer','CONDITIONAL_COMPOSITION','Targetvocabprojection/collectiveownership retainsallscoreconsumers;notprovedbygreedykernels alone.'),
        ('historical_identity','UNKNOWN','Deviceprofiles do notrecoveractualpointeralias/readiness orallCPUmetadataflags.'),
        ('gain','UNPRICED','No physicalHBM/network/criticalpathgain,NPUexperiment orimplementation claimed.')]
    write(O/'F35-sampling-condition-ledger-v28.json',dict(generated_utc=stamp,candidate_id='F35',conditions=[dict(id=f'F35-v28-{k+1:02d}',branch=x,state=y,evidence=z)for k,(x,y,z)in enumerate(conditions)],all_conditions_closed=False,overall_complete=False))
    write(O/'F35-target-sampling-candidate-v28.json',dict(generated_utc=stamp,candidate=f35,evidence_refs=refs,overall_complete=False))
    assert not(R/'candidates/F35.json').exists();write(R/'candidates/F35.json',f35)
    model=deepcopy(read(O/'candidate-profile-ledger-v27.json'));framework=deepcopy(read(O/'framework-candidate-ledger-v27.json'))
    mods=deepcopy(read(O/'module-profile-ledger-v27.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v27.json'))
    model['candidates'].append(f35);model['count']=35
    for c in model['candidates']:
        if c['id']in['F31','F32']:
            q=R/'candidates'/(c['id']+'.json');assert not q.exists()
            master=deepcopy(c);master.update(evidence_root=str(O),authoritative_record=str(O/'candidate-profile-ledger-v28.json')+'#'+c['id'],registry_note='Rehydratedexistingcandidatefornavigation;nonewprooforgain.',implementation=False,overall_complete=False);write(q,master)
    mm={m['id']:m for m in mods['modules']};bm={m['id']:m for m in matrix['modules']}
    new_links=[]
    for key,cid in [('M12','F04'),('M07','F32'),('M08','F32')]+[(key,'F35')for key in f35['modules']]:
        if cid not in mm[key]['candidate_links']:mm[key]['candidate_links'].append(cid);new_links.append(dict(module=key,candidate=cid))
    for m in mods['modules']:
        m['phase_branch_record']='module-phase-branch-matrix-v28.json#'+m['id']
        bm[m['id']]['candidate_links']=list(m['candidate_links'])
    notes={'M16':'F35targetscorestagingconsumerfrontierdistinguishedfromF30draftMarkov/vocabownership;noallvocabcommunicationremoval.',
           'M17':'Fresh840API/1144TASK/14fullcaller intake plusforcedbonusraw,rawtargetentropyconsumer andnot-all-random guard bound;remainingprobabilistic/metadataflags explicit.',
           'M21':'F35rowmaps12/24/14 andragged10/20targetrows requireactualspecmetadata;fill/IDs/acceptedcounts remain.',
           'M24':'F35outputplaceholders/acceptedtokenIDsfeednextiteration;noimplicitrollback/copylifetimeclosure.',
           'M12':'F04FFNnorm/castwasorphanedfrommodulecandidateinverseindex;restoreprimarylink,dualsourceconsumerreviewstillpending.',
           'M07':'RestoreF32inputgather/projectioncrossboundarylinkfromexistingv15/v16contracts;navigationrepairnotnewsemanticproof.',
           'M08':'RestoreF32C1/C2projectioninterfacecrossboundarylink;C2FP32andC1BF16domains remainseparate.'}
    for key,text in notes.items():
        mm[key]['review_v28']=dict(finding=text,evidence=['whole-coverage-frontier-audit-v28.json','M17-sampler-profile-contract-v28.json','M17-sampling-source-and-liveness-v28.json','F35-target-sampling-candidate-v28.json'],whole_module_closed=False)
        bm[key]['branches'].append(dict(branch='Wholecoveragefrontier/candidateinverseindex andtargetsamplerrawscore/greedybranches',state='SCOPED_REVIEW_OR_NAVIGATION_REPAIR;WHOLE_MODULE_OPEN',evidence=text,gap='Retainunreviewedbranch/flag/consumer/alias/event/resourcequestions.'))
    for obj,parent,dest in [(model,'candidate-profile-ledger-v27.json','candidate-profile-ledger-v28.json'),(framework,'framework-candidate-ledger-v27.json','framework-candidate-ledger-v28.json'),(mods,'module-profile-ledger-v27.json','module-profile-ledger-v28.json')]:
        obj.update(generated_utc=stamp,parent_ledger=parent,whole_modules_closed=0,overall_complete=False)
        if obj is model:obj['framework_candidate_ledger']='framework-candidate-ledger-v28.json'
        if obj is mods:obj.update(existing_model_candidates=35,framework_candidates=7)
        write(O/dest,obj)
    matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v27.json',
        history_scope_note='Branches span explicitold/currentcaptureversions andscopedreviews,notanexhaustiveconditionuniverse oronecurrentpopulation. Donotsumhistoricalcounts;useindividualevidencefiles. Candidate-links synchronizedfromcurrentmoduleledger.')
    write(O/'module-phase-branch-matrix-v28.json',matrix)
    for m in mods['modules']:assert set(m['candidate_links'])==set(bm[m['id']]['candidate_links'])
    allids={c['id']for c in model['candidates']+framework['candidates']};linked={c for m in mods['modules']for c in m['candidate_links']};assert allids<=linked
    registry=[]
    for c in model['candidates']:
        q=R/'candidates'/(c['id']+'.json');assert q.exists(),q
        registry.append(dict(id=c['id'],master_record=str(q),master_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),current_authority=str(O/'candidate-profile-ledger-v28.json')+'#'+c['id'],modules=[m['id']for m in mods['modules']if c['id']in m['candidate_links']]))
    write(O/'candidate-record-index-v28.json',dict(generated_utc=stamp,model_candidates=registry,framework_authority=str(O/'framework-candidate-ledger-v28.json'),unlinked_candidate_ids=[],whole_modules_closed=0,overall_complete=False))
    write(O/'coverage-link-reconciliation-v28.json',dict(generated_utc=stamp,existing_links_missing_before=front['missing_module_candidate_links'],existing_link_repairs=83,new_explicit_links=new_links,
        orphan_before=['F04'],orphan_after=[],rehydrated_master_records=['F31','F32'],new_candidate='F35',all_35_model_master_records_exist=True,
        structural_only_unknowns_retained=dict(nodes=39,edges=90,all_semantic_fields='UNKNOWN_IN_LEGACY_LEDGER'),
        current_graph_versions=front['graph_version_surfaces'],legacy_graph_scope_note='CoverageCSVmatchesbaseG1,notaux-contextG1. Neitherstructuralinclusionnorlinkrepairissemanticclosure.',overall_complete=False))
    remaining=deepcopy(read(O/'module-coverage-gaps-v27.json')['remaining'])
    remaining.insert(1,'v28frontieraudit:legacy623node/1029edgeledgerispartialhistoricalstructure,semanticfieldsallUNKNOWN;39node/90edge structuralunknownsretained,currentauxG1versionseparate. 83missingcandidate navigationlinks repaired,F04orphanlinkedM12,F31/F32masterrecordsrehydrated,35model/7framework nowindexed. Noneofthesearereasonsforwholemoduleclosure.')
    remaining.insert(2,'M17v28nowbinds840API/1144TASK/14fullchains,F35forcedbonusrawlogit/directgetter andrawtargetentropyconsumer. Actualoriginalmaxlogprobs/metadata/processor/entropyflags,fullnon-greedybody,aliases/generations remain. Do notinferall_greedyfromgreedyargmax:guardisnotall_random.')
    remaining.insert(3,'NextprofilefrontierF04:previouslyonlyRmsNormCastfamilyhits/scoped_detailNone,absentfrommoduleinverseindex. Querycurrentallranknative/fullcaller andgraphdualoutputshape/dtype plusrouter/sharedexpert consumers beforepointsource. Do notadvertisealreadyfusednorm/castasnewfusionorcastoneoutputaway.')
    write(O/'module-coverage-gaps-v28.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v27F33exact80orders/oldlatentWAR/20conditions',
        modules=26,original_phase_branch_rows=210,repaired_existing_candidate_links=83,new_explicit_links=len(new_links),rehydrated_masters=['F31','F32'],previously_orphan_candidate='F04',
        sampler_native_APIs=840,sampler_TASKs=1144,sampler_fullchains=14,new_model_candidate='F35',F35_conditions=len(conditions),
        forced_bonus_raw_logits_consumer_bound=True,target_raw_logits_entropy_consumer_bound=True,greedy_does_not_prove_all_greedy=True,
        model_candidates=35,framework_candidates=7,whole_modules_closed=0,overall_complete=False,
        no_new_model_NPU_service_or_health_work=True,all_jobs_terminal=True,evidence_refs=refs,remaining=remaining)
    write(O/'sampling-and-coverage-summary-v28.json',summary)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v28.json');gate.update(status='FRONTIER_PROFILE_AND_NECESSARY_POINT_SOURCE_REVIEWED',answered_by=['whole-coverage-frontier-audit-v28.json','M17-sampler-profile-contract-v28.json','M17-sampling-source-and-liveness-v28.json','F35-target-sampling-candidate-v28.json'],
        closed_scope='Navigationgaps/versionmismatch andselectedsamplingnativebranches/directrawscoreconsumers.',not_closed='Actualmetadataflags/fullnon-greedyprobability/allmodulegraph/consumercoverage.',stop_rule=source_contract['stop_rule']);write(O/'PROFILE-CODE-QUESTION-GATE-v28.json',gate)
    report='''本轮从整体覆盖表出发，核对26模块、34项原模型候选和7项框架候选，而非重复深化F33。发现210条阶段/分支记录混有不同采集版本和历轮局部审查，不能当作210个穷尽条件。24模块阶段表合计遗漏83条已有候选链接；F31/F32没有独立候选主记录；F04“FFN归一化与类型转换”在模块逆向索引中完全缺席。本轮同步83条导航链接，恢复F31/F32主记录，给F04恢复M12主链接并列为下一轮profile前沿，补F32的M07/M08边界链接。没有据此关闭任何语义范围。

旧结构表包含623节点行、1029边行，semantic_coverage全部UNKNOWN；39节点和90边还没有对应结构枚举覆盖。其指纹匹配336节点/685边的旧G1，而不匹配338节点/690边的aux-context G1。不同图版本、G5模板和跨模板边有重叠，不能相加为唯一机会数；旧结构表保留原状，新版本审计与未知单独登记。候选存在和结构枚举都不能代替全消费者、动态分支及允许改写规则审查。

具体取证转向此前证据较薄的M17。全部32份保存DB分片中，原生sampler/rejection祖先筛选得到840个API、1144个TASK、14条完整调用链，7种算子：Cast480、ArgMaxV2 160、IndexCheck/Index各144、TensorMove72、Fill72、rejection_greedy_sample_triton72。它们是实际target采样caller，不混入draft Markov采样。API/TASK数不是sampler调用次数；跨site数量相同也没强行配成同一次调用。行域包括12/24/14行完整logits、2/4行bonus、10/20行target，14→4+10说明不能一律按B×5处理。

沿这14条调用链定点读实际方法后，补齐重要消费者：RejectionSampler为bonus内部强制max_num_logprobs=-1，并选择raw_logits或processed_logits返回。Base Sampler先保存raw score，再建立FP32工作tensor、执行processors和采样。其调用方仅在原始max_num_logprobs非None的最终logprobs分支读取bonus整词表score。与此同时，target行FP32提升后，在raw mode会clone来保留原值；72个clone native记录正命中这条分支。原始target logits既可能供最终raw logprobs，也传给rejection helper；后者在enable_entropy_verify时softmax原始值。因此不返回客户端logprobs，不能单独推出target raw clone无用。

另纠正一个采样判据：base sample的greedy argmax与rejection greedy路径都可在not all_random时执行，混合采样同样可能命中；all_greedy为True才会提前返回。因此160条ArgMax和72次greedy rejection不能单独证明所有请求/阶段都是纯greedy。special uniform spec_len1分支、generic Triton分支、mixed/random、entropy/block/synthetic验证、draft_probs、logprob_token_ids和processor flags均明确区分。原始HTTP temp0仍是输入合同，不冒充每次resolved metadata快照。

新增F35：目标采样raw logits按需生成，以及行选择/转换/argmax/token-ID/拒绝输出的重组。A分支在最终无live score消费者时避免内部强制生成整bonus raw/processed score；B分支只有排除最终raw消费及entropy原值读取后，才研究去掉target保存clone；C分支合并行选择、必要类型提升和归约，保留FP32 processor算术、ragged索引、NaN/tie规则及其他score输出；D分支在无long-only index消费者时保留INT32 ID并研究与placeholder初始化重组，但必须写全所有可观察尾部并保持第一次拒绝/bonus/accepted-count语义。无条件删除raw scores已排除。它与F30的draft Markov/词表所有权、F31的metadata/下一步发布有关联，收益不能叠加计算。

首版源码摘录在跨度305行的rejection_sample方法触发行数保护并停止，原失败receipt保留；修正版只读方法入口、原生472/510/513上下文、完整短all_greedy body及控制祖先，未扩成整个概率实现。另一个参数用点工具曾把if测试自身读误标为orelse；源码复核后修成test，保留旧记录与修正说明。新查询只补enable_entropy_verify定义和已定位符号用点，没有读整个SDK/模型。

F35记录18项分支条件，模型候选变为35项、框架仍7项。35个模型主记录均可索引，所有候选均有模块导航；阶段表与模块表链接已一致。目标代码、服务及NPU未改动，没有新增健康检查，最后健康证据仍v12post。本轮全范围覆盖仍未闭合，26整模块均保持open。下一批先查F04的当前native/graph双输出norm/cast与router/shared消费者，避免把已经存在的复合算子重复报告为新融合。
'''
    (O/'PROFILE-FINDINGS-v28.md').write_text(report)
    pub=R/'publication-draft/FUSION-INVENTORY.md';(O/'FUSION-INVENTORY-before-v28.md').write_bytes(pub.read_bytes());text=pub.read_text()
    assert not any(line.startswith('| F35 |')for line in text.splitlines())
    lines=text.splitlines();pos=max(k for k,line in enumerate(lines)if line.startswith('| F34 |'))+1
    lines.insert(pos,'| F35 | 目标采样raw logits按需生成、行选择/转换/argmax与拒绝输出重组 | 条件 | 保留bonus、原始/processed logprobs、entropy、processor、ragged行映射、INT32/INT64索引及placeholder/accepted-count消费者；无量化收益。 |')
    lines=[line.replace('当前持续分析台账（v25）','当前持续分析台账（v28）').replace(str(O/'candidate-profile-ledger-v25.json'),str(O/'candidate-profile-ledger-v28.json')).replace(str(O/'PROFILE-FINDINGS-v25.md'),str(O/'PROFILE-FINDINGS-v28.md'))for line in lines]
    lines += ['','v28持续分析记录：35项模型候选、7项框架候选；表中旧G1/32项枚举统计仅属历史范围，未证明当前全图闭合。当前证据见 '+str(O/'sampling-and-coverage-summary-v28.json')+'。']
    pub.write_text('\n'.join(lines)+'\n')
    action=('继续Loop004，v28实质进展。先读sampling-and-coverage-summary-v28.json、whole-coverage-frontier-audit-v28.json、coverage-link-reconciliation-v28.json、M17-sampling-source-and-liveness-v28.json、module-coverage-gaps-v28.json。'
        '26模块旧210branchrows含跨版本/历轮scope，不是条件宇宙。24模块83missingnavigationlinks已同步，F04此前完全orphan已加M12主链接；F31/F32主record复原；F32加M07/M08；35model/7framework全可索引。'
        '旧coverage623nodes/1029edges semantic全UNKNOWN，39node/90edge结构unknown，匹配旧G1(336/685)不匹配auxG1(338/690)。别用候选数/枚举数冒充semantic/allconsumer穷尽，不合并重叠图版本计唯一机会。'
        'M17完整savednativecaller840API/1144TASK/14chains；Cast480,ArgMax160,IndexCheck144,Index144,TensorMove72,Fill72,greedyrejection72。全targetsample，不含draftMarkov；数量非invocations、不按相等数配call。'
        '新F35条件：bonus内部强制maxlogprobs=-1，但fullbonusrawscore只有原始maxlogprobs!=None最终getter读；目标rawclone217实际72，除finalrawlogprobs还给entropyverify436/461-462softmax，no-logprobs alone不够删clone。'
        'Sampler.forward73–150, sample244–303 SHA f95805b2c6d06218c724826aa5f881a606e8d0a534a566aee7c85948520dd216；Ascendgreedy83–84 SHA867f1770614bb594baf5ecfb858aa4a20dc413cbfbe4f94e06b3786900b581ee；rejectionSHA bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda。'
        'Greedyargmax/rejectionnativepositive只证notall_random/HAS_TRITON，不证明all_greedy，mixed也可走。保留maxlogprobs/logprob_token_ids/FP32processors/entropy/block/synthetic/RNG/draft_probs/actualraggedindices和ID-ABI/sentinel/acceptedcount。18conditions未priced，F35与F30/F31关联不叠加收益。'
        'M/samplerprofilev28 SHAfb126dbf4d4fd99b24ae785cf191ff63d8e6da73233d680bc4adb3aa6633eea6；samplersourcev28v2 SHA9673d6aa9e46a814a8685e7dc118bf7ff3d8828634c7eed7a939f2602eb7c9f3；rawconsumer correctedv2 SHA8f2449eb10bf7a98b358c780ddba7842bc9cc2d67feb05562348643be5d1c2fa。'
        '首版305行helper触260guard错误保留，v2只native局部/greedybody/ancestor，non-greedy完整body未审；rawguard v1test错标orelse已修test，旧artifact保留。'
        '下一批优先F04：目前仍仅RmsNormCastfamilyhits、scoped_detailNone，刚修orphan。用当前E nativefullchains/graph80×40block dualoutput shapes/dtypes及router/shared expert消费者先辨已融合norm/cast，不直接猜新fusion或砍BF16/FP32输出；必要再点对应norm helper与modelFFN入口。'
        '继续用有限模块/条件/原图边覆盖frontier补未审范围，不重复F33/H07/F34已回答局部问题。全模块仍0闭合，3136candidate PMU/849missing/16unboundgraphs/26theoryINCOMPLETE保留。'
        'CPU关键缺口诊断仍需既有授权freshhealth/ownership/frozenworkload/clone/restore。此次无NPU/model/service/health，lastv12post，Goalactive整体未完成。')
    jobs=[('review','pass','Local26module/210branch/34candidatefrontieraudit identifies83missinglinks,2missingmasters,legacygraphfingerprint/structuralunknowns;notsemanticclosure.',[N/'audit_whole_coverage_frontier_v28.py',O/'whole-coverage-frontier-audit-v28.json']),
          ('profile','pass','All32savedDBpartnativecaller query recovers840samplerAPIs/1144TASKs/14chains withrawidentities andscope limits.',[N/'query_sampler_branch_surfaces_v28_166.py',N/'sampler-profile166-v28.json']),
          ('review','error','Firstnamedsourceexcerpt stopsat260-lineguard on305-linehelper;retainoriginalscript/receipt ratherthanbroadenwholemethod.',[N/'read_sampler_native_branch_contract_v28_166.py',N/'sampler-source166-v28.json']),
          ('review','pass','Correctednamedsourcequery boundslargehelper toactualnativecontexts/shortgreedybody/controlancestors;fullprobabilitybody staysopen.',[N/'read_sampler_native_branch_contract_v28_166_v2.py',N/'sampler-source166-v28-v2.json']),
          ('review','pass','Initialori_target_logits usequery recoversentropyconsumer butif-testguardlabeldefectisexplicitlyretained andsuperseded byv2.',[N/'read_rejection_raw_logits_consumers_v28_166.py',N/'rejection-raw-consumer166-v28.json']),
          ('review','pass','Correctedtest/bodyclassification andentropyconfigdefinition boundonlybyknownidentifiercontexts;no model import.',[N/'read_rejection_raw_logits_consumers_v28_166_v2.py',N/'rejection-raw-consumer166-v28-v2.json']),
          ('review','pass','MainregistersF35/18conditions,forcedbonusraw/directentropyconsumer andmixed-vs-greedy distinction;repairsnavigation/masters withoutclosingmodules.',[O/'M17-sampling-source-and-liveness-v28.json',O/'F35-target-sampling-candidate-v28.json',O/'F35-sampling-condition-ledger-v28.json',O/'coverage-link-reconciliation-v28.json',R/'candidates/F31.json',R/'candidates/F32.json',R/'candidates/F35.json',pub,O/'PROFILE-FINDINGS-v28.md'])]
    runs=[]
    for k,(kind,status,desc,arts)in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-k)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv28retrospectiveregistration: '+desc,'--code-ref','Savedprofile/exactsamplingmethods;targetimplementationunchanged/noNPU/model/servicework','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',desc]
        for q in arts:assert q.exists(),q;args+=['--artifact',str(q)]
        ctl(*args);runs.append(dict(run_dir=run,status=status,description=desc))
    cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],
        current_verified_changes=['26modulefrontier/versionaudit repairs83existinglinks,F04orphan,F31/F32masters;all35modelrecordsindexedwithoutsemanticclosure.',
            '840nativeAPI/1144TASK/14samplingchains andactualforcedbonusraw/directentropyconsumer bound;greedy!=all_greedyguard distinction corrected.',
            'NewF35 with18conditions andpartialdataflowgraph preservesprocessors/logprobs/entropy/rowmaps/ID-ABI/placeholder consumers.'],
        summary=str(O/'sampling-and-coverage-summary-v28.json'),candidate_ledger=str(O/'candidate-profile-ledger-v28.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v28.json'),module_ledger=str(O/'module-profile-ledger-v28.json'),branch_matrix=str(O/'module-phase-branch-matrix-v28.json'),
        report=str(O/'PROFILE-FINDINGS-v28.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v28.json'),coverage_gaps=str(O/'module-coverage-gaps-v28.json'),pending_full_scope_domains=remaining,
        remote_job_status='Alljobs terminal:1localfrontierauditpass,4remoteCPUjobs pass/1source-sizeerrorretained;mainreviewpass.NoNPU/model/service/healthwork.',
        model_candidates=35,framework_candidates=7,next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v28.json',cp)
    for q in[L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(q,d)
    for q in[L/'status.md',T/'status.md']:
        q.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in q.read_text().splitlines())+'\n')
    (O/'profile-summary-before-v28.json').write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_sampling_coverage_v28']=dict(summary=cp['summary'],model_candidates=35,framework_candidates=7,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for q in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if q==R/'STATE.json'else'next_action']=action
        if q==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if q in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'M17-sampling-source-and-liveness-v28.json'),str(O/'coverage-link-reconciliation-v28.json'),cp['coverage_gaps']]
        write(q,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v28.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    arts=[q for q in O.glob('*v28*')if q.is_file()and not q.name.startswith('checkpoint-manifest')]
    arts += [N/(label+'.json')for label in['sampler-profile166-v28','sampler-source166-v28','sampler-source166-v28-v2','rejection-raw-consumer166-v28','rejection-raw-consumer166-v28-v2']]
    arts += [N/x for x in['audit_whole_coverage_frontier_v28.py','query_sampler_branch_surfaces_v28_166.py','read_sampler_native_branch_contract_v28_166.py','read_sampler_native_branch_contract_v28_166_v2.py','read_rejection_raw_logits_consumers_v28_166.py','read_rejection_raw_logits_consumers_v28_166_v2.py','finalize_sampling_and_coverage_v28.py']]
    arts += [q for q in N.glob('zcode-*v28*20261031*')if q.is_file()]
    arts += [R/'candidates/F31.json',R/'candidates/F32.json',R/'candidates/F35.json',pub,N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    arts=sorted(set(arts));write(O/'checkpoint-manifest-v28.json',dict(generated_utc=stamp,artifacts=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in arts],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v28.json'),manifest_files=len(arts),runs=len(runs),pass_runs=6,retained_error_runs=1,model_candidates=35,framework_candidates=7,existing_links_repaired=83,new_links=len(new_links),validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
