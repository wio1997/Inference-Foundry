"""Seal HC/C1-SWA conditional relation and delivery-cut costs once."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v135.json').exists()
    pre=read(O/'hc-c1-preflight-v135.json');assert pre['status']=='PASS'and pre['checks']>7000
    names=['hc-c1-inputs-v135.json','hc-c1-native-consumers-v135.json','hc-c1-source-and-precision-contract-v135.json','hc-c1-delivery-cuts-and-work-v135.json',
        'hc-c1-packet-readiness-models-v135.json','hc-c1-layout-and-old-value-witnesses-v135.json','hc-c1-joint-graph-and-clauses-v135.json']
    detail=dict(evidence=names,C1_eager_frontiers=24,consumer_roles_per_frontier=12,source_bound_native_consumer_TASKs=288,reported_geometry_cases=4,
        delivery_cuts=4,abstract_ordered_communicator_cases=3456,new_scoped_pair_relations=1,new_pair_clauses=1,new_groups=0,new_main_candidates=0,
        existing_v27_joint_projection_norm_WAR_not_new_discovery=True,actual_parameter_pointer_native_new_M_or_all_readers_bound=False,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v134.json')))
    def save(n,d):write(O/(n+'-v135.json'),d)
    claim=(
        'ComposeexternalpremixHCY/fullHnormwithC1/SWAshared-inputproducersandcarrierwhilecurrentfull4H/FLOAT-HF32coeffandPostresidualstaylive. Differentconstructedmatricesdonotprovevalueinequalityorequality;noautomaticCSEorsame-inputwideGEMMwithHCcoeff. '
        'KeepYBF16/HnormBF16/twoGEMMBF16/two512norms/indexWkBF16+128norm/RoPE/INT8key/FP16scaleboundaries. '
        'Chooseprojection-before-norm,normalizedlatent,index-normorencoded-endpointdeliverywithvariant-specificparam/nativeM/owner/position/readers/capturegates;recipient-specific512normsmaystayreceiver. '
        'JointpacketcanwaitonlateC1/encodingandloseSWAearlyservice;compareorderedseparatepackets/newcompletegraphwithHCcoeffdeadline,notaddlaunch/IOcredits. '
        'PreserveindexWkold-latentreadbeforeoverlappinglongRoPEwriteorpricecopy/outofplace;notwholeindexpublicationbarrier. Packed1024stride,allcachegenerations/metadata/readinessandoldHCXretirementremain.')
    rel=fork('candidate-relation-clauses');assert not any({r['a'],r['b']}=={'F01','F33'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL135-01',a='F01',b='F33',relation='HC_COEFF_FORK_C1_SWA_PRODUCER_AND_TYPED_DELIVERY_CUTS',operator_claim=claim,evidence=names,
        applicable_domain='SelectedtargetC1source20/externalpremixandpositivelyrecordedfallbackindexkey/FP16scalecachepaths;empty/!owns_k/packed/folded/C2areexplicitotherdomains.',
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,HC_C1_composition_v135=deepcopy(detail)))
    for gid,member in [('HG93-HC-ATTENTION-FORK-DEADLINE','F33'),('HG74-C1','F01')]:
        g=next(g for g in rel['higher_order_groups']if g['id']==gid);assert member not in g['members'];g['members'].append(member);g.update(all_subsets_analyzed=False,legal_joint_fusion_proved=False);g['HC_C1_composition_v135']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v134.json',new_pair_clauses=1,new_higher_order_groups=0,clause_count=143,selected_pair_count=143,HC_C1_composition_v135=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    cov=fork('candidate-relation-coverage');r=next(r for r in cov['pairs']if {r['a'],r['b']}=={'F01','F33'});r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL135-01'],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,HC_C1_composition_v135=deepcopy(detail))
    cov.update(parent='candidate-relation-coverage-v134.json',parent_relation_index='candidate-relation-clauses-v135.json',scoped_conditional_pairs=143,remaining_pair_relations=452,HC_C1_composition_v135=detail,
        scope='143scopedconditionalpairs/26groups/452remaining. HC-C1typedproducer/packet/old-valuecutreviewisconditional;allruntime/allvariantsandwholemodelremainopen.',overall_complete=False);save('candidate-relation-coverage',cov)
    nav=fork('candidate-pair-navigation');r=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F01','F33'});r.update(conditional_clause_ids=['REL135-01'],semantic_pair_reconciliation_done_in_stated_scope=True,whole_pair_all_variants_closed=False,HC_C1_composition_v135=deepcopy(detail),evidence=names)
    nav.update(parent='candidate-pair-navigation-v134.json',parent_footprints='candidate-operator-footprints-v135.json',parent_relation_index='candidate-relation-clauses-v135.json',scoped_conditional_pairs=143,remaining_central_pair_reconciliation=452,HC_C1_composition_v135=detail);save('candidate-pair-navigation',nav)
    touched=['F01','F11','F19','F32','F33'];fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['HC_C1_composition_v135']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='hc-c1-joint-graph-and-clauses-v135.json',all_runtime_or_native_parameters_bound=False)
    fp['parent']='candidate-operator-footprints-v134.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['HC_C1_composition_v135']=detail;save('operator-analysis-coverage-ledger',op)
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['HC_C1_composition_v135']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='143conditionalpairs/26groups/452centralpairsandallvariantsopen. HC-C1compositionmustchooseproduceranddeliverycut:rawtwo512projectionsor512normedlatents2048B/row,index-normcut2304B,encodedendpoints2178B;QA1280adds2560B/rowifneeded. MoreownerworkcanholdjointpacketandPostthroughHCresourcecontention;oldC1latent/WAR,packedstride/twonorms/currentHCcoeff/residual/readersstay. Thesearevaluecontent/abstractDAGsnotwire/HBM/latencybounds.'
        if r['id']=='G10':r['remaining']='Central143 conditional relations/26 groups leave452 pairs to reconcile;blank rows are not independence. Groupmembershipextensionsdonotproveallsubsetsorjointnativelegality;oneHC/C1baselinehasnoadditivesplit/packingcredit.'
        if r['id']=='G13':r['remaining']=(
            'v135F01-F33reviewhas24source-boundC1frontiersx12nativeconsumers,includingold-latentWk,128norm/RoPE,INT8quant,FLOAT16scalecastandpositivefallbackkey/scale/longstores. '
            'Fourdeliverycutspricedwithreceiver-specificnormoption;ownerencodedoutputsrequireloadedW/gamma/quant/position/slot/rank/nativeM/FP16rounding/captureandallrawlatentsobserversclosed. '
            'Two512norms/currentHCfull4Hcoeffremain;W_A=W_Bnotproven. Packed1024stridecannotinheritcontiguous512consumercontract. 3456abstractordered-packetDAGscannotidentifyactualC/Yservice,rankepisodesorbesthardwareplan. '
            'Nextprofile-firstpendingF01-F17:HCcurrentcoeff/YforkwithC2FLOATpairandglobalpool/history/scoreconsumers. Reuse72positiveeagerC2pairsandv88mixedprecisioncontracts;do notapplyC1row-localendpointcuttoC2window/statewithoutitsownboundaryproof. '
            '4472savedattentionSPsourcejoinsremaincomplete;old80capturedSWA/C1ambiguitiesandall9/1352native/globalbranch/graph/physicalcapacity/MFU/lowestgatesremain. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v134.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v134completed4472savedHC/SPsourcejoins;postcheckPASS123761',input_checkpoint=134,followup_checkpoint=135,HC_C1_composition_v135=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=143,central_pairs_not_reconciled=452,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v135.json',dict(status='POSITIVE_C1_NATIVE_OLD_VALUE_INDEX_QUANT_STORE_ROLES_THEN_HC_PACKET_AND_LAYOUT_CONTRACTS',questions=read(O/'hc-c1-inputs-v135.json')['questions'],
        source_scope='Reusev27consumer/WAR/distinctmatrixcontracts;onlypinnedHC/constructors/two512norms/indexer106-125/longRoPEdirectsubtree. No frameworksurvey/loaderaudit/newSDK/remote/deviceexecution.',
        no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-c1-validation-history-v135.json',dict(events=[
        'Predecessorv134revalidatedPASS123761/manifest106;previousgoalturnclosed960namedeagerjoins,notwait/statusonly.',
        'Priorv27alreadyreviewedseparatematrices/two512norms/indexold-latentWARandprojectionpacking. Reusedasanswers,notnewdiscoveries;newquestionisHCforkcombinedreadinessanddeliverycut.',
        '24eagerC1parentsfromv134nowhave12exactCID/nativeconsumerroleseach:two512norms,SWA/long/indexRoPE,IndexWk128/norm128/INT8quant/keyStore/FP16cast/scaleStore/longStore. Selectedsourceframeduplicatesfromsharedline110canonicalized,taskrecordsunchanged.',
        'Sourceconstructorsshowdistinctobjectsbutnoloadedvalueinequality/equalityproof. No parameterdedupclaimandnoall80capturedgraphrolepromotion.',
        'Fourdeliverycutsdistinguishrawprojectionswithrecipientnorms,normedlatents,indexnormoutputs,andlongKV+encodedindexkey/FP16scale. Encoded2178B/rowversuslatent2048Badds130Btoavoidreceiverworkunderexplicitextraowner/param/quant/position/readergates;QAadds2560B.',
        '3456independentDAGchecksincludeorderedcommunicators,jointhead-of-linewaitsandHCPostcoeffjoin;bothjointandsplitcanwininhypotheticalservice. No actualunlimitedoverlaporoptimumclaimed.',
        'Packedstride1024/old-valueWAR/twonormcounterexamplesandunionmatrix/privatecopyreferenceschecked. AllY/GEMM/norm/Index/FP16rounds,currentHCcoeff/pre4/Postresidualandactualcachevisibilityretained.',
        'Preflight'+str(pre['checks'])+'PASS. REL135-01addsF01-F33;HG93addsF33andHG74-C1addsF01withoutnewgroup/subsetclosure.143pairs/26groups/452pending,allnative/global/wholeuncertaintyretained.'],overall_complete=False))
    gap=('NeedactualW/gamma/eps/bias/quantizer/nativeM/position/CP-SP/slot/scale/capture/rankepochsandallpre-RoPElatent/raw/norm/readers. Separateobjects/shapesnotvalueproof. '
        'Deliverybefore512normmayleavegammaatrecipient;movingnorm/indexencodingneedstheirextragates. Priceownerwork,typedpacketalignment/queues,orderedgroupjoins,scratch/copy/strideandHCsharedresource/Postdeadlines. '
        'Preserveold-latentindexreadbeforerangewrite,notwholeindexpublicationbarrier;C1/C2andfallback/packed/empty/!owns_kremainseparate. No nativeparity/speedup/MFU/lowest/wholepair/allbranchclosure;9/1352andold80graphrolesremain.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v135F01-F33conditional:143pairs/26groups/452pending;24C1x12nativeconsumerroles,4cuts/3456DAGcases.4472savedSPjoinsretained;358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0wholeunchanged.')
    action=(
        'ContinueoriginalGoal;v135PROGRESSisF01-F33HCfork/C1-SWAproducer/delivery-cutcomposition,notnativeoptimizationorwholeclosure. Readhc-c1-summary/native-consumers/source/delivery-cuts/packet-readiness/layout-witnesses/joint-graph/currentobligations-v135;verify_hc_c1_relation_v135mustPASS. '
        '24C1eagerfrontiersx12newconsumerroles=288TASKs:two512norms,SWA/long/indexRoPE,IndexWk512→128/norm128/INT8quant,keyStore,FLOAT16scaleCast/Store,longStore. Source/CIDkeyspositive;old80capturedgraphrolesremainambiguous. Priorv27sameconsumerfactsnotnewdiscovery. '
        'HCraw4H/FLOAT-HF32coeffandnormalizedBF16YforC1/SWAaredifferentinputs;currentcoeff/post/comb/pre4andPostresidualstay. SeparateWconstructorsnotpostloadvalueinequality/equalityproof;twodistinct512normsnotone1024norm. '
        'DeliverycutsPROJECTION_BEFORE_NORMS2048B/row(recipientgammacanremainlocal),normalizedLATENT2048,INDEX_NORM2304,ENCODED_ENDPOINTS2178=SWA512BF16+longC1RoPE512BF16+index128INT8+scaleFP16. QA1280adds2560Bifrequired. Added130BencodedvslatenttradesownerworkforreceiverWk/norm/RoPE/quantwork,notguaranteedbenefit. AllnewM/quant/scale/position/slot/reader/paramreplicagatesexplicit. '
        '3456hypotheticalDAGswithoneorderedcommunicatorandHCcoeffPostjoinshowjoint548faster/split284faster/2624ties. Servicesnotmeasured,fullcarrierchosenmodelnotwhole-rowbarrier;nohardwareoptimality/actualoverlap/gain. Laterpre4/FFNnotpricedinthePostendpointtime. '
        'PreserveIndexerWkoldlatentreadbeforeoverlappinglongRoPEwriteorpricecopy/newoutofplace;notwholeindexpublicationbarrier. Packed1024rowstridecannotinheritcontiguous512views;optionaladapter8NDvalueIOandallreaders/banks/epochspriced. UnionmatrixcontentsnotHBM/physicalcapacity;onewarmgrantonlyindeclarednecessary-servicecut. '
        'REL135-01F01-F33;HG93addsF33,HG74-C1addsF01,allsubsets/jointlegalityopen.143conditionalpairs26groups452pending;358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowestunchanged.4472savedHC/SPjoinsstillcovered. '
        'NextpendingF01-F17HCcoeff/YforkwithC2FLOATpair/globalpool/history/score;use72eagerC2pairsandv88precision/sourceanswers. C1row-localencodedendpointcannotbeappliedtoC2state/windowwithoutneighbordemand/ownership/bankproof. '
        'Alllocaljobscomplete,no remote/Torch/CANN/NPU/model/capture/replay/implementation. Padv120consumed,nopendingapproval. Do noteditsealedv135/rerunfinalizer. Versionedpostcheck/CURRENTmatchingRECOVERY;TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F01–F33：HC 系数分支与 C1/SWA 投影、通信和旧值读者如何共同重组。只分析固定 profile 与源码，没有运行设备或修改模型。

从上一轮24个eager C1前沿出发，新增关联了每组12类原生消费者，共288个TASK：两个512维norm，SWA/index/long三种RoPE，indexer的512→128投影和128维norm，INT8量化、key写入、FP16 scale转换/写入以及long KV写入。所有关联均由同一forward内的源码栈和精确CID确认，不能替代历史tensor地址、权重副本或新M数值等价性。旧80个captured graph的SWA/C1歧义继续保留。

v27已确认的独立矩阵、两个512 norm及旧latent读写顺序是本轮前提。SWA与C1各自构造权重，不证明加载后的值相等或不等；共享输入分块不需要两矩阵相等，删除一份计算则需要另外的值/epoch/精度证明。HC系数读原始4H/FLOAT-HF32状态，C1/SWA读归一化后的BF16 Y，不能无条件合成同输入宽GEMM。原HC系数、post/comb、给后续FFN的pre4和Post残差需求仍保留。

| 交付切点 | 每sender行的值内容 | 主要接收端工作 |
|---|---:|---|
| 两份GEMM输出，norm留接收端 | 2048B | 两个512 norm及后续分支 |
| 两份norm后的latent | 2048B | index投影/norm/RoPE/量化与long-RoPE |
| long-RoPE结果与index norm结果 | 2304B | index-RoPE/量化及cache写入 |
| long-RoPE结果与编码index终点 | 2178B | key/scale/long cache写入 |

各方案均含512维BF16 SWA平面，SWA角色的RoPE/写入按声明位置保留；若还需传1280维Q_a，每行另加2560B。这些只是命名平面的值内容，不含header、对齐、slot映射或其它raw读者。只移动GEMM时可让接收端保留自身norm参数；移动norm或index编码则增加相应gamma、quantizer、position、nativeM与舍入条件，不能把更强条件误当作所有投影前移的限制。

编码终点方案比latent多130B/行，试图用owner的Wk/norm/RoPE/INT8quant/FP16 scale计算换掉接收端重复工作。它要求实际读者只需要这些终点，key/scale/long的映射和世代一致，原量化及FP16舍入精确保留，数据在当前QLI/SMLA需要时就绪。不能等接受结果后才交付当前所需cache；也不能将本轮fallback的编码布局直接套到packed、folded、empty或!owns_k分支。新接口的数值等价性和提速仍未验证。

3456组有限抽象DAG比较了有序分开发送、合包和owner编码交付。在给定假设服务中，合包较快548组，拆分较快284组，其余2624组相同。它们只说明较晚的C1/编码分支可能挡住SWA交付，省一次collective也可能有价值；不是实测比例或硬件最优时间。实际Cube/Vector/访存、队列、pack和HC系数竞争会改变服务，Post还要等本层系数。表中模型的终点是attention HCPost，后续pre4/FFN服务没有计入。

indexer的Wk必须读取未旋转C1 latent的相应值，再允许in-place long-RoPE覆盖同一范围；也可付出copy或改为out-of-place。只需要保留真实行/范围的旧值读者，不应添加“整个index发布完毕才能long-RoPE”的通用barrier。packed1024输出的两份512视图还带row stride1024，不能继承当前连续512缓冲区的合同；若需要转成两个连续输入，适配的逻辑值读写为8ND。两个512 norm、所有GEMM→BF16及index/FP16边界继续保留，不能改成一个1024 norm或未舍入输入。

新增REL135-01，条件关系142→143、待核对453→452；HG93补入F33、HG74-C1补入F01，联合组仍26，全部子集和运行时联合合法性未关闭。4472次固定profile SP前沿关联保持完成；35主候选、7历史框架记录、358语义行/26模块及854节点/1346边/48提示的部分图不变。9类/1352次原生字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义均保留，整模块关闭0，整体目标仍未完成。
'''
    inputs=[Path(r['path'])for r in read(O/'hc-c1-inputs-v135.json')['inputs']];refs=read(O/'hc-c1-native-consumers-v135.json')['databases'];inputs += [Path(r['raw_database']['path'])for r in refs]+[Path(r['cached_native']['path'])for r in refs]
    oldc={r['id']:r for r in read(O/'candidate-profile-ledger-v134.json')['candidates']};updates={cid:dict(review_v135=dict(evidence=names,relation_clause='REL135-01',pair=['F01','F33'],C1_frontiers=24,consumer_roles=12,delivery_cuts=4,packet_models=3456,one_HC_C1_baseline=True,no_native_fusion_or_physical_gain_proved=True),
        previous_conclusion_changing_gap_before_v135=oldc[cid].get('conclusion_changing_gap'),conclusion_changing_gap=gap)for cid in touched}
    persist_refinement(version=135,previous=134,title='HC/C1-SWA producer, packet readiness and encoded endpoint delivery cuts',
        summary=dict(filename='hc-c1-summary-v135.json',previous_turn_classification='PROGRESS:v1344472savedSPsourcejoincoverage;PASS123761',**detail,
            refined_candidates=5,refined_modules=5,branch_records=358,semantic_modules=26,relation_clauses=143,higher_order_groups=26,pending_central_pair_reconciliation=452,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='Alllocalrawprofile/source/conditionalDAG/valuecontentworkcomplete;no remote/device/Torch/CANN/model/replay/serviceswork.'),
        changes=['Bind24C1x12nativeconsumerrolesincludingunrotatedindexread,FP16scale/fallbackstoresandlongmutation;keepold80graphambiguities.',
            'ReconcileHCforkwithfourtypedC1/SWA/Indexdeliverycuts,orderedpacketreadiness,recipientnormoptionandallround/stride/oldreaderconstraints.',
            'AddoneconditionalpairREL135-01:143pairs/26groups/452pending;allnative/global/wholescopestatusremainsopen.'],candidate_updates=updates,
        module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in {
            'M06':'HCcoeff/Postresidualandoldpremix/YforkstaylivewhileC1packet/encoderreadinessmoves;onesharedbaselineandrealPost/nextFFNdeadlines.',
            'M07':'TwoBF16projections/two512normsretainrounds;projection-onlyowneroptioncanleaverecipient-specificnormparametersinplace.',
            'M08':'C1normalization/oldlatent/longRoPEbranchcombineswithSWAproducers;packedstrideandold-valuecopieshavecosts,notweightCSE.',
            'M09':'24positiveindexWk128/norm/RoPE/INT8quant/FP16scale/storechainsgroundnewownerendpointcuts;allparam/quant/nativeM/slot/visibilitygatesremain.',
            'M23':'Oneorderedcommunicator/jointpacketcanwaiteachother;encoded2178Bversuslatent2048Btradesbytes/readiness/compute. Typedcontentsnotwireorlatencygains.'}.items()},
        remaining=remaining,action=action,jobs=[
            dict(kind='profile',status='pass',description='24source-boundC1/SWA/Index/longconsumerfrontiersx12roles;rawAPI/CID/TASK/Compute/sourceandfallbackdtypefieldschecked.',artifacts=[O/'hc-c1-native-consumers-v135.json']+inputs),
            dict(kind='design-check',status='pass',description='Fourdeliverycuts,packed/oldvalue/normguards,3456independentpacketDAGchecksand'+str(pre['checks'])+'preflightPASS;oneconditionalHC-C1relation.',artifacts=[O/'hc-c1-preflight-v135.json',O/'hc-c1-validation-history-v135.json']+[O/n for n in names]+[O/'coverage-current-obligations-v135.json',O/'whole-scope-requirement-audit-v135.json'])],
        report=report,read_first=names+['coverage-current-obligations-v135.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_hc_c1_consumers_v135.py',N/'analyze_hc_c1_relation_v135.py',N/'verify_hc_c1_relation_v135.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-c1',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
