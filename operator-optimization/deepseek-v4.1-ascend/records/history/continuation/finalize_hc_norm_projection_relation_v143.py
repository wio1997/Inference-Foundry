"""Seal typed HC/input-RMS/projection relations and preserve all broader gaps."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v143.json').exists()
    pre=read(O/'hc-norm-projection-preflight-v143.json');assert pre['status']=='PASS'and pre['checks']>70000
    names=['hc-norm-projection-inputs-v143.json','hc-norm-projection-native-frontiers-v143.json','hc-norm-projection-source-contract-v143.json','hc-norm-projection-observed-and-work-v143.json','hc-norm-projection-stat-and-round-witnesses-v143.json','hc-norm-projection-readiness-DAGs-v143.json','hc-norm-projection-composition-v143.json'];comp=read(O/'hc-norm-projection-composition-v143.json')
    detail=dict(evidence=names,eager_frontiers=960,source_CID_TASK_bindings=10824,QR_source_role_bindings=1920,prior_192x2_QR_bindings_rechecked=384,C1_frontiers=24,C2_frontiers=72,retained_IndexWeight_reader_frontiers=192,reference_DAG_cases=3072,new_scoped_pair_relations=5,new_pair_clauses=5,new_groups=0,new_main_candidates=0,new_candidate_module_links=0,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v142.json')))
    def save(n,d):write(O/(n+'-v143.json'),d)
    rel=fork('candidate-relation-clauses')
    for c in comp['clauses']:
        assert not any({r['a'],r['b']}=={c['a'],c['b']}for r in rel['clauses'])
        rel['clauses'].append(dict(id=c['id'],a=c['a'],b=c['b'],relation={'F05':'INPUT_RMS_AND_QR_QUANT_DISTINCT_STATS_ROUNDS','F06':'OWNER_VERSUS_RECIPIENT_TYPED_NORM_PROJECTION_CARRIER','F07':'INPUT_RMS_HIGHER_ORDER_QB_ROPE_PRECISION_PATH','F08':'INPUT_RMS_SWA_TYPED_CACHE_PRODUCER','F33':'INPUT_RMS_C1_SWA_SEPARATE_GROUPS_AND_READERS'}[c['b']],operator_claim=c['contract'],evidence=names,applicable_domain='960savedpositive-query targeteager scopes10824rawTASKbindings;actualM/Q/NandC1/C2/IndexWeightreadersretained. Newtypednorm/projection/commhandoff,allreader/nativeprecision/emptyQandotherbranchesremainconditional.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,HC_norm_projection_v143=deepcopy(detail)))
    for gid,m in comp['group_extensions']:
        g=next(g for g in rel['higher_order_groups']if g['id']==gid);assert m not in g['members'];g['members'].append(m);g.update(all_subsets_analyzed=False,legal_joint_fusion_proved=False);g['HC_norm_projection_v143']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v142.json',new_pair_clauses=5,new_higher_order_groups=0,clause_count=167,selected_pair_count=167,HC_norm_projection_v143=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    cv=fork('candidate-relation-coverage');nav=fork('candidate-pair-navigation')
    for c in comp['clauses']:
        for d in [cv,nav]:
            r=next(r for r in d['pairs']if {r['a'],r['b']}=={c['a'],c['b']});r.update(whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,HC_norm_projection_v143=deepcopy(detail))
            if d is cv:r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']])
            else:r.update(conditional_clause_ids=[c['id']],semantic_pair_reconciliation_done_in_stated_scope=True,evidence=names)
    cv.update(parent='candidate-relation-coverage-v142.json',parent_relation_index='candidate-relation-clauses-v143.json',scoped_conditional_pairs=167,remaining_pair_relations=428,HC_norm_projection_v143=detail,scope='167scopedconditionalpairs/26groups/428pending;F02inputRMS/QA-quant/QB-RoPE/SWA/C1 typedstatandcarrieralternativesreviewed. F01star34scopednotwholeclosure retained.',overall_complete=False);save('candidate-relation-coverage',cv)
    nav.update(parent='candidate-pair-navigation-v142.json',parent_footprints='candidate-operator-footprints-v143.json',parent_relation_index='candidate-relation-clauses-v143.json',scoped_conditional_pairs=167,remaining_central_pair_reconciliation=428,HC_norm_projection_v143=detail);save('candidate-pair-navigation',nav)
    touched=['F01','F02','F05','F06','F07','F08','F09','F10','F32','F33'];fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['HC_norm_projection_v143']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='hc-norm-projection-composition-v143.json',all_runtime_or_new_native_fusion_gates_bound=False)
    fp['parent']='candidate-operator-footprints-v142.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['HC_norm_projection_v143']=detail;save('operator-analysis-coverage-ledger',op)
    gap=('NeedactualY/Z/gamma/eps/beta/rstd/weightparameter-generations,fullHreduction/rsqrt/BF16 andHF32nativeaccuracy/toleranceandlegalVector/Cubehandoff. Raw4HHCstatsnotYstats;initialrepeatstatreuseisarestrictednewABI,notalllayers. '
         'OwnernormalizedZ,recipientrawY+unionnorm andownerprojectedcarriersarealternatives;priceN-Mnormalizerrowduplication,CPqueryQvsNpanelwork,strides/queues/memoryandcollective/rankmatches. '
         'KeepC1/C2/192IndexWeight/QRnativequant/IndexQandallotherY/Z/rstdreaders orprovecompatibleextraoutputs;optionalbiasBF16Addandallnative/debug/hookobserversremain. '
         'Currentcoeff/Post/raw4HX/pre4andcacheCS/slot/epoch/lastreader+queuedwriterlifetimesremain. QueryRoPEtail64afterQBnotdirectinputRMSfold;preserveallrounds/scales/nativequant. '
         'No newnativeparity/physicalcapacity/MFU/lowest/actualspeedup/wholepair/allbranchesclosure;9/1352native849PMU16phase3136assignments andbroadercandidate gaps retained.')
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['HC_norm_projection_v143']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='167conditionalpairs/26groups/428pending. F02-F05/F06/F07/F08/F33 fullHinputRMSversusHCrawstats andQR/SWA/C1groupstats,typedrounds,owner-vsrecipientcarrier/work reviewed. '+gap
        if r['id']=='G10':r['remaining']='167conditionalpairs/26groups/428pending. HG73-QaddsF02/F07/F08/F33;HG87currentreadcommitaddsF02.3072arbitraryservicereferences528recipientfaster2544slower;noactualgain/optimum. RawYandZcarrierssharelocalinterfacecredit,normalizedinputreadunionrequiresallZreaders;higherorderallsubsets/nativelegalityopen.'
        if r['id']=='G13':r['remaining']=('v143960positiveeagerfrontiers10824rawCIDTASKbindings;QRnorm/quant1920rolesincluding384priorIndexsubset;C1 24/C2 72/IndexWeight192retained. '
            'HCraw4Hstat≠collapsedYstat;initialonehotrepeatrealstatidentitylimited,newnativehandoffnotprovided. ExactrealRMS-linear-RMSscalarcancellationrequiresadaptiveepsilonandstillinputstat;BF16Zroundcounterexampleproduces-.07421875vs0. Optionalbias/rstdobserversgated. '
            'OwnerHCnormremovesonechosen4MHprivateinterface;rawYrecipientunionnorm/projchangesgraphandnormalizesNnotM,choseninterface4MH+2QHbutcannotaddownercredit. QsubsetvsfullNpanelwork/C1C2IndexWeightfullreadersmustbeprovided. RawYvsZsameHcarrierdoesnotitselfreducepayload. '
            '3072two-projectionDAGs528faster2544slower,arbitraryservice/earlyKVVectorrepricingnotphysicalbounds;C1/C2/IndexWeightoutsidecutstillgated. '
            'NextpendingF02-F09/F10/F11/F17/F18/F19:inputnorm/Indexquery-Weight-Key/C2projection-pool-latentfanout;reusev135-v140row/state/rounding evidence,onlydirectsourceasneeded. '
            '358semantic26modules854nodes1346edges48hints2073regimes285598conditional9classes1352native849PMU16phase3136assignment0whole/physicalMFU-lowestunknown. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v142.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v142 HC-draft5relations/2496TASK/312layers;postcheckPASS26624',input_checkpoint=142,followup_checkpoint=143,HC_norm_projection_v143=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=167,central_pairs_not_reconciled=428,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v143.json',dict(status='PROFILE_INPUT_RMS_QR_TYPED_PROJECTION_THEN_STAT_ROUND_CARRIER_QUESTIONS',questions=read(O/'hc-norm-projection-inputs-v143.json')['questions'],source_scope='Direct model HC/inputNorm,DSA input/preprocess/quant/currentjoin,C1/C2project callers andRMSwrapper94-104 only;reusev27/v61/v94/v99/v133-v140. No framework/SDKsurvey/source download/targetchange.',no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-norm-projection-validation-history-v143.json',dict(events=[
        'v142authority/CURRENT-PASS26624/RECOVERY142and149manifestfilesmatched;previousGoalturnPROGRESS.',
        '960source/CID eagerscopes10824TASKs;1920QRnorm/nativequantroles including384priorIndexsubsetrawrechecked. NoAPI-onlyholes,newnativeclassesorfunctionalphasepromotion.',
        'C1 24/C2 72/IndexWeight192normalizedinputreadersretained. ActualM/Q/NdomainsandBF16->RMS->BF16->nativeDQINT8/FLOAT->QB->RoPEtypeschecked.',
        'HCraw4Hstatcounterexample;restrictedinitialrepeatidentity;exactrealadaptiveepsilonformulaandnonzeroepswitness. BF16Z→GEMMcounterexample-.07421875vs0;featuregamma/fullrowdependencycounterexamples.',
        'No-residualwrapperoptionalbias/rstdscopedgateexplicit;realrefactorassumesbeta0. NormalizedZ/rawY/projectedcarriersandallprivate/readcreditcountsarealternatives,notadditiveHBM/time.',
        '3072DAGs528recipientfaster2544slowerunderarbitrarynorm/Vectorservices;sourcepart3releasechangesresourceoverlapnotzero-cost. Fullfrontiermodelnotminimumbarrier.',
        'PreflightPASS'+str(pre['checks'])+';REL143-01..05,167pairs26groups428pending. Allbroadercandidate/native/global/F01wholegaps retained.'],overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v143167conditionalpairs/26groups/428pending;960frontiers10824TASK/1920QRroles/C1 24C2 72IndexWeight192/3072DAGs.4472SPjoins358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0whole unchanged.')
    action=('ContinueoriginalGoal;v143PROGRESSF02-F05/F06/F07/F08/F33. Read hc-norm-projection-summary/native-frontiers/source/observed-work/stat-round-witnesses/DAGs/composition/currentobligations-v143;verify_hc_norm_projection_relation_v143mustPASS. '
        '960positiveeagerfrontiers10824rawCIDTASKs;QRnorm+nativequant1920roles,384oldIndexsubsetrechecked. C1 24/C2 72/IndexWeight192Zreadersretained. M800Q800N6400x320,M816Q816N6528x320,M834Q834N6668x280,M834Q830N6668x40;countsnotactualsame-subrow/parameterproof. '
        'HCraw4HrsqrtstatnotweightedY5120stat;initialequalrepeat+onehotfinite/sameepsilonhasrestrictedrealstatidentitybutnativeorder/rstdABIunproved. InputRMS→BF16Z→QA/KV→BF16→groupnormBF16→QRDQINT8/FLOAT→QB→tail64RoPEallroundskept. QR1280/SWA512/C1512notonegroupnorm. '
        'Exactrealbeta0RMS-linear-RMScanremoveinputscalarronlywithadaptiveeps2/r^2,whichstillneedsinputfullHstat;unchangedepswrong. BF16Zplanesmakeprojection-.07421875vsunrounded0;gammafeaturecannotjustbeonescalar. OptionalwrapperbetaBF16Add/rstddebug/hook/allreader gatesremain;newVector-Cubehandoff/capacityneeded. '
        'OwnerHCnorm+Zcarrier vsrawYcarrier+recipientunionnorm/panelvsownerprojectedcarriersaredifferentgraphs. RecipientnormalizesNvsM(about8x),separatenormsN+Q;QAonallNovercomputesN-Q unlessnewmixedrowmap. Chosenlocalinterfacecreditowner4MH;rawYrecipientunion4MH+2QHbutshares4MH,notadditive. CarrierBF16Hpayloadunchanged;logicalnotHBM/alloc/time. AllC1/C2/IndexWeightandQRreadersmustinternalizeorreceivecompatibleextraoutputs. '
        '3072plain-QA/SWAreferences528recipientfaster2544slower;hypotheticalservice/noC1C2Indexreaderclosure. KVnormcanstartonKVdataearlierthanoldpart3QRjoinonlywithVectorrepricing/newcontract;currentQB/SWAjoinandQueryCSlastreaderremain. CurrentHCcoeff/rawXPost/pre4notretired. '
        'REL143-01..05;HG73-QaddsF02/F07/F08/F33,HG87commitaddsF02.167conditionalpairs26groups428pending;F01star34scopednotwhole.358/26semantic854/1346/48partialgraph2073/285598conditional9/1352native849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU-lowest unchanged;broadercandidate gaps retained. '
        'NextpendingF02-F09/F10/F11/F17/F18/F19 inputnorm-to-Indexquery/Weight/Key/C2projection/pool/latentfork;reusev135-v140evidence,actualnormalizedZ/roundedFLOAT/fullrow+requestpair/ringfuturestate/readerscentral. Allotherpendingpairsremain. '
        'Alllocalworkcomplete;no remote/device/Torch/CANN/model/capture/replay/implementation. Padv120grantconsumed,nopendingapproval. Do noteditsealedv143/rerunfinalizer;versionedpostcheck/CURRENTmustmatchRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F02–F05/F06/F07/F08/F33：HC 输入归一化与 Q/KV、量化、RoPE 和 C1 投影的条件组合。整体目标仍未完成。

960组正Query eager前沿关联10824个原生TASK，包含完整的1920个Q_a输出归一化/量化角色，其中384个与既有Indexer源层子集逐项复核。它们均通过原始API、调用栈、CID、TASK和Compute记录绑定，没有时间邻近补配或新增原生类别关闭。C1的24组、C2的72组及192组IndexWeight仍是归一化输入的独立读者。

当前链是HC加权合并Y_BF16→input RMS5120→Z_BF16→SP all-gather→Q_a1280与SWA KV512。之后还有Q_a norm1280、native DynamicQuant的INT8/FLOAT、Q_b BF16和Query tail64 RoPE。各归约和舍入点不能因融合而自动取消；Q_a1280、SWA512、C1 latent512也不能改成一次1792/2304维norm。无残差RMS wrapper的可选bias及其BF16 Add、rstd与其它观察者仍保留适用性条件。

HC原始4H数据的平方均值不能代替合并后Y的平方均值。反例中原始4H范数相同，但通道相加或相减使Y范数不同。只有首层严格相等的四通道repeat、onehot premix、有限数值等限定域中，二者实数均值才相同；相同norm_eps也不证明native归约顺序、rstd导出或新handoff已成立。

在无bias的精确实数域，令r=(mean(Y²)+ε1)^−1/2、v=B·diag(γ1)·Y，则下一RMS(r·v)可写成γ2·v/sqrt(mean(v²)+ε2/r²)。直接去掉r却沿用ε2会改变函数；保留自适应ε2/r²仍需要输入统计量。当前BF16 Z舍入还独立阻断这个实数重写：一个合法构造中，从舍入Z做矩阵乘得到−0.07421875，直接使用未舍入值则为0。它不是实际冻结权重的结果，但足以否定无条件代数等价。

可继续研究三种不同版本：owner上融合HC与input RMS后发送Z；发送原始Y后在recipient上融合RMS与Q/KV投影；owner上归一化并投影后发送较窄carrier。前两种Y/Z carrier的形状和BF16载荷相同，不会仅因改名而减少通信量。它们选择了不同的归一化位置，不能把收益直接相加。

以M=800、Q=800、N=6400为例，原owner每卡归一化800行；rawY recipient union方案需归一化6400行，分开给QA/KV做norm则为7200行。若为统一面板而把QA从Q行扩到N行，还会额外计算N−Q行；需要新的混合行映射，不能默认免费。

在声明的逻辑接口图中，owner HC/RMS融合可消去的私有Y读写参考为4MH B；rawY recipient union norm/panel相对同一基线的参考为4MH+2QH B，两者共享其中4MH，不能相加。QA和KV在真正共享的Q行上，一次共同输入读取最多对应这份图中的2QH B。这里的B是逻辑值读写，须先关闭其它读者；不是HBM流量、额外分配量或节省时间。C1、C2 FLOAT cast及IndexWeight若仍在图外，就需要保留Z或给出兼容的新输出。

当前多流调用让Q_a完成后KV与Q norm/quant重叠，part3又同时约束Q_b和KV norm。KV norm的数据条件只需要KV，但更早释放它会与QR工作竞争Vector，必须重新计价。3072个普通QA/SWA两投影参考DAG中，给定任意假设服务后recipient组合528组较快、2544组较慢；这不是实测胜率或物理上限。C1/C2/Indexer读者与更细行流水仍是额外条件。

F02到Query RoPE是包含多个非线性、量化和舍入边界的高阶组合，不是input RMS直接连接RoPE。当前没有post-Q_b RMS512可删，完整Query512仍需要；HC的原始X、当前系数、Post残差及下一FFN的pre4也不会因归一化或carrier提前完成而退休。

本轮条件关系162→167，待核对433→428，联合组仍26。35主候选、7历史框架记录、358语义行/26模块、4472次SP关联和854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图及3136项归属歧义仍待补齐；整模块关闭0，全模块物理MFU与最低时间尚未建立。没有新增服务器或NPU运行。
'''
    updates={cid:dict(review_v143=dict(evidence=names,pairs=comp['new_pairs'],native_TASK_bindings=10824,QR_source_roles=1920,scoped_conclusion_changing_gap=gap,existing_broad_conclusion_gap_preserved=True,no_native_fusion_or_physical_gain_proved=True))for cid in touched}
    inputs=[Path(x['path'])for x in read(O/'hc-norm-projection-inputs-v143.json')['inputs']];refs=read(O/'hc-norm-projection-native-frontiers-v143.json')['databases'];inputs +=[Path(x['raw_database']['path'])for x in refs]+[Path(x['cached_native']['path'])for x in refs]
    findings={'M06':'HCraw4Hstat versuscollapsedY5120stat;restrictedinitialrepeatidentity needsnewtypedhandoff,coeff/Post/pre4 rawX stayslive.','M07':'960eagerQRnorm/quant fullcohort;input5120/QR1280/SWA512/C1512 stats andBF16/nativequant/RoPE boundaries distinct.','M08':'C1 sameZ distinctweights/norm;C2 FLOAT castofroundedZ andstateprojections retained asoutsidepanelreaders.','M09':'192IndexWeight plusQR/IndexQ readers preventblind Z/QRprivateplaneelision.','M10':'EarlierKVnorm/cachepublication needsactualrows/slots/epoch andVectorcompetition;notnewfreeoverlap.','M21':'ActualCPquerysubset/globalcache/padding/emptyQ metadata remains;M/Q/N countsnotrowidentityproof.','M22':'OwnerZ/rawY/projectedcarriers chooseonegraph,legalAIV-AIC/GMhandoff andallreaders/capture/queuecapacity required.','M23':'RecipientnormNvsownerM workduplication andQvsNQAovercompute priced;sameBF16HrawYcarrierdoesnotreducecommunicationpayload.'}
    persist_refinement(version=143,previous=142,title='HC input-RMS typed projection,normalization statistics and carrier alternatives',summary=dict(filename='hc-norm-projection-summary-v143.json',previous_turn_classification='PROGRESS:v142 HC-draft5relations/2496TASK/312layers;PASS26624',**detail,refined_candidates=10,refined_modules=8,branch_records=358,semantic_modules=26,relation_clauses=167,higher_order_groups=26,pending_central_pair_reconciliation=428,partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='All local saved-profile/inputRMS-projection/stat-round/carrier/DAG work complete;no remote/device/Torch/CANN/model/replay/service work.'),changes=['Bind960positiveeagerfrontiers10824TASKs/1920QRnormquantroles;retainC1/C2/IndexWeightreaders andactualM/Q/N.', 'Deriveinputnormstat/adaptiveepsilon/BF16counterexamples andowner-vsrecipientcarrier/work alternatives;3072weightedreferenceDAGs.', 'AddREL143-01..05;167pairs26groups428pending;broadercandidate/native/global gapsunchanged.'],candidate_updates=updates,module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in findings.items()},remaining=remaining,action=action,jobs=[
        dict(kind='profile',status='pass',description='960source/CIDeagerfrontiers10824rawTASKs;QRnorm/quant1920rolesincl384priorrecords;typedQA/KV/C1/C2/IndexWeight/MQN checked.',artifacts=[O/'hc-norm-projection-native-frontiers-v143.json',O/'hc-norm-projection-observed-and-work-v143.json']+inputs),
        dict(kind='design-check',status='pass',description='3072DAGs,HCrawstat/initialrepeat/adaptiveepsilon/BF16/gamma/fullHcounterexamples and'+str(pre['checks'])+'preflightchecksPASS;fiveconditionalrelations.',artifacts=[O/'hc-norm-projection-preflight-v143.json',O/'hc-norm-projection-validation-history-v143.json']+[O/n for n in names]+[O/'coverage-current-obligations-v143.json',O/'whole-scope-requirement-audit-v143.json'])],report=report,read_first=names+['coverage-current-obligations-v143.json'],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_hc_norm_projection_native_v143.py',N/'analyze_hc_norm_projection_relation_v143.py',N/'verify_hc_norm_projection_relation_v143.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-norm-projection',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
