"""Seal HC/draft/metadata/sampling epoch composition without closing broad scope."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v142.json').exists()
    pre=read(O/'hc-draft-preflight-v142.json');assert pre['status']=='PASS'and pre['checks']>20000
    names=['hc-draft-inputs-v142.json','hc-draft-native-frontiers-v142.json','hc-draft-source-contract-v142.json','hc-draft-observed-frontiers-v142.json','hc-draft-composition-v142.json','hc-draft-readiness-DAGs-v142.json','hc-draft-value-epoch-and-demand-witnesses-v142.json','hc-draft-work-and-lifetimes-v142.json'];comp=read(O/'hc-draft-composition-v142.json')
    detail=dict(evidence=names,draft_frontiers=104,draft_layers=312,source_CID_TASK_bindings=2496,target_eager_aux_frontiers_reused=72,dummy_frontiers=16,proposal_ancestry_frontiers=88,reference_DAG_cases=1536,raw_DBs_with_unresolved_stack_IDs=13,unresolved_selected_positive_frames=0,new_scoped_pair_relations=5,new_pair_clauses=5,new_groups=0,new_main_candidates=0,new_candidate_module_links=0,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v141.json')))
    def save(n,d):write(O/(n+'-v142.json'),d)
    rel=fork('candidate-relation-clauses')
    for c in comp['clauses']:
        assert not any({r['a'],r['b']}=={c['a'],c['b']}for r in rel['clauses'])
        rel['clauses'].append(dict(id=c['id'],a=c['a'],b=c['b'],relation={'F28':'HC_DRAFT_INITIAL_REPEAT_AND_COEFF_INPUT','F29':'HC_DRAFT_CONTEXT_QUERY_READY_FORK','F30':'HC_DRAFT_RAW_COLLAPSE_MARKOV_CONFIDENCE_EPOCH','F31':'HC_CURRENT_INPUT_METADATA_VERSUS_FUTURE_PREFIX','F35':'HC_QUERY_DEMAND_THROUGH_TARGET_SAMPLE_STATE_CUT'}[c['b']],operator_claim=c['contract'],evidence=names,applicable_domain='104 saved direct draft source-call windows,312layers2496TASKs plus72 existing targetaux frontiers.16dummy None-metadata paths kept distinct from88proposal ancestries. No same-client/loaded-parameter/tensor-bank or future-epoch binding.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,HC_draft_v142=deepcopy(detail)))
    for gid,member in comp['group_extensions']:
        g=next(g for g in rel['higher_order_groups']if g['id']==gid);assert member not in g['members'];g['members'].append(member);g.update(all_subsets_analyzed=False,legal_joint_fusion_proved=False);g['HC_draft_v142']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v141.json',new_pair_clauses=5,new_higher_order_groups=0,clause_count=162,selected_pair_count=162,F01_star_scoped_conditional_pairs=34,whole_F01_closed=False,HC_draft_v142=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    cv=fork('candidate-relation-coverage');nav=fork('candidate-pair-navigation')
    for c in comp['clauses']:
        for d in [cv,nav]:
            r=next(r for r in d['pairs']if {r['a'],r['b']}=={c['a'],c['b']});r.update(whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,HC_draft_v142=deepcopy(detail))
            if d is cv:r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']])
            else:r.update(conditional_clause_ids=[c['id']],semantic_pair_reconciliation_done_in_stated_scope=True,evidence=names)
    cv.update(parent='candidate-relation-coverage-v141.json',parent_relation_index='candidate-relation-clauses-v142.json',scoped_conditional_pairs=162,remaining_pair_relations=433,HC_draft_v142=detail,F01_star_scoped_pairs=34,whole_F01_closed=False,scope='162scopedconditionalpairs/26groups/433pending. F01has34/34centralpairsconditionallyreviewedinexplicitdomains;itsnative/allvariants/allbranchesandwholecandidatearenotclosed.',overall_complete=False);save('candidate-relation-coverage',cv)
    nav.update(parent='candidate-pair-navigation-v141.json',parent_footprints='candidate-operator-footprints-v142.json',parent_relation_index='candidate-relation-clauses-v142.json',scoped_conditional_pairs=162,remaining_central_pair_reconciliation=433,HC_draft_v142=detail,F01_star_scoped_pairs=34,whole_F01_closed=False);save('candidate-pair-navigation',nav)
    touched=['F01','F02','F03','F26','F27','F28','F29','F30','F31','F35'];fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['HC_draft_v142']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='hc-draft-composition-v142.json',all_runtime_or_new_native_fusion_gates_bound=False)
    fp['parent']='candidate-operator-footprints-v141.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['HC_draft_v142']=detail;save('operator-analysis-coverage-ledger',op)
    gap=('Need actual same-call requestgeneration/row ownership/seed-ID/masks/positions/slot metadata andloaded target/draft/context/query weight epochs;source ancestry isnot same-client pairing. '
         'First broadcast/onehot/coeffprefold needs finite/numeric/HF32/newstride-ABI/firstPostresidual contracts. Early draft HC/query pluscontext needs demanded cache ranges,WAW/RAW/epochs/VectorCubequeue contention andlastreader/queuedwriter/capture contracts. '
         'Raw draft collapse remains optionalconfidence input;runtime confidence/dynamic flags andfull raw/score/RNG/acceptance observers unbound. Laterkeep/newacceptedcount cannot produce current metadata orauthorize earlierfullKV/state deletion. '
         '13rawDB string tables have unresolved stack IDs;selectedpositive2496TASK frames resolve,but wholecallchain completeness/absence isnotpromoted. '
         'No transformednative parity,physicalMFU/lowest/actualgain/fullF01/wholepair/allsubset closure;9/1352native849PMU16phase3136assignments andbroadercandidate gaps retained.')
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['HC_draft_v142']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='162conditionalpairs/26groups/433pending. F01-F28/F29/F30/F31/F35 distinct targetaux/context/draftembedding/rawcollapse andcurrent/future metadata/acceptance epochs reviewed. F01central34/34scopeddoesnotcloseF01allvariants/native/whole. '+gap
        if r['id']=='G10':r['remaining']='162conditionalpairs/26groups/433pending;HG93addsF28/F29/F30/F31/F35,HG85VERIFY/HG73TAILaddF01,HG86terminaladdsF31/F35.1536referenceDAGs992faster544slower,768overlappingcredits768equal underhypotheticalservices;allsubsets/nativelegalityopen.'
        if r['id']=='G13':r['remaining']=('v142104draftwindows/312HClayers/2496sourceCIDTASKs;16dummy/88proposalancestrydistinct;72existingtargetauxfrontiersrecheckedbeforeHC.13DBunresolvedrawstackIDsbut0unresolvedselectedframes;noabsencepromotion. '
            'Earlyembedding/HC/query versuscontextprecompute isconditional,saves0matrixwork. Delayedcontext-querypackextendscontextbankreaders;distinctversionnotfreeearlierretirement. Firstonehotfinite/broadcastspecializationretainscoeff/firstPostresidual;nonfinite/FP32counterexamples. '
            'RawdraftcollapseBF16neededoptionalconfidence;headnormnotreplacement. Laterconfidencekeepcannotprunecurrentnoncausaldecoder/metadata;targetsamplingnext-IDrequiresnewacceptanceoutcome. FullKV/state fenceslimitselectedlastquerydemand. '
            'Next pending F02-F05/F06/F07/F08/F33 norm-to-Q/KV/rope andC1projection composition;F02-F32alreadyreviewed,reuseit. Keep actual normalized input,full-row reductions,BF16rounds,SP/CP mapping and coefficient/rawresidual/pre4 readers central;profilefirst. '
            '358semantic/26modules/854nodes1346edges48hints/2073regimes285598conditional/9classes1352native/849PMU/16phase/3136assignment/0whole/physicalMFU-lowestunknown. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v141.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v141 HC-MoE4relations/8638TASK+2APIonly/936bridges;postcheckPASS90025',input_checkpoint=141,followup_checkpoint=142,HC_draft_v142=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=162,central_pairs_not_reconciled=433,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v142.json',dict(status='PROFILE_DRAFT_HC_CONTEXT_SEED_THEN_DIRECT_VALUE_EPOCH_READERS',questions=read(O/'hc-draft-inputs-v142.json')['questions'],source_scope='Only direct shared HC/draft forward/context/seed+input publication and sampled-output call sites. Reuse v35/v85/v86/v113/v117/v129/v130;no framework/SDKsurvey or source download.',no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-draft-local-correction-v142.json',dict(status='CORRECTED_WITH_MISSING_STRING_EVIDENCE_RETAINED',initial_error='Full raw stack dictionary lookup hit KeyError268437198:some original stack IDs have no STRING_IDS entry.',correction='Keep unresolved raw frames as valueNone,matchonly positive resolvedsourceframes;record13affectedDBs. All2496selectedTASKcallchainframesresolve;neither fabricate strings nor infer fullsourcecoverage/absence.',new_device_work=False,overall_complete=False))
    write(O/'hc-draft-validation-history-v142.json',dict(events=[
        'Predecessorv141manifest107digests/CURRENT-PASS90025/RECOVERY141matched;previous turnPROGRESS.',
        '104existingrepeat/seed/SP anchors bound to312draftlayer scopes:1872HCstageTASKs+416collapse+208repeat/seed=2496rawCIDTASKs.16dummy88proposalancestry;no client/episode promotion.',
        '72targeteageraux mean/currentHC/previousFFN fronts reused andrawverified;auxmeansprecedeHC,notweightedY/finalnorm.',
        'Raw missing-string lookup initially failed;13DBs retainunresolvedIDs.0selectedpositiveframesmissing;native/global gapsnotpromoted.',
        '1536three-layer DAGs992faster544slower,768overlappingcredits768equal;arbitraryservices/readycachedomainnotphysicalgain. Earlyschedule matrixworkcredit0.',
        'Firstonehot/nonfinite/prefoldFP32,mean-vsY,rawconfidence-vsnorm,selectedquery-vsKV,bankepoch andfuturecutoffcycle witnesses preserve all current/future readers.',
        'PreflightPASS'+str(pre['checks'])+';fiveconditionalrelations;162pairs26groups433pending. F01central34/34scopednotwholeclosure.'],overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v142162conditionalpairs/26groups/433pending;F01star34/34scopednotwhole.104draft312layers2496TASK/72targetaux/1536DAGs;4472SPjoins358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0whole unchanged.')
    action=('Continue originalGoal;v142PROGRESS F01-F28/F29/F30/F31/F35. Read hc-draft-summary/native-frontiers/source/observed/composition/DAGs/value-epoch-demand-witnesses/worklifetimes/currentobligations-v142;verify_hc_draft_relation_v142 mustPASS. '
        '104repeat/seed CPUwindows312draftlayersx6HCroles1872TASK+416collapse+208repeatseed=2496exactsource/CID/TASK;16dummy88proposalancestry,notclient/requestbinding.72targeteageraux fronts rawverifiedmean-beforeHC/previousFFN,notlaterdraftpairing.13rawDBmissingstackIDsretainedNone,0selectedpositiveframesmissing;neverabsencepromotion. '
        'TargetHCY,targetnorm,3auxmeans,maincontext,draftembedding/rawcollapse distinct. Firstdraftrepeat4+onehotpremix allowsconditionalfinite/broadcastspecializationbutcoeffs/firstPostresidual remain;Inf*0 andFP32prefoldcounterexamples. '
        'Earlyembedding/HC/query canoverlapunchangedcontextprecomputeundercorrectmetadata/seed/cacheWAW/RAW/epoch/queuegates,notreplacequeryinputwithcontext. Delayedcontext-querypackedprojectionisanalternativeextendingcontextbankreaders,notfreeearlierretirement. '
        'RawdraftcollapsedBF16optionalconfidenceinputnotheadRMS;confidence/keepafterdecoder+Markovcannotprunecurrentnoncausaldecoder/metadata. SeedalreadyfromsamplednextIDbeforedecoder;lateoutputcolumncopyisnotseedcreation. NexttargetacceptanceproducesnewnextID/count,notpriorproposalkeep;same-shapemask/slot/requestepochbankreuseunsafe. '
        'F35finalqueryselectionstopsatneededKV/statecut;earlierfullHC/KVrowsneededbyselectedqueriesremain. Keepallrawscores/RNG/bonus/acceptedstate/zeroQcacheeffects.1536arbitraryDAGs992faster544slower,768overlap768equal;earlyschedulematrixcredit0,logicalcontentnotHBM/alloc/time/physicalbounds. '
        'REL142-01..05;162conditionalpairs26groups433pending. F01star34/34centralpairsscopeddomainreviewed,notF01whole/native/allvariantsorallsubsetsclosed.358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU-lowest unchanged;broadercandidate gaps retained. '
        'Next pendingF02-F05/F06/F07/F08/F33norm-to-Q/KV/RoPE/C1projection;F02-F32alreadyreviewedreused. Follow actual normalizedinput/fullHreductions/BF16/SP-CP/currentcoeffrawresidualpre4 contracts,profilefirst;allotherpendingpairsremain. '
        'Alllocalworkcomplete,no remote/device/Torch/CANN/model/capture/replay/implementation. Padv120grantconsumed,nopendingapproval. Do noteditsealedv142/rerunfinalizer;versionedpostcheck/CURRENTmustmatchRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F01–F28/F29/F30/F31/F35：HC 与草稿初始化、context/query、Markov/confidence、metadata 及目标采样的条件关系。整体目标仍未完成。

104组已有草稿入口，通过repeat和seed的精确API/CID确定CPU调用窗口，关联312个decoder层：1872个HC阶段TASK、416个最终collapse TASK及208个repeat/seed TASK，共2496个。16组dummy与88组proposal调用栈分开；它们不是同一client/request/迭代或实际tensor bank身份的证明。另复核72组既有target辅助均值前沿，其mean在当前HC之前，输入来自上一FFNPost。

数据边界明确：target attention前的HC weighted Y、target最终归一化输出、三个checkpoint均值、main_proj/main_norm后的context、draft embedding以及draft最终未归一化collapse互不等价。5120维形状相同不能合并。辅助均值观察的是进入配置层的完整残差，不是该层HCPre的weighted Y。

草稿首层的四通道repeat和FP32 onehot premix允许研究有限数值域内的collapse捷径、广播残差表示及首系数投影重组。但数据相关系数仍供Post和后续pre4，firstPost仍需要初始残差；原算子的连续4H输入ABI不能自动接受新广播表示。Inf×0会使onehot表达式出现NaN；参考FP32累加反例也说明实数权重预折叠不自动保持舍入。没有宣称新的native数值等价。

目前三层context预计算都早于draft repeat。若seed、输入ID、mask、位置和slot metadata已就绪，可研究让embedding、HC及Query准备与context预计算重叠，在每层SWA真正读取前等待它需要的cache行，并保留重叠slot的WAW顺序。它不把context当成query输入，也不自动减少矩阵计算。将context投影延迟到每层query就绪再联合打包，是另一个版本；它会延长全局context bank的读者期限，不能同时领取提前退休的收益。三个draft层或target/draft的同形权重也没有被证明相同。

1536个任意服务时间的三层完整前沿DAG保留了系数/Post/FFN和context就绪期限。加入资源代价后992组较快、544组较慢；理想组合中768组的单项收益之和重复计算了共同缩短的时间，768组与单项之和相同。这些说明组合必须联合计价，不是实测胜率、可达速度或物理上限。当前模型采用充分的全前沿等待，没有把它说成不可拆的全算子barrier。dummy的None-metadata零输出仍单独保留。

draft最终collapse是BF16加权和，不是另一次HcPre。其未归一化结果在可选confidence路径中仍有读者；给LM head的RMS结果不能替代。构造例中两个不同原值经BF16 RMS可得到同一结果，而原值的confidence线性观察不同，因此不能仅因logits已经产生就删除raw collapse。

当前proposal的seed来自当前采样输出计算的next-ID，在decoder之前已写入输入准备；decoder之后复制到输出第0列不等于此时才生成seed。五步Markov有真实ID递归，之后的可选confidence/keep只可能影响后续兼容的验证前缀。若反用这个keep裁剪当前decoder或metadata，会形成“metadata→decoder→confidence→keep→metadata”的依赖环。随后target接受/拒绝还会产生新的next-ID/count，不能由旧proposal keep预测。相同shape、总行数或graph bucket也不足以复用mask、slot和请求世代bank。

目标采样的最终行需求可以有条件限制末层Query/Post/FFN结果，但不能越过必要KV和状态写入边界删除更早层的HC行。构造的causal attention只选择最后一个Query，仍需要另一token的KV：删掉它会把结果6改成10。原始score、重复/有序head行、RNG、bonus、accepted-state及zero-query的cache效果均保留。

原始profile的13个数据库有部分stack ID缺失字符串；本轮2496个选中TASK所需正帧全部解析成功。缺失ID被保留，不补造callchain，也不据此断言路径未运行。脚本修正及原始缺口已登记。

条件关系157→162，待核对438→433，联合组仍26。F01与其它34个主候选的中央两两关系均已在明确范围内条件核对；F01自身的全部分支、实现变体、native合法性及高阶组合仍未关闭。35主候选、7历史框架记录、358语义行/26模块、4472次SP关联、854节点/1346边/48提示的部分图不变。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍待补齐；整模块关闭0，全模块物理MFU与最低时间尚未建立。没有新增服务器或NPU运行。
'''
    updates={cid:dict(review_v142=dict(evidence=names,pairs=comp['new_pairs'],draft_TASK_bindings=2496,target_aux_frontiers_reused=72,scoped_conclusion_changing_gap=gap,existing_broad_conclusion_gap_preserved=True,F01_star_scoped_not_whole=True,no_native_fusion_or_physical_gain_proved=True))for cid in touched}
    inputs=[Path(x['path'])for x in read(O/'hc-draft-inputs-v142.json')['inputs']];refs=read(O/'hc-draft-native-frontiers-v142.json')['databases'];inputs +=[Path(x['raw_database']['path'])for x in refs]+[Path(x['cached_native']['path'])for x in refs]
    findings={'M06':'Distinct target/draft HC epochs/onehot andcoeff/Post/pre4 boundaries;all34F01centralpairs scopenotwholeclosure.','M15':'Raw target snapshot scopedclosure doesnotclose rawdraftcollapse optionalconfidence observer orheadRMS equivalence.','M17':'Target finalquery/output rowdemand stopsatnecessaryKV/state;acceptedoutcome remainsneeded fornext-ID.','M18':'Threeauxmeans enter main_proj/norm context,notweightedHCY;72targetsourcefrontiers retained.','M19':'312draftHC layers/2496TASKs;earlyembedding/HC/query overlapcontext isconditional,newbroadcastABI andinitialresidual retained.','M20':'Seed beforedecoder;latecopy notseedcreation;Markovrecurrence/rawconfidence/futurekeep causalepochs retained.','M21':'Currentinputs versusfutureprefix/acceptedstate metadata distinct;slot/mask/requestgeneration andzeroQeffectsgated.','M22':'RawunresolvedstackIDs13DBs retained;selectedpositiveframesresolve;newstream/alias/capture/lastreader+queuedwriter gatesopen.','M23':'Perlayer neededcontext/querycache deadlines andSPownership;no wholepublicationbarrieror freeearlyretirement.','M24':'Futurekeep cannotprunecurrentnoncausaldecoder;bankreuse requiresactualrequestgeneration andreader/writer retirement.'}
    persist_refinement(version=142,previous=141,title='HC draft/context/metadata/sampling distinct values and causal epochs',summary=dict(filename='hc-draft-summary-v142.json',previous_turn_classification='PROGRESS:v141 HC-MoE4relations/8638TASK+2APIonly/936bridges;PASS90025',**detail,refined_candidates=10,refined_modules=10,branch_records=358,semantic_modules=26,relation_clauses=162,higher_order_groups=26,pending_central_pair_reconciliation=433,F01_star_scoped_conditional_pairs=34,whole_F01_closed=False,partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='All local saved-profile/draftHC/value-epoch/DAG work complete;no remote/device/Torch/CANN/model/replay/service work.'),
        changes=['Bind104draftwindows312layers2496TASKs andreuse72targetaux fronts;16dummy/88proposalancestriesdistinct;rawmissingstrings retained.', 'Reconcile distinct HC/context/draft/rawconfidence inputs and current/futuremetadata/acceptedstate;1536DAGs andnumeric/demand/epochcounterexamples.', 'AddREL142-01..05;162conditionalpairs/26groups/433pending. F01star34/34scopednotwhole;allnative/global/broadercandidategaps retained.'],candidate_updates=updates,module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in findings.items()},remaining=remaining,action=action,jobs=[
        dict(kind='profile',status='error',description='Initialfullrawcallchain stringlookup hit missingSTACK_ID;retainNone/source-positiveframes and13DBmissingIDs,notfabricatednativeevidence.',artifacts=[O/'hc-draft-local-correction-v142.json']),
        dict(kind='profile',status='pass',description='104draftrepeat/seedCPUwindows312layers2496rawCIDTASKs;72reusedtargetaux fronts/typedHCcollapse/ancestries checked.',artifacts=[O/'hc-draft-native-frontiers-v142.json',O/'hc-draft-observed-frontiers-v142.json']+inputs),
        dict(kind='design-check',status='pass',description='1536weightedDAGs andfinite/nonfinite/FP32/rawconfidence/KVdemand/epochcycle witnesses;'+str(pre['checks'])+' preflightchecksPASS,fiveconditionalrelations.',artifacts=[O/'hc-draft-preflight-v142.json',O/'hc-draft-validation-history-v142.json']+[O/n for n in names]+[O/'coverage-current-obligations-v142.json',O/'whole-scope-requirement-audit-v142.json'])],report=report,read_first=names+['coverage-current-obligations-v142.json'],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_hc_draft_native_v142.py',N/'analyze_hc_draft_relation_v142.py',N/'verify_hc_draft_relation_v142.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-draft',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
