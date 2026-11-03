"""Persist explicit MC2return branches andmask-gated Vector work,not an implementation."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v116.json').exists()
    assert read(O/'combine-preflight-v116.json')['status']=='PASS'
    names=['combine-inputs-v116.json','combine-profile-branch-map-v116.json','combine-source-contract-v116.json',
        'combine-row-and-peer-interval-proof-v116.json','combine-mask-gated-work-models-v116.json','combine-guards-and-invalid-rewrites-v116.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v115.json')));d['MC2_return_branch_review_v116']=dict(evidence=names,
            source_predicted_V2=3512,source_predicted_V1=960,mask_state_unknown=True,new_pairs=0,new_groups=0,new_native_promotions=0)
        if base=='candidate-relation-clauses':
            g=next(g for g in d['higher_order_groups']if g['id']=='HG73-MOE')
            g['MC2_return_branch_review_v116']=dict(evidence=names,contract='Freezeoriginalexpert/rankclipselection androunding. V2activeCast/UBcompaction onlyunderexplicitmask/observer/epochcontract;GMreturnspans andoldGMhandoffcredit remainseparate.',whole_group_closed=False)
        write(O/(base+'-v116.json'),d)
    cov=read(O/'candidate-relation-coverage-v116.json')
    obligations=deepcopy(read(O/'coverage-current-obligations-v115.json'))
    for g in obligations['obligations']:
        if g['id']=='G00':g['remaining']=g['remaining'].replace('353 semantic','355 semantic')
        if g['id']=='G10':
            g['remaining']=f"Central{cov['scoped_conditional_pairs']} conditional relations/{cov['higher_order_groups']} groups leave{cov['remaining_pair_relations']} pairs to reconcile;"+g['remaining'].split(';',1)[1]
        if g['id']in ['G01','G04','G06','G07','G08','G09']:
            g['MC2_return_branch_review_v116']=dict(evidence=names,
                V1_V2sourceguards_andV2row_peerintervals_reviewed=True,
                installed_Cast_false_mask_ABI_pinned=True,actual_loaded_kernel_mask_andcopyextents_unbound=True,whole_requirement_closed=False)
        if g['id']=='G13':g['MC2_return_branch_review_v116']=dict(evidence=names,
            genericMC2fusion_capacity_clip_and_v56_CUT_FINE_are_prioranswers=True,
            V2subcore16row_path_now_reviewed=True,do_not_infer8192fromrepeat128withoutmaskmode=True,
            exact_current_mask_orcompiledentry_andreceiver_counts_missing=True)
    obligations.update(parent='coverage-current-obligations-v115.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v116.json',obligations)
    gap=('V1/V2path issource-predicted fromallocatedlocalM*topK,notobservedinternalbranch orreceiverrows. '
        'Needactualloadedsource/compiler/2201maskmode+bits,receiverexpertcounts/clipmask,copylibraryactualbyteextents andallUBpad/FPstatus/error/lifetimeobservers. '
        'FalseCastdoesnotsetmask;repeat128 gives8192onlyinfullNormaldomain. Repeattrimto4*actualm anddependent8192→4096privateUBlayout requirethatdomainandnootherpadreaders. '
        'Do notleakCountermaskintofalseMuls,changeFP16/FLOAT/BF16rounds/scaleepochs,logicalexpert/rankclippingselection,crosscoreflagsorpeerunpermute. '
        'GMread/returnstoresalreadyactualtilegeometry;UBinstructionwork ratiosnotHBM/occupancy/sourceTASKspeedup. Allphysicalcapacities/criticalpath/native/wholeclosures remainopen.')
    matrix=read(O/'module-phase-branch-matrix-v115.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M12'))
    existing=next(r for r in branches if r['id']=='M12-OP-13');existing['evidence']=list(dict.fromkeys(existing['evidence']+names))
    existing['MC2_return_branch_review_v116']=dict(explicit_return_branch_rows=['M12-OP-20','M12-OP-21'],evidence=names)
    for rid,label,count in [('M12-OP-20','V2 allocatedlocalM*topK<=4096:pairedAIV16rowpartition,perpeerintervalreturn andexternalmask-dependentCast/UBwork',3512),
        ('M12-OP-21','V1 allocatedlocalM*topK>4096:perexpert/perpeerreturn path retained;V2microtile/activeCastcredits nottransferred',960)]:
        assert all(r['id']!=rid for r in branches)
        branches.append(dict(id=rid,branch=label,state='SCOPED_CONDITIONAL_SOURCE_GUARD_AND_DATAFLOW_REVIEW;WHOLE_MODULE_OPEN',
            evidence=names,gap=gap,scope='Fixedcapturedinputgeometry predictsselectedsourcebranchonly;otherloadedimplementations/runtimeoperands remainunbound.',
            source_predicted_executions=count,actual_native_internal_branch_attested=False))
    assert len(branches)==21
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v115.json'));audit.update(previous_goal_turn='PROGRESS:v114normalhashrange/Hfactor+v115coveragecountcorrection;v115manifestPASS34',input_checkpoint=115,followup_checkpoint=116)
    r=next(r for r in audit['requirements']if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES');r['branch_records']=355;r.setdefault('evidence',[]).extend(names)
    audit['MC2_return_branch_review_v116']=dict(evidence=names,explicit_source_branch_rows_added=2,
        former_generic_MC2branch_retained=True,actual_mask_or_receivecounts_not_recovered=True,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v116.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v116.json',dict(status='FIVE_MC2_NATIVE_GEOMETRIES_THEN_ONLY_RETURN_BRANCH_AND_CAST_MASK_OPERATOR_CLAUSES',
        questions=['WhichreturnpathdoesallocatedM*topKguardpredict?',
            'Whichrows/peerintervalsandUB/GMbytesdoV2microtilesactuallydemand?',
            'DoesCastfalseuseitsliteral-1mask,andwhatmaskpremiseisneededforrepeat/UBcuts?'],
        old_frozen_MC2_parent_source_controls_match=True,installed_SDK_header_arch2201_only=True,
        source_retrieval='OneboundedCPUfilenameinventory then10archivefilecopies(9uniqueSDKpaths) withacl_rtSHAcontrol;SSHclosed',
        no_model_Torch_NPU_context_capture_replay_or_targetimplementation=True,
        native_Pad_probe='StillPREPARED_NOT_EXECUTED;PENDING_REPLY',overall_complete=False))
    write(O/'combine-validation-history-v116.json',dict(events=[
        'v115latestmanifest34files verified. Existingv56/v97/v98genericfusion/internalGM/FP16round/capacityclip/C65536 evidence reused,notclaimednew.',
        'Boundedlocaloperatorsearch foundCombineV2notexplicitlyinventoried. Partialv56archive lackschildV2epilogue;fullfrozencsrcsnapshotcontainsit,andbothparentkernel/headerSHAs matchv56copies.',
        'FiveexistingMC2geometries predictV2=3512,V1=960 underselectedsourceguard;inputMallocatednotactive/receiverM. No originalnativeattributionpromoted.',
        'Repeat128 initiallysuggestedfixed8192Castwork,butboolfalse requiresmaskABIcheck. Officialmaskguide andinstalledCANN9.2source2201frontend/impl proveexternalmask;actualinternalmaskremainsunrecorded. No unconditionalpaddingworkclaim.',
        'SDKfilenameinventory12secondcap finished0.178s/7595dirs/65051filenames;noTorch/CANNlibrary/device load.10exactarchivecopiesallREAD,9uniquepaths,acl_rtcontrolSHA62d127... matches. SSHclosed.',
        'Finalizercompilecheck caughtaninvalidslashinoneJSONmetadata keyword beforeanyscriptbody/sharedwrite;correctedandcompilevalidatedbeforepersistence. Operatoranalysisunchanged.',
        '514rowpartitionreferences and7565peerintervalreferences pass;independentDPchecks369fragmentationboundcases. NormalfullmaskCastpredicateandrewriteusepointwisedependency/roundingcontract,notnativeprecisiontest.',
        'Countermaskleaktoyexample changesactiveperrowvalues[2,3,4]to[2,6,24];noexecutednativebugclaim. Repeat-onlytrimunderfullNormalavoidsmaskmutation. UBcompactiondependsontailCasttrimandallalias/observercontracts.',
        'TwoexplicitM12returnguardrows added(353→355)withallmoduleclosuresopen. Relation/graph/native/PMU/phase/ambiguitycounts unchanged. OriginalPadprobe awaitsreply andwasnotrun.'],overall_complete=False))
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('Currentv116:355semanticrows/26modulesincludes2explicitMC2returnguards;sourcepredict3512V2/960V1. FalseCastexternalmask makes8192work/2..32paddingratio conditionalonfullNormal. Actualmask/receivercounts/copyextentsstillunknown. 134pairs/26groups/461pendingunchanged;Pad44Bprobe notrunpendingexplicitreply.')
    report='''本轮补查MC2返回路径，复用已有4472次DispatchFFNCombine输入几何，没有新增推理或归属提升。所选源码按allocated local M×topK是否大于4096选择CombineV1/V2，输入形状代入得到3512次V2、960次V1；这只是源码兼容条件下的分支预测，未观测到内部flag。M3×k6选择V2不等于接收端每个expert有3行，实际receiver counts与mask值仍未记录。

V2使用GMM2的128×256分块，再把每个M块划成16行微块，由一对AIV子核分担。ceil(m/16)个微块中，子核0取前floor(q/2)个，子核1取剩余，最后不足16行的块在子核1；1..16行的小M块只有子核1工作。这不是所有新融合实现的并行度上限。已检查514个行分配参考和7565个peer区间参考：每个有效接收行恰好覆盖一次，返回段是微块与各source-rank expert计数前缀的交集。它们证明源码坐标和语义store span，不能替代编译后DMA字节范围或实际地址/lifetime证据。

进一步定点读取V2 epilogue发现，两次Cast均写repeat=128，UB max_len为8192，实际GM输入/输出由微块有效行列决定。但Cast<float,half,false>及Cast<BF16,float,false>的false表示使用外部mask，调用的-1不设置mask。只读固定安装CANN9.2的2201前端与实现，确认该bool被传给CastImpl，false跳过SetMask并进入f162f32/f322bf16r；Normal/Counter模式还受外部状态控制。因此不能把repeat128直接当作实际8192元素或固定浪费。[官方mask说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0024.html)

在明确的Normal全64-lane mask域内，repeat128的确对应每方向8192个转换元素，而当前有效微块最多16×256=4096。H5120分20个列块，令R为clipping后实际总接收行，S=sum ceil(r_e/16)，该固定源码每方向转换元素数为163840S，有效FP16→FLOAT或FLOAT→BF16值转换为5120R；比值32S/R位于2..32。若已知R和正行数expert数Z，S的精确无额外expert容量限制范围为max(Z,ceil(R/16))到Z+floor((R-Z)/16)，独立动态规划复核369个有限案例。实际r_e缺失，不能用本地M3代入，也不能把元素数比例视为整个融合算子的提速。

条件优化可以具体收敛：H5120的每个列块为256列，有效行数m下的元素数256m正好是64的倍数；若入口确为Normal全mask，可把两个Cast的repeat从128改为4m，保持false、stride、FP16→FLOAT、每行FP32 scale和BF16 RINT边界。这样不修改mask状态。若改用Counter而没有恢复，后续Muls<false>也会继承Counter count；一个纯CPU参考中，原本三行分别得到[2,3,4]，错误遗留的Counter会得到[2,6,24]。这是对盲改规则的反例，不是已测到原生bug。实际Counter或局部mask域须单独建模。

私有UB布局缩小是依赖选项：先消除所有padding Cast访问，才可考虑把max_len8192改为4096。给定FP16输入、BF16输出、FLOAT中间和每行scale、双缓冲，源码局部布局参考从131328B变为65664B；旧repeat128若保留会越过缩小后的buffer。还要保持scale cache、事件、SwiGLU/return共享UB的旧读者和FPstatus/错误观察。这个数值不是GM workspace、HBM、VRAM释放量，也没有证明占用率改善。原GM读和peer返回存储已按实际微块范围执行，因此仅裁剪UB转换不减少这些GM字节。

其他组合约束继续保留：receiver clipping选择依赖原expert和rank顺序，盲目重排即使保留R也可能保留不同assignments；应先冻结旧keep rows/offset，再讨论物理排程。C/C2、A/activation别名、跨核flag、FP16舍入、per-token scale epoch、peer bank和最终weighted unpermute不能随意删除。此前v56的细分流水/工作量切分、v97容量模型和v98 C65536证据均是已有答案；本轮没有重复计为新融合机会。

为核对Cast ABI，在原166做了一次限12秒的CPU SDK文件名查询，实际0.178秒结束；随后只读复制10份文件（9个唯一路径），acl_rt SHA控制匹配，SSH已关闭。没有Torch/NPU库加载、应用NPU上下文、图捕获、replay或模型执行。现有源码父kernel/header与完整frozen树SHA一致，但当前SDK/源码不证明历史执行加载的二进制或内部mask状态。

M12语义台账新增V2/V1两条显式guard记录，353→355条，26个暂定模块保持open；134条条件关系、26个联合组、461对待核对不变。9类/1352次原生执行、849缺PMU、16未绑定phase图和3136含糊归属不变，0个整模块关闭，物理时间上界/MFU/最低模块仍未确立。原生Pad小探针继续等待单独明确回复，本轮未运行；整体目标继续进行。
'''
    action=('继续原Goal，v116PROGRESS，先读combine-summary/report/branchmap/source/rowproof/maskwork/invalidrewrites/current-obligations-v116；verify_combine_v116须PASS。'
        'v115manifestPASS34，v116preflightPASS645；新M12-OP20/21显式returnguard使353→355semanticrows26modules。'
        '已有4472MC2几何按selectedsourceallocatedM*k>4096预测V1960/V23512，非actualinternalflag，Mlocal3≠receiverexpertrows。'
        'V2每128×256块按16rows分pairedAIV，最后partial在subcore1。514row/7565peerintervalrefs，369independentfragmentDP；只语义span不是compiledDMAextent。'
        'V2两Castfalse repeat128/maxlen8192，SDK9.2 __NPU_ARCH__2201frontend→dav_c220 false不SetMask，-1placeholder。实际maskunknown，不能断言8192。只在Normalfull64域每方向8192，H5120下163840S vs5120R，ratio32S/R∈[2,32]，R/r_e实际缺。'
        '条件candidate repeat128→4*actualm不改mask保留F16→FLOATscale→BF16RINT；先裁尾Cast后才可UB8192→4096（双bank131328→65664B布局，不是GM/HBM/VRAM/occupancygain）。Countermask泄露到falseMuls反例[2,6,24]，flags/alias/readers/FPstatus/scaleepochs全部保留。'
        '原expert/rankclipselection冻结再讨论physicalreorder。v56FINE/CUT、v97C/workspace、v98C65536是prioranswer，别重读泛化为未分析。'
        '远程CPUfilenameinventory0.178s/7595dirs65051filenames+10SDKarchivecopies(9unique)已结束SSHclosed，无Torch/NPUlib/context/capture/model；allserverjobsdone。'
        '继续真实未分析operator关系/分支或补actualmask/loadedABI/receivercounts关键证据；No source-onlyclaim转为固定负载已证。134/26groups/461、355/26modules、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU均保持。'
        'v111新Pad44BBOOL20→24/valueTrue async仍待明确reply，禁止run；v109最小probe授权已完成消耗。若新获准用v117独立记录，按v111idle/container/SHAguards一capture/noReplay/noModel/noopaqueGPUderef。'
        'TaskACTIVE Loop004EVALUATING GoalACTIVE；不改已完成v116或重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=116,previous=115,title='MC2return branch,row writes andmask-gated Vector work',
        summary=dict(filename='combine-summary-v116.json',previous_turn_classification='PROGRESS:v114normalhash/Hfactor+v115coveragecountcorrection;v115manifestPASS34',
            existing_MC2_executions=4472,source_predicted_V2_executions=3512,source_predicted_V1_executions=960,
            row_partition_cases=514,peer_segment_cases=7565,independent_fragmentation_cases=369,
            installed_SDK_archive_copies=10,unique_SDK_source_paths=9,explicit_return_branch_rows_added=2,
            conditional_full_Normal_Cast_padding_ratio_range=[2,32],actual_mask_state_attested=False,
            refined_candidates=1,refined_modules=1,branch_records=355,semantic_modules=26,new_pairs=0,new_groups=0,
            relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            new_device_work=False,implementation=False,native_test=False,targeted_Pad_probe='PREPARED_NOT_EXECUTED;PENDING_ASYNC_REPLY',
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            remote_job_status='OneboundedCPUfilenameinventory and10exactSDKarchivesfinished;SSHclosed. No model/Torch/NPUlib/context/capture/replay. Padprobe unexecuted.'),
        changes=['AddtwoexplicitMC2returnguardbranches:existinginputgeometries predict3512V2/960V1,withoutinternalflag/receiver-countpromotion.',
            'PinV2paired16row/expert-peerintervalpath andinstalledSDKCastfalseexternalmaskABI;rejectunconditional8192elementinferencefromrepeat128.',
            'Derivemask-gatedactive-repeat/privateUBlayoutandfragmentationworkcases withrounding/alias/FPstatus/scale/event/clippinggates;noGMbyte/MFU/speedupclaim.'],
        candidate_updates={'F21':dict(review_v116=dict(evidence=names,source_predicted_V2=3512,source_predicted_V1=960,
            row_reference_cases=514,peer_segment_cases=7565,mask_actual_unknown=True,
            full_Normal_active_Cast_anddependent_UBlayout_candidate=True,actual_loaded_ABI_writeextents_or_precision_attested=False,
            new_device_work=False,implementation=False),conclusion_changing_gap=gap)},
        module_updates={'M12':dict(candidates=['F21'],finding='TwoexplicitMC2returnguards andV2paired16row/peerwrite/maskconditions nowinventoried. Repeattrim/privateUBlayout conditionalonfullNormalmask;noactualreceiver/mask/GMtraffic orruntimegainclaim.',evidence=names,gap=gap)},
        semantic_branches={'M12':branches},remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='ReusefiveMC2geometries/4472exec,selectedfrozenparentSHAcontrols and10installedSDKarchivecopies;sourceguardpredict3512V2/960V1,actualmaskunbound.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='514rowpartitions7565peerintervalrefs369independentfragmentbounds;Castfalsemaskgate andactive-repeat/UBdependency/counterexamples;twoexplicitsemanticrowsadded.',artifacts=[O/'combine-preflight-v116.json',O/'combine-validation-history-v116.json',O/'whole-scope-requirement-audit-v116.json',O/'coverage-current-obligations-v116.json'])],
        report=report,read_first=names+['coverage-current-obligations-v116.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_combine_v116.py',N/'verify_combine_v116.py',N/'read_combine_sdk_inventory166_v116.py',N/'read_combine_sdk166_v116.py',N/'read_combine_sdk_forward166_v116.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='combine',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
