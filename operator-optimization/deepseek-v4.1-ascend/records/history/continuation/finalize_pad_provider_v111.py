"""Persist native provider protocol evidence and a prepared,unexecuted targeted probe."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v111.json').exists()
    assert read(O/'pad-provider-preflight-v111.json')['status']=='PASS'
    names=['pad-provider-inputs-v111.json','pad-provider-source-contract-v111.json','pad-provider-metadata-contract-v111.json',
        'pad-provider-native-reconciliation-v111.json','MemSet-default-zero-metadata-model-v111.json','pad-barrier-fusion-conditions-v111.json','native-pad-probe-plan-v111.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-operator-footprints','candidate-pair-navigation','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v110.json')));d['native_pad_provider_review_v111']=dict(evidence=names,
            original_MemSet_executions_reused=1272,installed_provider_only=True,new_pairs=0,new_native_promotions=0)
        write(O/(base+'-v111.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v110.json'))
    for r in obligations['obligations']:
        if r['id']in ['G01','G03','G05','G07','G09']:
            r['native_pad_provider_review_v111']=dict(evidence=names,source_current_A3_barrier_workspace_bytes=1536,
                eight_PadV3_metadata_and16MemSet_variants=True,historical_loaded_binary_addresses_mode_or_extent_bound=False,
                new_targeted_native_probe='PREPARED_NOT_EXECUTED;PENDING_REPLY')
        if r['id']=='G13':
            r['native_pad_provider_review_v111']=dict(evidence=names,legacy_dynamic_provider_reset_barrier_source_reviewed=True,
                do_not_repeat_old_v72_owner_or_fullwriter_findings=True,actual_native_packets_and_guards_missing=True)
    obligations.update(parent='coverage-current-obligations-v110.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v111.json',obligations)
    gap=('Pinned currentA3legacyPadV3 metadata declares one1536B zero-initialized barrier workspace;source MODE4 skips its only barrier reference,non4fills/barriers/copies. '
        'This identifies a specific source/provider subdomain,not the historicalloadedkernel/guard/destination/writeextent. '
        'Original7MemSet/1272 and2BOOL24/80 remain pending. ActualPadmode,packet/compilerABI,relativeinlineflowtable andall workspace reader/alias/retirement/nextreset epochs require proof. '
        'No skip/reset-hoist or wholeoutput-byte attribution fromname/blockcount alone. CPU96B record model coversdefaultFLOAT32zeroonly;packetpacking/inputorder remainsa hypothesis. '
        'New44BBOOL20→24/valueTrue probe prepared,not executed,pending explicitreply;v109probe finished andnot thisauthorization. No physicalMFU/speedup/wholeclosure.')
    matrix=read(O/'module-phase-branch-matrix-v110.json');semantic={}
    selected={'M12':['M12-OP-11'],'M19':['M19-OP-05']}
    for mid,ids in selected.items():
        branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']==mid))
        for r in branches:
            if r['id']in ids:
                r['evidence']=list(dict.fromkeys(r['evidence']+names));r['gap']=gap
                r['native_pad_provider_review_v111']=dict(source_protocol_cases=['MODE4_no_barrier_reference','NON4_fill_barrier_copy'],
                    current_source_workspace_bytes=1536,actual_original_operand_or_guard_bound=False,probe='PREPARED_NOT_EXECUTED',evidence=names)
        semantic[mid]=branches
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v110.json'));audit.update(previous_goal_turn='PROGRESS:v110 histogram/penalty source andrequest-prefix representations;postcheck PASS376',input_checkpoint=110,followup_checkpoint=111)
    audit['native_pad_provider_review_v111']=dict(evidence=names,source_current_provider_and_declared_workspace_pinned=True,
        original_call_addresses_or_guards_not_recovered=True,metadata_models_not_physical_measurements=True,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v111.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v111.json',dict(status='EXACT_SAVED_MEMSET_NAME_THEN_ONLY_MATCHING_INSTALLED_PAD_MEMSET_PROVIDER',
        questions=['Is this native PadV3 initializer the same route as the prior byte/D32 MEMCPY probe?',
            'Which private storage/initializer does the installed matchingA3 provider declare?',
            'When does the source read the barrier workspace,andwhat metadata determines its writes?',
            'What exact tiny native operator capture would discriminate the remaining ABI/guard questions?'],
        read_only_archives_and_one_bounded_filename_inventory=True,unrelated_arch35_or_grad_source_not_read=True,
        cached_original_SP_MASK_valueTrue_checked=True,new_device_work=False,target_implementation=False,
        source_package_not_historical_loaded_ABI=True,overall_complete=False))
    write(O/'pad-provider-public-search-status-v111.json',dict(queries=['PadV3AiCore_MemSet','MemSet mem_set.py Ascend','MemSetKernelResource','MemSetTilingData'],
        primary_entry='https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/aolapi/atlasascendc_api_07_1118.html',
        result='Reserved/versioned resource-API documentation did not establish installedkernel ABI. No such function was called;installed sources andmatching metadata/binary hashes provide the scoped evidence instead.',
        no_public_browser_or_CUA=True,search_failure_not_operator_impossibility=True,overall_complete=False))
    write(O/'pad-provider-validation-history-v111.json',dict(events=[
        'Six guessed original provider paths were absent;one bounded filename inventory found ops_legacy placement. Missing paths retained,not an absence-of-source claim.',
        'CPU analyzer initially passed a string to the repository Path-only JSON reader;stopped before shared/checkpoint writes and was corrected.',
        'Prepared probe initially usedpadFalse;originalSP_MASK source showsTrue. Corrected the unexecuted script/plan andvalidated before persistence;shape/operator/budget ofpendingquestionunchanged.',
        'Source/metadata offsets andthreebinary checksums independently checked;all original native operand gaps remain.'],new_device_work=False,overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Current obligations-v111. New nativeBOOL20→24 padTrue probe is prepared andawaiting explicitreply;do not run beforeyes. v109completedprobe was a different MEMCPY/zeros route. Current1536B source workspace,96B tiling record andopParaSize are distinct quantities;none is historical measuredMemSettraffic.')
    report='''本轮回到原profile的7类MemSet/1272次执行。它们全部记录为aclnnConstantPadNd_PadV3AiCore_MemSet，已有v72父调用归属复用，未重新提升归属。全部1272个保存实例的blockNum为48、mixBlockNum为0；这不能说明48个block都在有效写入，也不能给出字节数。

只读取得安装包的原生provider材料。六个初始猜测路径不存在，经一次限时文件名查询找到ops_legacy位置；没有读取无关arch35/grad实现正文。固定了16份A3 MemSet编译元数据、8份相关dtype/contiguity的PadV3元数据，以及与元数据内SHA一致的3个二进制。另定点核对原SP_MASK调用的value=True；本轮没有加载NPU库、运行模型、捕获或replay图。

这组当前A3legacy PadV3元数据均声明一个1536字节workspace，parameters[4]标为float32零初始化。源码对应INT64[48*4]的sync_workspace，使用workspace/atomic标记。它为prependedMemSet的用途提供了具体候选：它可能清理屏障状态，而不是按整个padding输出体积写零。它仍是当前provider的声明，不能替代原profile每次调用的地址、实际加载binary和tiling证据。

新的条件分支会改变融合结论。选定源码在非MODE4时填充输出，再调用block_barrier(sync_workspace)，随后复制输入；MODE4跳过这个外层fill和该文件唯一的sync_workspace屏障读者。因此，精确MODE4/privateworkspace/all-reader/后续reset与retirement合同成立时，可研究跳过无用reset；其他分支不能把逐核清零随意搬进PadV3。若使用新epoch协议，必须证明初态、wrap、参与核和跨调用读者；已有v72的disjoint fullwriter可以改换整个协议，但不是本轮新发现或已实现方案。

写入量也必须按新图重算。选定非4的显式源码参考中，输入I字节、输出O字节，加独立reset W，逻辑访问为2I+O+W；disjoint fullwriter的显式输出参考是I+O，另计映射和必要状态。W=1536目前只是上述provider子域的声明。不能将差额当作HBM、峰值内存、可回收TASK时间或可与v72重复相加的收益。MODE4必须使用自己的成本图，不能沿用外层fill/barrier开销。

动态MemSet的默认FLOAT32零初始化源路径每workspace使用96字节flowtable：24个INT32存储槽，包含18个混合INT32/INT64逻辑字段。已建立CPU字段布局及相对store-span模型。相同1536字节可以由一个writer或48个writer提交；100字节尾部在支持data_move_pad的路径写100字节，在对齐DMA参考路径可写128字节。这些是给定字段后的源码提交范围，不是已证明合法的原生tiling、语义必写量或物理流量。

16份元数据的workspace数量为1..16，parameters个数为N+2，opParaSize=96N+8。opParaSize、参数包大小、96N flowtable和tensor填充范围不能混同。另准备了只解析已经rendered参数包的CPU方法：只有匹配元数据、包长和唯一有界in-packet指针时才定位flowtable，完全不解引用设备指针。这个包布局仍是假设，须受控native dump验证，不能用于强解原profile缺失字段。

为验证这条ABI通路，已准备一次BOOL[20]→BOOL[24]、padding[0,4]/valueTrue的原生ConstantPadNd小图探针，匹配保存SP_MASK的算子形状和值。应用输入/输出共44B，另有内部cast/workspace、驱动和分配器开销；只读取自身任务元数据，不加载模型、不replay、不读取张量内容或opaque设备地址。当前探针未运行，新的明确授权问题待回复；先前v109已完成的小图使用不同MEMCPY路由，不替代这次验证。

本轮细化F20/F28/F31及M12/M19的条件provider合同。35项模型候选、353条语义记录/26暂定模块、132条条件关系/25组/463对待核对均不变；原7类MemSet及2类BOOL[24]共9类/1352次执行仍缺实际字段。没有整模块关闭、物理MFU、实测提速或整体完成结论。
'''
    action=('继续原Goal,v111PROGRESS：先读pad-provider-summary/report/source/metadata/native-reconciliation/96Bmodel/fusionconditions和native-pad-probe-plan/current-obligations-v111。v110PASS376。'
        '已读原166安装CANN9.2 ops_legacy provider，16MemSetN1..16meta/8A3PadV3meta/3.o SHA match；一次boundedfilename inventory后archive，SSHclosed无NPU/model/newcapture。'
        '新实质证据：8Padmeta workspace1 size1536 parameters[4]FLOAT32init0，源码sync_workspace INT64[48*4]。非MODE4 fill→blockbarrier→copy，MODE4跳过唯一sync读；条件deadreset/epoch重组不是已批准删除，v72fullwriter是prioranswer。'
        '全部1272原MemSet block48/mix0同名，12typedPadcohorts，48blocks≠activewriters/bytes；historicalkernelhash/calladdr/mode未bind，9/1352remain。'
        'DefaultF32zero非long动态MemSet96B/18mixedfields sourceoffsets已建CPUspans，K1/K48都可写1536；tail100pad vs128aligned；不是HBM/actualtiling。MetaopPara96N+8，parametersN+2，inlinepacketpointer/schema尚待nativevalidation，不读deviceopaque。'
        '用户新的Padprobe async问题待回复，affirmative前不得run。准备脚本diagnostic-native-pad-probe-v111/inspect_pad_graph.py，固定原SPmask20→24 BOOL padTrue，44BappIO(非totalalloc)，3meta+binary/lib/headerhashguard，own/tmp，45stimeout，空闲device0，一capture/noReplay/noModel/noTensorPayload/Deref。'
        '旧v109最小探针已完成只得到MEMCPY/BOOLmetadata，不授权此新Pad。若批准按plan先freshoccupancy/containerrunning，不占卡不重启，复制exactscript stdin到独立输出；若占用跳过。若kernel/args协议不匹配保留UNKNOWN，不能盲读GPUtiling。'
        '新运行结果须单独v112记录，别改已完成v111。优先验证MemSetinputaddr vsPadworkspace tag和boundedinline tiling，再决定原7regime能否任何提升；newowncall不自动等同historicalall1272。另G08physicalcapacity仍open，可继续定点离线。'
        '35/353/26、132/25/463、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU不变。GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_pad_provider_v111必须PASS；不重跑完成finalizer或改完成artifacts。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    updates={cid:dict(review_v111=dict(evidence=names,current_source_barrier_workspace_bytes=1536,
        conditional_mode4_unused_reset_and_non4_phase_contract=True,historical_native_operands_bound=False,
        metadata_record_bytes=96,packet_ABI_validated=False,new_device_work=False,implementation=False),conclusion_changing_gap=gap)for cid in ['F20','F28','F31']}
    persist_refinement(version=111,previous=110,title='Native PadV3 barrier-workspace and MemSet metadata contract',
        summary=dict(filename='pad-provider-summary-v111.json',previous_turn_classification='PROGRESS:v110 penalty histogram/source andcrossiteration contracts;postcheck PASS376',
            MemSet_metadata_variants=16,PadV3_metadata_variants=8,binaries_verified=3,existing_MemSet_executions=1272,typed_Pad_cohorts=12,
            source_tiling_bytes_per_workspace=96,source_layout_fields=18,metadata_reference_cases=5,current_A3_declared_barrier_workspace_bytes=1536,
            targeted_probe='PREPARED_NOT_EXECUTED;PENDING_ASYNC_REPLY',new_device_work=False,implementation=False,native_test=False,
            refined_candidates=3,refined_modules=2,branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,
            relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            remote_job_status='Read-only operator-file archives andonebounded filename inventory finished;SSHclosed. CPU metadata models finished;no newNPU/model/library/capture/replay. NewtargetedPadprobe prepared only.'),
        changes=['Pin matching A3legacy PadV3/MemSet sources,16/8metadata andthreebinary checksums;identify declared1536Bbarrier workspace rather than infer fulloutput clear.',
            'Separate MODE4 no-barrier-reference andnon4fill/barrier/copy reset-removal/epoch conditions;retain v72directfullwriter as prioranswer.',
            'Derive source96B mixed-field layout,relative-write examples andguardedinline-packet hypothesis;prepare exact44BnativePadcase without running it.'],
        candidate_updates=updates,module_updates={'M12':dict(candidates=['F20'],finding='Currentprovider gives a concrete1536Bzeroedbarrierworkspace candidate forPad-ownedMemSet,withMODE4/non4conditions. Historicalcallfields/guard andallreaders stillunbound.',evidence=names,gap=gap),
            'M19':dict(candidates=['F28','F31'],finding='SPmask originalpadTrue andprivatePadV3scratch are distinct. Source/reset/readiness cases nowpinned;prepared20→24nativePadprobe targets actualprovider family,notv109MEMCPYroute.',evidence=names,gap=gap)},
        semantic_branches=semantic,remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reuse1272exactPadownedMemSet executions;boundedinstalledsource retrieval16MemSet+8Padmeta+3binarycontrols. Originaloperands remain unbound.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Sourcebarrier/resetmodecontract,96Bfieldlayout andfiveCPUspan/packetreferences;new44Bprobe remains unexecuted pendingreply.',artifacts=[O/'pad-provider-preflight-v111.json',O/'pad-provider-validation-history-v111.json',O/'whole-scope-requirement-audit-v111.json',O/'coverage-current-obligations-v111.json'])],
        report=report,read_first=names+['coverage-current-obligations-v111.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_pad_provider_v111.py',N/'verify_pad_provider_v111.py',N/'decode_memset_metadata_v111.py',N/'read_pad_provider166_v111.py',N/'read_discovered_pad_provider166_v111.py',N/'read_pad_metadata166_v111.py',N/'read_pad_binaries166_v111.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='pad-provider',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
