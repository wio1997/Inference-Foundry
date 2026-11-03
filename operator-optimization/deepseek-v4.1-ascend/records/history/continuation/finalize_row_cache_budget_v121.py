"""Reconcile the explicit F24/F34 memory-resource relation without claiming freed VRAM."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v121.json').exists()
    assert read(O/'row-cache-budget-preflight-v121.json')['status']=='PASS'
    names=['row-cache-budget-inputs-v121.json','row-cache-representation-contract-v121.json',
        'row-cache-common-budget-models-v121.json','row-cache-miss-traffic-and-memo-demand-v121.json','row-cache-retirement-and-numerical-contract-v121.json']
    detail=dict(evidence=names,new_relation='REL121-01',budget_models=144,capacity_reference_models=36,
        actual_memory_credit_hitrate_or_allocator_binding=False,new_native_promotions=0,whole_requirement_closed=False)
    rel=deepcopy(read(O/'candidate-relation-clauses-v120.json'))
    assert not any({r['a'],r['b']}=={'F24','F34'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL121-01',a='F24',b='F34',relation='SHARED_DEVICE_BUDGET_COMPETITION_AND_CONDITIONAL_STORAGE_REUSE',
        operator_claim='EncodedEngramrowcachecompeteswithprivateC2banks/WKVmemo/commonKVblocksinthe sameNPUarena. CurrentevenringviewdoesnotfreeHBM;onlygenuinecommonIDreleaseplusonebudgetremaindercanoffsetselectedcacheallocation. Cachehitsandretirement/readinessremainseparate.',
        evidence=names,applicable_domain='Normalv79unpackedpool/nooverrideandselectedexactencodedrowcachelayout/samerank-ownerbudget;noactualnewcacheorcredit.',
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    g=next(g for g in rel['higher_order_groups']if g['id']=='HG85-SHARED-MEMORY-BUDGET');assert 'F24'not in g['members'];g['members'].append('F24')
    g['evidence']=list(dict.fromkeys(g['evidence']+names));g['encoded_row_cache_budget_v121']=dict(**detail,
        contract='Oneco-residentpeakbudgetforrowcacheK,fullWKVmemoJ,privateC2owners,MC2workspaceandprotocol/migrationstorage. Selectupstreamdemandfirst;nohostreadcreditforbothrowcacheandmemoifmemoalreadyeliminatesthelookup.')
    selected={tuple(sorted((r['a'],r['b'])))for r in rel['clauses']};assert len(selected)==135 and len(rel['higher_order_groups'])==26
    rel.update(parent_relation_index='candidate-relation-clauses-v120.json',clause_count=len(rel['clauses']),selected_pair_count=len(selected),
        new_pair_clauses=1,new_higher_order_groups=0,encoded_row_cache_budget_v121=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v121.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v120.json'))
    for p in cov['pairs']:
        ids=[r['id']for r in rel['clauses']if {r['a'],r['b']}=={p['a'],p['b']}]
        if {p['a'],p['b']}=={'F24','F34'}:p.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=ids,whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v120.json',parent_relation_index='candidate-relation-clauses-v121.json',
        scoped_conditional_pairs=len(selected),remaining_pair_relations=595-len(selected),higher_order_groups=26,
        scope='135scopedconditionalpairs/26groups/35modelcandidates;460centralpairreconciliationsremain. F24/F34resourcebudgetreviewdoesnotproveactualcredit,cachehitsoralljointvariants.',encoded_row_cache_budget_v121=detail,overall_complete=False)
    write(O/'candidate-relation-coverage-v121.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v120.json'));p=next(p for p in nav['pairs']if{p['a'],p['b']}=={'F24','F34'})
    p.update(conditional_clause_ids=['REL121-01'],semantic_pair_reconciliation_done_in_stated_scope=True,whole_pair_all_variants_closed=False,
        resource_coupling='OneNPUarena:encodedrowcacheallocationversusC2privatebank/common-IDcapacity;notsharedoperatoridentity.',evidence=names)
    nav.update(parent='candidate-pair-navigation-v120.json',parent_footprints='candidate-operator-footprints-v121.json',parent_relation_index='candidate-relation-clauses-v121.json',
        scoped_conditional_pairs=135,remaining_central_pair_reconciliation=460,encoded_row_cache_budget_v121=detail)
    write(O/'candidate-pair-navigation-v121.json',nav)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v120.json'))
    for r in footprints['candidates']:
        if r['id']in ['F24','F34']:r['encoded_row_cache_budget_v121']=dict(evidence=names,resource_coupling_not_work_identity=True,new_target_operator_atoms=0)
    footprints.update(parent='candidate-operator-footprints-v120.json');write(O/'candidate-operator-footprints-v121.json',footprints)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v120.json'));op['encoded_row_cache_budget_v121']=detail;write(O/'operator-analysis-coverage-ledger-v121.json',op)
    obligations=deepcopy(read(O/'coverage-current-obligations-v120.json'))
    for r in obligations['obligations']:
        if r['id']=='G10':r['remaining']=f"Central{cov['scoped_conditional_pairs']} conditional relations/{cov['higher_order_groups']} groups leave{cov['remaining_pair_relations']} pairs to reconcile;"+r['remaining'].split(';',1)[1]
        if r['id']in ['G05','G06','G07','G08','G09','G10','G13']:r['encoded_row_cache_budget_v121']=deepcopy(detail)
        if r['id']=='G13':r['remaining']=('F24/F34encodedrowcache/common-IDbudgetrelationnowreviewed;currentevenviewC2compactionisnotfreeHBM. Needactualsame-deviceB/r,live/residentS/U,releasedIDs,allco-residentstorage/retirement andformat/keyepoch/rowaccess/hitrate/allocatorproof. '
            'v120Padone-shotiscomplete/consumed;v119observerjointcounts,v118Engrammask,v117descriptorcapandv85F26rawcreditareprioranswers,notunreadoperatorbodies.')
    obligations.update(parent='coverage-current-obligations-v120.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v121.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v120.json'));audit.update(previous_goal_turn='PROGRESS:v120oneauthorizednativePadcaptureandbounded152/96/32contract;postcheckPASS152',input_checkpoint=120,followup_checkpoint=121)
    audit['encoded_row_cache_budget_v121']=deepcopy(detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=135,higher_order_groups=26,central_pairs_not_reconciled=460)
        if r['id']in ['CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v121.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v121.json',dict(status='EXISTING_UVA_FORMAT_AND_RING_POOL_PROFILES_THEN_ONLY_COMMON_BUDGET_AND_READER_CONTRACTS',
        questions=read(O/names[0])['operator_questions'],existing_source_and_profile_contracts_reused=True,
        no_new_target_source_files_or_remote_or_device_work=True,Pad_authorization='GRANTED_AND_CONSUMED;noequivalentprobe',overall_complete=False))
    write(O/'row-cache-budget-validation-history-v121.json',dict(events=[
        'v120revalidatedPASS152/manifest62. InitialDSparkbankquestionswerealreadycoveredconditionallyinv113;notcountedasnewscope.',
        'CentralF24/F34pairwasexplicitlypending,sharedM21/M22/M24associationsbutnopriorrelation. ExistingHG85containsWKVmemo/privateC2butnotencodedlookuprowcache. Namedmemoryarena/allocatorinteractionselected,notarbitraryblankpairenumeration.',
        'v79proveevenringviewcompactiondoesnotchangepoolC540928orallocatedN;three2.268GBviewsarenotindependentallocations. NoactualVRAMcreditclaimed. v85rawcredit335MBisF26tailproposal,notF34.',
        'Reusev62payloadsBF16512/INT8group32288/MXFP8264. Explicitillustrative32Balignment+64Bmetadata yields576/352/352Bslots;actualformat/layout/metadata/hitsunbound.',
        '144jointbudgetcasesand36no-old-free-ID-lossfrontiersuseonecommonremainder. S=U4privateevenringplus4096INT8slots costs2228224B andmaylose1globalIDorequalbaseline;D00isillustrativeandunknownrealoverheadcanworsenit.',
        'SameA6/U3/K2LRUtracescanhave6or3misses. CorrecteddisabledK0fill-then-readreferencebeforepersistence:inapplicableisnull,notanexistingcacheaccess.',
        'RowcachevsfullWKVmemo demandunionprecedescredit;cachecode/scaleepochs,pinning/eviction,C2oldqueuedwriters,capture/physicalextent/migrationpeakandotherKVpoolownersretained.',
        'Independent1800smalljointcapacityfrontiersandpreflight603checksPASS. AddREL121-01,extendexistingHG85memberF24;135relations/26groups/460pending,allnative/semantic/graphcountsunchanged.',
        'NoNPU/model/remote/targetimplementation. v120Padprobeauthorizationconsumedandnotreused. TaskACTIVE/LoopEVALUATING/allwholeclosuresopen.'],overall_complete=False))
    gap=('Needactualsame-NPUarena/B/remainder,runtimeformat,cachelayout/metadata,hitrate/accessorder,table/scale/shardepochsandallreaderretirement. '
        'F34evenviewisnotfreeHBM;private/shared-ownerABIreleaseandalloldwriters/captureaddressmigrationmustbeproved. '
        'RowcacheK/WKVmemoJ/MC2workspace/protocolpeakshareonebudget,notindependentcredits. '
        'C2IDsarenotembeddingkeys;cachecode/scalesmustcommitoneepochandpinagainstasync eviction. '
        '144examples/36layoutcapacityreferencesarenotactualallocation,hardwarecapacity,MFU,HBMtrafficorperformancegain. Allnativeandwholevariantgatesopen.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v121newREL121-01F24/F34:135conditionalrelations26groups460pending;encodedrowcache/privateC2/WKVmemoonebudget,notfreeVRAM.144cases/36selected-layoutcapacityreferences;actualmemorycredit/hitsstillunknown.356semanticrows26modules/allnativecountsunchanged.')
    report='''本轮补入一条此前未登记的跨模块关系：F24 Engram设备行缓存与F34 C2状态重组竞争同一个NPU设备的存储预算，并扩展已有HG85联合预算。没有把空白关系一概认定为独立，也没有重新认领v62行缓存、v79 ring压缩或v85预算公式本身。

关键限制先明确：在v79的当前unpacked/no-override缓存池模型中，C2偶数行逻辑视图压缩不改变common-ID的540928B预算除数，也不改变allocated block数。三个2.268GB ring视图不是三份独立分配，不能把它们相加当可释放显存。新的私有状态库或paired-owner协议可能释放common-ID预约，但仍要计算新bank、metadata、迁移和其它KV容量；global ID也不是一块可直接交给普通Tensor allocator的空闲显存。

行缓存按精确保留encoded codes及配套scale分析。v62给出的每行payload为BF16 512B、INT8 group32 288B、MXFP8 group32 264B。仅在明确选定“32B对齐payload+64B slot metadata”的示例布局中，slot为576/352/352B；64B不是已实现ABI或必要最小值，真实index/状态/pinning/epoch/event/allocator等开销还须计入D0。键必须区分table/layer/head、完整rowID、table/scale generation、格式/布局及owner配置；C2 request/page owner ID与embedding row ID不是同一种语义。

令B=N*C+r、0≤r<C，L为真正释放的旧common-ID预约，D为私有C2、K个row slots、J个WKV memo项及其它新增开销的合计。一个共同预算下，净空闲ID变化为 L+floor((B+DeltaB-D)/C)-floor(B/C)。本轮示例DeltaB均为0；v85的335MB假定raw credit属于F26尾部快照方案，须有它自己的读者/退役证明，不能归给ring视图压缩。若要求不减少原空闲ID，选定布局须满足 K*q+J*w+D_ring+D0≤L*C+r。K、J不能分别领取同一份最大容量。

144组预算参考和36组容量边界给出一个具体反例：S=U=4、采用新私有even-ring库时，三个层的ring需786432B；4096个INT8示例slot再需1441792B，合计2228224B。在D0暂设0、未知同一个r的条件下，净空闲ID变化为-1或0，不能承诺增加KV容量。示例D0不代表真实metadata免费。若再放256项既有示例WKV结果+ID-key memo，总额变为15360000B，净空闲ID变化为-25或-24。它可以是明确付出KV容量的取舍，不是同一容量上的多个免费收益。预算只属于同一设备/allocator；其它芯片显存不能相加，要求共同block数的参与组还应按rank瓶颈约束。

容量也不等于命中率。两条均为6次访问、3个不同key、容量2的冷LRU示例，按交错或连续排列会分别miss6次和3次。对选定full-row读取路径，host逻辑payload是p*M；写入缓存增加p*I设备写，miss直接转发或填充后再读取又有不同设备读代价。它们不是物理UVA交易/HBM字节；硬件cache、合并访问、实际owned比例及格式仍未知。slot容量模型不能给出数值加速比或硬件带宽上限。

与F25 WKV memo的组合必须先做全消费者需求并集：只有memo命中且其它读者与TP/ready协议都允许时，那份lookup才能消失；行缓存随后只针对仍需求的row accesses计价。同一份已被memo删除的host读取不能再算row-cache节省。v118的false gate通常仍需求value特征，v114的hash/history/DEAD/commit rollback和全局bank/collective参与也不能自动删除。

跨迭代复用还需要明确退役：代码与scale若在逐出期间来自不同generation，2×3=6和4×5=20的两种合法内容可能被混成10或12；必须先完成内容复制再发布VALID，并保留读者直到dequant/扇出/异步消费者结束。C2旧reader结束也不够，queued writer、初始化/zero、物理padding store、capture地址及request-ID重用都要退出旧存储；稳态容量不能提前用于尚未退役前的lookup。迁移期间old+new bank共存和workspace须纳入同一峰值预算。

前置603项核对、1800个独立小规模联合容量集合通过。新增REL121-01，HG85加入F24，中央条件关系134→135、待核对461→460，联合组仍26。356条语义记录/26模块、9类/1352次原生关键字段、849缺PMU、16未绑定phase图及3136含糊归属均不变，整模块关闭0。没有新设备工作或实现；v120 Pad授权已完成消耗。实际预算/命中率、最低MFU及整体理论数值上限仍未建立，整体目标继续。
'''
    action=('继续原Goal，v121PROGRESS；先读row-cache-budget-summary/common-budget/miss-traffic/retirement/current-obligations-v121，verify_row_cache_budget_v121须PASS。'
        'v120manifest62/PASS152；v121preflight603/1800independentjointfrontiers。新REL121-01F24/F34显式same-devicearena存储竞争，已有HG85加入F24（仍26groups），135conditionalpairs460pending。'
        'v79evenviewC2不改变C540928/N，三个2.268GBviews非独立allocation；无freeHBM。新private/pairedowner释放LcommonIDs仍chargeDr+rowK*q+WKVJ*w+D0，同一个r。若no-old-ID-loss Kq+Jw+Dr+D0<=L*C+r。'
        'v62rowpayloadBF16512/INT8288/MXFP8264；示例align32+tag64slot576/352/352，非actualABI/minimum。S=U4 privateevenDr786432+K4096Int8row1441792=2228224，D0=0未知r下ΔID[-1,0]；加J256rawWKV+keybytes51296/entry总15360000，Δ[-25,-24]。F26raw335MBcredit不归F34。'
        '容量≠hitrate，同A6U3K2coldLRU miss6vs3；host pM logical、device填充/读/metadata分层，不当HBM或speedup。K0fillthenreadinapplicable=null。'
        '先消费需求并集再算rowcache/WKVmemo，不能重复省hostlookup；falsegatevalue/otherreaders/hashhistory/collectives保留。Encodedcachekey含table/head/row/scaleepoch/format-owner，非C2ownerID；codes/scales同代提交、pin/retire防mixed10/12。'
        '旧C2queuedwriters/init/zero/physicalpadding/captureaddress/迁移old+new峰值全保留；onlysame-devicepool，其他芯片HBM不相加，commonblockgroup按瓶颈。ActualB/r/L/S/U/D0/format/hits/precision/owners尚unknown。'
        '356semanticrows26modules,813/1297/46graph,2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallMFU不变。'
        'Alllocaljobsdone，无server/NPU/model/implementation；Padv120onecapture授权GRANTED_AND_CONSUMED，不要等reply或重跑probe。v119observerjoint已恢复，v118mask/v117descriptorcap/v85budget基础为prioranswers。继续真正未展开operator关系/条件分支或call-bound关键证据。TaskACTIVE Loop004EVALUATING GoalACTIVE，不改封存v121或重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=121,previous=120,title='C2 and encoded Engram row-cache common-budget composition',
        summary=dict(filename='row-cache-budget-summary-v121.json',previous_turn_classification='PROGRESS:v120authorizedPadpacket/source32Bcontract;PASS152',
            **detail,refined_candidates=2,refined_modules=3,new_pairs=1,new_groups=0,
            branch_records=356,semantic_modules=26,relation_clauses=135,higher_order_groups=26,pending_central_pair_reconciliation=460,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            independent_joint_frontier_cases=1800,new_device_work=False,implementation=False,native_test=False,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',
            remote_job_status='Alllocalbudget/source-contract/synthetictraceanalysisfinished;no remote/NPU/model/implementation. CompletedPadprobe notrerun.'),
        changes=['ReconcilepreviouslypendingF24/F34same-devicebudgetrelationandextendexistingHG85withencodedrowcache.',
            'Compute144jointbudget/36selectedlayoutcapacityreferenceswithonecommonremainder;currentringviewcompactionisnotfreeHBM,norindependentcachebudgets.',
            'Separatecapacity,hitrate,tieredmiss/filltraffic,fullWKVconsumerdemandandencodedcontent/retirementepochs;noactualnative/memory/throughputpromotion.'],
        candidate_updates={c:dict(review_v121=deepcopy(detail),conclusion_changing_gap=gap)for c in ['F24','F34']},
        module_updates={'M08':dict(candidates=['F34'],finding='ExistingC2representationchangesdo notautomaticallyreleaseHBM;newprivate/paired-ownerIDreleasesmustbepricedwithencodedrowcacheandallco-residentbanks.',evidence=names,gap=gap),
            'M14':dict(candidates=['F24'],finding='Exactencodedrowcachehasformat/alignment/tag/epochandmiss/fillcosts;sharedbudgetwithC2andWKVmemoprecedesanyhostreadbenefit.',evidence=names,gap=gap),
            'M22':dict(candidates=['F24','F34'],finding='OneNPUarena/common-IDremainderandretirement/migrationpeaknowexplicitlycoupleencodedEngramcache,C2privatebanks,andexistingmemo/workspaces;noactualallocatorcreditproved.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reusev62UVAformat/profileandv79/v85/v84/v97pool/memocontracts;selectnamedF24/F34resourceinteractionwithoutnativecountpromotion.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='144budget36capacitymodels1800independentjointfrontierspluscacheorder/mixedepochcounterexamples;newREL121-01andHG85extension.',artifacts=[O/'row-cache-budget-preflight-v121.json',O/'row-cache-budget-validation-history-v121.json',O/'coverage-current-obligations-v121.json',O/'whole-scope-requirement-audit-v121.json'])],
        report=report,read_first=names+['coverage-current-obligations-v121.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_row_cache_budget_v121.py',N/'verify_row_cache_budget_v121.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='row-cache-budget',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
