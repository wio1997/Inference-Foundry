"""Persist route inverse/group metadata analysis with explicit unclosed native gates."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v96.json').exists()
    assert read(O/'route-map-analysis-preflight-v96.json')['status']=='PASS'
    names=['route-map-operator-source-contract-v96.json','route-map-source-work-and-interfaces-v96.json','route-map-native-frontiers-v96.json',
      'route-map-semantic-and-ceiling-contract-v96.json','route-map-variants-v96.json','DataCopyPad-primary-evidence-v96.json']
    members=['F20','F21']
    gap=('Local inverse identity preserves the exact first-sort permutation;native backend/alias/queue/copy and loaded tiling/precision costs remain unbound. '
      'Sharing a first grouping map or replacing it by histogram needs matching tie/capacity semantics,postmask ID mutation visibility,sentinel counts,current output/capture epochs and all readers/flags. '
      'Source full-load Sort call counts are not native instruction counters or necessary work for all algorithms. Count/cumsum reuse and inverse reuse have different value keys. '
      'New map reads/publication/alignment/cache costs and changed partition parallelism must be priced with v95 quant reuse once. GM bytes/block-touch models are not HBM/wire traffic or speedup. '
      'Loaded binary/capacity/critical path/full candidate and whole module coverage remain open.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v95.json'));c=next(r for r in rel['clauses']if r['id']=='REL73-07')
    c['route_map_refinement_v96']=dict(evidence=names,
      contract='Router score TopK and MC2 postmask expert-ID grouping are different operations. Keep first grouping sigma to replace inverse sorting by pi[sigma[j]]=j;partial or shared maps preserve current mask/sentinel/inplace effects,capacity/order and unpermute probabilities. Combine partition/quant/map graphs before costing.',
      inverse_identity_in_declared_domain=True,all_runtime_or_whole_pair_gates_closed=False)
    for g in rel['higher_order_groups']:
        if g['id']in ['HG73-MOE','HG80-A3-LOCAL-PATH','HG84-HC-FFN-FORK']:
            g['route_map_refinement_v96']=dict(evidence=names,contract=c['route_map_refinement_v96']['contract'],whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v95.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False);write(O/'candidate-relation-clauses-v96.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v95.json'));cov.update(parent='candidate-relation-coverage-v95.json',parent_relation_index='candidate-relation-clauses-v96.json',
      route_map_refinement_v96=dict(existing_clause='REL73-07',new_pairs=0,new_groups=0,evidence=names),overall_complete=False);write(O/'candidate-relation-coverage-v96.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v95.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v95.json')['model_candidates']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=95
        if r['id']in members:r['operator_atom_union']+=['moe.route.masked_ID_group_inverse'];r['route_map_refinement_v96']=dict(evidence=names,source_work_not_new_profile_operator=True)
    fps.update(parent='candidate-operator-footprints-v95.json',overall_complete=False);write(O/'candidate-operator-footprints-v96.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v95.json'));fm={r['id']:r for r in fps['candidates']}
    for pair in nav['pairs']:pair['shared_operator_atom_union']=sorted(set(fm[pair['a']]['operator_atom_union'])&set(fm[pair['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v95.json',parent_footprints='candidate-operator-footprints-v96.json',overall_complete=False);write(O/'candidate-pair-navigation-v96.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v95.json'))
    sigs={r['saved_input_shapes']for r in read(O/'route-quant-tiling-and-work-models-v95.json')['cases']if r['expected_source_key']==21000}
    regs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['signature']['inputShapes_text']in sigs]
    assert sum(r['executions']for r in regs)==3512
    ids={r['regime_id']for r in regs}
    for r in op['regimes']:
        if r['regime_id']in ids:assert r['state']=='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED';r['scoped_conditional_review_evidence']+=names
    op['route_map_refinement_v96']=dict(existing_full_load_MC2_calls=3512,excluded_gathered_calls=960,native_TopK_MC2_frontiers=3200,reused_TASKs=9600,
      source_Sort_reference=120240,local_inverse_reference=60120,new_regime_execution_or_assignment_promotions=0)
    write(O/'operator-analysis-coverage-ledger-v96.json',op)
    matrix=read(O/'module-phase-branch-matrix-v95.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M12'))
    assert not any(r['id']=='M12-OP-18'for r in branches)
    branches.append(dict(id='M12-OP-18',branch='TopK→postmask expert grouping→inverse map/cumsum→dispatch/unpermute;local inverse construction,consumer-demand windows,shared publication and value-keyed count/map reuse.',
      finding='Keeping first-sort sigma makes inverse[sigma[j]]=j an algebraic replacement of the second sort in the stated valid permutation/index domain. Three small full-load cohorts3512 calls;large960 gathered calls already use indexed inverse writes. Postmask inplace IDs/sentinel,cumsum mode2 and clipping/order gates are explicit;24 joint graphs avoid duplicate work credits.',
      state='SCOPED_PERMUTATION_IDENTITY_AND_CONDITIONAL_GROUP_PUBLICATION_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
      scope='Frozen source algorithm/current small topK3/6 cohorts and declared representation/copy models;not loaded inner Sort timings,all backend implementations or whole candidate closure.'))
    mods={mid:dict(finding='MoE inverse map is now separated from score TopK/group sort,with exact inversion identity,postmask mutation/sentinel and capacity/order gates. Shared-map/read costs and v95 partition interaction are explicit.',
      evidence=names,candidates=[cid for cid in members if cid in next(m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
      profile_scope='Operator original/transformed map semantics and scoped work/representation limits;no framework expansion.')for mid in ['M12','M22','M23','M24']}
    updates={cid:dict(review_v96=dict(evidence=names,native_frontiers=3200,reused_TASKs=9600,full_load_saved_calls=3512,gathered_calls_not_credited=960,
      exact_inverse_identity_in_stated_domain=True,current_native_permutation_or_tie_or_scatter_cost_proved=False,
      source_Sort_calls_reference=120240,inverse_only_reference=60120,source_counts_not_all_algorithm_lowerwork=True,
      new_pairs=0,new_groups=0,implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v95.json'));audit.update(previous_goal_turn='PROGRESS:v95routing-quant/source-partition/interface contracts persisted;150postchecks',input_checkpoint=95,followup_checkpoint=96)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=358
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['route_map_refinement_v96']=dict(local_algebraic_identity_not_native_speedup=True,inverse_sort_not_universal_necessary_work=True,
      all_module_or_goal_closure=False)
    write(O/'whole-scope-requirement-audit-v96.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v96.json',dict(status='PROFILE_TOPK_MC2_EDGE_THEN_PINPOINT_GROUP_INVERSE_MASK_CONSUMERS',
      questions=['Is the second full-load sort necessary to invert the first permutation?',
        'Which inverse positions/counts does each local core and global unpermute consume?',
        'Can group metadata be constructed before MC2 applies its active mask?',
        'Can map sharing be priced independently of token partition/quant reuse?'],
      cached_pinpoint_source_files=5,new_target_source_files=0,new_public_API_context_documents=1,no_framework_wide_research=True,
      no_model_Torch_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'route-map-validation-history-v96.json',dict(corrected_assumptions=[
      dict(topic='TopK output envelope',initial_check='Only two3x6 outputs expected.',
        saved_descriptor='FLOAT[3,6];INT32[3,6];FLOAT[3,384]',correction='Retain all three descriptors;third full-score plane is not a new write proof. Existing v63 outflag contract remains separate;no output elision credit added.'),
      dict(topic='Copy units/alignment',correction='Checked official DataCopyPad context:DataCopyParams blockLen is byte for this API;GM address need not32B alignment. 32B per-copy block exposure is a sensitivity model,not an actual DMA/HBM bound or a mandatory round-up assertion.'),
      dict(topic='Cost interaction',correction='M3K6 source36Sortcalls;tokenaligned6,inverseonly18,joint3. Standalone savings cannot add. Larger gathered cases excluded from full-load inverse-sort credit.')],
      arithmetic_models_not_native_tests=True,all_local_jobs_finished=True,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v95.json')['remaining'])
    remaining.append('v96: ExistingF20/F21group/inverse boundary refined. Full-load21000 repeats2high-levelSortcalls/core;keepingfirstsigma givesexactinverse pi[sigma[j]]=j,so secondSortnotnecessaryworkinthispermutationdomain. Current3smallcohorts3512calls source120240SortAPIreference→60120inverseonly,notnativeinstructions/gain;960gatheredcallsalreadyindexedmappingexcluded.3200TopK/MC2edges9600oldTASKs root2560unbound640;3TopKoutputplanesretained. ApplyXMaskinplaceIDs→sentinelbeforeSort;earlymetadata/mappedIDpublication mustpreserveeffect/observers/epochs. Sharedfirstmap/histneedsnativeorder-or-rowwise-equivariance/nonbindingcapacityproof;clippingcounterexample retained. Count/cumsumonlysamehistcanreusebutinverse needsorder,knownfalsepaddingtailadds(P-N)k sentinels notrowpruning. 192copyalignmentexposurecases/24jointmap-partitiongraphs;M2T6logical96Bsavecanincrease32Bblocktouch32B,notHBM. M3K6Sort36→tokenaligned6/inverse18/joint3;noadditivecredits. ExistingREL73-07/HG73/HG80/HG84refined122/25/473unchanged,358records24semantic,9/1352pending849/16/3136uncertainty0wholeclosed/lowestallmoduleMFUunestablished.')
    report='''本轮核对 Router TopK→MC2 内部路由分组与逆映射。结论是：保留第一次排序结果后，第二次用于求逆的排序不是这段语义必需的工作。尚未实现替换，也未据此给出提速。

```mermaid
flowchart LR
  A[TopK IDs] --> B[active mask 原地改写为 sentinel]
  B --> C[按 expert ID 分组排序]
  C --> S[σ：sorted 到 source]
  S --> D[原第二次排序求逆]
  D --> P[π：source 到 sorted]
  S -. 条件替换 .-> I[直接构造 πσj 等于 j]
  I --> P
  P --> Q[routing 行复制]
  P --> U[按原 topK 顺序加权 unpermute]
```

第一次排序返回排列σ，现有第二次排序以-σ[j]为唯一key、j为payload，故其有效输出恰好满足π[σ[j]]=j。直接按该等式构造逆映射保留原σ，因此不需要改变第一次expert同分排序。已检查T≤8的全部46234个排列及当前T=6/9/18的局部消费者窗口模型；这是数学模型验证，没有原生Scatter速度或加载内核精度证明。大的816/834行gather分支已有按索引写逆映射，未重复领取“去第二次排序”收益。

full-load的每个有效分区都读取完整expert-ID向量并做两次高层Sort。3行/TopK6有18个分区，3行/TopK3有9个，2行/TopK3有6个。三种组合3512次既有调用，对应条件源码参考120240次高层Sort；仅去逆排序后的参考为60120次。这不是原生指令统计、实测耗时，也不是所有等价算法的必要工作下界。固定显式GM逆映射表示仍须发布4T字节，容量未知，仅保留相应符号界。

局部方案也可区分消费者：core0保留完整逆映射供全局unpermute，其他core只构造自己assignment区间会读的位置。共享一份全局映射则需新读取和就绪关系，还要证明原各core的分组排列一致。当前源码计数/cumsum只由最后一个有效core写出，并没有按全部core重复写出；该部分未计为重复收益。

| M=3、TopK=6的条件计算图 | 高层Sort调用参考 | 内部行量化参考 |
|---|---:|---:|
| 当前源码分区 | 36 | 18 |
| 仅直接构造逆映射 | 18 | 18 |
| 仅按token对齐分区 | 6 | 3 |
| 两者结合 | 3 | 3 |

这是同一基线的替代图；两项独立节省不能相加。按token对齐也改变并行度，若再共享量化或全局映射，必须按同一组选定分区重新计算。

需要特别保留active mask边界：MC2先把inactive token的expert ID在GM中改写为额外sentinel，再同步、排序和计数。例子[0,1]在第一行inactive后变为[384,1]，逆映射从[0,1]变为[1,0]，计数也不同。不能直接用mask之前的TopK输出生成可复用映射，或悄悄提前这个可观察改写。Router TopK按每行score选expert，MC2按所有assignment的有效expert ID分组，它们的key与数据域不同。

若进一步以histogram/prefix替换第一次分组排序，必须保留tie/capacity语义。声明的逐行expert模型在receiver容量为1时，两个同expert token的顺序互换，会改变留下哪个token。无容量裁剪且所有处理对行排列等变时可另行建立等价域；当前原生精度、alias和实际容量尚未据此关闭。直接保留第一次σ的局部逆映射方案没有这项重排。

跨层/跨迭代复用也分输出：当前mode2的cumsum取决于有效ID直方图，但逆映射还依赖assignment顺序。[0,1]与[1,0]计数相同，逆映射不同。已知false的padding尾部可给直方图增加(P-N)·k个sentinel；这条公式不删除原有map、记录、mask副作用或通信。映射相同也不说明hidden、量化值或expert输出相同。验证相等与发布新epoch仍须计价。

共享映射的值模型还保留了新GM读取。在T=6、6个分区时，所列逻辑值访问可少96B；但声明的每次copy触及32B块、无跨copy缓存合并且base offset为0的敏感性模型，块覆盖从224B变为256B。它不是实际DMA或HBM流量，说明不能仅按值字节确定收益。官方[DataCopyPad文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)支持非对齐逻辑长度，blockLen单位为Byte；当前文档是API语义参考，未作为9.2原生实现或物理带宽证明。

原生层联接3200组router/TopK/MC2边界，复用9600条旧TASK；2560组有阶段归属、640组仍未知。TopK的三路输出描述全部保留，内部Sort没有被虚构成新增可计时TASK。已有REL73-07及三组联合合同得到细化，中央计数仍122条/25组、473对待汇入或复核；分支台账358条、24个语义模块。9个regime/1352次执行的关键字段缺口、849条PMU缺失、16张阶段未知图、3136条模块归属歧义未改变。整体目标及全模块最低MFU仍未完成。
'''
    action=('继续原Goal，v96PROGRESS，v95PROGRESS已150checks。先读route-map-summary-v96/PROFILE-FINDINGS/source/work-interfaces/native/semantic/variants及wholeaudit。'
      '用户聚焦算子fusion/split/reorg/theory，profile优先pinpoint，不扩framework。'
      '本轮fullLoad21000每core两Sort；首次σ保持则逆π[σ[j]]=j代替第二Sort的数学身份成立，不需改expert ties。源码T<2^24 current6/9/18，46234allT0..8perms+21000partialmodels；notnative性能或整个F21closure。3smallcohort3512calls sourceSortAPI120240→60120inverse-only，960largegatheredalreadyindexedinverse不领收益。'
      '3200RouterTopKMC2joins9600oldTASKs root2560/640unknown；TopK descriptorFLOAT[3,6],INT32[3,6],FLOAT[3,384]保留，3rdplane不是writeproof。ApplyXMask在MC2前原地 IDs→sentinel然后SyncAll；早metadata须postmask/副作用可见性。分组按ID不是Router scoreTopK共排序。'
      '局部inverse保留σ+allprefix；core0globalT其他localdemand可裁，nativealias/ABI/queues/perfcost待定。共享σ/π则nativeTie一致/Order/capacity/reader/flag/epoch待证；receiverclip可让同expert重排改变结果。Currentcumsum仅lastcore一次，无重复countwrite收益。'
      'Exacteffectivehist可复用cumsum但同hist不同inverse；falsepaddingtail(P-N)k sentinel公式不删map/record/collective；同ID不缓存hidden/q/expertoutputs。'
      'M3K6sourceSort36,onlyinverse18,onlytokenalign6,joint3；24jointcases不加独立收益。192copyoffset模型M2T6logicalsave96可32Bblocktouch224→256，notactualDMA/HBM；DataCopyPad官方8.5语义doc blockLenByte/GM非32align参考非9.2binary/rate。'
      'REL73-07/HG73MOE/HG80/HG84refined0newpairs/groups=>122/25/473；358branches24semantic，2073/285598review9/1352pending849/16/3136unchanged0wholeclosed/noallmodulelowestMFU。'
      '下一步优先尚未定案且影响全范围融合/理论的直接算子条件，避免重放QR/HC/inverse代数。MoE receiver容量与行重排数值条件可作后续定点问题；没有证据时不得宣称clip不触发或新顺序位级相同。保持offlineprofile/frozen/math/必要primary,no server/model/Torch/NPU/capture/implementation/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING。verify_route_maps_v96必须PASS，不重跑已成finalizer。')
    sources=[Path(p['path'])for p in read(O/names[0])['parts']]
    jobs=[dict(kind='profile',status='pass',description='Join3200existingrouterTopKMC2boundaries9600TASKs;preserve3TopKdescriptors/phaseambiguity,nomade-upinnerSortTASKtiming.',artifacts=[O/names[0],O/names[2]]+sources),
      dict(kind='design-check',status='pass',description='46234inversepermutations21000partialwindows810inactivehistcases,192copyoffsetand24jointgraphs;postmask/tie/capacity/cumsum/protocol/representationcostgatesretained.',
        artifacts=[O/names[1],O/names[3],O/names[4],O/names[5],O/'DataCopyPad-primary-source-v96.html',O/'route-map-analysis-preflight-v96.json',O/'route-map-validation-history-v96.json',O/'whole-scope-requirement-audit-v96.json',R/'fusion-scan.json'])]
    persist_refinement(version=96,previous=95,title='MoE grouped inverse construction and postmask metadata contracts',
      summary=dict(filename='route-map-summary-v96.json',previous_turn_classification='PROGRESS:v95routingquantandpartitioncontractsverified',
        native_frontiers=3200,reused_TASKs=9600,full_load_saved_calls=3512,gathered_calls_not_credited=960,source_Sort_call_reference=120240,inverse_only_Sort_call_reference=60120,
        inverse_permutation_models=46234,partial_reader_models=21000,inactive_prefix_hist_models=810,copy_offset_models=192,joint_partition_map_models=24,
        new_target_source_files=0,new_public_API_context_documents=1,refined_candidates=2,refined_modules=4,new_semantic_families=1,branch_records=358,semantic_modules=24,
        new_pairs=0,new_groups=0,relation_clauses=122,higher_order_groups=25,pending_central_pair_reconciliation=473,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
        actual_hardware_ceiling_or_speedup_or_whole_candidate_closed=False,implementation=False,native_test=False,
        remote_job_status='Alllocalprofile/source/math/APIcontextworkfinished;no server/model/Torch/NPU/capture/serviceorimplementation.'),
      changes=['Separateexpertgroupingandunique-indexinversion;localinverseidentitypreservesfirstSortanditsorder.',
        'BindpostmaskIDmutation/sentinel/cumsum/weightedunpermute/capacitygates;mapversuscountvaluekeysandknowninactiveprefixformulaprovedinstateddomains.',
        'Price24jointpartition/mapgraphs and192copyoffsetmodels;retainnewreads/waits,nofullSortnecessaryworkoradditivegainclaim.',
        'RefineexistingREL73-07/threejointgroupswithoutcountinflation;358records24semantic,fullgoalremainsactiveandincomplete.'],
      candidate_updates=updates,module_updates=mods,semantic_branches={'M12':branches},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['whole-scope-requirement-audit-v96.json'],
      publication_rows={'F21':'| F21 | MoE dispatch、专家FFN、combine及内部路由映射 | 已有复合边界／条件重组 | 已有dispatch到combine融合保留。小行数full-load的第二次排序只求逆，可在保留首个排列时直接构造；共享映射或前移到TopK另需mask原地副作用、sentinel、容量/顺序和发布合同，未证明提速。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_route_maps_v96.py',N/'verify_route_maps_v96.py',N/'persist_profile_refinement_v90.py'],executor_prefix='route-map',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
