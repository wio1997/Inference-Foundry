"""Persist exact padding ownership and existing-candidate cost refinements."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N, O, read, write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v72.json').exists()
    assert read(O / 'pad-owner-invocations-v72.json')['owner_API_calls'] == 1272
    assert read(O / 'pad-fusion-variants-v72.json')['count'] == 9
    e = ['pad-owner-invocations-v72.json','pad-helper-regime-refinement-v72.json',
         'pad-owner-API-namespace-reconciliation-v72.json','pad-operator-contract-v72.json',
         'pad-fusion-costs-v72.json','pad-fusion-variants-v72.json','pad-value-map-validation-v72.json',
         'operator-analysis-coverage-ledger-v72.json','pad-branch-inventory-v72.json','pad-protected-module-state-v72.json']
    c = deepcopy(read(O / 'operator-analysis-coverage-ledger-v71.json'))
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v71.json',
             newly_reconciled_regimes=0,newly_reconciled_executions=0,
             new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,
             prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0,
             enclosing_pad_owner_refinement_v72=dict(regimes=7,executions=1272,owner_bound=True,independent_destination_or_size_bound=False,evidence=e[:7],pending_promoted=0),
             overall_complete=False)
    write(O / e[7],c)
    old = {m['id']:m for m in read(O / 'module-phase-branch-matrix-v71.json')['modules']}
    branches = {mid:deepcopy(old[mid]['branches']) for mid in ['M12','M19']}
    details = {
        'M12-OP-09':'320 occurrences of each hidden/router/mask pad 800→816; 960 independent API owners, 2880 ComputeInfo TASKs plus 960 nonCompute MIX tasks. Joint pure-pad reference payload19031632B, known BOOL input/output conversion budget3232B. Existing F20 pad/direct-producer alternative refined; readiness and full output/native MC2 counts retained.',
        'M12-OP-11':'All960 MC2 MemSets exactly owned by PYTORCH_API aclnnConstantPadNd through CONNECTION_IDS sets. Prior v63 CANN_API ROWID lookup had a namespace error and is explicitly reconciled. Destination/length/overlap/workspace identity still absent; three regimes stay pending. Replacing the complete pure pad call is a different conditional cut from deleting MemSet alone.',
        'M19-OP-01':'The104 prior initializer groups contain15 ComputeInfo TASKs each; exact three-pad ownership additionally exposes3 nonCompute KERNEL_MIX_AIV records per initializer. Native TASK/control records are kept separate from ComputeInfo coverage and host launch counts.',
        'M19-OP-02':'Existing v67 SP-before-HC4-repeat value mapping/costs independently reproduced; N20 original2826240B, global2Dpad+localTile604160B, budget2222080B/card. Not a newly discovered mechanism or additional saving to add to direct-local budget.',
        'M19-OP-03':'Existing v67 direct local initializer revalidated for each SP rank: N20P24C3, V3 gives153600B HC4 read/write and2672640B budget; rank6V2 gives143360B, rank7V0 gives122880B. Actual output mask true tail, IDs0, BF16 copies and premix onehot preserved. New owner evidence supports a whole-pad/helper cut; all-reader/bank/capture/frontier conditions open.',
        'M19-OP-05':'All312 draft pad MemSets exactly bound to312 PYTORCH_API constant-pad owners (104 hidden/104 IDs/104 mask). Ownership no longer unknown, but addresses/length/overlap/private write roles remain unbound. Four regimes stay pending; no independent helper deletion or guessed physical bytes.',
    }
    replacements = {
        'M12-OP-11':'3MemSet regimes960 executions: exact ConstantPadNd API owners now bound; destination/length/overlap still absent, independent helper deletion and physical bytes pending.',
        'M19-OP-01':'104 draft initialization groups15 ComputeInfo TASKs each; three pad calls additionally bind3 nonCompute MIX records per group. Separate TASK coverage and host launch counts.',
        'M19-OP-05':'4ConstantPadNd MemSet regimes312 calls: exact parent API invocations bound; destination/length/workspace-vs-payload still unproved. Whole-pad replacement reviewed conditionally, independent MemSet deletion pending.',
    }
    for mid, rows in branches.items():
        assert len(rows) == (15 if mid == 'M12' else 13)
        for b in rows:
            if b['id'] in details:
                b['evidence'] += e[:8]
                b['operator_owner_review_v72'] = details[b['id']]
            if b['id'] in replacements:
                b['branch'] = replacements[b['id']]
            if b['id'] in ['M12-OP-11','M19-OP-05']:
                b['state'] = 'EXACT_PAD_API_OWNER_BOUND;MEMSET_DESTINATION_SIZE_OVERLAP_PENDING'
                b['gap'] = 'Exact constant-pad owner proved; actual MemSet destination, length, overlap and private-workspace/write role remain unknown. Whole-call semantic replacement is conditional; no independent MemSet deletion or HBM traffic claim.'
    write(O / e[8],dict(modules=branches,extended_families=sorted(details),retained_families={'M12':15,'M19':13},scope='Keep all28 semantic IDs; enrich6. Preserve24 other complete module records.',whole_modules_closed=0,overall_complete=False))
    write(O / e[9],dict(modules={mid:m for mid,m in old.items() if mid not in branches},previous_modules={mid:old[mid] for mid in branches},whole_modules_closed=0,overall_complete=False))
    write(O / 'PROFILE-CODE-QUESTION-GATE-v72.json',dict(status='EXACT_CONSTANTPAD_OWNER_AND_FULL_OPERATOR_REPLACEMENT_REFINEMENT',previous_turn_classification='PROGRESS:v71planneralgorithmkeyschemapublication',questions=['Which exact API invocation owns each of the1272 MemSet tasks, without temporal pairing?','Does the earlier CANN launch label use the correct API table namespace?','What is the full pad output contract and which helper TASK records belong to a whole-call replacement?','Does direct SP-local HC initialization preserve existing v67 value maps and cost budgets?','What cold materialized-output reference bound can be stated without inventing MemSet bytes or HBM residency?'],allowed_source='Saved native TASK/CONNECTION_IDS/PYTORCH_API and three previously pinned local padding/HC clauses. No framework survey, imports, remote read, model/device/container work or captures.',new_remote_source_read=False,local_operator_clause_review=True,independent_MemSet_operand_proved=False,overall_complete=False))
    gap = 'Seven MemSet1272 owners bound but destination/length/overlap unresolved; two BOOL24 fields80 and their readers unresolved. Actual HBM/cache/Vector/launch capacity, all-reader/generation/alias/frontier/native loaded implementation and full35-candidate interactions/applicable branches remain open.'
    updates = {}
    for cid in ['F20','F28']:
        updates[cid] = dict(review_v72=dict(owner_invocations=str(O/e[0]),namespace_correction=str(O/e[2]),operator_contract=str(O/e[3]),costs=str(O/e[4]),variants=str(O/e[5]),value_map_checks=str(O/e[6]),existing_candidate_refinement=True,new_mechanism_claimed=False,new_remote_source_read=False,implementation=False,native_test=False,new_performance_result=False),profile_review_state='EXACT_PAD_OWNER_FULL_REPLACEMENT_REVIEWED_CONDITIONAL;OPERAND_AND_ALL_READER_GAPS_OPEN',conclusion_changing_gap=gap,next_profile_or_contract_action='Use saved operator IR/allocation/argument evidence only if it can bind destination/extent/readers. Independently audit fixed-case branches and candidate overlaps; do not repeat blind missing-table or framework inspection.')
    mods = {
        'M12':dict(finding='960 MemSet owners exactly resolved to three MC2 pad calls per320 observed occurrences,800→816. Full pad3 payload reference19031632B and known BOOL conversion budget3232B. Existing F20 mechanism refined, helper operands/ABI/physical constraints open.',evidence=e[:9],candidates=['F20'],gap=gap,profile_scope='Padding, producer/consumer shape and precision/readiness only; no framework audit.'),
        'M19':dict(finding='312 pad owner invocations plus312 nonCompute MIX records bound across104 prior initializer groups. Existing F28 local SP+HC4 value maps and v67 budgets preserved, not rediscovered or added together. V0 still owns output fills and global KV/collective participation.',evidence=e[:9],candidates=['F28'],gap=gap,profile_scope='SP/HC copy map, pad tails, alias and reader contracts only.')}
    remaining = []
    for s in read(O / 'module-coverage-gaps-v71.json')['remaining']:
        if s.startswith('v71new'):
            continue
        remaining.append(s)
    remaining.append('v72 exact1272 MemSet→ConstantPadNd PYTORCH_API owners via connection sets (960MC2/312SP), correction of prior CANN_API ROWID namespace lookup.5088 nativeTASK=3816ComputeInfo+1272nonComputeMIX. Whole pure-pad replacement conditional; no destination/length/overlap identity and no independent MemSet deletion. Seven regimes1272 remain pending plus2BOOL80. MC2 actual800→816/H5120/E384:19031632Blogicalreference and3232B known BOOL conversion budget. Existing v67 F28 directlocal/HC4 budgets unchanged, not new candidates.2073/285598conditionalreview,9/1352pending,849/16/3136 uncertainty and0moduleclosures protected.')
    action = ('继续原Goal，最新v72 PROGRESS，v71亦PROGRESS。分析只围绕算子融合/拆分重组和理论上限，profile优先，具体问题才读算子代码，不研究框架。'
              '先读operator-analysis-coverage-ledger-v72/pad-owner-invocations/pad-owner-API-namespace-reconciliation/pad-operator-contract/pad-fusion-costs/pad-fusion-variants/pad-branch-inventory-v72。'
              '1272 MemSet通过TASK.connectionId→CONNECTION_IDS runtimeID→setID→PYTORCH_API.connectionId精确归属1272个aclnnConstantPadNd，不能用跨表ROWID或时邻配对。v63 linked_CANN_API_names=launch为错误命名空间查表，本轮留档更正；不外推所有旧关联无效。'
              '5088nativeTASK=3816ComputeInfo(Cast/MemSet/Pad各1272)+1272nonComputeKERNEL_MIX_AIV，后者不加入286950ComputeInfo分母。960MC2pad800→816，312SPpadN10/20→P16/24。整纯padAPI替换可条件移除其privatehelpers，但单MemSet地址/大小/重叠/工作区角色仍未知，7regimes1272不升格。'
              'MC2三路逻辑输入9421600B+输出9610032B=19031632B；已知BOOL数据和输出Cast额外3232B，scalarCast/实际MemSet未定不猜字节不重复加mandatoryout。按1.6TB/s冷输入新鲜全输出物化域仅作条件参考11.89477us，不是实测或全域下界。输出dtype/false或0尾/capture/allreader保留，不能直接MC2改816→800。'
              'F20/F28旧机制细化不是新候选。v67 SP-before-repeat及directlocal数值全一致：N20globalTile+pad2826240B→global2Dpad+localTile604160B；directlocalHC4V3=153600B预算2672640B、rank6V2=143360B、rank7V0=122880B。不要把这些相互替代预算相加；V0不等于整个decoder无KV/collective。'
              '当前2073/285598有条件分析，9/1352仍pending(7MemSet1272+2M05|M21BOOL80)。0整模块闭环、35model+7legacyframework、849missingPMU、16phase-unbound、3136ambiguity保持。'
              '下一轮不要因剩9profile形态而阻塞全目标：优先审查35候选相互重叠/依赖/替代/跨层卡迭代关系和固定case全部适用分支，补typed物理上限关键证据；已有IR/args若能绑定MemSet或BOOL再定点取证。已审v66原始DB无args/memorytable不重复盲扫。服务日志只有compiler config大行，可定向解析配置指向已保存IR，但不是框架研究。'
              '本轮无remote/服务/容器/NPU/模型/Torch/采集/进程操作；仅旧资料和纯数学检查。verify_pad_owners_v72应PASS。Goal保持active，不能把报告成稿或2082profile形态review当完整完成。')
    report = '''本轮把1,272次MemSet全部关联到各自的`aclnnConstantPadNd`调用：960次来自MC2准备，312次来自草稿SP初始化。依据是原始TASK、CONNECTION_IDS集合和PYTORCH_API的精确连接关系，不是时间相邻。每个调用都有独立API标识；BOOL路径包含三次Cast。共保留5,088个原始TASK，其中3,816个有ComputeInfo，另有1,272个KERNEL_MIX_AIV记录。后者没有加入286,950次ComputeInfo执行的覆盖分母。

旧v63记录把关联的PYTORCH_API行号放到CANN_API命名空间查询，因而写成“launch”。本轮保存更正记录并重建全部关联。原有“没有MemSet地址和长度”的结论仍成立；本更正只作用于这项查询。

| 算子子图 | 融合或重组机会 | 本轮可支持的量化结论 |
|---|---|---|
| MC2的hidden、router、mask三路padding | 联合写三路输出，或由各自上游直接写出padding结果 | 实际800→816，H=5120、E=384；一次读取输入并完整写出输出的逻辑参考量为19,031,632B；已知BOOL数据及输出转换另有3,232B参考预算 |
| 草稿SP padding与HC4初始化 | 直接输出本卡拥有的行，同时完成HC4复制、ID和mask尾值 | 延续v67既有方案；N20时旧Tile+全量HC4Pad为2,826,240B，拥有3个有效行的卡直接初始化为153,600B，接口预算差2,672,640B |
| 完整ConstantPadNd调用及其内部辅助任务 | 替换整个纯padding调用，在新实现中保留完整输出契约 | 已确定哪些辅助TASK属于该调用；不再需要猜测它们与哪次Pad有关。合法性仍取决于全部读者、别名、代际、图捕获地址和就绪事件 |

SP-before-repeat的2,222,080B预算，以及直接本地初始化的2,672,640B预算，都是v67已登记的互相替代方案，本轮只补强调用归属并独立复核，没有把它们当作新机会或相加。N20时rank6拥有2个有效行，直接HC4初始化参考量为143,360B；rank7没有有效query行，但仍须输出122,880B的本地HC4零尾，且不能据此删去其全局KV或集合通信工作。None-mask分支可生成false前缀；传入mask时必须保留前缀原值，包括True。padding尾mask为True，ID和hidden尾为零。

纯数学检查覆盖26,520个行映射条件、79,560个mask条件，并检查全部65,536种BF16位型的无算术复制表示。这些是重排行为的有限见证，没有执行Torch、模型或设备算子，也没有证明实际bank和所有消费者协议。

理论参考必须固定实现域。若输入只在HBM、输出是初值任意的新鲜HBM分配，所有输出必须按原dtype完整物化并被外部读取，且不允许压缩、别名视图或已有零bank不变式，则可用输入必读与输出必写量建立条件访存下界。以公开的每die 1.6TB/s容量作为适用上容量假设，三路MC2 padding参考值为11.89477µs。该数不适用于缓存命中、虚拟padding或其他输出表示，也没有证明本次实际HBM流量、Vector/启动容量或请求耗时上限。[硬件参考](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1)。不能把MemSet未知字节再加到完整输出写入中，因为二者可能重叠；更不能把原始TASK耗时相加当作可获得的请求收益。

整段替换padding与单独删除MemSet是两个不同判断。后者仍缺目标地址、长度、重叠关系及工作区角色。本轮没有因此升级7类MemSet、1,272次执行的覆盖状态。另有2类BOOL字段、80次执行的具体身份和读者未确定。累计仍是2,073类、285,598次已有条件分析，9类、1,352次关键证据待补；35项模型候选和7项历史框架候选保留，整模块闭环仍为0。完整适用分支、候选交互和物理容量证据仍需继续，Goal保持进行中。
'''
    jobs = [
        dict(kind='profile',status='pass',description='Exact1272 ConstantPadNd owners independently joined via connection SETs/PYTORCH_API;5088unique nativeTASK=3816ComputeInfo+1272nonComputeMIX;priorCANNROWIDnamespacecorrection retained. No helper operand/address/extent inferred.',artifacts=[O/e[i] for i in [0,1,2]]+[N/'analyze_pad_owners_v72.py']),
        dict(kind='design-check',status='pass',description='Three pinned local operator clauses,9existing-candidate variants/exclusions;MC2 actual800→81619.031632MB logical reference/3232B known BOOLcast budget. Existingv67SP/HC4rank budgets unchanged;26520map/79560mask/65536BF16finite witnesses, no native execution or physical HBM claim.',artifacts=[O/e[i] for i in [3,4,5,6]]+[N/'model_pad_owners_v72.py']),
        dict(kind='review',status='pass',description='Coverage unchanged2073/285598review9/1352pending;7MemSet1272 remain pending despite exact parent;2BOOL80 and849/16/3136 protected. Retain28 M12/M19families enrich6 and24othermodulesidentical;35model7framework0wholeclosures.',artifacts=[O/e[i] for i in [7,8,9]])]
    persist_refinement(version=72,previous=71,title='Exact padding invocation owners and full-operator replacement bounds',summary=dict(filename='pad-ceiling-summary-v72.json',previous_turn_classification='PROGRESS:v71planneralgorithmkeyschemapublication',refined_pending_regimes=7,refined_pending_executions=1272,newly_reviewed_regimes=0,newly_reviewed_executions=0,exact_pad_API_owners=1272,native_TASKs=5088,ComputeInfo_TASKs=3816,nonCompute_TASKs=1272,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_model_candidates=0,existing_candidates_refined=['F20','F28'],retained_semantic_families=28,enriched_semantic_families=6,protected_other_modules=24,new_remote_source_read=False,implementation=False,native_test=False,new_performance_result=False,actual_physical_numeric_ceilings_proved=False,remote_job_status='No remote jobs or reads this turn; no model/Torch/NPU/capture/container/service/process operations.'),changes=['Bound all1272 MemSet owners to exact PYTORCH_API constantpad invocations through connection sets; corrected prior cross-tableROWID label without hiding operand gaps.','Conditional whole-pad/private-helper cuts and9existingF20/F28 alternatives/exclusions;MC2 actual800→816 cost and v67 direct-local HC4 budgets independently reviewed, no new candidate or double counting.','All28 M12/M19families retained6 enriched24othermodule records identical;2073/285598review9/1352pending849/16/3136uncertainty0wholeclosures unchanged.'],candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[7],e[0],e[2],e[3],e[4],e[5],e[8]],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/x for x in ['analyze_pad_owners_v72.py','model_pad_owners_v72.py','verify_pad_owners_v72.py']],executor_prefix='zcode-pad-owners',main_script=Path(__file__),source_read=False)


if __name__ == '__main__':
    main()
