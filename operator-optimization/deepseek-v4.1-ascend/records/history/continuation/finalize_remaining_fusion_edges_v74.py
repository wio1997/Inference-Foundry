"""Persist extra operator relation edges and the head-RMS applicability correction."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v74.json').exists()
    rel=read(O/'candidate-relation-clauses-v74.json');new=rel['clauses'][28:]
    assert len(new)==15 and len(rel['touched_candidates'])==35
    e=['QB-head-RMS-RoPE-applicability-v74.json','QB-to-query-RoPE-native-visibility-v74.json','remaining-fusion-edge-costs-v74.json','remaining-fusion-edge-math-v74.json','remaining-fusion-edge-source-index-v74.json','candidate-relation-clauses-v74.json','candidate-relation-coverage-v74.json','candidate-stale-gap-reconciliation-v74.json','operator-analysis-coverage-ledger-v74.json','remaining-fusion-branch-inventory-v74.json']
    candidates={c['id']:c for c in read(O/'candidate-profile-ledger-v73.json')['candidates']}
    updates={}
    for cid in sorted({x for r in new for x in [r['a'],r['b']]}):
        rows=[r for r in new if cid in [r['a'],r['b']]]
        updates[cid]=dict(review_v74=dict(relation_clause_ids=[r['id']for r in rows],peers=sorted({r['b']if r['a']==cid else r['a']for r in rows}),central_relations=str(O/e[5]),relation_index=str(O/e[6]),source_contract=str(O/e[0]),joint_costs=str(O/e[2]),math_checks=str(O/e[3]),scope='Fifteen selected relation clauses and three higher-order groups only; all whole-pair/full-model gates remain open.',implementation=False,native_test=False,new_performance_result=False))
    updates['F07'].update(profile_review_state='FULL_HEAD_RMS_PATH_CONTRACT_CORRECTED;HISTORICAL_GRAPH_APPLICABILITY_OPEN',conclusion_changing_gap='Pinned CP caller explicitly has QB BF16→full512/head RMS→BF16→tail64 RoPE. Earlier direct QB→RoPE description omitted this stage and is superseded for this caller.3200saved graph brackets show no separately identified ComputeInfo head-RMS, but neither absence nor fusion nor historical loaded-caller identity is proved. Full norm/RoPE numeric/reduction/alias/rstd/reader/capture and Vector/Cube/cache capacity conditions remain.',next_profile_or_contract_action='Resolve head-RMS historical applicability using existing exact operator-level captured graph/compiled kernel/loaded caller evidence if available. Preserve separate with-head-RMS and separately attested no-head-RMS domains; do not infer identity/absence from ComputeInfo omission or inspect framework lifecycle.')
    stale={
        'F27':dict(current_gap='F27 main15360 contraction/full5120 norm/ordered selector/aux mean and lifetime/row-vs-feature partition/all context-bank readers and native reduction/physical capacity conditions remain. The21planner regimes608 were conditionally reviewed in v71 and prior config/constructor support current166 hashes reconciled; these are no longer pending planner/source-file-version gaps. Historical loaded code and actual values/aliases/generations remain unattested.7MemSet1272 and2BOOL80 are separate global pending scopes, not a reason to withhold F27 conditional operator analysis.',current_action='Continue F27-specific observer/ordered-I/late-selective ownership, full5120 norm/global-statistic precision, all context-bank query readers and resource bounds. Reconcile remaining central pair clauses/whole-model branches; do not repeat completed v71 planner algorithm or three support-file byte matches.'),
        'F34':dict(current_gap='F34 observable ring/history/page/rollback/lastreader generations, pool/update ordering and native precision/Vector-memory capacity remain conditional. All828unambiguous observed M21 regimes76943 now have conditional review, including21planner regimes608 in v71. This does not close whole ring/model branches or physical state capacities; remaining7MemSet1272/2BOOL80 retain separate operand/field gaps.',current_action='Continue F34-specific observable ring/write-reader/rollback and lifetime proofs or valid typed capacity evidence; incorporate existing pair/higher-order relations. Do not repeat obsolete242M21 or529plan/79zero pending counts; prior standalone planner/source-file byte-binding work is complete conditionally.')}
    gaprows={}
    for cid,z in stale.items():
        gaprows[cid]=dict(prior_gap=candidates[cid].get('conclusion_changing_gap'),prior_action=candidates[cid].get('next_profile_or_contract_action'),current_gap=z['current_gap'],current_action=z['current_action'],evidence=['planner-regime-models-v71.json','planner-source-algorithm-contract-v71.json','prior-support-file-binding-reconciliation-v71.json','operator-analysis-coverage-ledger-v73.json'],candidate_closed=False,scope='Correct stale historical frontier descriptions only; no new whole-module/physical/loaded-process attestation.')
        updates.setdefault(cid,{})['conclusion_changing_gap']=z['current_gap'];updates[cid]['next_profile_or_contract_action']=z['current_action']
        updates[cid].setdefault('review_v74',dict(scope='Stale plan/source support gap reconciliation only',evidence=str(O/e[7]),implementation=False,native_test=False,new_performance_result=False))
    write(O/e[7],dict(candidates=gaprows,overall_complete=False,whole_modules_closed=0))
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v73.json')['modules']};bs=deepcopy(old['M07']['branches']);assert len(bs)==11
    for b in bs:
        if b['id']in ['M07-OP-03','M07-OP-11']:
            b['evidence']+=e[:7];b['head_norm_applicability_review_v74']='Current pinned CP source includes full512/head RMS after QB and before tail64 RoPE. Earlier two-hop description superseded for this path;3200native brackets do not identify standalone head RMS and do not attest historical loaded path. Both norm and BF16 rounding/readiness must be preserved.'
        if b['id']=='M07-OP-03':
            b['branch']='INT8 QB1280→32768 BF16 output; pinned CP path then full512/head RMS before64-tail RoPE. Direct two-stage QB/RoPE shortcut superseded for this path; historical graph applicability open.'
            b['state']='QB_HEAD_RMS_ROPE_SOURCE_DOMAIN_REVIEWED;HISTORICAL_NATIVE_BINDING_OPEN'
    bs.append(dict(id='M07-OP-12',branch='Post-QB unweighted head RMS512 distinct from earlier QR RMS1280; full512 reduction/normalization and BF16 output before partial64 RoPE.',state='SOURCE_OPERATOR_CONTRACT_BOUND;NATIVE_APPLICABILITY_NOT_ATTESTED',evidence=e[:7],gap='No separately identified head-RMS ComputeInfo in3200saved graph brackets; do not infer absence/fusion. Historical source/kernel/caller applicability and all readers/precision/rstd/cache capacities remain unproved.',scope='One added source-semantic family only; no invented captured execution/regime or module closure.'))
    write(O/e[9],dict(modules={'M07':bs},retained_M07_families=11,new_family='M07-OP-12',extended_previous_families=['M07-OP-03','M07-OP-11'],protected_other_modules=25,whole_modules_closed=0,overall_complete=False))
    cov=deepcopy(read(O/'operator-analysis-coverage-ledger-v73.json'));cov.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v73.json',newly_reconciled_regimes=0,newly_reconciled_executions=0,new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0,QB_head_RMS_applicability_v74=dict(contract=e[0],native_visibility=e[1],new_source_semantic_family=True,new_captured_regimes=0,scope='Native regime rows unchanged; a new source/trace applicability gate is not invented as an observed operator.'),overall_complete=False);write(O/e[8],cov)
    write(O/'PROFILE-CODE-QUESTION-GATE-v74.json',dict(status='QB_HEAD_RMS_SOURCE_PROFILE_APPLICABILITY_AND_REMAINING_FUSION_EDGES',previous_turn_classification='PROGRESS:v73relationsandjointbounds',questions=['Does QB really connect directly to RoPE, or is full512/head RMS in the pinned operator path?','Can a64-tail-only fusion reproduce512-head RMS and BF16 stage order?','What does saved native visibility establish without assuming a historical loaded implementation?','How do C1 two-norm/fork and auxiliary context5120/threeKV512 domains constrain joint fusion/partition?','Which candidate gaps still incorrectly refer to completed planner/source byte checks?'],allowed_source='18saved operator contracts and four exact old pinned local operator files/clauses;3200saved graph/native brackets. No remote/model/Torch/device/capture/container/service/process work or framework lifecycle reading.',new_remote_source_read=False,new_source_files_obtained=0,local_pinned_operator_clauses_read=True,overall_complete=False))
    gap=updates['F07']['conclusion_changing_gap']
    mods={'M07':dict(finding='Correct QB→RoPE applicability for pinned CP caller: explicit post-QB512/head RMS needs whole512 reduction and BF16 boundary.3200native bracket visibility lacks standalone identified RMS but does not prove absence/fusion/history. Preserve11 families/add explicit12th,full/split fusion budgets priced separately.',evidence=e[:10],candidates=['F07'],gap=gap,profile_scope='Only operator norm/rotary/projection value/dependency clauses and saved native execution visibility; no framework audit.')}
    remaining=[]
    for s in read(O/'module-coverage-gaps-v73.json')['remaining']:
        if s.startswith('v73 central relation'):
            s='v74 central35candidate595pair index now43scoped relations/8higher-order groups,all35nodes touched.552pairs not centrally reconciled;old clauses may exist and blank rows do not prove independence/no fusion. All-pair/full-variant/whole-model opportunity and branch discovery still open.'
        remaining.append(s)
    remaining.append('v74 critical F07 correction:currentpinnedCPsourceQB→headRMS512→partial64RoPE,notdirectQB→RoPE. Full512 tail-only counterexample and explicit BF16/RNE stage-interchange counterexample.3200native QB/queryRoPE brackets have no separately identified head-RMS ComputeInfo;this does not prove absence, fusion inside QB, or historical loaded-source identity. NewM07semanticfamily12 explicitly carries binding/precision/readers/capacity gap; no native regime invented.')
    remaining.append('v74 adds15operator relation clauses/3higher-order groups for remainingF01F07F27F29F33edges;18oldcontracts4pinnedlocalfiles,no remote/device/framework work. C1 joint projection preserves two512norms/preRoPEfork;M6400 sharedinput65.536MB reference saving leaves direct-compute reference unchanged. F27/F34stale pending529plan79zero/oldM21/source-byte gaps corrected to v71/v73authority;0wholemodules,2073/285598review9/1352pending849/16/3136remain.')
    action=('继续原Goal，最新v74 PROGRESS，v73亦PROGRESS。用户目标只围绕算子融合/拆分重组和理论上限，profile优先，不研究框架。'
            '先读QB-head-RMS-RoPE-applicability/native-visibility-v74、remaining-fusion-edge-costs/math-v74、candidate-relation-coverage/clauses-v74、remaining-fusion-branch-inventory-v74和candidate-stale-gap-reconciliation-v74。'
            '本轮发现并更正F07具体契约遗漏：当前pinned dsa_cp._forward1858在QB BF16后有apply_dsa_q_rms512/head，再tail64RoPE；model.py406/638提供unweightedRMS512，device_op724-737非空callable或triton/fallback均norm。之前v61/v63直接QB→RoPE候选在此路径须被三段契约替代，或把RoPE接真正headNormproducer。QR1280是另一activation/rstd，不能共享代替。'
            '3200savedgraphQB与queryRoPE同streambracket无中间ComputeInfoop/standaloneRMS，但不能反推norm不存在/已融合进QB或当前source历史实际loaded。NativeAI_CORE913与QB914同taskId38为runtime/task包装关系样例，不可猜成norm，也不能时间相加。下一轮优先找已保存capturedIR/精确执行方/编译核代码或历史加载证据定点定案；若没有就保留两个适用域而推进其他完整分支，不能框架生命周期排查或新采集。'
            'M07保留11family加OP12 head512，OP03/11补条件，25其他matrixmodule完全不变。profile2082regimes286950ComputeInfo仍2073/285598conditionalreview9/1352pending，新sourcefamily不是新capturedregime。'
            '数学反例固定相同tail64而prefix448变0→2，headRMS尾结果变化，证明需要full512。scalarFP32/BF16RNE域tailpair[.125,3]×32、prefix0、cos=sin=FLOATsqrt(.5)、eps1e-6时norm→Rope最后[-2.703125,2.953125]，换序[-2.71875,2.9375]，不是nativeaccum实测。'
            '新图预算F=64M*512、R=64M*64：QB+headNorm+Rope物化6F+4R字节；仅Norm+Rope6F省4R；三段joint2F省4F+4R但full512reduce/allBF16round/otherreaders/readiness/Vector容量仍条件。逆Rope+packconsumer同样不能自动删full512planes，attentionepilogue+directpack才有更强预算，未证明每个M实际有pack。'
            '43selectedpair clauses+8higher-ordergroups已touchall35candidate，552pair尚未中央汇总并不等于此前全未分析/无融合；所有wholepair/allvariant/fullopportunityclosurefalse。C1两独立512权重/norm/preRopeIndexKeyfork；aux三均值/target采样后I/全5120norm/三独立KV512norm/slotbank和权重读者保留。'
            'F27/F34旧gap/action仍写18plan529+3zero79未review及2support待SHA，本轮已由v71planner608与三supportcurrent166SHA、all828M21/76943conditionalreview证据更正；historical loadedcode/actualvalues/allreaders和physical能力仍未知。不要重做这些旧pending。'
            'Goalactive，0整模块闭环35model7legacyframework/849missingPMU/16phaseunbound/3136ambiguity/7MemSet1272+2BOOL80未填；继续全模型适用分支/跨层跨卡跨迭代/真实typed容量和候选边补齐，不能因索引报告成稿结束。verify_remaining_fusion_edges_v74应PASS。本轮无remote/服务器/模型/Torch/NPU/采集/容器/服务/进程工作。')
    report='''本轮补充15条算子候选关系和3组高阶组合：累计43条具体关系、8组多候选约束，35项既有模型候选均至少有一条关系。另更正一处会影响F07融合判断的契约遗漏。

固定的CP调用源码在Q_B的BF16结果后，显式执行512维head RMS，再执行每head末64维RoPE；此前“Q_B直接接RoPE”的两段描述没有保留这一步。当前路径需要改为Q_B→head RMS512→RoPE64的三段契约，或把RoPE融合到真正的head-RMS生产者。更早的QR RMS1280使用不同激活和归一化统计量，不能替代head RMS。

已有3,200个图内Q_B/query-RoPE序列的同stream区间，没有单独识别出的head-RMS ComputeInfo。这项观察不足以证明RMS不存在、已经融合进Q_B，或当前固定源码就是历史执行代码。新增M07-OP-12把该适用性缺口单列；保留原11个语义分支，不给profile凭空增加执行。

融合判断有两个有限数学见证。一是保持64维tail相同，把其余448维从0改成2，会改变整个512维RMS分母，因此tail-only实现不能复现当前head归一化。二是在明确的FP32逐步运算、BF16 nearest-even边界模型中，tail重复[0.125,3]、其余维度为0，cos=sin=FLOAT(sqrt(0.5))、eps=1e-6时，norm后RoPE得到尾对[-2.703125,2.953125]，交换顺序得到[-2.71875,2.9375]。这不是原生内核精度实测，但足以排除在该算术域内无条件换序。

令F=M×64×512为完整query元素数，R=M×64×64为旋转尾元素数，仅按BF16值的逻辑物化参考计算：

| 所选算子图 | 逻辑参考字节 | 必须保留的工作 |
|---|---:|---|
| 分立Q_B、head RMS、RoPE | 6F+4R | 两次BF16生产边界及全部512维归约 |
| Q_B分立，融合head RMS＋RoPE | 6F | Q_B完整结果仍写回并被读入；只省尾段4R接口 |
| Q_B＋head RMS＋RoPE整体片上衔接 | 2F | 完整512维归约、各次BF16舍入、全部读者、同步和资源条件仍需满足 |

M=3时前者为638,976B，第三种为196,608B，预算差442,368B。这些口径不含cos/sin、辅助rstd、队列和额外协调，也不是实际HBM流量或请求收益。输出侧逆RoPE与pack同样区分“合并两个消费者”和“Attention直接生成最终send布局”：前者仍须读写完整输出，不能自动获得后者的物化预算。没有据此声称每一种M都存在独立pack任务。

C1/SWA联合投影继续保留两个不同权重、两组512维norm，以及C1归一化latent在long-RoPE前供IndexKey读取的分叉。共享输入与前移通信可以组合，但不能改为单个1024维norm或提前覆盖旧latent。对M=6400的直接稠密GEMM参考，少读一次输入的逻辑预算为65.536MB；在376TFLOP/s、1.6TB/s作为适用容量假设的冷物化模型中，联合前后仍受相同计算参考项支配。这不是融合后的实测时延结论，Vector、缓存和调度能力仍未绑定。[硬件容量参考](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1)。

辅助上下文组合保留三个不同checkpoint均值、采样后才得到的有序选择I、全5120维归一化、三个独立KV512映射/norm及其位置和slot发布。48个纯数学选行例子覆盖重排和重复索引。晚选行需要保留相应均值直到I就绪；保留值的工作集估算没有被解释成现有图池显存回收。

F27和F34的当前缺口文字曾继续列出已在v71完成条件分析的529次计划与79次清零，以及已完成当前文件SHA绑定的支持源码。本轮保留旧文并更正当前台账，避免重复调查。历史加载代码、实际值、读者/代际和物理容量没有因此被证明。

中央595个候选对中仍有552对尚未在此汇总全部既有证据与变体；空白项不代表独立、不能融合或此前没有分析。全部候选对、变体、更高阶组合、完整机会发现及适用分支仍未闭环。原profile覆盖保持2,073类、285,598次已有条件分析，9类、1,352次关键证据待补；整模块闭环为0，849次缺PMU、16张阶段未绑定图和3,136次功能归属歧义保留。Goal继续进行中。本轮仅使用旧资料和定点算子源码条款，没有远程、模型、Torch、设备或服务操作。
'''
    src=read(O/e[4])
    jobs=[
        dict(kind='profile',status='pass',description='3200existingQB/queryRoPEsame-streamgraphbracketsreviewed;no separately identified head-RMS ComputeInfo. Ordering is visibility only,not tensor identity or absence/fusion/current-loaded-source attestation. Profile denominators unchanged.',artifacts=[O/e[1],O/'projection-graph-chains-v61.json',O/'router-vector-regime-models-v63.json']),
        dict(kind='design-check',status='pass',description='Four exact old pinned operator clauses/18savedcontracts;correct F07 omitted explicit512headRMS path,price6Mdomains/sixheadRMSworkmodels/C1two512norm/auxretention. Fullhead/FP32BF16stage-order/inverse-cancel witnesses and48orderedselectioncases.15newrelationclauses/3newgroups,all35nodesnowtouchedbutclosureopen.',artifacts=[O/e[i]for i in [0,2,3,4,5,6]]+[N/'analyze_remaining_fusion_edges_v74.py']+[Path(x['path'])for x in src['local_operator_files']]),
        dict(kind='review',status='pass',description='F27/F34stale529plan79zero/supportSHApending descriptions corrected to v71authority. M07 retains11families/enrich2/addhead512family12;25othermodulesunchanged. Profile2073/285598review9/1352pending/global84916/3136/35model7framework0wholeclosures retained.',artifacts=[O/e[i]for i in [7,8,9]])]
    persist_refinement(version=74,previous=73,title='Head RMS applicability correction and remaining operator fusion edges',summary=dict(filename='remaining-fusion-edge-summary-v74.json',previous_turn_classification='PROGRESS:v73relationsandjointbounds',new_relation_clauses=15,total_scoped_relations=43,pending_central_pair_reconciliation=552,all35_candidate_nodes_touched=True,higher_order_groups=8,QB_head_RMS_applicability_corrected=True,historical_loaded_call_path_attested=False,graph_visibility_sequences=3200,new_source_semantic_families=1,M07_semantic_families=12,protected_other_modules=25,stale_candidate_gap_corrections=['F27','F34'],newly_reviewed_regimes=0,newly_reviewed_executions=0,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_remote_source_read=False,new_source_files_obtained=0,local_pinned_operator_clauses_read=True,implementation=False,native_test=False,new_performance_result=False,actual_physical_numeric_ceilings_proved=False,remote_job_status='No remote jobs/reads;no Torch/model/NPU/capture/container/service/process work.'),changes=['Correct omitted post-QB head-RMS512 in pinned CP F07 path,retain source-vs-historical-native applicability gate and price full-head/reordered precision/whole-output materialization domains.','Add15specific relationclauses/3higher-ordergroups:43centralrelations/all35nodes/8groups,552centralpairreconciliationsopen;C1two512norm/preRoPEfork andauxfull5120/threeKV512constraints retained.','Correct F27/F34stale planner/support pending text,retain11M07families/enrich2/addexplicithead512family12/25othermodulesidentical;profile2073/285598+9/1352/global84916/3136/0wholeclosuresunchanged.'],candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches={'M07':bs},remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[0],e[1],e[2],e[3],e[6],e[5],e[9],e[7],e[8]],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_remaining_fusion_edges_v74.py',N/'verify_remaining_fusion_edges_v74.py'],executor_prefix='zcode-remaining-fusion-edges',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
