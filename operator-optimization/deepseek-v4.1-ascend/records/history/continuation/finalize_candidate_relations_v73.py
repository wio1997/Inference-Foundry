"""Persist bounded joint-candidate analysis with an honest open relation index."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v73.json').exists()
    graph=read(O/'candidate-relation-coverage-v73.json')
    clauses=read(O/'candidate-relation-clauses-v73.json')
    assert (graph['model_candidates'],graph['pair_universe'],graph['scoped_conditional_pairs'])==(35,595,28)
    e=['candidate-relation-source-index-v73.json','candidate-relation-clauses-v73.json','candidate-relation-coverage-v73.json','joint-candidate-work-models-v73.json','candidate-composition-math-validation-v73.json','operator-analysis-coverage-ledger-v73.json']
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v72.json'))
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v72.json',newly_reconciled_regimes=0,newly_reconciled_executions=0,new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0,candidate_relation_review_v73=dict(pair_index=e[2],conditional_pairs=28,pending_central_pair_reconciliation=567,higher_order_groups=5,profile_regimes_promoted=0),overall_complete=False)
    write(O/e[5],c)
    write(O/'PROFILE-CODE-QUESTION-GATE-v73.json',dict(status='OPERATOR_CANDIDATE_RELATION_AND_JOINT_BOUND_REVIEW',previous_turn_classification='PROGRESS:v72exactpadownersandcostrefinement',questions=['Which existing fusion candidates share the same removable interface, row work or quant producer?','What numerical/fanout/WAR/partition gates survive a joint graph of three or more candidates?','How do owner filtering and ID dedup compose without summing overlapping lookup savings?','Can observer row demands invalidate an earlier head row filter, and do fewer rows remove the weight floor?','Does equal metadata plan identity imply equal Q/K/V or attention output?'],allowed_source='Eleven existing operator-specific contracts/variants; no new target/framework source reads, remote queries, device/model/capture/container/service/process work.',new_target_source_read=False,new_remote_source_read=False,pure_math_and_saved_operator_evidence_only=True,overall_complete=False))
    updates={}
    for cid in clauses['touched_candidates']:
        local=[r for r in clauses['clauses'] if cid in [r['a'],r['b']]]
        updates[cid]=dict(review_v73=dict(relation_clause_ids=[r['id'] for r in local],peers=sorted({r['b'] if r['a']==cid else r['a'] for r in local}),clauses=str(O/e[1]),central_pair_index=str(O/e[2]),joint_work_models=str(O/e[3]),math_checks=str(O/e[4]),scope='Selected scoped pair/multi-candidate composition only. Existing single-candidate facts preserved; whole pair variants, all candidate pairs and whole-model discovery not closed.',implementation=False,native_test=False,new_performance_result=False))
    remaining=[]
    for s in read(O/'module-coverage-gaps-v72.json')['remaining']:
        if s.startswith('v72 exact'):
            s='v72 exact1272 MemSet parent ConstantPadNd owners bound;actual destination/length/overlap unknown. Full pad replacement conditional with private helpers;7MemSet1272+2BOOL80 still pending.5088nativeTASK=3816ComputeInfo+1272nonComputeMIX;MC2actual800→81619.031632MB logical ref/3232B BOOLcast;v67F28budgets unchanged. Prior CANNROWIDnamespace name corrected and retained.'
        remaining.append(s)
    remaining.append('v73 central relation index35candidate nodes595pairs,28scoped pair clauses touching30candidates and5higher-order operator composition groups.567pairs not centrally reconciled here;older clauses may exist and need incorporation, not567proven-new-unanalysed optimization mechanisms. All-pair/all-variant/higher-order and whole-model opportunity closure remains open. F01/F07/F27/F29/F33 not touched by this selected relation pass. Need continue graph edge/observer/typed-capacity proof around those and remaining semantic branches, rather than treating35candidates as complete universe of opportunities.')
    remaining.append('v73 joint lower bounds require one legal selected partition/representation, complete reader-demand closure, precision/generation equality, unique necessary work per resource and new key/pack/reduction/weight/queue costs. Logical savings and standalone gains cannot simply add. Actual FLOAT/HF32/A2/Vector/INT64/UVA/L2/network/AICPU/launch capacities and request critical path remain unknown. Profile2073/285598conditionalreview+9/1352pending,849/16/3136uncertainty0wholemoduleclosures unchanged.')
    action=('继续原Goal，最新v73 PROGRESS；v72亦PROGRESS。当前目标只围绕算子融合/拆分重组和理论上限，profile优先，具体契约不足才定点读源码，不研究框架。'
            '先读candidate-relation-coverage/clauses/source-index-v73、joint-candidate-work-models-v73、operator-analysis-coverage-ledger-v73。'
            '新增全35候选595pair中央索引；28条具体算子关系触及30候选、5个多算子组合组。567pair只表示本中央索引尚未汇入其全部旧证据/分析，不能说过去全未分析或无融合；所有whole_pair/allvariants/higherorderclosure仍false。F01/F07/F27/F29/F33本轮未触及，下一轮定点补相关边/读者/适用分支并汇入旧关系。'
            '关键共同基线：F04/F20/F21/F22/F23保留roundedBF16共享平面，padABI/quantizer/merge两次round和HC读者；QR共享quant一次；C2投影两FLOAT输出/globalpool/前RoPElatentfork/oldringWAR不能被融合或换分区绕掉；输出先选WA前移或owner两级分区再算复制权重和通信，不能相加旧pack/RS节省。'
            'F24owner+dedup合并按Uowned计，构造A96/Aowned12/Uowned5/Uglobal40时baseline27648B→1440B，jointsave26208B，standalone相加40320B大于baseline而错误；不是实际host/UVA流量或命中率。'
            'F26/F35head提前选行/lazy/rejection必须取所有raw/prob/confidence等读者需求的并集。nativehead已有选行，假设24/12/6仅重叠预算见证，不是新增实际24→6证据。冷权重单次物化参考H5120/Vp16160W165478400B，M2/6/12/24的HBM参考约103.48/103.58/103.74/104.06us，减少FLOPs不等于相同倍率时延，缓存/分区变化另定域。'
            'F13/F31同plan不等于attention值：固定QSL/K/零logits，V[0,0]→[0,1]输出0→0.5见证；planner饱和长度复用只能复用调度，不能删新Q/K/V值计算。F12索引也需自己score/value/generation键。'
            '联合理论式T>=max_r(Qnecessary_r/Cupper_r)，固定direct算法/精度域，不把所有GEMM MAC当任意算法下界；同资源累加一次，不同资源允许重叠，只有证明不可避免的依赖才加时间。先证明新图合法/活跃读者/分区，再重新推界，接口budget并集本身还不是物理必省证明。'
            '2073/285598已有条件分析，9/1352仍pending(7MemSet1272 exactowner但地址/长度/重叠缺，2M05|M21BOOL80 field/reader缺)。849missingPMU/16phase-unbound/3136ambiguity，35model7legacyframework0整模块闭环保持；全26modulematrix本轮完全不变。'
            '本轮无remote/目标代码/框架新读取，无Torch/模型/NPU/采集/容器/服务/进程；只11旧算子证据与87381集合代数等纯数学检查。verify_candidate_relations_v73应PASS；Goalactive，不因报告或中央关系索引成稿标完成。')
    report='''本轮建立35项既有模型候选的中央关系索引，并围绕具体算子子图复核28条条件关系、5个多候选组合组。涉及30项候选，使用11份已保存的算子契约；没有新增框架或目标源码读取，没有执行模型、Torch或设备工作。

| 子图 | 联合判断必须保留的边界 |
|---|---|
| FFN norm→router→TopK→MC2，与共享expert汇合 | rounded BF16公共平面、独立FLOAT/logit接口、padding/count ABI、同一量化器、Down与merge两次BF16舍入和HC读者 |
| Q投影、QR量化、IndexQ/IndexWeight及输入通信 | 一份QR量化生产者；本地query与全局KV行域不同；QA的1280维norm不能扩成拼接后的1312维 |
| Compressor投影、pool、norm、IndexKey/long-KV与ring维护 | 两路FLOAT投影输出、全局pool、BF16边界、pre-RoPE latent分叉，以及旧ring先读后写 |
| Attention恢复、WA/WB、ReduceScatter与HCPost | 先选定计算/通信分区，再算权重复制、传输和buffer；保留各BF16阶段及原生归约，接口消除只计一次 |
| tail/head、Markov、拒绝与accepted-state更新 | 全部读者的行需求并集；target与draft计算分开；Markov串行依赖、raw hidden/confidence、RNG和旧状态快照仍然有效 |

两个构造例子说明为什么不能相加独立收益。第一，INT8 Engram查表若有96次引用，其中本卡拥有12次、5个唯一ID，全局有40个唯一ID，按每行256B数据和32B scale的逻辑参考口径：原始27,648B，先过滤owner再去重后为1,440B，共省26,208B。把两项各自在原始基线上计算的收益相加，会得到40,320B，超过原始工作量。这里没有声称实际ID分布、UVA流量或命中率。

第二，假设同一批head行工作从24行先选到12行，另一项单独方案只留6行，不能把分别少算的12行与18行加成30行。组合结果还必须保留其他观察者。例如raw或其他消费者需要第10、20行时，需求应取并集；第20行甚至要求重新审查第一个过滤器。该例仅验证预算与读者闭包，原生head已经存在选行，没有据此宣称新增24→6的实际机会。

减少行数还需要重新推导理论界。以H=5120、每卡Vp=16160的BF16直接稠密head为例，单份权重为165,478,400B。在“冷权重读取一次、输入输出完整HBM物化、固定直接GEMM算法”的受限参考域，以376TFLOP/s和1.6TB/s作容量假设，M=2、6、12、24的计算参考量分别约0.88、2.64、5.28、10.56µs，访存参考量约103.48、103.58、103.74、104.06µs。这个例子只说明行数与FLOPs下降不自动消除权重项，不能据此预测实测时延或可达加速比；缓存或分区变化必须另建实现域。[硬件容量参考](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1)。

跨迭代复用也区分调度与数据。固定query数、KV长度和零logits，值从[0,0]变成[0,1]时，数学Attention结果从0变成0.5；调度计划可以相同。长度饱和后的plan复用不能用来删除新Q/K/V的计算，TopK索引也需要自己的分数、值和代际证据。

联合理论模型采用T≥max_r(Q_r/C_r)：在选定的合法图、直接算法和精度域内，同资源的必要工作只累计一次，不同资源允许重叠；新引入的查键、打包、归约、队列、复制权重和就绪等待都必须进入模型。接口逻辑预算的并集本身还没有证明所有接口均可删除，也不是物理流量或请求收益。

本次核对87,381个小集合的并集与观察者关系。中央索引有595个候选对，其中28对纳入了本轮具体关系，其余567对尚未在此汇总既有证据和变体；这不代表过去全部没有分析，更不代表它们彼此独立或无法融合。F01、F07、F27、F29、F33未被本轮选定关系触及。全部候选对的全部变体、更高阶组合、完整模型机会发现与适用分支仍未闭环。

原始profile覆盖保持2,073类、285,598次已有条件分析，9类、1,352次关键证据待补。全部26份模块记录不变，整模块闭环仍为0；849次缺PMU、16张阶段未绑定图和3,136次功能归属歧义保留。Goal继续进行中。
'''
    source=read(O/e[0])
    jobs=[
        dict(kind='review',status='pass',description='Reconcile28specific operator relation clauses touching30of35candidates and5higher-order groups using11saved contracts.595pair index explicitly leaves567central reconciliations open;prior clauses not invalidated,no title-based independence or complete opportunity discovery claim.',artifacts=[O/e[i] for i in [0,1,2]]+[Path(x['path']) for x in source['files']]),
        dict(kind='design-check',status='pass',description='Joint owner-dedup population feasible,26208B versus invalid40320B savings;row-observer union witness,4coldhead dense reference domains with165478400B weight,plan-equality-not-activation witness,87381setalgebra cases. No native execution or runtime gain claim.',artifacts=[O/e[i] for i in [3,4]]+[N/'analyze_candidate_relations_v73.py']),
        dict(kind='review',status='pass',description='All26modulematrix records unchanged;selected30candidate records gain scopedrelationlinks only5othersunchanged. Profile2073/285598review9/1352pending,849/16/3136uncertainty,35model7framework0wholeclosures protected.',artifacts=[O/e[5],O/e[2]])]
    persist_refinement(version=73,previous=72,title='Joint operator candidate relations, observer closure and nonadditive bounds',summary=dict(filename='candidate-composition-summary-v73.json',previous_turn_classification='PROGRESS:v72exactpadownersandcostrefinement',candidate_relation_nodes=35,pair_index_size=595,scoped_conditional_pair_relations=28,pending_central_pair_reconciliation=567,candidate_nodes_touched=30,higher_order_groups=5,all_pair_variants_closed=False,all_higher_order_combinations_analyzed=False,new_model_candidates=0,newly_reviewed_regimes=0,newly_reviewed_executions=0,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,all26module_records_unchanged=True,new_target_source_read=False,new_remote_source_read=False,implementation=False,native_test=False,new_performance_result=False,actual_physical_numeric_ceilings_proved=False,remote_job_status='No remote jobs/queries or target source reads; no model/Torch/device/profile/container/service/process operations.'),changes=['Central35-node595pair index plus28specific conditional clauses and5higher-order groups;567not centrally reconciled here,prior evidence retained.','Joint owner-dedup and head-observer union costs prevent double counting;head weight term and plan-versus-activation distinction quantified under explicit domains,87381finite set checks.','Thirty candidates gain scopedrelationlinks;all26module records/profilecoverage2073/285598+9/1352/global849/16/3136and0wholeclosures unchanged.'],candidate_updates=updates,framework_updates={},module_updates={},semantic_branches={},remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[2],e[1],e[3],e[0],e[5]],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_candidate_relations_v73.py',N/'verify_candidate_relations_v73.py'],executor_prefix='zcode-candidate-relations',main_script=Path(__file__),source_read=False)


if __name__=='__main__':
    main()
