"""Persist exact penalty/histogram fusion and representation scope;no target implementation."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v110.json').exists()
    assert read(O/'penalty-histogram-preflight-v110.json')['status']=='PASS'
    names=['penalty-histogram-inputs-v110.json','penalty-profile-applicability-v110.json','penalty-source-contract-v110.json',
        'penalty-backend-row-and-dtype-contract-v110.json','penalty-histogram-representation-v110.json','penalty-parameter-demand-v110.json',
        'penalty-integer-reference-validation-v110.json','penalty-precision-and-invalid-rewrites-v110.json',
        'penalty-work-and-resource-models-v110.json','penalty-cold-query-symbolic-ceiling-v110.json','penalty-state-and-composition-v110.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-operator-footprints','candidate-pair-navigation','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v109.json')));d['penalty_histogram_review_v110']=dict(evidence=names,
            source_callsite='F35 target penalty/histogram with explicit Triton/fallback row contracts',new_pairs=0,new_native_regimes=0,new_native_attributions=0)
        write(O/(base+'-v110.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v109.json'))
    for r in obligations['obligations']:
        if r['id']=='G13':
            r['remaining']='Penalty histogram producers/consumer andeight parameter-demand patterns are now source/conditional-work reviewed;fallback utility body retrieved and INT64/row-expansion/custom-repetition differences made explicit. Do not reread these as missing bodies. Actual guards/row ownership,loaded fallback custom-op provider,bank identities andall readers remain named evidence questions;v92filter/v101recovery/v106projection/v108MinTokens are prior answers.'
            r['penalty_histogram_review_v110']=dict(evidence=names,source_callsite_questions_scoped=True,actual_provider_or_full_sampler_closed=False)
        if r['id']in ['G02','G05','G06','G07','G08']:
            r['penalty_histogram_review_v110']=dict(evidence=names[3:],
                request_prefix_state_and_parameter_dependencies_priced=True,cold_query_bound='SYMBOLIC_CEILING',native_capacities_or_runtime_flags_attested=False)
    obligations.update(parent='coverage-current-obligations-v109.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v110.json',obligations)
    branches=deepcopy(next(m['branches']for m in read(O/'module-phase-branch-matrix-v109.json')['modules']if m['id']=='M17'))
    gap=('Selectedhistogram/penaltysourceandfallbackutilityarenowread:exactnative-castcount/union/outputpresence roles,request+precedingdraftprefix andactualemitted-tokenstatecontracts. '
        'ActualnoPenalties/HAS_TRITON/parameterrows,runtimeowner/epochs andnativecustomrepetition/round/NaN/FMA/cast policy remainunbound. '
        'Fallback sourcepassesrequest arrayswithoutTritonrepeatgathers;B=Tnotownerproof,INT64V+1counts≠INT32V. '
        'Private-score fusionmustretainallraw/processed/logprob/entropy/random/mutation/error/publicationreaders;packed/unionbank construction,rollback/provenance andextendedlifetimecosts charged. '
        'Logical32→30→22TV refs andcold wordquery boundsnotnativeHBM/MFU/speedup;noPenaltiesTrue means0currentgain. Priorv108MinTokens/finiteforcinggatesretained. Otheractualnativebank/field/capacity/fullscopegaps remain.')
    for r in branches:
        if r['id']=='M17-OP-09':
            r['evidence']=list(dict.fromkeys(r['evidence']+names));r['gap']=gap
            r['penalty_histogram_review_v110']=dict(source_count_and_penalty_bodies_reviewed=True,source_files=8,new_utility_files=1,
                parameter_patterns=8,request_contexts=81120,prefix_row_checks=206856,ragged_cases=64,
                actual_flags_or_all_readers_closed=False,evidence=names)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v109.json'))
    audit.update(previous_goal_turn='PROGRESS:v109 authorized tiny graph metadata probe with exact limits;postcheck PASS149',input_checkpoint=109,followup_checkpoint=110)
    audit['penalty_histogram_review_v110']=dict(evidence=names,operator_scope='Count/presence producers -> penalty -> selected mask/argmax;request-prefix andcross-iteration representations',
        coefficient_patterns=8,fallback_row_and_dtype_difference_retained=True,cold_query_ceiling='SYMBOLIC_CEILING',new_device_work=False,
        native_activation_or_physical_gain=False,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v110.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v110.json',dict(status='SAVED_F35_BOUNDARY_AND_NAME_NAVIGATION_THEN_EXACT_HISTOGRAM_PENALTY_HELPERS',
        questions=['Which semantic information is actually consumed by repetition,frequency andpresence?',
            'Can prompt/output masks,countcast,penalty andargmax share boundaries without losing rows or precision?',
            'Can draft histories be represented as per-request base plus preceding draft prefix and safelymaintained acrossiterations?',
            'Does HAS_TRITONFalse receive the same row domain and count/precision provider?',
            'What work/storage remains after choosing the new representation?'],
        saved_databases=32,cached_operator_source_files=7,new_operator_utility_files=1,read_only_caller_controls=2,
        no_target_implementation_or_new_device_work=True,new_Torch_NPU_capture_or_model=False,
        official_public_tensor_contracts='Web text available;direct HTTP403 retained;no Browser/Chrome/CUA;versioned docs not deployed-runtime attestation.',overall_complete=False))
    write(O/'penalty-validation-history-v110.json',dict(events=[
        'Initial CPU analyzer had a missing whitespace in a conditional expression;syntax error occurred before any analyzer output/shared checkpoint write. Corrected before analysis.',
        'Strengthened the initial arbitrary-FLOAT adversary to BF16-representable input/alternatives and added both stored FLOAT and retainedBF16 input interfaces;do not transfer a4V cut into the2V graph.',
        'Expanded the scope to the actual fallback utility after discovering its different request-row delegation;one utility retrieved read-only withtwoexactcallercontrols.',
        'Direct official-doc HTTP requests returned403;official Web-reader contract text saved instead. No browser escalation needed.',
        'Independent verifier covers exact source/owner/state/typed-interface/adversary conditions;all current-model attribution and capacity gaps remain.'],new_device_work=False,overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Current obligations-v110. v109minimal probe is finished,not pending andnot authorization for further/full-modeldevicework. No equivalent probe repeat. Penalty/native execution is unbound;current9operandregimes/physicalMFU/fullscopegaps remain.')
    report='''本轮围绕F35的直方图 → penalty → 采样归约子图分析融合、拆分和跨迭代重组。32份既有profile投影中没有匹配的penalty/bincount名称标签；这只是检索结果，不能证明分支未执行或no_penalties实际值。若no_penalties=True，源码本来就提前返回，该子图当前收益为0；temperature0/all_greedy本身不等于无penalty。

Triton调用中，prompt侧只有“出现过”参与repetition；output侧的count转FLOAT参与frequency，output出现位参与presence。不能通用地把output count退化为BOOL，也不能把prompt次数加入frequency。8种系数组合给出充分统计表示（不宣称所有输入或实现都必须按该表示存储/执行）：rho!=1需要联合出现位U=prompt OR output；frequency!=0需要原生cast后的计数值；仅presence需要output出现位。frequency计数可导出output出现位。仅repetition分支可以合并prompt/output出现位；全中性系数不再依赖历史，但删除零项/恒等运算本身仍须处理signed-zero/FTZ/NaN及原生精度合同。

在合法token0..V（V为padding）、输出-1先转V、每bin不溢出INT32的声明域，request j的第d条target draft行具有C[j,d,v]=C_committed[j,v]+count(spec[j,:d],v)。零draft请求不产生target-history行；不能用固定B×K推断所有row owner。可条件性共享每请求prompt出现位和已提交output基表，再加draft前缀差分。直接按每v扫描前缀会新增sum(V*n*(n-1)/2)次相等判断；小直方图/列表/bitmap也有构建、寻址、同步和发布成本，不能把复用看成零成本。

跨迭代bank只提交验证后实际输出的token，包含实际发出的recovered/bonus token；不能持久化全部draft提案。回滚/截断要按重数删除或重建，槽位复用要切换request/prompt epoch。只有union bitset时，重复token删除和prompt保留需要额外重数/来源或重建。新持久bank延长生命周期，不能直接把每调用数组的差额当作峰值内存节省。

该整数重组不允许重排penalty的FLOAT运算。两个参考反例中，BF16值0.10009765625和FLOAT rho≈1.7采用“先舍入倒数再乘”与直接除法结果不同；BF16值2^25、frequency=presence=count=1时，按序两次FLOAT减1得到33554432，先合并成减2得到33554430。它们是声明IEEE参考策略的反例，未证明当前原生编译器采用哪个reciprocal/FMA/FTZ策略。

条件性融合点包括：把output count>0放进已有count消费者，省掉独立BOOL plane；prompt producer按所需出现位/联合位特化；把penalty结果直接交给选定mask/MinTokens/有限forcing/argmax图。后者仍保留v108的恢复谓词与A/B/H状态，不能省掉有读者的完整处理后logits、raw、entropy、random/recovery或logprob结果，全部mutation/error/参数/捕获epoch仍需绑定。

显式Triton逻辑基线（I32/BOOL/FLOAT为4/1/4B）中，两路zero/count→mask和penalty为32TV字节，加独立argmax为36TV。内联output_mask消除6TV接口预算，成为30TV；再在全部读者允许时移除penalty→argmax私有FLOAT写读8TV，成为22TV，另计新谓词/归约/元数据。该数字未计历史读取、原子更新、contiguous/gather/H2D/padding、参数、bitmap构建或必需输出，不能当作HBM、实测时间或可独立叠加的F35收益。

源码私有辅助plane值大小为10TV；每请求I32基计数加packed prompt位图的参考大小为4BV+4B*ceil(V/32)，另有draft差分与状态成本。这些是所选表示的值大小，不能当作实际峰值/保留内存。B/T=2/10、4/10、4/20只沿用保存的shape/capacity参考，不证明实际有效draft数。

Triton缺失分支不能直接继承上述合同。只读取回一个缺失的utility文件，并核对两份caller SHA：fallback使用INT64的[T,V+1]计数，再返回[:,:V]视图，并把repetition交给独立custom op；其辅助count/mask值大小参考为18TV+16T，不能套用Triton的32/36/30/22TV流量预算。Ascend的HAS_TRITONFalse直接调用Sampler.apply_penalties，未做主路径的repeat_indices gathers；若参数仍是请求B行，B=T也不证明owner相同。例如draft数[0,2]的两条target行都属请求1，而未展开请求行是[0,1]。

对这条fallback的形状推导，以[官方scatter契约](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.scatter_add_.html)及[广播/原地运算契约](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html)为条件：scatter索引不自动扩展请求行，普通广播也不等于按draft owner展开。文档版本不证明目标Torch/Ascend实现；实际fallback可达性、是否有已展开输入、custom repetition的原生dispatch/精度仍未绑定。本轮没有执行或修复该分支，未宣称当前负载存在运行错误。

理论上限为SYMBOLIC_CEILING：V>=3、1<=L<=2^20、rho1/frequency1/presence0、无旧计数/摘要/缓存、精确word-query、任意BF16可表示logits的声明域。历史全为token2、logits=[0.5,0,-1024,...]时ID0胜出；任何未读历史位置改为token0会使ID1胜出，改变任何未读logit也能改变ID，且可保持BF16表示。因此这个域的最坏输入需要读取V个logit和L个INT64历史word。已物化FLOAT输入为4V+8L字节，保留BF16并精确widen的另一输入图为2V+8L。选定一个共同存储边界且容量上界为U时，T>=(s*V+8L)/U（s=4或2）。不把旧FLOAT界套到BF16图；有warm bank、隐式producer、压缩或预先摘要时须重建界。U尚未绑定，没有数值物理时间、MFU或提速。

CPU参考覆盖81120个请求上下文、206856条精确prefix计数核对、64组含零draft的ragged布局、8种参数角色、4个错误整数改写和2个FLOAT策略反例。它们检验局部公式与证据一致性，不证明实际输入/flags或原生kernel。未新增NPU工作、模型执行、候选关系或原profile归属；原9类/1352次关键字段、全模块最低MFU与整体理论物理上限仍未完成。
'''
    action=('继续原Goal,v110PROGRESS：先读penalty-histogram-summary/report/source/backend-row-dtype/representation/parameter-demand/work/cold-ceiling/state及current-obligations-v110。v109PASS149，本轮preflight以验证文件为准。'
        'F35/M17OP9 count/penalty bodies8files已审7cached+1newutility；原166只读archive3files控Sampler/penwrapper两SHA，无containerprocess/lib/Torch/NPU/model/新capture。'
        'TritonI32T,V，prompt只presence/outputcastcount；8coeffpatterns Uifrho!=1,Ccastiffreq!=0,Oifpres!=0且freq0，neutral无history依赖非删arithproof。'
        'C_jdv=Ccommitted_jv+count(spec[:d],v)，81120contexts/206856rows/64ragged；只commitactualemittedinclrecoveredbonus，rollback/duplicate/slotepochs/oldreaders/persistentlifetime需证明。'
        '32TVpenalty+4TVargmax=36TV，inlineOmask去6TV→30TV，私有penaltyargmax去8TV→22TV需allreaders+新cost，一creditunion非HBM。Aux10TV vsI32base4BV+bitset，fallbackI64V+1aux18TV+16T别混budget。'
        'FallbackHASfalse直接Sampler无repeatgather，请求B与T维/owner另证，B=T在n[0,2]不够，customrepetitionprovider/precision未bind；sourceconditionalcase不是已执行bug。'
        '2FLOATcounter参考：BF16.10009765625/rhoF321.7 reciprocal×vsdivide不同；2^25两次minus1=33554432 vsminus2=33554430。Actualnativepolicy未知。'
        'SYMBOLIC coldwordquery V>=3 L<=2^20 BF16logits[.5,0,-1024...] historyall2：每logit/historyword可改ID。存量F32 4V+8L vs retainedBF16 2V+8L共同tier初态无cache，T>=bytes/Uupper；不适用warmbank/implicitproducer/compressed/allalgo，实际U未知。'
        '32savedDB labelregex0不是noPenalty实际flagproof；noPenTrue此部分gain0，BOOL24不可能是此logicalTVmask但不排除nativebackendtemps/不识别field。'
        '不要继续重复v92pivot/v101recovery/v106projection/v108MinTokens/v110countbody。优先回到关键G03/G04字段与G08硬件capacity：从已有raw/cachedprofile精确MemSet provider/kernel/ABI来源或当前finitegraphbank着手；只查会改变operator结论的代码。必要新device/model诊断先具名最小字段和有限可审计划，v109最小probe授权已完成不含fullmodel/replay。'
        '35/353/26、132/25/463、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU不变。SSHcontrol已关无livejob；GoalACTIVE TaskACTIVE Loop004EVALUATING，verify_penalty_histogram_v110必须PASS，不改完成artifacts/重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=110,previous=109,title='Penalty histogram demand,request-prefix reuse and symbolic ceilings',
        summary=dict(filename='penalty-histogram-summary-v110.json',previous_turn_classification='PROGRESS:v109 approved own-graph metadata probe and capability limits;postcheck PASS149',
            cached_source_files=7,new_operator_utility_files=1,read_only_caller_controls=2,saved_databases=32,new_device_work=False,
            coefficient_demand_patterns=8,request_contexts=81120,prefix_row_checks=206856,ragged_cases=64,invalid_integer_rewrites=4,FLOAT_reference_counters=2,
            refined_candidates=1,refined_modules=1,branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,
            relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            ceiling_status='SYMBOLIC_CEILING',actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
            remote_job_status='Read-only original166 source archive copy andlocalCPU operator/source/algebra checks finished;SSHclosed. No newNPU/model/Torch/capture/replay/service changes;v109probe alreadyterminal.'),
        changes=['Derive parameter-dependent native-cast count/union/outputpresence demand and scoped count→penalty→argmax fusion interfaces.',
            'Model per-request history plus precedingdraftprefix andactual-emitted-token crossiteration state;price new metadata/prefix/retirement work.',
            'Read exact fallback utility with callercontrols;keep I64V+1/customrepetition andmissing request expansion distinct from TritonI32.',
            'Bound cold query input with BF16-representable adversary andseparate4V/2Vgraphs;no actual branch activation,physicalMFU or speedup claim.'],
        candidate_updates={'F35':dict(review_v110=dict(evidence=names,source_provider_count_and_penalty_review=True,
            coefficient_patterns=8,prefix_row_checks=206856,ragged_cases=64,fallback_dtype_and_owner_contract_separate=True,
            ceiling_status='SYMBOLIC_CEILING',actual_flags_native_policy_all_consumers_closed=False,new_native_promotions=0,implementation=False,native_test=False),conclusion_changing_gap=gap)},
        module_updates={'M17':dict(candidates=['F35'],finding='Scoped histogram/penalty source now supports conditional presence/count specialization,per-request prefix/state reuse andconsumer fusion. FallbackI64/nativecustomop andrequest-row delegation have distinct conditions. Logical interfaces andcoldquery bounds are priced without actualgain/nativeactivation.',evidence=names,gap=gap)},
        semantic_branches={'M17':branches},remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reuse32savedDB labelnavigation andF35boundaries;read one missing utility withtwoexactcallercontrols;no new native role/flag attribution.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Eight parameter-demand patterns,206856prefix rows/64ragged layouts,integer/FLOAT counterexamples,typed interface budgets andBF16-compatible coldquery symbolic bound.',artifacts=[O/'penalty-histogram-preflight-v110.json',O/'penalty-validation-history-v110.json',O/'whole-scope-requirement-audit-v110.json',O/'coverage-current-obligations-v110.json'])],
        report=report,read_first=names+['coverage-current-obligations-v110.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_penalty_histogram_v110.py',N/'verify_penalty_histogram_v110.py',N/'read_penalty_fallback166_v110.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='penalty-histogram',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
