"""Persist rejection-sensitive DSpark cache demand and one new conditional relation."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v113.json').exists()
    assert read(O/'draft-cache-join-preflight-v113.json')['status']=='PASS'
    names=['draft-cache-join-inputs-v113.json','draft-cache-join-source-contract-v113.json','draft-cache-join-native-frontiers-v113.json',
        'draft-cache-slot-conflict-algebra-v113.json','draft-cache-order-and-witnesses-v113.json',
        'draft-cache-demand-and-readiness-v113.json','draft-cache-joint-work-and-ceiling-v113.json','draft-cache-composition-v113.json']
    packet=read(O/names[-1]);clause=packet['clause']
    rel=deepcopy(read(O/'candidate-relation-clauses-v112.json'))
    assert all({r['a'],r['b']}!={'F27','F28'}for r in rel['clauses'])
    rel['clauses'].append(deepcopy(clause))
    g=next(g for g in rel['higher_order_groups']if g['id']==packet['refine_group'])
    g['draft_cache_demand_review_v113']=dict(evidence=names,
        contract='Metadata/queryslot readiness can backpropagate finalcachewrite demand to contextKV andthe union of all maincontext readers. Current r/q mapping can force Wc→Wq conflicts. Preserve oldreader/nextiteration/COW/group/numeric epochs;do not count v35rowpacking or perlayerlazy prep as new.',
        actual_masks_and_all_runtime_readers_proved=False,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v112.json',clause_count=133,selected_pair_count=133,new_pair_clauses=1,new_higher_order_groups=0,overall_complete=False)
    write(O/'candidate-relation-clauses-v113.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v112.json'));row=next(r for r in cov['pairs']if{r['a'],r['b']}=={'F27','F28'});assert row['clauses']==[]
    row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[clause['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v112.json',parent_relation_index='candidate-relation-clauses-v113.json',scoped_conditional_pairs=133,remaining_pair_relations=462,higher_order_groups=25,
        scope='133scopedconditionalpairs/25groups/35modelcandidates;462centralpairreconciliations andallvariantgates remain. NewF27/F28backwards cachewrite-demand relation isconditional;v35contextqueryrowpacking isprioranswer.',
        draft_cache_demand_review_v113=dict(evidence=names,new_pairs=1,new_groups=0),overall_complete=False)
    write(O/'candidate-relation-coverage-v113.json',cov)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v112.json'));idx={r['id']:r for r in read(O/'candidate-record-index-v112.json')['model_candidates']}
    for r in footprints['candidates']:
        h=hashlib.sha256((R/'candidates'/(r['id']+'.json')).read_bytes()).hexdigest();assert h==idx[r['id']]['master_sha256'];r.update(captured_master_sha256=h,captured_master_version=112)
        if r['id']in ['F27','F28','F29','F31']:r['draft_cache_demand_review_v113']=dict(evidence=names,new_native_atoms=0,actual_slot_aliases_not_attested=True)
    footprints.update(parent='candidate-operator-footprints-v112.json',overall_complete=False);write(O/'candidate-operator-footprints-v113.json',footprints)
    nav=deepcopy(read(O/'candidate-pair-navigation-v112.json'));row=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F27','F28'});assert row['conditional_clause_ids']==[]
    row.update(conditional_clause_ids=[clause['id']],semantic_pair_reconciliation_done_in_stated_scope=True,
        draft_cache_demand_review_v113=dict(evidence=names,backwards_demand_not_shared_arithmetic_atom=True))
    nav.update(parent='candidate-pair-navigation-v112.json',parent_footprints='candidate-operator-footprints-v113.json',parent_relation_index='candidate-relation-clauses-v113.json',
        scoped_conditional_pairs=133,remaining_central_pair_reconciliation=462,overall_complete=False)
    write(O/'candidate-pair-navigation-v113.json',nav)
    native=deepcopy(read(O/'operator-analysis-coverage-ledger-v112.json'));native['draft_cache_demand_review_v113']=dict(evidence=names,
        source_order_frontiers=104,existing_context_cycles_rechecked=312,preceding_main_chains=88,new_native_regimes=0,new_native_executions=0,new_native_attribution_or_phase_promotions=0,new_conditional_relation='REL113-01')
    write(O/'operator-analysis-coverage-ledger-v113.json',native)
    obligations=deepcopy(read(O/'coverage-current-obligations-v112.json'))
    for r in obligations['obligations']:
        if r['id']=='G10':r['remaining']=r['remaining'].replace('132 conditional','133 conditional').replace('463 pairs','462 pairs')
        if r['id']in ['G02','G05','G07','G09','G10']:
            r['draft_cache_demand_review_v113']=dict(evidence=names,
                rejection_sensitive_slot_WAW_and_alllayer_backward_demand_scoped_reviewed=True,
                actual_request_slots_r_q_andallbank_readers_unbound=True,whole_requirement_closed=False)
        if r['id']=='G13':r['draft_cache_demand_review_v113']=dict(evidence=names,
            v34_v35_lazy_context_andsameWKVrowpack_are_prioranswers=True,
            new_exact_conflict_and_backwards_main_demand_formula=True,actual_masks_andretirement_missing=True)
    obligations.update(parent='coverage-current-obligations-v112.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v113.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v112.json'));audit.update(previous_goal_turn='PROGRESS:v112 installedclockfield/A3semantics and14operatorclockcertificates;postcheck PASS191',input_checkpoint=112,followup_checkpoint=113)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=133,higher_order_groups=25,central_pairs_not_reconciled=462)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','THEORY_AND_MEASURED_COMPARISON']:r.setdefault('evidence',[]).extend(names)
    audit['draft_cache_demand_review_v113']=dict(evidence=names,source_order_frontiers=104,
        conditional_conflict_write_demand_andmain_union_proved=True,new_physical_time_or_actual_masks=False,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v113.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v113.json',dict(status='EXISTING_CONTEXT_DRAFT_NATIVE_PROFILE_THEN_ONLY_SLOT_KERNEL_AND_CACHE_JOIN_CLAUSES',
        questions=['Does rejection filter currentcontextslotcopy orshift queryslots only?',
            'What precise WAW/RAW edges constrain lazy context/query overlap?',
            'When can removing overwritten contextKV backpropagate into mainprojection/fullrow norm?'],
        old_pipeline_and_rowpack_answers_checked_before_extension=True,new_public_or_remote_source_files=0,
        no_frameworkwide_audit=True,no_model_Torch_NPU_capture_replay_or_implementation=True,
        original_Pad_probe='StillPREPARED_NOT_EXECUTED;PENDING_REPLY',overall_complete=False))
    write(O/'draft-cache-join-validation-history-v113.json',dict(events=[
        'v112 predecessor revalidated PASS191. Pendingcentralpairs have nohistoricalsharedatom/interfaceedge;emptyfields were nottreated as independence.',
        'v34/v35alreadycontainperlayerlazy contextKV/sameWKVrowpacking andcontext-before-query duplicatewrite warnings. Those opportunities areprioranswers,not newcounts.',
        'Exactsourceutilityshowsfullcontextslotcopy andqueryeffective_seq_len=seq_len-rejected. Newwork derives min(r,q) overlap onlyunderdeclaredlinear/injective/COWsafe mapping andusesactualquerycount,not universalK+1.',
        '104typedsourceorderfrontiers,312contextcycles and88precedingmainchains crosschecked against32saved task projections. This doesnotbindsame actualrequest/bank/slotvalues orcounterfactualcriticalpath.',
        '25344CPUmapfixtures pass;independent3240intervalcases and6000DCPcompactinversecases pass. Cache-state schedules keep both immediate read andfinalpersistentstate obligations.',
        'Firstpreflight usedonelexicalliteral forcachebank identity andfailed:contextsourcefirstaliases swa_cache_layer thenindexes itskv_cache. Checker now resolves that exactASTaliasandcallargument. No runtime/source change;actualloadedbank identity remains unknown.',
        'Metadata splitting is independent ofcontextactivationvalues,butmethodmutations/copies/seed/errors/capture cannot move wholesale. RequiredWAWedges areperslot;no forcedalllatercontextlayers barrier.',
        'ContextKVrowpruning reachesF27 onlythrough union ofalllayer andothercontextbankreaders. Preserve5120mainnorm/512KVnorm/numericstages;priceconflictmap/packing/publication once. Symbolicwork,notphysicalHBM/MFU/speedup.',
        'REL113-01F27/F28 is newbackwardsdemand relation;HG85alreadycontainsallmembers,so no newgroup.133conditionalpairs/25groups/462pending;allnative/wholeclosuregates retained.'],overall_complete=False))
    gap=('Newsource-scoped min(r,q) overlap andcontext/query WAW/RAW/demand model requiresactualr/q,request/source-row maps,validslots,consistentCP/group/page/COWepochs andsame fullbytecachewriter. '
        'Originalprofiledoesnotrecordtheseoperands. Native104sourceorderfrontiers/312cycles/88mainchains aretimings/schema,notactualbankidentity,allreaderclosure orcausalbenefit. '
        'Onlyunobservedfull-overwrittencontextKVrowscanbepruned;mainprojection/full5120norm requiredeadrows acrossunionofalllayerandotherbankreaders. '
        'Preserveemptycontext/dummy/no-localQ-globalKV/anchor/configquerycounts/errors/rounding/collectives andretirement. Price onegraphincludingnewmaps/masks/packing/publication;oldv35rowpack/lazyopportunitiesnotnew. '
        'No nativepromotions,physicaltime/MFU/speedup orwholeclosure.')
    updates={cid:dict(review_v113=dict(evidence=names,source_order_frontiers=104,context_cycles=312,preceding_main_chains=88,
        CPU_slot_mapping_cases=25344,new_relation='REL113-01',refine_group='HG85-VERIFY-CONTEXT-DRAFT',
        actual_runtime_slots_masks_allreaders_andepochs_bound=False,new_device_work=False,implementation=False),conclusion_changing_gap=gap)for cid in ['F27','F28','F29','F31']}
    modules={'M18':dict(candidates=['F27','F29'],finding='Contextrow demand maybackpropagate throughprivateKVdeadstores into mainprojection onlyafteralllayer/otherbankreader union;actualrejected/slot/alias evidence remains missing.',evidence=names,gap=gap),
        'M19':dict(candidates=['F28','F31'],finding='Earlymetadata/queryinitialization has a distinctreadiness path;context/querycachewrites mayoverlap rejectedslots andrequireperslotprecedence. No-localQ canstillpublishglobalKV.',evidence=names,gap=gap)}
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Currentv113:conditional F27/F28 backwards cache-demand relation added;133pairs/25groups/462pending. Formula O=summin(r,q) onlyinsideexplicitconsistentinjectivemappingdomain;actualoperands/unions/epochs unknown. NewPadprobe stillawaits explicitreply,notrun. All14wholeobligations remainopen.')
    report='''本轮围绕DSpark上下文预计算与query路径的缓存发布关系。先检查旧台账：v34/v35已提出按层延迟准备和同一WKV权重拼行，并已提示context-before-query写入顺序。本轮不把这些旧机会重新计为发现，而是补齐拒绝后槽位冲突的条件公式，以及它对上游context主投影工作量的影响。

复核32份已保存任务投影后，得到104组严格的源码顺序模式：每组3个context KV计算/写入周期随后是draft repeat，其中88组之前有context主投影链，16组为dummy调用域。312个已有context周期的任务键、开始/结束时间和stream均逐项核对。这只证明保存的执行顺序，不能证明实际request、槽位地址或可回收关键路径时间；原始执行归属不提升。

定点源码显示，metadata kernel把全部context positions和slot mapping复制到输出；num_rejected并不裁剪这个复制。query的线性cache位置从seq_len-rejected开始，RoPE位置则从最后有效context位置推导，二者不能互代。context和query的标准writer指向同层swa_cache_layer的kv_cache[0]；实际加载对象、地址、字节写入范围和group/page epoch仍须运行证据。初始化时sample_from_anchor选择配置K或1+K个query，公式使用实际query_count，不把它固定为当前scheduled K+1。

在每请求context逻辑位置为[s-n,s)、拒绝数0<=r<=n、query位置为[s-r,s-r+q)，并且slot映射一致、页映射无别名、跨请求所有权/COW正确的域内，交集为[s-r,min(s,s-r+q))，重叠数恰为min(r,q)。DCP owner/local压缩映射有显式逆式，因此跨owner相加仍得到该数量；单rank不能直接套用总数。实际metadata若有无效槽、页别名或代际不一致，须计算物理slot/字节集合，不能直接套公式。本轮25,344个CPU映射参考用例，以及独立3,240个区间用例和6,000个DCP逆映射用例通过，均不属于原生运行验证。

具体反例中，context位置[7,8,9]、r=2、query位置[8,9,10,11,12]在8/9冲突。让旧context写入晚于query，会覆盖新的query值。只让即时attention读到正确值仍不够：晚到的context写入还可能破坏持久cache最终状态。保留两条物理写路径时，应保留冲突字节上的context→query顺序及各reader的RAW依赖；这不要求所有无关槽或后续层的context工作全局串行。no-localQ也不等于无query KV写入，旧profile的cache-only rank分支继续保留全局KV工作。

新的工作量机会是沿读者需求反向裁剪。若某context写入在被保证完整query覆盖前没有任何读者或状态/错误观察，其私有KV投影、512D norm和RoPE工作才可能随该死写入删除。更上游的F27主投影/5120D norm是否可删，取决于所有层剩余context需求与其他bank读者的并集；只在一层被覆盖不够。metadata/ID计算不读context activation，可研究从当前包含bank copy的入口中拆出以提前获得需求掩码，但不能直接搬动整个带seed、metadata和bank副作用的方法。dummy、无context、无完整overwrite保证的分支不据此裁剪。

在普通完整GEMM、无预计算输出复用、精度边界相同的声明域中，所选图的WKV参考工作为2*5120*512*各层剩余context/query行数之和，main参考工作为2*15360*5120*全体剩余main需求行数。两组纯参考例子中，context/query行数12/10与24/20、给定拒绝数组后分别有3/8行覆盖：单层cache显式值写入差额为3072/8192B，WKV差额为15,728,640/41,943,040 OPS。这些不是原profile实际拒绝数、HBM流量、物理时间下界或提速。冲突掩码、逆行映射、压缩/拼行、发布、并行竞争和丢失的复用都要计入同一图，不能与旧rowpack、cache写入和上游裁剪重复计收益。

新增条件关系REL113-01（F27/F28），细化已有HG85组以及F29/F31合同；它表达query最终写入需求向context主投影的反向传播。现在133条条件关系、25组、462对仍待核对；35项候选、353条语义记录/26暂定模块保持不变。9类/1352次原生执行、849缺PMU、16未绑定phase图和3136含糊归属均不变，0个整模块关闭。原生Pad小探针仍等待单独明确回复，本轮没有远程操作、模型、NPU计算、捕获、replay或实现修改。整体目标继续进行。
'''
    action=('继续原Goal，v113PROGRESS，先读draft-cache-join-summary/report/source/nativefrontiers/conflictalgebra/demand/work/composition/current-obligations-v113，verify_draft_cache_join_v113须PASS。'
        'v112PASS191，v113CPUpreflightPASS687；32savedTASKjson复核104context3cycles→draftrepeat，88main/16dummy，不是request/bank/criticalpath证明。'
        '新证据utils全量copycontextslots而querylinearpos=seq_len-r+q，normalconsistentinjectiveperreqmap重叠min(r,q)，DCPinverse与25344map/3240independentinterval/6000inversecases。实际slots/r/q未录；不要把位置或shape当地址。'
        'Wc→Wq冲突边和RAW分别保留，finalpersistentstate也必须正确；no-localQ仍可写globalQKV。私有contextKV被保证完整覆盖且无中间/未来读者才可删；F27main需求是所有层与otherbankreader的union，不能只看某层。拆metadata/IDready与bankcopy后可提前需求，但不能搬整个mutatingmethod；所有round/errors/COW/epochs/protocol/newcost保留。'
        'REL113-01F27/F28新增1，HG85原有组细化，133/25/462；旧v34/v35lazyperlayer/sameWrowpack是prioranswer。CPUcheck初用literal未解析cachealias失败，已AST精确校正，不是nativebug。'
        '下一步继续有证据的未分析算子关系/条件分支，避免重复已答的rowpack/频率/operandAPI问题；若需要进一步裁剪，先查实际全层D_main需求和bank生命周期证据，不把条件结果当当前负载已成立。'
        '新Pad44BBOOL20→24 padTrue的v111async仍待明确reply，禁止run；v109最小probe授权已完成消耗。若新获准在v114记录，按v111plan空闲/容器/SHAguards一capture,noReplay/noModel/noopaqueGPUderef，不修改已完成v113。'
        '35/353/26、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU仍不变；TaskACTIVE Loop004EVALUATING GoalACTIVE，不重跑完成finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=113,previous=112,title='DSpark rejected-slot cache joins and upstream row demand',
        summary=dict(filename='draft-cache-join-summary-v113.json',previous_turn_classification='PROGRESS:v112 exactclockfield/A3semantics and14operatorcertificates;postcheck PASS191',
            saved_source_order_frontiers=104,existing_context_cycles_rechecked=312,preceding_main_chains=88,dummy_frontiers=16,
            CPU_source_map_cases=25344,independent_interval_cases=3240,independent_compact_inverse_cases=6000,
            refined_candidates=4,refined_modules=2,branch_records=353,semantic_modules=26,new_pairs=1,new_groups=0,
            relation_clauses=133,higher_order_groups=25,pending_central_pair_reconciliation=462,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            new_device_work=False,implementation=False,native_test=False,targeted_Pad_probe='PREPARED_NOT_EXECUTED;PENDING_ASYNC_REPLY',
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            remote_job_status='No remote process ornewNPU/model/capture/replay. CPUalgebra/source/savedprofile jobs finished;Padprobe remains unexecuted.'),
        changes=['Recheck104typed source-order context/queryfrontiers and312priorcycles;derive rejection-inducedmin(r,q) slotconflict in consistentinjectivemappingdomain,notactualslotattestation.',
            'Fix per-byteWAW/RAW andfinalpersistentstate conditions;derive all-layer/other-reader union before upstreamcontextmainprojection pruning.',
            'Add REL113-01F27/F28backwards cache-demand relation,refineHG85andF29/F31;oldv35rowpack/lazycontext variants reused,notnewdiscoveries.'],
        candidate_updates=updates,module_updates=modules,remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Crosscheck32savedTASKprojections,104context→repeatfrontiers/312cycles/88mainchains;noactualslots/newattribution.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Exactslotcopy/querymapping,25344CPUmapcases,3240independentintervals/6000inversecases,WAW/read/finalstate andbackwards-demand/costcontracts.',artifacts=[O/'draft-cache-join-preflight-v113.json',O/'draft-cache-join-validation-history-v113.json',O/'whole-scope-requirement-audit-v113.json',O/'coverage-current-obligations-v113.json'])],
        report=report,read_first=names+['coverage-current-obligations-v113.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_draft_cache_join_v113.py',N/'verify_draft_cache_join_v113.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='draft-cache-join',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
