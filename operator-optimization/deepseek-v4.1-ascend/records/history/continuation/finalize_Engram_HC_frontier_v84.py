"""Versioned operator-only continuation; preserve unresolved full-scope evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v84.json').exists()
    names=['Engram-HC-operator-source-contract-v84.json','Engram-HC-native-frontiers-v84.json',
        'Engram-WKV-hoist-and-memo-variants-v84.json','Engram-WKV-storage-and-work-models-v84.json',
        'Engram-HC-rounding-witnesses-v84.json','Engram-HC-joint-frontier-models-v84.json',
        'torch-split-primary-evidence-v84.json','HC-Engram-composition-clauses-v84.json',
        'Engram-HC-transfer-credit-v84.json']
    profile=read(O/names[1]);assert(profile['graph_frontier_count'],profile['root_bound_frontiers'],profile['phase_unbound_frontiers'])==(160,128,32)
    credits=[]
    for rows in[2,3,800,816,830,834]:
        extent=rows*4*5120
        credits.append(dict(M=rows,E_elements=extent,
            original_gate_temp_write_bytes=2*extent,original_copy_temp_read_bytes=2*extent,
            conditional_direct_store_unique_credit_bytes=4*extent,
            updated_hidden_destination_write_retained_bytes=2*extent,
            original_HC_collapse_updated_hidden_read_bytes=2*extent,
            HC_Y_write_and_norm_read_private_bytes=4*rows*5120,
            direct_store_plus_collapse_norm_distinct_transfer_union_bytes=6*extent+4*rows*5120,
            HC_coefficient_FLOAT_store_read_bridge_retained_bytes=8*extent,
            coefficient_BF16_source_reread_not_credited_without_new_ABI_bytes=2*extent,
            duplicate_previous_ENG_FINAL_STORE_credit_must_not_be_added=True,
            actual_MEMCPY_address_length_or_gain=None))
    write(O/names[-1],dict(models=credits,
        domain='One legal joint graph with exact old-hidden-read retirement, original BF16 gate+Y rounding, direct gate output into hidden[:n], contained collapse/norm readers and original current coefficient output path. Different rows/generations/partitions cannot share these credit IDs.',
        credit_identity_rule='Directstore saves original private gate-temp WRITE and its copy READ; final updated-hidden destination WRITE remains. PriorENG-FINAL-STORE is the same pair, not an additional4ThH credit. HCcollapse updated-hidden READ andY WRITE/Norm READ are separate conditional credits.',
        no_deleted_updated_residual_or_FLOAT_CV_bridge_claim=True,actual_HBM_or_critical_path_gain=None,overall_complete=False))
    variants=read(O/names[2]);memo=next(v for v in variants['variants']if v['id']=='F25-WKV-MEMO')
    memo['conditions'].append('Exact reuse requires deterministic or otherwise proved equivalent original BF16 WKV output, not just identical row bytes. Native determinism/shape-dependent algorithm equivalence is not newly attested.')
    write(O/names[2],variants)
    additions=read(O/names[7]);relations=deepcopy(read(O/'candidate-relation-clauses-v83.json'))
    relations['clauses']+=deepcopy(additions['clauses']);relations['higher_order_groups']+=deepcopy(additions['higher_order_groups'])
    old=next(c for c in relations['clauses']if c['id']=='REL73-20')
    old['pure_WKV_refinement_v84']=dict(contract=additions['refined_existing_F24_F25_relation'],evidence=names,
        does_not_reuse_hidden_gate_or_HC_coefficients=True,no_actual_hits_or_gain=True)
    relations.update(clause_count=84,selected_pair_count=84,no_new_maincandidate=True,overall_complete=False)
    write(O/'candidate-relation-clauses-v84.json',relations)
    coverage=deepcopy(read(O/'candidate-relation-coverage-v83.json'))
    for c in additions['clauses']:
        pair=next(p for p in coverage['pairs']if {p['a'],p['b']}=={c['a'],c['b']});assert not pair['clauses']
        pair.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    coverage.update(scoped_conditional_pairs=84,remaining_pair_relations=511,higher_order_groups=16,
        parent_relation_index='candidate-relation-clauses-v83.json',parent_coverage='candidate-relation-coverage-v83.json',
        HC_Engram_refinement_v84=dict(new_pair_clauses=15,new_higher_order_groups=2,exact_graph_frontiers=160,evidence=names))
    write(O/'candidate-relation-coverage-v84.json',coverage)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v83.json'))
    registry={r['id']:r for r in read(O/'candidate-record-index-v83.json')['model_candidates']}
    for row in footprints['candidates']:
        path=R/'candidates'/(row['id']+'.json');blob=path.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==registry[row['id']]['master_sha256']
        row['captured_master_sha256']=registry[row['id']]['master_sha256']
        if row['id']=='F25':
            row['operator_atom_union']+=['engram.WKV','engram.key_rstd','state.WKV_output_memo']
            row['variant_scope_v84']='TargetEngram1/14;WKVpure-rows hoist/memo separatefromhidden-dependentgate;ordinarykey-rstdhandoffonly,notunrotatednativeABI.'
        if row['id']=='F31':row['operator_atom_union']+=['metadata.WKV_generation']
    footprints.update(parent='candidate-operator-footprints-v83.json',source='Exactv83masterhashescapturedbeforev84mutation;F25pureWKV/key-rstd/memoandF31generationnavigationexpanded;unionnotasimultaneousgraph.',overall_complete=False)
    write(O/'candidate-operator-footprints-v84.json',footprints)
    navigation=deepcopy(read(O/'candidate-pair-navigation-v83.json'));byid={r['id']:r for r in footprints['candidates']}
    for pair in navigation['pairs']:
        pair['conditional_clause_ids']=[c['id']for c in relations['clauses']if {c['a'],c['b']}=={pair['a'],pair['b']}]
        pair['semantic_pair_reconciliation_done_in_stated_scope']=bool(pair['conditional_clause_ids'])
        pair['shared_operator_atom_union']=sorted(set(byid[pair['a']]['operator_atom_union'])&set(byid[pair['b']]['operator_atom_union']))
    navigation.update(scoped_conditional_pairs=84,remaining_central_pair_reconciliation=511,parent='candidate-pair-navigation-v83.json',overall_complete=False)
    write(O/'candidate-pair-navigation-v84.json',navigation)
    operators=deepcopy(read(O/'operator-analysis-coverage-ledger-v83.json'))
    operators['HC_Engram_refinement_v84']=dict(existing_WKV_regimes=5,existing_WKV_executions=208,exact_graph_frontiers=160,
        root_bound_frontiers=128,phase_unbound_frontiers=32,newly_promoted_regimes=0,newly_promoted_executions=0,evidence=names)
    write(O/'operator-analysis-coverage-ledger-v84.json',operators)
    matrix=read(O/'module-phase-branch-matrix-v83.json')
    selected={
      'M06':{f'M06-OP-{i:02d}'for i in range(6,13)},
      'M12':{'M12-OP-06','M12-OP-13','M12-OP-14','M12-OP-15'},
      'M13':{'M13-OP-01','M13-OP-06','M13-OP-10','M13-OP-12','M13-OP-14'},
      'M14':{'M14-OP-12','M14-OP-13','M14-OP-14','M14-OP-15','M14-OP-16','M14-OP-17','M14-OP-18','M14-OP-19'},
      'M22':{'M22-S12','M22-S13','M22-S14'},
      'M23':{'M23-OP-14','M23-OP-16','M23-OP-18'},
      'M24':{'M24-OP-07','M24-OP-09'}}
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selected}
    for mid,rows in branches.items():
        for row in rows:
            if row['id']in selected[mid]:row['evidence']+=names;row['HC_Engram_review_v84']='Lookup-onlyWKVandordinarykey-rstdcanreorderbeforehidden;currentgate/HcPrestilluseupdatedBF16hidden. Certifiedmemoisdistinctfromhoist/directstore. CostcachedrawKV/backing/keys/misses/extraevents/physicalresourcecompetition;allsource/nativeprecision/fanout/generationgatesremainconditional.'
    gap='Conditional operator question analyzed;actual lookup-ready time/hoist window,contenthit rate/cachebudget,native determinism/alias/address/length/newABI,actualHF32/Vector/GM-L2-HBM/networkmixedcapacityandfullbranch/compositionclosure remain unproved. No implementation/benchmark required for the bounded conditional analysis.'
    for mid,identifier,branch,finding in[
        ('M14','M14-OP-20','Lookup-ready pureWKV/key-rstd hoist and content-generation WKV output memo',
         'WKV(rows) is independent of currenthidden;ordinarykey-rstdscalaralsopure. Hoist extendscompleteKVliveness;memo removesWKVonlyforcertifiedhits,notcurrentgate/HCcoeff. Preserve originalFP32 association/BF16round and native algorithm policy.'),
        ('M24','M24-OP-12','Immutable pureWKV output cache keyed by row/table/weight/numerical generation',
         'Unlikehiddenactivationreuse,WKVcanreuseidentical6144BF16rowcontentunderfixedlayer/weight/numerics. Entry51200B+96BexplicitIDkey,optional16Brstd. Partialmisseskeepfullweightplane;packing/assembly/cacheinsert/lookupprotocol andallreader/writerretirementpriced.')]:
        assert not any(r['id']==identifier for r in branches[mid])
        branches[mid].append(dict(id=identifier,branch=branch,state='SCOPED_OPERATOR_DEPENDENCY_CACHE_AND_COST_ANALYZED_CONDITIONAL;WHOLE_MODULE_OPEN',
            evidence=names,finding=finding,gap=gap,scope='FixedtargetEngramordinaryBF16WKV/currentrotatedgateandspecifiedconditionalhoist/memo/storage domains;notframeworkimplementation.'))
    write(O/'HC-Engram-branch-inventory-v84.json',dict(modules=branches,new_families=['M14-OP-20','M24-OP-12'],
        retained_prior_family_IDs=True,global_records=344,semantic_modules=24,refined_modules=7,untouched_other_module_matrices=19,
        whole_modules_closed=0,overall_complete=False))
    members=sorted({c[k]for c in additions['clauses']for k in['a','b']})
    updates={cid:dict(review_v84=dict(clause_ids=[c['id']for c in additions['clauses']if cid in[c['a'],c['b']]],
        source_contract=names[0],native_frontiers=names[1],pure_WKV_variants=names[2],work_storage_models=names[3],joint_credit=names[-1],
        conditional_operator_analysis_deliverable=True,implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    updates['F25'].update(profile_review_state='PURE_WKV_KEY_RSTD_HOIST_MEMO_AND_UPDATED_HIDDEN_HC_JOIN_REVIEWED_CONDITIONAL',
        next_profile_or_contract_action='BoundedWKVpurelookup/hiddenjoinquestiondone:5variants,5observedregimes208exec,160graphfrontiers128root32phaseunbound. Noactualhit/lookupreadywindow/aliasorβmeasured. PreservegateBF16/hcYBF16andrawdotassociation. CurrentMEMCPYconsistentwithsourcecopybutnotpointerbound;directstoreoldcopybudgetcountonce. Reconcileotherremainingglobaloperatorsusingcurrentledger;do not demandimplementationtocloseconditionalanalysis.')
    updates['F24']['next_profile_or_contract_action']='Row/headpartition+lookupready nowconnecttopureWKVhoist/memo underREL73-20refinement;onehash/DPproducerforallEngramalreadybaseline. Memo plan keeps currentUVA/collectives unlessseparateallreader/allrankproof;cachegenerationmustcertifyactual6144BF16rows,notrawIDoraddressalone.'
    mods={mid:dict(finding='PureWKV/ordinarykey-rstd dependency split,160graph gate-copy-HcPrefrontiers,updatedhiddenBF16counterexampleandFP32association,hoist/memo/directstorecostand15HC/FFN/Engramrelationsreviewed. Two newoperator-statefamiliesregisterthepureoutputcachequestion;not anewframeworkscope.',
        evidence=names,candidates=[cid for cid in members if any(cid in m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
        profile_scope='Reusedsavedprofilesand3pinnedoperatorcallerfiles;no model/device/frameworkexecution.')for mid in selected}
    write(O/'PROFILE-CODE-QUESTION-GATE-v84.json',dict(status='PROFILE_BOUND_ENGRAM_PURE_PROJECTION_AND_HIDDEN_HC_JOIN',
        previous_turn_classification='PROGRESS:v83operatorcompositionandjointcapacityproof',
        questions=['DoesWKVdependoncurrenthidden,oronlylookuprowsandweights?','Whichkey-onlyscalarcanhoistwithoutFP32reassociation?',
            'WhatBF16andcopy/residual/coefffrontiersmustsurviveEngram->HCfusion?','Howdohoist,memoanddirectstorechangedependencies,storageandnecessaryworkdifferently?'],
        reused_local_operator_files=3,new_target_source_files_obtained=0,new_remote_source_read=False,
        one_public_API_page='PyTorch2.8splitviewcontractonlyforoperatorKVbackinglifetime,notserverversionattestation.',
        source_read_purpose='Namedoperatorfusion/dependency/precision/upper-boundquestionsonly;no frameworkcatalogaudit.',
        new_model_Torch_NPU_capture_or_server_work=False,overall_complete=False))
    write(O/'Engram-HC-validation-history-v84.json',dict(errors=[
        dict(kind='LOCAL_STALE_FILENAME_READ',error_type='FileNotFoundError',path='HC-work-traffic-and-reference-bounds-v59.json',
            resolved_by='rgfoundactualHC-work-traffic-and-reference-bounds-v58.json;v59holdsclarification/numericalcontract,notduplicateworkmodel.',resolved=True,no_state_mutation=True),
        dict(kind='LOCAL_DISCOVERY_SNIPPET_SYNTAX',error_type='SyntaxError',description='Unused discoverycomprehensioncontainedandnot;parserstoppedbeforeexecution. Correctedboundedfileinventorycompleted.',resolved=True,no_state_mutation=True)],
        pre_checkpoint_analysis='PASS',new_device_work=False,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v83.json')['remaining'])
    remaining.append('v84pureWKV/key-rstdfrontier:3localpinnedoperatorfiles+160exactgraphtriples128root32phaseunboundand5WKVregimes208exec. WKVrows-onlycanhoistbeforehidden;key-rstdscalarcanhoistordinarybranchbutpreweighted/normalizedkeyreassociatesFP32. Memoidentical6144rowpossiblewithoutsamehiddenbutneedsdeterminism/numericgeneration,table/ID/weights/ownership/full-keyproofandbankretirement;hitrate/lookupreadywindow/budgetunknown. Entry51200B+96Bkey,rawM3result153600B;partialmisskeeps314572800Bweightplane. PreservegateBF16beforeHCY(5vs4witness),updatedresidualandFLOATCVbridge;160MEMCPYnotpointerbound,oldENGFINALSTOREsamecredit.15newclauses=>84pairs16groups511centralpending;M14OP20/M24OP12add2records=>344/24semantic. Profile2073/285598review9/1352pending/global84916/3136/0wholeclosures unchanged.')
    action=('继续原Goal，v84实质PROGRESS(前v83PROGRESS)。只围绕算子融合/理论上限：profile先，代码定点。'
        '先读Engram-HC-summary-v84、PROFILE-FINDINGS-v84和Engram-HC-operator-source-contract/native-frontiers/Engram-WKV-hoist-and-memo-variants/storage-and-work-models/Engram-HC-rounding-witnesses/joint-frontier-models/transfer-credit-v84。'
        '160graphFrontiers128root32phaseunbound，layers1/14各80；priorFFNHcPost→38taskEngrambody→CastBF16→MEMCPY_ASYNC+AI_COREcarrier→HcPre→RMS。形状相等和source-slicecopy不是native地址/长度/alias证明，48DIRECTWKV不强拼。5observedWKVregimes208execM3/800/816/834。'
        '3frozenoperatorfiles：model101f96...,layerdc9457...,common4797b8...。WKV(rows)独立hidden，commonkeyrsqrt(mean(key.float^2)+eps)独立hidden；原dot=sum((rotH*channelweight)*key)*[rstdH*rstdK]*H^-0.5，不能preweight/normalizekey重排。unrotatednativefusedgate无证明兼容rstdhandoff。当前target1/14有Engram，draft构造排除。'
        '新增5分域：lookupreadyWKVhoist、key-rstdhoist、WKVmemo、gate→HCcollapseNorm、directhiddenstore。前两只是reorder不删work；memo只有content+layer/table/weight/native算法determinism同等性认证hit才删WKV，不复用hidden/gate/HCcoeff，不自动删UVA/DP/TP原生产。GateBF16→HCYBF16保序，标量5vs4反例；preweightFP32溢出Infvsfinite反例。'
        'WKV每row输出51200B，24IDkeys96B，M3一bank153600B/M80040.96MB；普通keyrstd4FP32额外16B/row。key,value=kv.split是view域：保FLOATkey+valueview仍可能保完整BF16KV，133136Mlive；独立value拷贝后92176Mlive但额外20480Mcopy，非actualallocation/UB承诺。PyTorch2.8官方split页1份快照只支持view契约不是serverTorch版本。'
        'Memo referenceMrows/Udistinctmiss：Base=W+aM+bM；选定missbuf+assemble+cacheplan=1[U>0]W+aU+2bU+2bM+96M，W314572800,a12288,b51200。partialhit不按比例省完整weightplane；U0可取消该plane但添cache/load/assembly/key成本。actualhitrate/readywindow/freebudget/BW/β未知；L1Bwarmfloor域保GMdirect/otheralgorithm例外，别造HBM/MFU/提速。'
        'Directstore原private gateTempWrite+copyRead合计4ThH省额，destWrite仍需；与旧ENGFINALSTORE相同不能重复。collapseRead2ThH、Ywrite/read4TH另域可union；updatedHCresidual/coeffFLOATGM8ThH保留。现currentHcPre全completionSyncAll，earlyY需新handoff，可行schedule不是理论不可超越上界。'
        '15新HC/FFN/Engram关系+2groups、旧F24F25REL73-20更新：总84pairs16groups511central待汇入，35footprints595navigation。M14OP20+M24OP12新条件家族使344records24semantic，7modules其余19矩阵保留。'
        '下一批优先target/draft head/context/accept/reject与metadata的真实数据依赖和共享代价，利用v84pairnavigation汇入F26/27/28/29/30/31/35旧合同；同时审阅哪些固定case分支只是历史框架导航、哪些仍是算子结论真缺口，不扩大研究框架。补影响结论的typedcapacity/reader/generation证据，不为条件分析强要实施。'
        '2073/285598conditionalreview9/1352pending、849PMU16phase3136ambiguity和0wholeclosure保护；未全覆盖不complete，Goalactive。两个localread/discover错误均已记录并修复，不影响target/证据。所有localwork已结束，无server/model/Torch/NPU/newcapture/service/processwork。verify_Engram_HC_frontier_v84应PASS。')
    report='''本轮围绕Engram投影、门控与mHC交界处补齐了新的跨层及跨迭代机会，没有执行模型或设备实验。

最具体的新发现是：WKV投影只依赖lookup行和该层权重，不依赖当前隐藏状态。因此，在lookup及TP/SP行映射已经就绪的条件下，可以分析提前计算WKV；普通门控路径中的key倒均方根也能独立提前计算。当前隐藏状态、门控以及其后的HcPre系数仍需逐层产生。提前安排工作本身不会删除计算，也没有已测得的可重叠时间收益。

保存的profile核对了160组Engram门控到HcPre的图边界：128组有请求阶段归属，32组仍阶段未知。每组BF16门控转换后有一条MEMCPY_ASYNC及一条运行载体记录，再到HcPre。源码确有写回hidden[:n]的切片赋值，但profile没有地址和长度，所以不能直接把这些复制认定为可删除。另48次DIRECT WKV未强行拼入该图边界。

WKV输出还可以单独分析内容缓存：即使当前隐藏状态改变，只要lookup行、层权重及数值算法的输出同等性得到认证，投影结果仍可能复用；门控和HC系数继续重新计算。每个token、每层完整BF16结果为51,200B，24个INT32索引组成的显式键另需96B。M=3时一个结果bank为153,600B，M=800时为40.96MB。当前没有实际命中率、可用缓存预算或lookup就绪时间证据。

部分命中不能直接按比例折算访存收益：只要仍有一个未命中行，常规完整权重投影仍需314,572,800B的权重值面；缓存还增加结果读取、索引查询、miss组装及写入开销。全部命中、部分命中和提前计算已分别建模，逻辑GM字节未当作HBM流量或提速。

key/value底层存储也计入了模型。[PyTorch 2.8的split契约](https://docs.pytorch.org/docs/2.8/generated/torch.split.html)说明分块是原张量的视图。在这一存储域中，保留value视图可能继续保留整个BF16 KV底层数组；提前保存FLOAT key并不自动释放它。若将value复制到独立存储，需要另计复制及所有旧读者退役条件。这不是服务器实际地址或分配快照证明。

数值上保留两条边界：门控BF16舍入之后才能进入HC collapse；有限FP32/BF16-RNE模型给出保留舍入结果5、取消舍入结果4的反例。原点积先算(rotated_hidden×channel_weight)再乘key，预先归一化或加权key会改变FP32结合顺序，也有反例。安全候选仅是原顺序下独立的key倒均方根标量提前，不外推到未暴露兼容接口的原生非旋转门控。

本轮新增15条条件关系及2组多候选合同，累计84条两两关系、16组多候选关系，中央仍有511个候选对待汇入或复核。新增2个算子状态家族记录WKV提前计算及纯投影结果缓存，当前344条分支、24个语义模块；原有profile缺口和整体未完成状态保留。
'''
    preflight=O/'Engram-HC-analysis-preflight-v84.json';assert read(preflight)['status']=='PASS'
    sourcepaths=[Path(x['path'])for x in read(O/names[0])['source_files']]
    jobs=[dict(kind='profile',status='pass',description='160exactgraphEngram→HcPrefrontiers128root32phaseunbound;208WKVtypedcalls/5regimes.3reusedpinnedoperatorfilesresolvepureWKV/key-rstd/currentupdatedhidden dependencies;no source/framework/server/deviceexecution.',artifacts=[O/names[0],O/names[1]]+sourcepaths),
        dict(kind='design-check',status='pass',description='Fivehoist/memo/directstore/fusiondomains,6shapemodels11misscases,storageviewlivenessand2numericalcounterexamples;onepublicsplitAPIreferencewithsha. Actualhit/window/capacity/alias/gainunproved.',artifacts=[O/x for x in names[2:7]]+[O/names[-1],preflight]),
        dict(kind='review',status='pass',description='15newconditionalpairclauses+2groups,existingF24/F25refinement;84pairs16groups511pending.2newsource-operatorstatefamilies344records24semantic;regime/globaluncertaintiespreserved.',artifacts=[O/names[7],O/'HC-Engram-branch-inventory-v84.json',O/'candidate-pair-navigation-v84.json']),
        dict(kind='review',status='error',description='ResolvedlocalreadFileNotFoundError:workmodelactualv58notv59;rgdiscoveryandcorrectreadcompletedwithoutstatemutation.',artifacts=[O/'Engram-HC-validation-history-v84.json']),
        dict(kind='review',status='error',description='ResolvedlocaldiscoverySnippetSyntaxErrorbeforeexecution;correctedboundedlocalinventorycompleted,notarget/evidencemutation.',artifacts=[O/'Engram-HC-validation-history-v84.json'])]
    persist_refinement(version=84,previous=83,title='Engram pure WKV hoist/memo and updated hidden to HC frontier',
        summary=dict(filename='Engram-HC-summary-v84.json',previous_turn_classification='PROGRESS:v83operatorcompositionandsharedtypedcapacitysupportproof',
            reused_local_operator_files=3,new_target_source_files=0,public_API_pages=1,exact_graph_frontiers=160,root_bound_frontiers=128,phase_unbound_frontiers=32,
            existing_WKV_regimes=5,existing_WKV_executions=208,conditional_variants=5,storage_shape_models=6,memo_scenarios=11,numerical_counterexamples=2,
            new_conditional_pair_clauses=15,relation_clauses=84,higher_order_groups=16,pending_central_pair_reconciliation=511,
            model_candidates=35,operator_footprints=35,pair_navigation_rows=595,refined_candidates=len(members),refined_modules=7,
            new_semantic_families=2,branch_records=344,semantic_modules=24,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_remote_source_read=False,implementation=False,native_test=False,
            actual_hit_rate_or_hoist_gain_proved=False,actual_lowest_MFU_module_established=False,retained_resolved_local_inspection_errors=2,
            remote_job_status='Alllocalsource/profile/mathworkfinished;no server/model/Torch/NPU/capture/service/processwork.'),
        changes=['WKVrows-onlyandordinarykey-rstdpuredependencyprovedinthreepinpointfrozencallers;160typedsource/graphfrontierspreservephase/aliasuncertainty.',
            '5hoist/memo/gateHC/directstoredomainsplusrawKV/viewlifetime/cachemisscosts/2numericcounterexamples;no unmeasuredgainorphysicalcapacityclaim.',
            '15newclauses2groups;84pairs16groups511pending;M14OP20/M24OP12newfamilies344records24semantic;allregime/globaluncertaintiesretained.'],
        candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names+['candidate-pair-navigation-v84.json','candidate-relation-clauses-v84.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_Engram_HC_frontier_v84.py',N/'model_Engram_WKV_frontier_v84.py',N/'reconcile_HC_Engram_relations_v84.py',N/'verify_Engram_HC_frontier_v84.py'],
        executor_prefix='Engram-HC',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
