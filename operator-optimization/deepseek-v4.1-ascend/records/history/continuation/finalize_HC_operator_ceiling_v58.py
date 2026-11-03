"""Persist operator-only HC fusion/split and conditional theoretical-bound analysis."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v58.json').exists()
    p=read(O/'HC-operator-profile-and-edges-v58.json');b=read(O/'HC-work-traffic-and-reference-bounds-v58.json')
    v=read(O/'HC-fusion-split-reorganization-variants-v58.json');c=read(O/'HC-operator-numerical-and-dependency-contract-v58.json')
    inputs=read(O/'HC-operator-inputs-v58.json');ref=read(O/'operator-regime-refinement-v58.json')
    assert p['regime_count']==16 and p['executions']=={'HcPre':8944,'HcPost':8944}
    assert len(v['variants'])==7 and ref['refined_execution_count']==17888
    m3=next(r for r in b['models'] if r['M']==3);m800=next(r for r in b['models'] if r['M']==800)
    assert m3['pre_to_norm_removable_BF16_Y_write_read_bytes']==61440
    assert m3['source_conditioned_tiling']['source_cast_FP32_workspace_write_read_bytes']==491520
    assert m3['source_conditioned_tiling']['Gram16x16_requested_ops']==10485760
    assert m3['pre_semantic_sumsq_conventional_ops']==122880
    assert m800['post_minimum_descriptor_value_bytes']==73792000
    e=['HC-operator-profile-and-edges-v58.json','HC-work-traffic-and-reference-bounds-v58.json',
       'HC-operator-numerical-and-dependency-contract-v58.json','HC-fusion-split-reorganization-variants-v58.json',
       'operator-regime-refinement-v58.json','HC-hardware-reference-scope-v58.json','HC-operator-inputs-v58.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v58.json',dict(status='HC_OPERATOR_WORK_NUMERICS_DELAYED_COLLAPSE_FUSION_SPLIT_AND_THEORETICAL_BOUNDS',
      previous_turn_classification='PROGRESS:v57shared_expert_operator_ceilings_and_singleton_merge',
      profile_evidence=['operator-shape-ceiling-ledger-v55.json',e[0]],
      questions=['Does currentY usecurrentpre or externalpre_mix, andwhichcoefficientoutputs remainlive?',
        'WhatnativeHF32/Gram/FP32/BF16 work,workspace androunding constrainfusion orinternalretile?',
        'WhichPost→Pre/Pre→Norm edges exist inallcapturedgraphs,andwhichcrosslayeredgescontainotheroperatorwork?'],
      allowed_source='8 exactHCoperatorwrapper/kernel/tilingfiles pluspinnedmodeldataflow/config;4followeddirectA3kernelincludes. Noindependentframework/lifecycle/loader/APIcatalogresearch.',
      opaque_profile_attr_positions_not_used_as_semantic_schema=True,
      no_model_Torch_NPU_newcapture_service_health_process_work=True,whole_module_closed=False,overall_complete=False))
    updates={}
    for cid in ['F02','F03','F23']:
      updates[cid]=dict(review_v58=dict(profile_edges=str(O/e[0]),work_traffic_bounds=str(O/e[1]),numerical_contract=str(O/e[2]),conditional_variants=str(O/e[3]),
        relation='HC16regimes/8944eachPrePost,3200graphlayers reconciled. Externalpre_mix makescurrentY independentofcurrentcoefficientprojection/Sinkhorn;7conditionalvariants includeY/coeffsplit+norm,Post→collapse/norm,FP32castworkspace andGramdiagonal workreduction. RetainBF16barriers/residual/newcoeffoutputs/otherconsumers.',
        implementation=False,native_test=False,new_performance_result=False,conditional_operator_analysis_deliverable=True),
        profile_review_state='HC_OPERATOR_FORMULA_WORK_IO_AND_CONDITIONAL_FUSION_SPLIT_REVIEWED',
        conclusion_changing_gap='FullnumericphysicalHF32/A2/Vectorrates/cachetiers/internalphasecost andnewkernelallconsumer/readiness/precisionconditions remainunbound. Capturedregime conditionalanalysis isdelivered;do nottreatnewimplementation/nativebenchmarkabsence asunfinishedconditionalproposal.',
        next_profile_or_contract_action='AdvanceSparseFlashMla/Indexer/QKV/attention-output operatorwork/traffic/typedceilings andstillunreviewedregimes;no furtherframework orHCAPIcatalogaudit. ReopenHCsource onlyforanamedunansweredoperatorchoice.')
    mods={'M06':dict(finding='16capturedHCregimes and8944eachPrePost modeled;all80graphs/3200layers exactroleedges. Externalpre_mix delayedcollapse canbesplitfromcurrentcoefficients;HF32projection/16x16Gram/split-Kworkspace/39normalizationrounds/FP32Post+BFloat16round identified. 7conditionaloperatorvariants assessed withnonadditiveinterface/readiness limits.',evidence=e,candidates=['F02','F03','F23'],gap='Conditionalcapturedoperatoranalysis delivered. NumericphysicalHF32/A2/Vectorcapacity/cachetiers/phasecost,actualnewkernelallconsumer/generation/precision andotherunobservedoperatorbranches stillnotclosed.'),
      'M13':dict(finding='ExactoriginalHCPost confirmsseparateFP32Mul/Addi-order followedRINTBF16. F23Merge→Post mustpreserveDownBF16 thenMergeBF16 rounds beforeHCwiden;ONEADD/DownAdd/MergePost shareoneAddstage andmustnotbecountedindependently.',evidence=[e[2],e[3],'F23-singleton-merge-specialization-v57.json'],candidates=['F23'],gap='No newimplementation/performanceclaim. Allproducer/readiness/otherconsumer conditions andremainingmodeloperatorceilings retained.')}
    remaining=deepcopy(read(O/'module-coverage-gaps-v57.json')['remaining'])
    remaining.append('M06v58operatorreview:16regimes/HcPre8944/HcPost8944 acrossM2/3/800/816/834;80graph/3200layers exactedges,16phase-unboundgraphs retained. Allobservedexternalpre_mix meanscurrentY independentofcurrentcoeff;splitcollapse+norm/currentcoeff isconditionalpriority. SourceHF32AB,16x16A2Gramdiag,FP32castGMworkspace and39Sinkhornnormalizations known;7conditionalvariants reviewed. NumericHF32/A2/Vectorcapacity/cachetier/internalphasetimes notbound,so coldHBMvaluefloors are notspeedup/MFU. Sourceepsilonpacked/alignedrowcondition recordedwithoutactivebinarydefectclaim. Crosslayer400/3120edges containinterveningcompute-by-start andzero-count isnotidle. Nextadvanceattention/indexer andotherunreviewedoperatorceilings;avoidHC/frameworkauditchurn.')
    action=(
      '继续原operator Goal，最新v58。用户目标是算子能否融合/拆分/重组及理论上限；profile优先，只有决定具体算子结论的问题才定点查代码。不要独立研究framework、HTTP/SSE、请求生命周期、loader或SDK目录。'
      '先读HC-operator-profile-and-edges-v58、HC-work-traffic-and-reference-bounds-v58、HC-operator-numerical-and-dependency-contract-v58、HC-fusion-split-reorganization-variants-v58及operator-regime-refinement-v58。16HCregimes、HcPre/HcPost各8944、5个M(2/3/800/816/834)已做本轮条件算子分析，不能称全模块闭环。'
      '80图3200层，root2560/unbound640；PreAttention→RMSNorm、PreFFN→RmsNormCast、intralayerPost→Pre、Merge→Post各3200，crosslayerPost→Pre3120。跨层400条有其它Compute按start落在区间，不能擦掉Engram/aux/gather；无interveningcount也不等于idle/张量identity。root阶段medianus attnPre35.401/norm3.96/post7.46，FFNPre34.94/normcast3.16/post7.42；64rootgraph HC曝光份额median7.02036%不是严格全请求收益上限。'
      '原166stage只读docker cp取得8个HCwrapper/kernel/tiling文件：wrapper与历史snapshot哈希匹配，native loadedobject仍不假定绑定。HcPre已融合投影、平方和/RMS、pre/post/comb、39轮Sinkhorn归一化及Ycollapse；内部仍有FP32castGM写读、splitK投影/16列Gram输出、stage2重读BF16。所有16profile signature有externalpre_mix。当前Y用externalpre_mix而非本次newpre，本次newpre/post/comb仍供下一HC/本次HCpost使用；所以Ycollapse+norm可条件拆出并与currentcoeff支并行，保原BF16Yround，不能删除coeff工作。全kernel完成门控需新split/handoff，不是简单改launch顺序。'
      'HcPreAB显式SetHF32Mode(1)，不能套BF16/INT8peak或通用fullFP32dot。A2每16行算16x16Gram取diag；source24AIC条件M3ABrequested15.72864Mops，Gram10.48576Mops，而必要projection2.94912M、平方和conventional.12288M。源tile重构不是每executionnativeissue证明；向量替换Gram需保数值归约/资源，不是无条件bitwise优化。'
      'HcPost公式每j先x*post_j，随后按i顺序把res_i*comb_i,j单独Mul再Add进FP32，最后RINT BF16。F23保Down BF16→Merge BF16→HC FP32的两个round；不把未舍入accumulator直接加进去。Post输出仍供后续HCPost residual/aux/Engram/tail，Post→Pre融合不能宣称所有residual写读都删除。'
      'Qpre_min=4KC+2MK+2MH+4M(C+h)+4C+12，K=hH=20480,C24；Qpost_min=2MH(1+2h)+4M(h+h²)。M3pre2,120,124B/cold1.32508us，post276,720B/cold.17295us；M800pre43,015,788B/cold26.88487us，post73,792,000B/cold46.12us。仅1.6TB/s参考valueIO条件，不是实HBM/MFU/可实现gain。Pre→Norm可条件省4MH(M3 61,440B)；castworkspace8MK(M3 491,520B)。HF32/A2/Vector typedphysicalrate和specialfunction/reduction关键路径仍无可靠数值，不强行填BF16peak。'
      '7变体：YCOEFF-SPLIT、PRENORM、POST-COLLAPSE-NORM、CAST-WORKSPACE、GRAM-DIAGONAL、MERGE-POST、拒绝无条件ALGEBRAIC-PREFOLD。split与norm的4MH省字节只算一次；MergePost/ONEADD/DownAdd共享Add不得叠加。epsilon源码列归约stride8却长度curRows*4的条件已记录独立索引反例，host可增加rowFactor；未证明当前loadedbinary/rowtile触发，勿报现有模型故障或另起未经授权修复。'
      '下一步转向SparseFlashMla/Indexer/QKV/attention-output及其它未定量算子上限，沿83类型/2082regimes/35模型候选补覆盖，不继续重复已回答HC/shared/MC2局部问题。实际物理typedcapacity/缓存/跨rank关键路径/全部消费者及未覆盖算子分支仍开；849PMU缺、16unbound图、3136module歧义保持。M01/M21/M22/M25语义不改；35模型+7legacy框架候选、0whole modules closed。'
      '条件方案分析不需要先实现/benchmark；无模型/Torch/NPU新capture/health/process/kill/实现，Goalactive。优先本地完整savedprofile/cache。必要原166文件读取可用显式override ssh166-v56-xinyufeng.sock，账号授权已有不重复问。')
    report='''本轮围绕 HC 算子的融合、拆分、重组与理论上限，覆盖16个捕获形状/角色/上下文组合，HcPre、HcPost各8,944次执行，M为2、3、800、816、834。代码仅定点读取8份HC算子wrapper、kernel及tiling文件，未研究框架生命周期，未运行模型或新增设备测试。

最值得进一步验证的方案是把“旧pre_mix驱动的collapse＋归一化”与“当前系数投影＋Sinkhorn”拆开。所有捕获HcPre都提供external pre_mix；原始源码确认当前Y使用它，而新pre是下一HC的输入。因此当前Y不必在数据依赖上等待本次投影及系数生成，但当前post/comb仍必须在本次HcPost前就绪，新pre也必须保留。拆分需要明确的算子边界与同步，并计算新增launch和Vector/访存资源竞争。

```mermaid
flowchart LR
 R[BF16残差R] --> Y[用旧pre_mix collapse\n保留BF16舍入]
 OLD[旧pre_mix] --> Y
 Y --> N[RMSNorm或RmsNormCast]
 N --> A[Attention或MLP]
 R --> C[当前投影与系数生成\npre / post / comb]
 A --> P[HcPost]
 R --> P
 C --> P
 C --> NEXT[新pre供下一HC]
```

| M=3、root绑定图层阶段 | 中位设备耗时 |
|---|---:|
| Attention HcPre / Norm / HcPost | 35.401 / 3.960 / 7.460 μs |
| FFN HcPre / NormCast / HcPost | 34.940 / 3.160 / 7.420 μs |

80个图、3,200层均有精确记录ID：两类Pre→Norm、层内Post→Pre、Merge→Post各3,200条；跨层Post→Pre有3,120条。其中400条跨层区间还包含其它Compute起点，涉及Engram/辅助状态及通信，不能统一跨层折叠。其余区间未记录中间Compute起点也不证明空闲、别名或融合合法。64个root图中，HC设备任务时段之和占图device control span的比例中位7.0204%，只是曝光份额，不能当成端到端收益上限。

HcPre已经融合了系数投影、平方和/RMS、pre/post/comb生成、Sinkhorn和collapse。内部仍先把BF16输入转为FP32，经GM工作区交给Cube，并存取split-K部分结果，collapse再读一次BF16输入。投影AB显式启用HF32；官方API说明该模式在Mmad前把FLOAT32输入舍入到HF32，因而不能拿BF16峰值替代其计算能力。[华为HF32接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/cube_compute/asc_enable_hf32.md)

平方和路径还有可分析的内部重组空间：原始源码执行16×16 Gram块，再取对角线。在24 AIC的源码条件重构中，M=3请求的投影工作量为15.72864M次操作，Gram为10.48576M次操作；对应必要dense投影为2.94912M，普通逐元素平方归约为0.12288M。请求矩阵尺寸并非完整硬件发射证明，改成向量归约也不自动保持原Cube归约精度。两者的资源成本和数值合同要分别评估。

HcPost先将BF16输入扩成FP32，计算x×post，再按残差通道i的原顺序分别乘comb并累加，最后RINT回BF16。h=4时，每token每hidden元素为20次乘法、16次加法。合并上游Add必须保留Down的BF16舍入、Merge的BF16舍入，再进入HC的FP32计算。Post输出还会作为后续Post的残差，并可能供Engram、辅助状态或尾部使用，不能因为与Pre融合就宣称整份残差存储消失。

| 逻辑值读写模型 | M=3 | M=800 |
|---|---:|---:|
| HcPre最小描述符值字节量 | 2,120,124B | 43,015,788B |
| 对应冷HBM参考访存下界 | 1.3251μs | 26.8849μs |
| HcPost最小描述符值字节量 | 276,720B | 73,792,000B |
| 对应冷HBM参考访存下界 | 0.1730μs | 46.1200μs |

上述数字使用此前注明的每die 1.6TB/s参考，假设这些值实际经过HBM；真实缓存层级、内部重复访问、填充、同步和计算依赖尚未计全，不能拿它们与实测耗时相除报加速或MFU。完整计算下界仍须分别使用HF32、A2 Gram、Vector及特殊函数的能力，并计入20次Sinkhorn迭代中的39轮依赖归一化。当前没有足够证据把这些能力填成可靠的本机数值。

共记录7个方案/决策：延迟collapse与系数分支拆分、Pre＋Norm、Post＋下一collapse/Norm、消除FP32 cast工作区、Gram只求必要对角线、Merge＋Post，以及拒绝无条件代数折叠。Pre→Norm条件成立时，M=3可减少61,440B的BF16中间写读；cast工作区接口为491,520B逻辑写读。这些都不是已测得的HBM节省或时间收益。前两个方案共用的61,440B只能计一次，Merge＋Post也不能再叠加singleton Add和Down＋Add的同一项收益。

改变每次处理行数还需检查原始HC源码中列归约缓冲的epsilon对齐：当前读取的源码具有stride8而部分Adds长度按rows×4计算的条件，已保存独立索引反例；未证明当前捕获的loaded binary及row tiling触发，未诊断现有模型故障或修改代码。

本轮条件分析已入台账；全范围仍未完成。下一步转向SparseFlashMla、Indexer及投影链的工作量、访存、融合边界和计算上限，继续补齐其它算子与未覆盖条件。
'''
    src=[Path(r['local_path']) for r in inputs['files'] if r['status']=='READ']
    jobs=[
      dict(kind='profile',status='pass',description='16HCregimes/8944eachPrePost conserved;80graphs/3200layers exactroleedges,3120crosslayerpairs andinterveningoperators reconciled. Noidle/gaininference.',artifacts=[O/e[0],O/e[4],N/'analyze_HC_operator_ceiling_v58.py']),
      dict(kind='review',status='pass',description='8exactHCwrapper/kernel/tilingfiles readwithoutcontainerexecution;wrapperhistoricalhashmatches. Externalpre_mix/HF32/Gram/FP32PostBF16rounding contracts found;loadedbinarynotassumed.',artifacts=[O/e[2],O/e[6],N/'HC-operator-source166-v58.json',N/'HC-dependencies-source166-v58.json',N/'read_HC_operator_contract_v58.py',N/'read_HC_dependencies_v58.py',Path(c['model_config']['path'])]+src),
      dict(kind='design-check',status='pass',description='5Mdomains work/IO/source-conditionedtilemodels,7operatorfusion/splitvariants,retainedoutputs/rounding/residual andnonadditivity constraints checked. Typedphysicalcomputepeaks unbound;onlyconditionalcoldIOreference.',artifacts=[O/e[1],O/e[3],O/e[5],N/'refine_HC_fusion_costs_v58.py']),
      dict(kind='review',status='pass',description='Operator-onlyincrementalregimecoverage andexistingF02/F03/F23 refined;noframeworksemanticclosure,implementation/nativebenchmark orwholemodulecompletion claim.',artifacts=[O/e[4],O/e[3]]),
    ]
    persist_refinement(version=58,previous=57,title='HC operator fusion, delayed-collapse split, native work and conditional theoretical bounds',
      summary=dict(filename='HC-operator-ceiling-summary-v58.json',previous_turn_classification='PROGRESS:v57shared_expert_operator_ceilings_and_singleton_merge',
        captured_HC_regimes=16,HcPre_executions=8944,HcPost_executions=8944,graph_layers=3200,root_bound_graph_layers=2560,
        crosslayer_edges=3120,crosslayer_edges_with_intervening_compute_starts=400,conditional_variants=7,
        original_operator_files_read=8,all_observed_external_pre_mix_present=True,current_Y_depends_on_current_coefficients=False,
        M3_PreNorm_removable_logical_bytes=61440,M3_cast_workspace_logical_write_read_bytes=491520,
        target_implementation_changed=False,native_test=False,new_performance_result=False,
        remote_job_status='Onlyexactexisting166stageHCsource docker-cp reads completed. Localimmutableprofile/modelanalysis only;no model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
      changes=['HC16regimes/17888executions andexactgraphoperatorroleedges reconciled.',
        'Delayedexternal-pre_mix collapse/currentcoeff independentbranches,HF32projection/Gramworkspace andFP32PostBF16round contracts quantified.',
        '7conditionalfusion/split/internalreorganization decisions with5Mwork/IOmodels;typedphysicalcomputeandallremainingoperatorscopegaps retained.'],
      candidate_updates=updates,framework_updates={},module_updates=mods,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/x for x in ['read_HC_operator_contract_v58.py','read_HC_dependencies_v58.py','analyze_HC_operator_ceiling_v58.py','refine_HC_fusion_costs_v58.py']],
      executor_prefix='zcode-HC-ceiling',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
