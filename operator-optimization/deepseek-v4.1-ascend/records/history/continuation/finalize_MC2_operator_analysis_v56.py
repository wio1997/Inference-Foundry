"""Persist MC2 internal fusion, receiver cost and communication-cut analysis."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v56.json').exists()
    p=read(O/'MC2-operator-profile-v56.json');b=read(O/'MC2-kernel-boundary-and-tiling-contract-v56.json');m=read(O/'MC2-receiver-work-and-communication-model-v56.json');inputs=read(O/'MC2-operator-inputs-v56.json')
    assert p['MC2_executions']==4472 and len(p['root_bound_graph_layers'])==2560 and len(p['phase_unbound_graph_layers'])==640
    assert len(m['cohorts'])==87 and len(m['conditional_variants'])==5 and len(b['parts'])==11
    assert m['per_expert_INT8_weight_bytes']==35389440 and m['per_assignment_GEMM_ops']==70778880
    assert m['per_assignment_internal_logical_interface_bytes']==43520
    e=['MC2-operator-profile-v56.json','MC2-kernel-boundary-and-tiling-contract-v56.json','MC2-receiver-work-and-communication-model-v56.json','MC2-operator-inputs-v56.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v56.json',dict(status='PROFILE_CANNOT_SPLIT_INSIDE_EXISTING_MC2_KERNEL;POINT_OPERATOR_SOURCE_ONLY',
        previous_turn_classification='PROGRESS:v55operator_fusion_and_shape_ceiling',
        profile_evidence=['operator-fusion-profile-v55.json','shared-MC2-observed-order-contract-v20.json',e[0]],
        questions=['Whichoperations/internalGMbuffers/numericalbarriers arealreadyinsideDispatchFFNCombine?',
            'Does zeroexpertworkloadreadallallocatedweights,andwhichrecord/dataformat crossesrank?',
            'What pipelinecut/alias/flag requirements constrain finerfusion andcommunication coalescing?'],
        allowed_source='Eight exactMC2kernel/tiling/dependencyfiles fromknown166stage;W8historicalcontrolmatch. Noframework/API/default/plugin audit.',
        answered_contract=str(O/e[1]),supporting_saved_count_models=str(O/e[2]),
        no_model_Torch_NPU_newcapture_service_health_process_or_implementation_work=True,whole_module_closed=False,overall_complete=False))
    candidate={}
    for cid in ['F21','F22','F23']:
        candidate[cid]=dict(review_v56=dict(profile=str(O/e[0]),kernel=str(O/e[1]),receiver_models=str(O/e[2]),
            relation='MC2alreadyfusesrouteddispatch/GMM1/SwiGLUquant/GMM2/combine;sharedactivation isseparate. InternalFP16/INT8/FP32barriers andGMaliasedbuffers bound. Work usesreceiverassignments/activeexperts;5conditionalvariants refineexistingcandidates. Sharedoverlaporacle excludesdependentfinaladd.',
            implementation=False,native_test=False,new_performance_result=False),
            profile_review_state='MC2_INTERNAL_OPERATOR_FUSION_RECEIVER_WORK_AND_COMMUNICATION_BOUNDS_ANALYZED',
            conclusion_changing_gap='Actualhostprof n_e/token-destination jointchoices,internalphase/physicalmemory/linkcost,allnewtiling/precision/alias/flag/nativebinaryconditions. Knownkernelboundary/internalFP16/zeroexpert/L2hint/default21+3 andGMreusecontracts areanswered;do not re-auditloaders/framework trees.',
            next_profile_or_contract_action='Useoperator-specificshape/work/capacitymodels;advance sharedexpert quant-GEMM-activation-GEMM andHc/attention/indexer chains. Only pointread kernel/tiling/consumer/sync that changes a namedfusion/ceiling conclusion;do not requireimplementation merelyforconditionalanalysis.')
    modules={
       'M12':dict(finding='MC2alreadylarge-fusedbutretainsGM C1/activated/C2 interfaces;INT8GMMlocaloutputsFP16,activationFP32withRINTrequant. ReceiverA_r andactiveE_r model6HI*A_r ops,3HI*E_r weights;zeroexpertskip/sourceL2hint/default21+3cut nowknown. Fiveoperatorvariants registered.',evidence=e,candidates=['F21'],gap='Exactcurrentpercallroutes,internalphase/hardware/memory/link model andallnewnativeprecision/tiling/lastreaderconditions;count/profilepopulations notjoined.'),
       'M13':dict(finding='2560root-boundgraphlayersputshared5stagesafterMC2;route median304.513us,sharedpremerge57.301us,endpoint384.4475us. 57.301us isunlimited-resource/no-work-changeoverlapcap median,notfreegain;finalrouted/sharedaddexcludedfromoverlap.',evidence=e,candidates=['F22','F23'],gap='ActualcompatibleMC2sharedschedule,resourceinterference,allscale/round/ownership/events andsharedoperatorroofline remain.'),
       'M23':dict(finding='Observer87formalcohorts conserveall16rank sender/receiver counts. CurrentcodedispatchrecordH+32=5152B,returnBF16H=10240B;unique-token/destinationcountboundedfrommarginals,notexactdeduprate. Coalescing needsnewmetadata/localfanout;pre-returnweightedaggregation changesround/sumorder.',evidence=e,candidates=['F21'],gap='ActualhostprofjointtopKroutes/physicalwiretraffic/linkcapacities/cutcriticalpath;no native16rankjoin fromtime/ordinal.'),
       'M24':dict(finding='MC2workspacealiasesC1/C2andinput/activatedINT8banks. Finerpipeline/morebuffers requireexplicitlast-reader,rows/stride/scale/credit-generation ownership;sourceflags alreadyexist. No address-onlyreuse orfreepipelinebanks.',evidence=e[1:3],candidates=['F21'],gap='Actualnativealias/readiness generations andallcrossiterationobservers/newprotocol proof;no independentframework lifecycle audit.'),
    }
    remaining=deepcopy(read(O/'module-coverage-gaps-v55.json')['remaining'])
    old=remaining[5];assert old.startswith('MoE/MC2operatornext:')
    remaining[5]='MC2v56operatorlocalcontractsanswered:alreadyfusedroutequant/dispatch/GMM1/SwiGLUquant/GMM2/dequantreturn/unpermute;GMaliasedC1/C2andA/activation;internalFP16notexternalBF16;zeroMskip,L2bypasshint<=128,21+3target/5+3draftfirstcut. Receivercountmodels87separateformalcohorts nowderive6HI*A_r,activeweight3HI*Eplus andGMinterface(10I+4H)*A_r pluspayload/cutbounds. Currenthostprofactualn_e/jointtokenroutes/internalphase/cache/link/tiling/nativeprecision remaingenuinegaps. No samecallmodel/profilejoin or speedup claim;do not repeat answeredABI/loader/kernelboundary questions.'
    remaining.append('F21/F22v56:5conditionaloperatorvariants—finerC/Vhandoff,work-awareepiloguecut,token-destinationcoalescing,conditionalpre-returnpartialcombine,andnewcompatible sharedoverlap. ExactrootboundMC2median304.513us;sharedpremerge57.301us/endpoint384.4475us. Unlimitedresourcecapexcludesdependentadd andcannotignore24Cube48Vector/BWcontention. Observerfirst21 targetexpertscarrymedian93.069%rows;not temporal latency fraction. Activeexperts/weights/packets andreferencepeak models do not priceactualHBM/nettraffic orcurrentrequestgain.')
    write(O/'MC2-gap-reconciliation-v56.json',dict(before=old,after=remaining[5],scope='Retireexactoperatorboundary/internaldtype/zeroexpert/GMalias/defaultcut gaps;retainactualpercall/keycapacity/otheroperator coverage.',overall_complete=False))
    action=(
      '继续原Goal：只围绕operator fusion/split/reorganization/theoreticalceilings，不研究框架。v55→v56是MC2实际kernel边界、profile和计数理论进展。先读MC2-kernel-boundary-and-tiling-contract-v56、MC2-operator-profile-v56、MC2-receiver-work-and-communication-model-v56和module-coverage-gaps-v56。'
      '已有4472MC2分5shape；3200graph层中2560rootbound/640unbound。rootboundMC2median304.513us，共享分支合并前span57.301us，MC2start→sharedaddend384.4475us；当前同流串行，MC2==1已有sharedoverlap禁用是已知，不再泛读选择框架。overlaporaclecap必须排除依赖两支的最终add，且无资源冲突假设不实际可达。'
      '8个原166stage operator/tiling依赖与2个历史hash匹配EPcount完整文件缓存N/MC2-operator-source-v56，W8controla676通过；新ssh56重新验证166/host105root和4corehash。source不是loadedbinary证明，但局部条件算子分析不需先实现benchmark。'
      '重要kernel事实：INT8GMM1/GMM2内部CType=float16_t，FP16中间→FP32pertokenscale/clamp/SiLU→RINT INT8+FP32scale；外部输入/returnBF16不代表内部BF16。C1与C2同GMbank，input与INT8activation同bank；不能省掉舍入或忽略WAR。n_e=0跳过blockMmad，n_e<=128请求权重L2bypass；不能把24expertbank全计作每call读。L1tile128x256x512不是所有tinyM必执行128rows的证明。'
      '默认epiloguecut=E_local-3(target21+3,draft5+3)，不是半分或按token；observerformal80target/7draft cohort(与hostprof不同run)first21行份额median93.069%，weightactive份额86.667%，值得F21-CUT条件方案，不是93%耗时。receiveractualA_r来自所有sender，zeroSender仍可能receiver忙，参与者不能删。'
      '理论：H5120/I2304每activeexpertINT8weight35389440B+packedscales77824B；每receiverassignment6HI=70778880 INT8op(2/MAC)。GM逻辑接口往返8I+2I+4H=43520B/assignment，不是HBM实测。Tref>=max_r(compute/Pint8,activeweights/BHBM,cutbytes/Bnet,deps)，用v55reference752TOPS/1.6TB/s有条件，不能把不同时序计数/耗时除为效率。'
      'Dispatch实际recordH+32=5152B(含scale/pad)，语义INT8+FP32scale为H+4=5124B，returnBF16H10240B。sender-receiver边A_sr→unique token-dest D_sr区间ceil(A_sr/min(K,E_local))..min(V_s,A_sr)；marginals不能给实际D。F21-COALESCE需新协议/ownership/quant规模，F21-PARTIAL提前weightedcombine不能当bitwise等价(舍入/求和顺序/传weights/宽度)。'
      '下一步直接推进共享expert DQ→INT8GEMM→SwiGLUquant→GEMM→merge的operatorwork/traffic/epilogue模型，或HcPre/HcPost/attention/indexer尚未给出定量ceil的链；不要重复MC2加载/empty专家/FP16/现有大融合已答问题。再按全2082regimes及35model候选补全所有原operator分支、消费者和跨层跨卡跨迭代模型。'
      '849PMU缺/16phase-unbound/3136歧义ComputeInfo/全模块真实未分析项保留；不以candidate数、完整表、报告或无implementationbenchmark判定Goal完成。优先读本地缓存；必要原文件通道ssh166-v56-xinyufeng.sock已用ControlPersist3600延长复用，新script显式override transport.CONTROL=该socket，不改pinned旧helper/不silentfallback。无模型/Torch/NPU/新profile/health/process/kill/实现，Goal active。')
    report=(
      '本轮直接分析DispatchFFNCombine内部融合、接收端工作量和通信下界。读取8个定点operator/kernel/tiling依赖、2个历史哈希匹配的完整EP计数资料；没有模型、设备或新采集。\n\n'
      'DispatchFFNCombine已经融合routing量化、跨卡dispatch、GMM1、SwiGLU/再量化、GMM2、dequant-return和weighted unpermute。profile旁边的DequantSwigluQuant是共享专家支路，不能重复算成routed FFN待融合激活。\n\n'
      '内部仍经GM：GMM1的FP16结果→FP32激活/scale→INT8结果→GMM2 FP16结果→BF16返回。外部BF16 dtype不能替代内部FP16边界。C1/C2复用同一GM bank，input/activation也复用；细化流水必须保护旧读者和行/stride/scale边界。每个receiver assignment涉及的三处逻辑中间接口往返为8I+2I+4H=43,520B；这些不是实测HBM字节。\n\n'
      '按H=5120、I=2304计算，每个活跃expert的两组INT8权重共35,389,440B，另有77,824B packed scale；两次GEMM每接收一个token-assignment需70,778,880整数运算（每MAC计2）。源码对空专家不调用blockMmad，对n_e≤128的权重请求绕过L2。因此不能把分配的24个expert权重全部当成每次读取。\n\n'
      '流水第一切点固定为本地expert数减3，即target21+3、draft5+3。保存的80个正式target计数cohort中，前21个expert包含接收行的中位93.069%；按实际工作量选择切点值得研究。它是计数份额，不是耗时份额。\n\n'
      '2560个root关联图层：MC2设备耗时中位304.513μs，共享支路合并前span中位57.301μs，MC2开始到最终add结束span中位384.4475μs。即使不改变工作、假设输入独立就绪且资源无限，能重叠掉的共享支路也至多受该premerge span限制；最终add不能被并行到两支完成之前。实际两支共享24Cube/48Vector及带宽，不能把57.301μs直接当收益。\n\n'
      '通信模型区分assignment与唯一token-destination。当前dispatch读取H+32=5152B的INT8/scale记录，BF16返回数据为2H=10240B。可研究同一token发往同一rank时只发送一次、在本rank fanout；但现有计数矩阵只有边缘统计，只能给唯一pair数量区间，不能给实际去重率。提前在receiver做weighted partial combine还会改变舍入和求和顺序，并需要传递weights，不能作为无条件等价融合。\n\n'
      '本轮形成5个条件方案：更细的Cube/Vector中间交接、按接收工作量切流水、按目的rank合并token传输、受数值条件约束的提前combine、与MC2兼容的共享专家重叠。它们细化现有F21/F22/F23，没有新增主候选或实测加速。\n\n'
      '理论形式为T≥max_r(6HI·A_r/P_INT8，活跃权重/内存带宽，必要跨rank字节/链路容量，依赖下界)。87个计数cohort与后来的profile不是同一轮，参考硬件容量也未冒充当前实测，不能相除报利用率。下一步继续共享专家及其它热点算子的定量融合/上限分析，整体尚未完成。\n')
    sources=[Path(r['local_path'])for r in inputs['operator_source']+inputs['saved_evidence']]
    jobs=[
      dict(kind='review',status='pass',description='Exactoriginal166root/corehashrevalidated;2savedEPfullanalyses historicalhashmatch and8boundedMC2operator/tilingfiles copiedwithoutcontainerexec/model/device/servicequery. No frameworkaudit.',artifacts=[O/e[3],N/'MC2-operator-source166-v56.json',N/'original166-xinyufeng-recheck-v56.json',N/'reauth_original166_v56.py',N/'read_MC2_operator_evidence_v56.py']+sources),
      dict(kind='profile',status='pass',description='4472MC2shape timings and3200exactgraphMC2/shared chains;2560rootbound/640unbound retained. Sharedpremerge overlaporacle excludesdependentfinaladd,not measuredgain.',artifacts=[O/e[0],N/'analyze_MC2_operator_bounds_v56.py']),
      dict(kind='design-check',status='pass',description='11pointoperatorranges bind internalFP16/FP32/RINT,GMaliasing,zeroexpertskip,L2hint,21+3/5+3cuts andH+32records. Nativebinary/pipelinecost unknown preserved.',artifacts=[O/e[1]]),
      dict(kind='design-check',status='pass',description='87formal16rankcohorts conserveassignment/expert/senderreceiver counts;receivercompute/activeweight/GMinterface andconservativeunique-token-dest bounds modeled. Counts andhostprof timings explicitlyunpaired.',artifacts=[O/e[2],N/'observer-final-evidence166.json',O/'EP-routing-and-placement-contract-v17.json',O/'hardware-reference-scope-v55.json']),
      dict(kind='review',status='pass',description='Fiveconditionaloperatorvariants refineexistingF21/F22/F23;preciseMC2gapsretiredwhileallremainingoperator/physicalkeyevidence gaps remainopen.',artifacts=[O/'MC2-gap-reconciliation-v56.json',O/e[2]]),
    ]
    persist_refinement(version=56,previous=55,title='MC2 internal operator fusion, receiver work and communication lower bounds',
      summary=dict(filename='MC2-operator-analysis-summary-v56.json',previous_turn_classification='PROGRESS:v55operator_fusion_and_shape_ceiling',
        MC2_profile_executions=4472,root_bound_graph_layers=2560,phase_unbound_graph_layers=640,
        source_parts=11,formal_count_cohorts=87,conditional_operator_variants=5,
        per_active_expert_INT8_weight_bytes=35389440,per_assignment_GEMM_integer_ops=70778880,
        target_first21_row_fraction_median=m['summary']['target']['first_epilogue_row_fraction']['median'],
        native_test=False,target_implementation_changed=False,new_performance_result=False,
        remote_job_status='Authorized166reauth/saved2files/dockercp8operatorfiles complete;localanalysiscomplete. No model/Torch/NPU/newcapture/containerexec/health/process/kill/implementation oranalysisjobleft running.'),
      changes=['MC2alreadyfusedboundary/internalFP16/GMaliasedinterfaces/zeroexpertskip/L2hint/sourcepipelinecut nowknown.',
        '4472MC2profiles and3200sharedgraphchains separated;sharedoverlapcapexcludesdependentmerge andresourcecontentionstillopen.',
        '87formalcohort receiverwork/activeweight/communicationcutmodels and5conditionalvariants boundwithoutcrossrun timingjoin orspeedupclaim.'],
      candidate_updates=candidate,framework_updates={},module_updates=modules,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/'read_MC2_operator_evidence_v56.py',N/'analyze_MC2_operator_bounds_v56.py'],
      executor_prefix='zcode-MC2-operator',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
