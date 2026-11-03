"""Persist scoped clock evidence;retain every native and whole-scope closure gate."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v112 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v112.json').exists()
    assert read(O/'clock-capacity-preflight-v112.json')['status']=='PASS'
    names=['clock-capacity-inputs-v112.json','clock-field-primary-contract-v112.json','clock-query-capability-v112.json',
        'clock-report-frequency-reconciliation-v112.json','operator-time-conversion-certificate-v112.json',
        'operator-clock-gated-cases-v112.json','clock-invalid-inference-witnesses-v112.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-operator-footprints','candidate-pair-navigation','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v111.json')));d['clock_capacity_review_v112']=dict(evidence=names,
            original_cycle_results_reused=True,physical_clock_upper_bound_proved=False,new_pairs=0,new_native_promotions=0)
        write(O/(base+'-v112.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v111.json'))
    for g in obligations['obligations']:
        if g['id']in ['G01','G08','G11','G12']:
            g['clock_capacity_review_v112']=dict(evidence=names,installed_enum_name_not_physical_fmax=True,
                exact_installed_header_and_A3_official_field_semantics=True,public_document_same_driver_release_not_attested=True,
                saved_1800MHz_retains_report_metadata_role=True,Vector_support_and_common_clock_upper_unbound=True,
                no_new_hardware_frequency_values_or_HBM_byte_capacity=True,whole_requirement_closed=False)
        if g['id']=='G13':g['clock_capacity_review_v112']=dict(evidence=names,
            v18_frequency_report_conversion_and_v89_typed_rates_are_prior_answers=True,
            do_not_repeat_rejected_freq_CLI_or_temperature_queries=True,need_new_capacity_contract_not_more_report_identity_tests=True)
    obligations.update(parent='coverage-current-obligations-v111.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v112.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v111.json'));audit.update(
        previous_goal_turn='PROGRESS:v111 nativePad provider source/metadata,1536Bbarrier workspace and96Brecord;postcheck PASS336',
        input_checkpoint=111,followup_checkpoint=112)
    audit['clock_capacity_review_v112']=dict(evidence=names,exact_header_and_field_meaning_new=True,
        report_unit_and_existing_typed_cut_results_reused_not_new=True,
        new_device_compute=False,new_physical_frequency_upper_bounds=0,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v112.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v112.json',dict(status='CAPACITY_QUESTION_ONLY_EXACT_DCMI_HEADER_AND_OFFICIAL_A3_FIELD_DOCS',
        operator_questions=read(O/names[0])['operator_questions'],
        new_framework_source_read=False,new_source_read='One installed driver interface header;officialA3DCMI field semantics',
        admin_queries='Four bounded npu-smi info commands;two return215 retained;temperature-only sensors result retained',
        no_model_tensor_compute_capture_replay_or_configuration_change=True,new_device_work=False,
        no_manual_DCMI_init_or_function_calls=True,public_browser_or_CUA_used=False,
        new_Pad_probe='StillPREPARED_NOT_EXECUTED;PENDING_REPLY',overall_complete=False))
    gap=('Exactinstalled DCMI header names type9 AICORE_MAX,but officialA3DCMI describesrated sustainable-in-TDP frequency;'
        'neither name/report1800/current sample establishes F_AIC/F_AIV upper envelopes. Vector12 isenumerated butnotinthisA3supportedlist;notqueried. '
        'No current/rated/maxvalues recovered byboundedCLI:freqselector rejected,helpavailable,sensorsonlytemps,commonwithcard/chip rejected. '
        'Existingtypedcycles andv18reportidentities remain priorconditionalanswers. Need declaredpath/numeric/workdomain plusphysicalcapacity envelope '
        'forseconds/MFU;use perengine conversion thenmax,or a provedcommonupperenvelope. NoHBM bytecapacity fromclock/utilization. No allmoduleMFU/wholeclosure.')
    updates={cid:dict(review_v112=dict(evidence=names,typed_cycle_results_reused=True,
        physical_clock_upper_bound_attested=False,same_engine_cut_order_clockfree=True,
        raw_cross_engine_cycle_MFU_ranking=False,new_native_promotions=0),conclusion_changing_gap=gap)for cid in ['F05','F10','F20']}
    modules={mid:dict(candidates=candidates,finding=finding,evidence=names,gap=gap)for mid,candidates,finding in [
        ('M07',['F05','F10'],'Preserve selectedAIC projection cyclecuts;M<=48 sharedweightcut andM>=49IndexQcompute cut ordering is clockfree withinfixedBdomain. No numericphysicaltime ceiling.'),
        ('M09',['F10'],'IndexWeight has distinctBF16work andsharedweight-ingress cut;one AIC clock-envelope certificate needed forphysicaltime. Itsrawcycles cannotrankVectorutilization.'),
        ('M12',['F20'],'Sevenrouterregimeinstructioncuts remain AIV-clock/normType2lowering conditional. NeitherAICnominal/MAXlabel norAICPUmaxattestsVectorFmax;no universalTopKceiling.') ]}
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('Currentv112clockevidence:installedMAX9isratedperA3docs;CLIrecoverednofrequencyvalues. PhysicalAIC/AIV/HBMceilings andallmoduleMFU remainunestablished. Priorv18PMUconversion/v89/v106cuts reused. OriginalnativePad44Bprobe stillawaits explicitreply;notexecuted.')
    report='''本轮补的是算子理论上限所需的时钟证据，没有扩展到框架研究。只读取得166（实际105）的安装版DCMI头文件，并固定四份A3官方接口文档的内容与SHA。查询了4个有界npu-smi info命令：freq选择器被拒绝，help成功，sensors只返回4项温度，带card/chip的common查询被拒绝。失败返回码均保留。未调用DCMI初始化或函数、未运行模型、张量计算、图捕获、replay或配置修改，SSH已关闭。

安装头文件将type9命名为DCMI_FREQ_AICORE_MAX，但A3官方说明将其定义为AI Core额定频率，表示TDP条件下可持续的频率；该说明没有给出所有允许运行条件下的硬频率上界。AICPU的max_freq属于不同引擎，不能作为Cube/Vector频率上界。Vector当前频率type12虽在枚举中，A3文档列出的支持类型只有2、6、7、9。本轮未测试type12，也不把这份公开文档当作已证明与安装驱动同一版本。[A3频率接口](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_049.html) [AICPU字段](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_039.html)

原profile的1800MHz继续保留为报告换算元数据。v18已对286101组PMU检查周期换算和报告时间，本轮没有重跑或将旧答案计为新发现。匹配报告公式不证明物理最大频率、完整时钟积分或Vector时钟域关系。HBM接口的频率、容量和利用率字段也没有给出匹配存储层的峰值字节带宽。[HBM字段](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_050.html)

已把14个已有算子容量案例加上明确的时间转换条件：7个AIC投影案例与7个AIV router案例。若必要工作为W、引擎数量不超过N、每周期能力不超过R，且运行期间频率始终不超过F，才可推出T>=W/(N*R*F)。已有周期下界C可在相同域转换为C/F。多个引擎先各用自己的有效频率上界转换，再取max；若有同时覆盖两者的共同频率上界，也可得到较弱的共同上界转换，不要求瞬时频率相等。没有共享资源或必需串行的证明时不相加。

频率未知不妨碍同一AIC域内的下界比较。已有IndexQ/IndexWeight普通完整乘法、未压缩权重、固定B从L1进入L0B、24AIC、一次初始grant的模型中，Q计算下界为80M/3周期，Weight为5M/3，共享权重进入下界为1302周期。交点为M=48.825；整数M<=48时后者更强，M>=49时Q计算下界更强。保存的M3属于前者，M800/816/830/834属于后者；两种QA/SWA/Weight ragged panel案例的权重下界也强于其计算下界。这里比较的是选定通路内已有下界的强弱，不是实测瓶颈、所有算法最优解或新增融合提速。

三个纯代数反例固定了推断边界：额定频率不是上界时转换不成立；不同引擎的周期排序可在各自频率换算后反转；相同内存频率不能唯一确定字节带宽。均不声称166具有反例中的假设频率或配置。

本轮细化F05/F10/F20及M07/M09/M12的容量证据合同，未新增候选、关系或原生执行归属。35项模型候选、353条语义记录/26暂定模块、132条条件关系/25组/463对待核对均不变。9类/1352次原生执行仍待补证，14项整体义务均未关闭，0个整模块关闭，全部模块的可比较MFU与最低模块仍为NOT_ESTABLISHED。目标保持进行中。

v109获准的最小探针已完成；v111另行准备的BOOL20→24/valueTrue原生Pad探针仍等待明确回复，本轮没有运行。当前没有新的完整模型测试授权。
'''
    action=('继续原Goal，v112PROGRESS，先读clock-capacity-summary/report/fieldcontract/timecertificate/14cases/current-obligations-v112；verify_clock_capacity_v112须PASS。'
        'v111PASS336，当前v112只读hostDCMIheader SHAa7be9d5f...与A3HDK2610四doc，4info查询中freq/common返回215、help成功、sensors四temps，无频率新值；SSHclock已closed。'
        '新证据是MAX9额定字段/Vector12非支持列表/AICPUmax异引擎；非新答案是v18全部PMU1800换算、v89typedrates、v106cyclecuts。物理F_AIC/F_AIV/HBM容量仍缺，别继续重复报告公式或CLI失败。'
        '14旧算子案例加perengineclockenvelopecertificate；commonupperbound足够不要求instantclockequal。IndexQWeight fixedB域交点48.825（M<=48权重cut更强，>=49Qcompute更强），只是已有cuts比较不是实测bottleneck/新gain。'
        'v111准备nativePad44BBOOL20→24 padTrue新async问题仍未收到明确reply，禁止run。原v109最小probe已完成消耗授权，不可视为全model或新Pad授权。'
        '若新的Pad明确获准，可在新v113而非已完成v112记录，按v111plan freshidle/container/header/lib/metadatahashguard，只自身一次capture、不replay、不model/noopaqueGPUderef；否则继续有证据的未分析operator关系/分支，不为报告成稿关Goal。'
        '35/353/26、132/25/463、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU均不变。TaskACTIVE Loop004EVALUATING，禁止改完成artifacts或重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=112,previous=111,title='Typed operator clock-envelope and physical-ceiling evidence',
        summary=dict(filename='clock-capacity-summary-v112.json',previous_turn_classification='PROGRESS:v111 PadV3 source/metadata and96Brecord;postcheck PASS336',
            installed_driver_header_files=1,official_A3_document_files=4,bounded_read_only_admin_queries=4,retained_rejected_query_results=2,
            new_hardware_frequency_values=0,new_physical_frequency_upper_bounds=0,operator_clock_cases=14,
            IndexQ_Weight_clockfree_fixedB_integer_crossover=49,new_device_work=False,implementation=False,native_test=False,
            refined_candidates=3,refined_modules=3,branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,
            relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            targeted_Pad_probe='PREPARED_NOT_EXECUTED;PENDING_ASYNC_REPLY',actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            remote_job_status='Four read-only npu-smi info queries andoneheader retrieval finished;SSHclosed. No model/tensor/compute/capture/replay/configuration change;newPadprobe unexecuted.'),
        changes=['Bind exact installedMAX9enum to officialA3rated-frequency semantics;do not substitute AICPUmax orenumerated unsupportedVector12 for a physicalAIC/AIV upperclock.',
            'Retain all four boundedCLIreceipts including two rejectedqueries;no hardwarefrequency recovered,no manualDCMI calls,andSSHclosed.',
            'Attach perengineclock-envelope proof conditions to14prior operatorcuts;derive sameAICfixedB M48/49cut crossover withoutphysicalMFU orspeedupclaims.'],
        candidate_updates=updates,module_updates=modules,remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Exactinstalleddriverheader plusfourprimaryA3documents andfourboundedadminqueryreceipts;two return215 retained,nofrequencyupper recovered.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='14existingoperatorclockcertificates,M1..2048fixedBcutcrossing and36clockenvelopealgebrafixtures;no model ornewprobe.',artifacts=[O/'clock-capacity-preflight-v112.json',O/'whole-scope-requirement-audit-v112.json',O/'coverage-current-obligations-v112.json'])],
        report=report,read_first=names+['coverage-current-obligations-v112.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_clock_capacity_v112.py',N/'verify_clock_capacity_v112.py',N/'read_clock166_v112.py',N/'read_clock_sensors166_v112.py',N/'read_clock_common166_v112.py',N/'persist_profile_refinement_v112.py'],
        executor_prefix='clock-capacity',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
