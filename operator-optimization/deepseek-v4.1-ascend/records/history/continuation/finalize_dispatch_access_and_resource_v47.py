"""Persist login facts, bounded missing dispatch definitions and corrected global resource units."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v47.json').exists()
    for a in read(O/'checkpoint-manifest-v46.json')['artifacts']:
        b=Path(a['path']).read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],a['path']
    access=read(O/'original-profile-access-status-v47.json');hw=read(O/'M21-hardware-and-DSACP-dispatch-contract-v47.json')
    resources=read(O/'whole-scope-resource-denominator-audit-v47.json');sem=read(O/'M21-semantic-branch-inventory-v47.json');audit=read(O/'M21-original-requirement-completion-audit-v47.json')
    assert access['login_outcomes'][0]['application_status']=='REMOTE_IDENTITY_VERIFIED'
    assert all(r['application_status']=='BASTION_ASSET_NOT_MATCHED'and not r['remote_python_executed']for r in access['login_outcomes'][1:])
    assert not access['original_dataset_access_restored']
    assert len(hw['normal_factory_cases'])==4 and len(sem['branches'])==18 and len(sem['static_branch_sites'])==268
    assert len(resources['modules'])==26 and resources['theory_comparisons_still_incomplete']==26
    count=resources['canonical_count_units'];assert count['certain_single_module_assignment_executions']+count['ambiguous_module_assignment_executions']==count['ComputeInfo_executions']==286950
    assert count['ambiguous_module_assignment_executions']==3136 and count['shape_role_context_aggregate_regimes']==2082
    assert len(audit['requirements'])==8 and not any(r['completion_proved']for r in audit['requirements'])
    evidence=['M21-hardware-adaptor-and-DSACP-source-v47.json','M21-hardware-and-DSACP-dispatch-contract-v47.json',
        'original-profile-access-status-v47.json','whole-scope-resource-denominator-audit-v47.json',
        'M21-semantic-branch-inventory-v47.json','M21-original-requirement-completion-audit-v47.json','M22-profile-access-revalidation-v47.json']
    common=dict(hardware_dispatch=str(O/evidence[1]),data_access=str(O/evidence[2]),resource_units=str(O/evidence[3]),implementation=False,new_performance_result=False)
    candidates={
        'F31':dict(review_v47=dict(common,relation='Missing DeviceOperator/adaptor and canonical DSACP definitions answered locally. NormalA3 factory packed getterNone is a conditional frozen-source result;actual166dependency/build snapshot remains unproved. Current3136ambiguity is module execution membership,not priced candidate rows.'),
            profile_review_state='FROZEN_ADAPTOR_DSACP_PREDICATES_ANALYZED;ORIGINAL_RAW_DATA_ROUTE_PENDING',
            conclusion_changing_gap='Remaining exact PCP selectors/current166generated build/dependency binding and actual graph/native bank/epochs/phase/cost need original sources/data. Newaccount193works but original166/105notmatched;193doesnotreplace oldcapture.',
            next_profile_or_contract_action='First resolve original asset/shared backup path;verify old E/index/graph/model/capture hashes on access. Until then use existing evidence and declared conditional denominators,not more unrelated source or stale-credential retries.'),
        'F34':dict(review_v47=dict(common,relation='Packed cache A5 software adaptor is not normal A3 getter;source historical A3condition does not attest current166generatedfile. Existing unchanged-SWA-anchor memory rejection remains;new whole-pool layout requires its own owner/ABI/actual memory denominator,not typed slot counts.')),
        'F35':dict(review_v47=dict(common,relation='F34/F35entered model ledger afterv20,but resource assignment is by26modules rather than33/35priced optimization candidates. Reuse actual sampler/head evidence with explicit per-variant selector/consumer/denominator;do not infer zero cost or automatic complete pricing from ledger age.')),
    }
    fw={
        'H02':dict(review_v47=dict(common,relation='Sixzero-primaryComputeInfo modules include control/process/graph work;no zero CPU/launch cost or MAC roofline inferred. Preserve host/device/copy hierarchy and actual producer-ready evidence.')),
        'H06':dict(review_v47=dict(common,relation='Normal architecture namespace/adaptor selection and resource-denominator units clarified.193login is not environment reproduction or replacement of oldDP/TP/EP capture evidence.')),
    }
    modules={
        'M21':dict(finding='Device/adaptor packed-cache getter and canonicalDSACP source definitions nowread:normalA3→STANDARD/Base→None,A5→packed;four supported profile cases documented conditionally. Old A3build artifact is historical,not current166snapshot.18families/268sites preserved;original8gates stillopen.',
            evidence=evidence,candidates=['F31','F34'],gap='Exact PCP predicates/current166generated/dependency hashes and actual native bank/allreader/phase/resource/process scopes remain.193login succeeds but original166/105asset lookup fails;originalrawdataset not restored.',scope_audit=str(O/evidence[5]),profile_scope=str(O/'M21-original-profile-gap-reconciliation-v41.json')),
        'M22':dict(finding='Originaldataset availability now has concrete new-account result:193host verified,no knownoriginalprofileprojectpath found;166oldIPandknown105入口SSHrc0butbastionasset-matchfailure,no remotePython. Resource scope differentiates primary0 from graph-child132000/hostwork.',
            evidence=evidence,candidates=['F31'],gap='Need original asset or accessible saved backup;no193dataset substitution. Existing110coreguards remain local source evidence,not actual bank/native epoch/16phase/resource/process completion.'),
    }
    old_remaining=deepcopy(read(O/'module-coverage-gaps-v46.json')['remaining']);retired=[r for r in old_remaining if r.startswith('NextM22')]
    remaining=[r for r in old_remaining if r not in retired]
    substitutions={
        '3136candidate-setresource rows':'3136ambiguous-module ComputeInfo executions',
        '3136candidate资源行':'3136条模块归属歧义计算执行',
        '3136resource':'3136ambiguous-module ComputeInfo executions',
        '3136资源行':'3136条模块归属歧义计算执行',
        '3136资源':'3136条模块归属歧义计算执行',
    }
    changed=[]
    for i,s in enumerate(remaining):
        before=s
        for a,b in substitutions.items():s=s.replace(a,b)
        if s!=before:changed.append(dict(before=before,after=s))
        remaining[i]=s
    write(O/'resource-terminology-and-access-history-before-v47.json',dict(previous_version=46,retired_progress_paragraphs=retired,
        current_terminology_replacements=changed,reason='Clarify ambiguous module-membership execution units and actual new-account routing;preserve original counts/source scopes and historical reports.',overall_complete=False))
    remaining.insert(18,'v47global resource units:286950ComputeInfo executions=283814certain singletonmodule+3136ambiguous modulemembership,4ambiguitysets;2082shape/role/context aggregate regimes.3136is notoptimizationcandidate count or3136pricedcandidate rows.35model7framework indexing is distinct from per-variant pricing. Sixprimary0modules M00/M01/M02/M03/M22/M25are notzeroCPU/graph/allocator/transport cost;hierarchical parents andCOMM/ComputeInfo need explicit denominators. All26theory comparisons stayINCOMPLETE;no inventedzero/peak/gain.')
    remaining.insert(19,'Nextoriginal-profile access:193newaccountloginverified;old166andknown172.16.10.105bothbastionassetnotmatched despiteSSHrc0,not successfulremoteexecution.193knownprojectprofilepaths absent;boundedpubliclookup no candidate found,notexhaustivefilesystemabsence. Useraskedfororiginalasset/account/alias orsharedbackup path;fresh193recapture onlyifuserchoosesnewcontrolledcollection,do notsubstituteoldcase. Nooldcredentialretries orother-sessionchange. Untilrouteavailable,continueuseful existing-evidence/uncertainty audits without broad code/zero-filling. Remainingoriginalmodule/capture/process/resource andnativeepoch/16phase scopes persist.')
    action=('继续原Goal，v47为实际进展：新登录结果+旧source缺失predicate补证+全26模块资源口径审计。先读dispatch-access-resource-summary-v47.json、original-profile-access-status-v47.json、M21-hardware-and-DSACP-dispatch-contract-v47.json、whole-scope-resource-denominator-audit-v47.json、M21-original-requirement-completion-audit-v47.json及module-coverage-gaps-v47.json。'
        '用户新给fanyichao@root@172.16.10.193登录，已一次认证root193/hostname193成功；同新账号原10.55.0.166及实际主机已知172.16.10.105都返回match asset failed(SSHrc0不代表成功)，未执行remotePython。193原P0/profile、已知wio/tiankuan路径不存在，public只限1000topentries过滤未找到，不是全盘不存在证据。新问题已问原资产可用账号/别名、193可访问sharedbackup路径，或由用户选择193重采。不要试随机资产/IP/密码<REDACTED_CREDENTIAL>'
        '密码只经echo-disabled TTY stdin→anonymouspipeFD传sshpass，未存workspace/argv/receipt；只publicknown_hosts和600s控制socket，连接按asset绑定，不能换用户名共用193socket去执行166命令。现控制socket可过期，future只检查已活socket并用muxonly(ProxyCommand=false)防新authfallback；需要新登录仍用隐蔽stdin，不从代码/argv写密码。与其他会话无改动，无Ns/processkill/环境迁移/模型或容器传输。'
        '早期keychain旧xinyufeng自Oct26未变，原cfgpass与authfail约600s可能是ControlPersist到期，一致不是服务端原因证明；这个旧缺口已升级为newaccount资产匹配/资料位置问题，不再反复问同旧密码<REDACTED_CREDENTIAL>ile-access-revalidation-v47是新凭据前的检查。'
        '本地补defs: DeviceOperator=get_device_adaptor()在module初始化，依据_CURRENT_HARDWARE_PROFILE.family；DeviceConfig缓存built __device_type__，显式unknown报错，缺字段才走legacySOC；A2/A3STANDARD→BasegetterNone，310P兼容继承None，A5FP8_OPTIMIZED→MixedQuantPackedCacheOps。hardware_profile/device_config目前冻结source新SHA并非新166attest，A3build只在历史source169 _build_info(65ed...)有记录；normalA3 conditionalproof与现legacy529metadata一致，不当actual166snapshot。utils.enable_dsa_cp1589-1596读validatedAscendConfig.singleton，v42CLIflag&&hasIndexer&&seqparallelmoe规范化仍保留，不能rawadditionalTrue代实际。Exactenable_pcp/common_cpdecode-shard仍本地缺。无需再找不存在的DeviceOperatorFactory/get_device_type符号或整SDK。'
        '资源口径纠正：3136是4组模块归属歧义的ComputeInfo执行，不是优化candidate资源行；283814certain+3136=286950，2082是shape/role/context aggregates，PMU286101/缺849；35模型候选和7框架只是indexed，不代表全部同口径定价。v20当时33model，后F34/F35可用原module证据但需variant身份/边界绑定才报数值gain/rank，不可直接说新candidate零资源或33已priced。'
        '全26模块成本比较已按control/compute/compound/memory-state/collective等声明正确需求，不改变旧26INCOMPLETE或补P0值。6primary0模块M00/M01/M02/M03/M22/M25并非零CPU/graph/allocator/transport成本；M19func1560不等nativeancestry14191，M21func76943不等84297关联scope，M22func0不等132000graphchild或host0；COMM intervals/HCCLComputeInfo/MC2内部层级不可相加。ShapeMAC是算法条件模型，MAC0不等无attention/MoE算量，missingPMU不填0。下一步依证据分母准备critical补读/必要来源，而非继续堆model代码。'
        '用户若给原始入口/backup，先verify原E/index SHAa3aa47...、graphSHAa035c4.../32part/capturecase/modelCANN合同后读完整资料，绝不按newhost/ordinal最近时间把16unbound补配。用户若明确选择193新采，先从服务器可用资料完整复现CANN9.2/固定代码模型负载并核资源使用条件，原限制本地资料不上传；别在环境未复现/当前任务未授权负载时直接启动Ns或改服务。新的capture另label，不冒充旧数据集。'
        '当前M21 18family268sites、M22 16family1101sites/110localcoreguards保留；所有原8gates/full26source-branch-consumer-process/resource scopes未全证，0整module。可独立继续明确有限的原scope证据/条件分析，不因认证一项就blocked；真正无有意义进展且同阻塞≥3turn才blocked。没有实现优化或测速，Goalactive不因报告完成。'
        '本轮6runs4PASS2ERROR：193identity/projectlookupPASS，166/105资产匹配应用层ERROR各1(不要看rc0当成功)，boundedsourcePASS，4hardware条件case+26module/resourceunit审计PASS，formal8gate/完整ledger一致审核PASS。无新inference/NPU/service/health，remoteprobes全terminal；profile rawaccess未恢复。Usepersist_profile_refinement_v3，勿改已pinned旧helper。')
    report='''193已成功登录，root及主机身份匹配。用同一新账号核验原10.55.0.166和记录中的172.16.10.105，均返回“未发现匹配的资产”；虽然SSH退出码为0，远端Python并未执行，不能标成成功。193上原profile及已知项目路径不存在，限定的public目录查找也未发现匹配项；这不是全盘无资料的证明。已询问原资产可用入口或共享/备份路径。尚未将193替代原数据集，也没有迁移环境、改服务或启动模型负载。

密码仅经关闭回显的stdin和匿名pipe FD用于认证，未写入工作目录、进程参数或结果文件。连接测试和资料路径读取均已结束；其它会话未改动。

独立分析补齐了部分旧predicate定义。normal factory按HardwareProfile选择adapter：A2/A3使用标准Base getter返回None，310P兼容分支也继承None，A5分支才返回MixedQuantPackedCacheOps。DeviceConfig按构建字段选择，profile及DeviceOperator在模块初始化时绑定；canonical enable_dsa_cp读取validated AscendConfig singleton。已有历史A3 build资料与legacy SMLA/QLI轨迹相符，但历史source169 build-info和新读的冻结依赖文件没有冒充新166运行时attestation。当前是否存在外部override、生成文件变化及精确PCP selectors仍保留缺口。A3可复用某纯坐标helper，不等于可以直接选择完整A5 packed ABI。

本轮还校正了全局资源口径。之前“3136资源行”的表述不准确：它是4组模块归属仍有歧义的ComputeInfo执行数。283814条确定单模块归属，加3136条歧义执行，共286950；2082才是shape/role/context聚合regime数。35项模型候选、7项框架候选的索引覆盖，与每个变体是否完成资源定价是不同问题；不能用模型候选数或旧资源表完成度代替后者。

26模块已建立明确的成本分母需求：控制/进程、条件计算模型、复合动态算量、内存/状态、通信层级分别需要不同证据。M00/M01/M02/M03/M22/M25没有ComputeInfo主归属记录，不意味着零CPU、graph、allocator或transport开销。M19功能bucket1560不等于14191条native ancestry；M21功能76943不等于84297关联导航；M22主归属0不等于132000图内子任务或host开销为0。也不能把COMM区间、HCCL命名ComputeInfo和MC2内部通信相加。没有将MAC0、缺PMU或缺容量补成0，没有新增理论峰值/收益结论；26理论比较仍INCOMPLETE。

6项登记为4pass、2error。错误是原166/105两次资产匹配失败，已按应用层结果记录。193只读核验、定点source、条件硬件分派与全26分母审计、正式持久化审核通过。35模型候选、7框架候选、0整模块关闭；286950计算执行、3136模块归属歧义执行、849缺PMU、16phase-unbound保持。

整体目标仍未完成。原始完整profile入口、实际graph/bank/native epoch及全部读者、剩余模块/分支/进程边界和同分母资源上界仍待补证。Goal active，不以新服务器登录、局部source证明或报告成稿代替整体完成。
'''
    jobs=[
        dict(kind='review',status='pass',description='Newuser193rootlogin/hostnameverified and boundedknownproject/publicmetadata lookupviaexistingauthenticatedmaster;originaldata paths notfound,no NPU/model/service changes.',artifacts=[N/'probe_updated_bastion_login_v47.py',N/'updated-login-193-v47.json',N/'193-existing-saved-materials-v47.json',O/evidence[2],O/evidence[6]]),
        dict(kind='review',status='error',description='Original166new-account probeSSHrc0butbastion asset-matchfailure;remotePythonnotexecuted,resultclassifiedERRORnot success.',artifacts=[N/'updated-login-166-v47.json',O/evidence[2]]),
        dict(kind='review',status='error',description='Known originalhost172.16.10.105 alternateentrynew-account probealsoasset-matchfailure despiteSSHrc0;nootherhostguessing.',artifacts=[N/'updated-login-166-via105-v47.json',O/evidence[2]]),
        dict(kind='review',status='pass',description='Boundedfrozenadapter/config/profile/DSACPdefinitions;historicalA3build metadata explicitlynotfresh166attestation,no module import.',artifacts=[N/'read_M21_hardware_dispatch_points_v47.py',O/evidence[0],O/evidence[1]]),
        dict(kind='review',status='pass',description='Fourhardwareconditionalfactorycases and26module resource-denominatoraudit;3136ambiguous executions/2082regimes/35candidate indexing distinguished,no numericgain or pricedcandidate claim.',artifacts=[N/'build_dispatch_and_resource_evidence_v47.py',O/evidence[1],O/evidence[3]]),
        dict(kind='review',status='pass',description='Previous46filemanifest verified;M21stable18families268sites and8open gates retained,originalcase notsubstituted,newaccount/accesspending and canonical metrics persisted.',artifacts=[O/evidence[4],O/evidence[5],N/'persist_profile_refinement_v3.py']),
    ]
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v47.json');gate.update(no_new_SSH_login=False,
        authorization_update='User supplied new193account and requested trying it;knownoriginal166/105 entries were read-only checked for originalprofile access.',
        original_no_new_login_rule_applied_before_user_credential_update=True,no_new_model_torch_NPU_service_health_work=True)
    write(O/'PROFILE-CODE-QUESTION-GATE-v47.json',gate)
    summary=dict(filename='dispatch-access-resource-summary-v47.json',previous_turn_classification='PROGRESS:v46modeDP41guards',
        login193_verified=True,original166_asset_matched=False,known105_asset_matched=False,original_profile_access_restored=False,
        normal_factory_hardware_cases=4,resource_module_denominators=26,conditional_new_variants=0,
        compute_executions=286950,certain_single_module_assignment_executions=283814,ambiguous_module_assignment_executions=3136,
        shape_role_context_aggregate_regimes=2082,PMU_groups=286101,missing_PMU=849,phase_unbound_graphs=16,
        model_indexing_not_individual_variant_pricing=True,theory_comparisons_still_incomplete=26,
        related_model_candidates=['F31','F34','F35'],related_framework_candidates=['H02','H06'],new_model_candidates=[],
        remote_login_probes=3,remote_project_lookup=1,remote_identity_pass=1,remote_application_errors=2,
        original_data_location_question_pending=True,
        remote_job_status='All3authorizedloginprobesand193existing-mastermetadata readterminal:193identityPASS,166/105bastionassetNO_MATCH despiteSSHrc0. No raworiginalprofileaccess restored,no runningmodel/profile jobs. Local source/evidence auditsPASS.',
        no_new_model_NPU_service_or_health_work=True)
    persist_refinement(version=47,previous=46,title='v47 original-profile access, hardware dispatch and resource units',summary=summary,
        changes=['193newloginidentityverified;original166andknown105assetsnotmatched,SSHrc0notremoteexecutionsuccess;originalprofileaccessnotrestored andlocationquestionpending.',
            'Fourconditionalhardwareadaptorpaths/canonicalDSACPgetter locallyanalyzed,historicalA3metadata notfresh166attestation;missingPCP/live dependency binding retained.',
            'Correctglobalunits:3136ambiguous module-membership computeexecutions,2082regimes;26module denominatorneeds declared,35candidate indexing not individual pricing. No gain/modulecompletion claim.'],
        candidate_updates=candidates,framework_updates=fw,module_updates=modules,semantic_branches={'M21':sem['branches']},source_read=True,
        remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=['original-profile-access-status-v47.json','M21-hardware-and-DSACP-dispatch-contract-v47.json','whole-scope-resource-denominator-audit-v47.json','M21-original-requirement-completion-audit-v47.json'],
        publication_rows={
            'F31':'| F31 | Metadata、slot、draft与graph发布重组 | 条件 | v47补normal adapter与canonicalDSACP源码；A3 packed getterNone为冻结normal-path条件证明，不冒充当前166依赖/生成值。193已连但原166/105资产未匹配、原rawprofile未恢复。3136为模块归属歧义执行而非已定价候选资源行；实际bank/epoch/分母和完整scope仍未闭合。 |',
            'F34':'| F34 | 压缩KV状态、slot与物理缓存布局 | 条件 | v47区分normalA3与A5packed软件分派；已有SWA anchor不变时不能承诺单独slot压缩省显存，整体pool/ABI变体需实际所有权/布局/内存分母。历史A3构建不是新166runtime快照。 |',
            'F35':'| F35 | 目标采样、词表所有权与概率消费者重组 | 条件 | 复用既有sampler/head证据，并为各变体绑定实际选择范围和成本分母；全局模块资源守恒或35候选索引不证明每个变体已定价，仍保留真实采样flag、消费者与native代际条件。 |',
        },
        gate_updates=dict(status='LOCAL_DISPATCH_DEFS_AND_GLOBAL_RESOURCE_UNITS_ANALYZED;ORIGINAL_DATA_ROUTE_OPEN',answered_by=evidence,
            closed_scope='Frozen normalhardwareadaptor/canonicalDSACPdefinitions;explicit193login vs166/105assetno-match;globalcountunitanddenominator audit.',
            not_closed='Originalrawprofileaccess,actual166dependency/build/PCP/native bank epochs,remainingmodule branch/consumer/process/resource scopes.',
            stop_rule='Do not repeatoldpasswordor misreadrc0assetfailure as access;do not replace originalcapturewith193 or price variants by module ambiguity counts. Originaldata route takes priority over broad newsource reads.'),
        scripts=[N/'probe_updated_bastion_login_v47.py',N/'read_M21_hardware_dispatch_points_v47.py',N/'build_dispatch_and_resource_evidence_v47.py'],
        executor_prefix='zcode-dispatch-',main_script=N/'finalize_dispatch_access_and_resource_v47.py')


if __name__=='__main__':main()
