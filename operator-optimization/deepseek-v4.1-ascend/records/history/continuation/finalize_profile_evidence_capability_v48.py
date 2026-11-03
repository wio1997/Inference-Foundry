"""Persist the scoped cache-capability audit; do not run the prepared remote query."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N, O, read
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v48.json').exists()
    audit = read(O / 'saved-profile-evidence-capability-audit-v48.json')
    proof = read(O / 'saved-profile-local-proof-index-v48.json')
    validation = read(O / 'saved-only-query-validation-v48.json')
    assert len(audit['rows']) == 13 and len(proof['input_receipts']) == 15
    assert validation['status'] == 'PREPARED_STATIC_VALIDATION_ONLY'
    assert hashlib.sha256(Path(validation['script']).read_bytes()).hexdigest() == validation['script_sha256']
    assert not audit['new_remote_query'] and not validation['remote_execution']
    assert all(not row['requires_new_capture_now'] for row in audit['rows'])
    evidence = ['saved-profile-evidence-capability-audit-v48.json',
                'saved-profile-local-proof-index-v48.json', 'saved-only-query-validation-v48.json']
    review = dict(evidence_capabilities=str(O / evidence[0]),
                  local_proof_index=str(O / evidence[1]),
                  scope='Cached receipts and bounded analysis-query logic;no new model/SDK source or remote read.',
                  implementation=False, new_performance_result=False)
    candidate_updates = {
        'F31': dict(review_v48=dict(review,
            relation='Shapes/stacks do not by themselves prove storage/version/native event generations. Table-field absence and first-record disabled memory/args flags do not establish full-capture absence.16unboundgraphs have0sidecarenvelope candidates,not16unique candidates awaiting clock promotion.'),
            next_profile_or_contract_action='Resolve original saved-profile route,then read only a specific missing metadata/identity surface under pinned hashes. Continue finite branch/consumer source questions independently. Do not require implementation/native validation to count a declared conditional variant semantically analyzed;do not claim actual gain or deployment legality without their evidence.'),
    }
    framework_updates = {
        'H02': dict(review_v48=dict(review,
            relation='Core24/48 and reference1.8GHz conversion are cached with empirical286101group agreement. All28savedPMUfields read;849groups remain missing,not a four-column omission. Worker wall/helper DB inventories do not provide current EngineCore onCPU;no blanket recollection or invented denominator.')),
        'H06': dict(review_v48=dict(review,
            relation='Saved metadata/config census query is prepared-only,not executed on193or166.16phase-unboundgraphs have no saved sidecar envelope candidate;metadata/clock inspection is not promised to bind them. Conditional semantic analysis is distinct from actual capture ABI/epoch eligibility.')),
        'H07': dict(review_v48=dict(review,
            relation='56same-threadnonoverlapping sync pairs and no observed writer between are host-order facts,not native event-generation equality or all-future-reader proof. Preserve actual bank/input generation obligations when considering reuse or sync reorganization.')),
    }
    modules = {
        'M21': dict(
            finding='Profile-first evidence audit separates13question domains;fullshape/stack identity is not storage/scalar/native epoch identity. All28PMUfields and cached core/clock reference retained. No new source sites or branches declared analyzed.',
            evidence=evidence, candidates=['F31'],
            gap='Existing finite eligibility/consumer branches and original8formal gates remain. Rawcaptureabsence not inferred from one schema;conditional semantic analysis does not require implementation/native benchmark. Originaldataset route still pending.'),
        'M22': dict(
            finding='Graph phase gap sharpened:64same-call bindings,16zero-sidecar-envelope-candidate graphs,0previously-unbound unique candidates. All32selectedsix-table schema agreement does not cover everymetadata schema/value;bounded saved-only query prepared,not executed.',
            evidence=evidence, candidates=['F31'],
            gap='No new graph/bank/native handle/generation/rankwise values obtained;110localcoreguards and16semanticfamilies preserved. Originalroute and allremainingmodule/process/resource/consumer scope remain open.'),
    }
    remaining = deepcopy(read(O / 'module-coverage-gaps-v47.json')['remaining'])
    remaining.append('v48evidence-capability audit:13question domains/15historicallypinned receipts. Cached16device24Cube/48Vector and documented1.8GHz reference with286101group agreement already exist;all28PMUfields read,849executiongroups missing. Avoid repeated core/clock queries or four-column/blanket recapture assumptions.16unboundgraphs havezero savedsidecarenvelope candidates;clock inspection alone not proven to bind. PYTORCH_API lacks explicitstorage/versionfields;firstconfig memoryFalse/record_argsFalse onlyfirstrecord,notfullcapturenegativeproof. Queryonlythe missing savedmetadata/config/identitysurface afteroriginalaccess;preparedqueryNOTEXECUTED. Conditionalsemanticcompletion is separate from actualdeployment/nativevalidation/gain;finiteoriginalbranch/consumer/process/theorygaps remain.')
    action = (
        '继续原Goal，v48是缓存能力/证据边界审计进展，不是整模块完成。先读saved-profile-evidence-capability-audit-v48.json、saved-profile-local-proof-index-v48.json、original-profile-access-status-v47.json及module-coverage-gaps-v48.json。'
        '193新账号登录root/hostname193已验证；同账号原10.55.0.166和实际105均堡垒机asset-no-match，SSHrc0不是远端成功；193已知项目/profile路径未找到，不等于全盘不存在。用户待答原资产可用入口或共享/备份路径；不重试旧密码<REDACTED_CREDENTIAL>'
        '已缓存16device每设备24Cube/48Vector、captureinfo24/48及1800、CANN参考1800→1.8e9Hz、286101组双引擎周期公式一致；这部分不要再标成待采。28项PMU全字段已读，849缺组不补零。16phase-unbound graphs的sidecar候选数均0，不能说只差时钟就可绑定。'
        'profile API shapes/stacks可导航，不能独立证明tensor storage/version/alias/plan标量/native event generation；PYTORCH_API无显式字段+首个config memoryFalse/record_argsFalse只证明限定视图，不证明全部raw中没有。'
        'query_saved_metadata_and_capture_flags_v48.py仅准备和静态校验，未执行。原资料路由恢复后先验证capsSHA79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095/EindexSHAa3aa47...及旧capture/model/input/serve身份；按需补读保存的32config和五类小metadata，不启动模型/torch/NPU/health/process/kills，不用旧remote_run误连169。'
        '无原资料时继续可推进的有限source语义问题，不为报告再造范围。条件优化点可在声明条件下完成语义分析，勿以未实现/未native验证/未benchmark本身当未分析；实际reader/branch/bounds缺口仍须补。'
        '35模型/7框架/0整模块；M21十八semantic families268static sites、M22十六families1101sites/110coreguards原样。3136是模块归属歧义执行，2082regimes；26理论比较INCOMPLETE。不可因报告/登录/能力表成稿标Goal完成。'
    )
    report = (
        '本轮按profile优先审计了13类证据问题，并核对15份本地回执的历史manifest哈希。没有新连服务器、读取模型/SDK代码或启动设备工作。\n\n'
        '核数及周期换算已有证据：原166缓存的16设备属性均为每设备24Cube/48Vector，capture metadata也覆盖ID0至15；CANN9.2参考将记录1800换为1.8e9Hz。此前286101组在双引擎上的周期/时间一致性检查已全通过。这些事实无需重复补采，亦不能据此宣称物理峰值、MFU或可删除时延。\n\n'
        '完整PMU已包含28项、8010828行、286101组，不是只提取4列。849条ComputeInfo执行缺PMU仍保留；缺组不能补零，重采相同配置也不保证能取得不发计数器的工作。\n\n'
        '16个phase-unbound graph在现有sidecar envelope中的候选均为0，已绑定的64个各为1；没有原先未绑定而现在出现唯一候选的graph。因此不能假定校准时钟就能完成绑定。\n\n'
        'PYTORCH_API提供shape/dtype/stack及执行连接字段，没有显式pointer/storage/version列。首个profiler配置关闭memory及op-args记录；这一首记录事实没有扩大成全32配置普查，更没有扩大成所有raw文件均无身份信息。attrInfo解码也不直接证明本算子实际标量值。调用栈能够定位机会，不能单独定案别名、捕获bank或native event代际。\n\n'
        '已准备一个带原capture/index/DB哈希、主机及路径约束的保存资料查询，仅补读32配置及NPU/HOST/META/SESSION/RANK元数据，限制输出且只读。当前只通过静态校验，未执行。193登录在上一轮已成功，但原166/105资产匹配仍未恢复；待用户提供原资产入口或共享备份路径，不将193冒充旧capture。\n\n'
        '条件优化点的语义分析、部署条件及实测收益分别记录；未实现、未native验证或未benchmark本身不算未分析。尚未解决的全分支、全部消费者、进程范围、16阶段绑定、实际代际及资源上界仍保留。35模型候选、7框架候选、0整模块关闭，整体Goal继续进行。\n'
    )
    jobs = [
        dict(kind='review', status='error', description='Local builder initially assumed all historical manifests had artifacts;legacyv3/v4/v5 use files. Failed before any output/shared-state mutation;compatibility correction retained.',
             artifacts=[O / 'evidence-capability-builder-error-v48.json']),
        dict(kind='review', status='error', description='Added utility historical-pin assertion found no manifest entry for local inventory utility. Failed before overwriting audit/shared state;now records current SHA without claiming historical remote script authentication.15data-receipt pins remain mandatory.',
             artifacts=[O / 'evidence-capability-utility-pin-error-v48.json']),
        dict(kind='review', status='error', description='First persistence attempt used taskctl kindanalysis,which is unsupported. Registration stopped before first run/checkpoint/RECOVERY/task-state update;F31master/publication writes were archived and restored exactly to v47 before retry with supported kindreview.',
             artifacts=[O / 'profile-capability-persistence-recovery-v48.json']),
        dict(kind='review', status='pass', description='Audit13saved-profile evidence questions against15historically pinned receipts;preserve cached denominators,28field PMU scope,zero-candidate16graphs and qualified table/config absence. No new model/SDK/remote work.',
             artifacts=[O / evidence[0], O / evidence[1], N / 'build_profile_evidence_capability_v48.py',
                        N / 'inventory_host_profile_v12_166_v2.py', N / 'query_complete_PMU_fields_v18_166.py']),
        dict(kind='review', status='pass', description='Prepared bounded hash-pinned immutable metadata/config query;AST/import/header-hash checks pass. STATIC ONLY:remote contents and execution have not been tested.',
             artifacts=[N / 'query_saved_metadata_and_capture_flags_v48.py', O / evidence[2]]),
    ]
    persist_refinement(version=48, previous=47,
        title='Saved-profile evidence capability and cache-first gap audit',
        summary=dict(filename='profile-evidence-capability-summary-v48.json',
            previous_turn_classification='PROGRESS:v47access_dispatch_resource_units',
            evidence_question_count=13, input_receipts_historically_verified=15,
            no_new_model_or_SDK_source_read=True, saved_only_query_executed=False,
            cached_devices=16, cached_Cube_per_device=24, cached_Vector_per_device=48,
            PMU_metric_fields=28, missing_PMU=849, phase_unbound_without_sidecar_candidate=16,
            remote_job_status='No new remote operation or background job launched in v48. v47completed193identityPASS and166/105asset-no-match facts retained;current server workload/health not polled.',
            original_profile_access_restored=False, target_implementation_changed=False,
            new_performance_result=False),
        changes=[
            '13evidence questions bounded against15pinned receipts;cached core/clock reference and all28PMUfields distinguished from actual missing849groups.',
            '16unboundgraphs have0savedsidecarenvelope candidates,not merely pendingpromotion unique matches;table/first-config absence not inflated into fullcapture absence.',
            'Saved-only32config/fivemetadata-table query prepared and staticallychecked,NOT EXECUTED;conditional semantic completion separated from implementation/native/gain. No newmoduleclosure.'],
        candidate_updates=candidate_updates, framework_updates=framework_updates,
        module_updates=modules, remaining=remaining, action=action, jobs=jobs,
        report=report, read_first=evidence + ['original-profile-access-status-v47.json'],
        publication_rows={}, gate_updates=dict(overall_complete=False),
        scripts=[N / 'build_profile_evidence_capability_v48.py', N / 'query_saved_metadata_and_capture_flags_v48.py'],
        executor_prefix='zcode-profile-capability', main_script=Path(__file__), source_read=False)


if __name__ == '__main__':
    main()
