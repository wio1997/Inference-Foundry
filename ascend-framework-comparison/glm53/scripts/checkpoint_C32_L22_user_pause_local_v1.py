"""Close the completed L22 diagnostic locally and honor the user's explicit pause."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/20261019T-L22-user-pause-closeout-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    RUN.mkdir(exist_ok=False)
    before = RUN / 'before'
    before.mkdir()
    names = ['STATE.json', 'STATUS_BRIEF.json', 'HANDOFF.md', 'PLAN.md', 'README.md', 'RECOVERY_LATEST.md', 'PERFORMANCE_CHECKPOINT.md']
    receipt = {'project_id': 'glm53', 'model_requests': 0, 'remote_operations': 0, 'new_profiles': 0, 'Git_operations': 0, 'before': {}, 'after': {}}
    for name in names:
        (before / name).write_bytes((ROOT / name).read_bytes())
        receipt['before'][name] = sha(before / name)
    analysis_path = ROOT / 'results/C32-L22-paired-trace-primary-analysis-v2.json'
    assert sha(analysis_path) == '7169905ba2e3766cd2965f5462b4a71e38dd7f61ad2db355935ab0619c2fffa9'
    analysis = json.loads(analysis_path.read_bytes())
    previous_ledger = ROOT / 'results/model-request-ledger-through-L17-formal-v2.json'
    assert sha(previous_ledger) == 'b86b1b11d582522df50ed15835f29d72373c8e015662cf413e61f418d8a48cfb'
    old = json.loads(previous_ledger.read_bytes())
    cohort_summaries = {}
    evidence = [analysis_path, previous_ledger]
    for framework, expected_sha in [('vllm', '3e9df6a2d5216f8c7d3c2440d8c648c17f52f9942afdd9c334dccbce973a95ca'), ('sglang', 'a9c06546ee5e52a47f245670ce0b018be699c85787549d9835a894719821d53f')]:
        folder = ROOT / f'runs/20261019T-L22-{framework}-C32-paired-diagnostic-v1'
        report_path = folder / 'diagnostic-result.remote.json'
        assert sha(report_path) == expected_sha
        d = json.loads(report_path.read_bytes())
        rpc = json.loads((folder / 'controller-rpc-state.json').read_bytes())
        assert rpc['phase'] == 'DONE' and rpc['client_rc'] == rpc['cpu_connection_rc'] == 0
        assert rpc['report_sha256'] == expected_sha
        assert len(d['rows']) == 32 and all(row['count_and_terminal_pass'] for row in d['rows'])
        assert len({r['request_id'] for r in d['rows']}) == 32 and not d['pending_terminal_ids']
        assert d['raw_cache_observation_passes'] == 32 and d['actual']['all_64_backend_successes']
        assert d['actual']['backend_HTTP_attempts'] == 64 and d['actual']['all_successful_backend_output_tokens'] == 19232
        assert d['diagnostic_only'] and not d['performance_certificate_allowed']
        if framework == 'sglang':
            assert d['profile_control']['start_response']['status'] == 200
            assert d['profile_control']['stop_response']['status'] == 500
            assert d['profile_control']['pass_control'] is False
        else:
            assert d['profile_control']['pass_control'] is True
        cohort_summaries[framework] = {'run_id': d['run_id'], 'report_sha256': expected_sha, 'true_terminal': True,
            'logical_requests': 32, 'backend_HTTP_attempts': 64, 'count_terminal_successes_backend': 64,
            'successful_backend_output_tokens': 19232, 'prefix_cache_pass_logical': 32,
            'wall_seconds_profiled_diagnostic_only': d['wall_seconds'],
            'sampled_peak_D_running': d['D_metrics']['sampled_peak_total_D_running'],
            'profile_control_pass': d['profile_control']['pass_control'], 'formal_performance_certificate': False}
        evidence.extend([report_path, folder / 'controller-rpc-state.json'])
    native_path = ROOT / 'runs/20261019T-L22-native-evidence-collect-sglang-v3/receipt.json'
    native = json.loads(native_path.read_bytes())
    assert native['pass'] and native['model_requests'] == native['new_profiles'] == native['remote_writes'] == 0
    ranks = []
    for node in [36, 41]:
        path = native_path.parent / f'node-{node}-report.json'
        node_report = json.loads(path.read_bytes())
        assert all(node_report['checks'].values()) and len(node_report['worker_folders']) == 16
        assert all(w['worker_complete'] for w in node_report['worker_folders'])
        ranks.extend(w['native_rank_id'] for w in node_report['worker_folders'])
        evidence.append(path)
    assert sorted(ranks) == list(range(32))
    evidence.extend([native_path, ROOT / 'results/C32-L22-vllm-native-raw-capture-primary-acceptance-v1.json'])
    for node in [43, 45]:
        path = ROOT / f'runs/20261019T-L22-vllm-representative-offline-export-v1/node-{node}-export.json'
        export = json.loads(path.read_bytes())
        assert export['pass'] and export['model_requests'] == export['new_profiles'] == 0
        evidence.append(path)
    for representative in analysis['representatives']:
        path = ROOT / representative['source']
        assert sha(path) == representative['sha256']
        evidence.append(path)
    acceptance = {'project_id': 'glm53', 'decision': 'L22_TERMINAL_COUNTS_CACHE_AND_NATIVE_CAPTURE_ACCEPTED_DIAGNOSTIC_ONLY',
        'cohorts': cohort_summaries,
        'native_profile': {'sglang': '32 distinct native ranks 0..31, six required CSVs complete/stable on every worker; two representative traces copied locally.',
                           'vllm': '32 raw worker captures independently verified; exactly two representatives parsed from copies and copied locally, not all32 parsed.'},
        'SGLang_stop': {'HTTP_status': 500, 'raw_control_pass': False, 'waived': False, 'reason': 'UNKNOWN; file completion is independently proven and does not turn HTTP500 into success.'},
        'new_full_per_request_SDMA_transport_qualification': False,
        'performance_certificate_allowed': False, 'new_candidate_deployed': False,
        'trace_analysis': str(analysis_path.relative_to(ROOT)), 'trace_analysis_sha256': sha(analysis_path),
        'evidence': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in evidence]}
    acceptance_path = ROOT / 'results/C32-L22-terminal-primary-acceptance-v1.json'
    assert not acceptance_path.exists()
    write(acceptance_path, acceptance)
    totals = dict(old['totals'])
    totals['backend_HTTP_attempts'] += 64
    totals['pending_native_backend_inferences'] += 64
    totals['pending_native_successful_output_tokens'] += 19232
    assert totals['backend_HTTP_attempts'] == totals['qualified_successful_backend_inferences'] + totals['backend_attempts_without_qualified_success'] + totals['pending_native_backend_inferences']
    ledger = {'project_id': 'glm53', 'framework': 'sglang', 'classification': 'L22_ACTUAL_COUNT_CACHE_PASS_FULL_TRANSPORT_QUALIFICATION_NOT_NEWLY_REPEATED',
        'previous_ledger': str(previous_ledger.relative_to(ROOT)), 'previous_ledger_sha256': sha(previous_ledger),
        'totals': totals, 'L22_delta': cohort_summaries['sglang'],
        'new_count_terminal_and_cache_pass_backend': 64,
        'new_full_native_qualified_backend': 0,
        'pending_native_note': 'Old L17 v1 pending64/64tokens preserved; new L22 64/19232 count-terminal/cache successes separately pending full per-request transport qualification. Completed profiling does not certify the entire P-to-D transport path anew.',
        'no_double_count': True, 'vllm_counts_added_to_SGLang': False,
        'performance_certificate_allowed': False}
    ledger_path = ROOT / 'results/model-request-ledger-through-L22-diagnostic-v1.json'
    assert not ledger_path.exists()
    write(ledger_path, ledger)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary_path = 'diagnostics/model-profile/C32-L22-pause-summary-20261019.md'
    pause = {'requested_by': 'human_user', 'instruction': '最新的这个跑完就先停一下 总结归纳 然后到github', 'as_of': now,
             'pause_new_experiments': True, 'auto_resume_allowed': False, 'services_retained': True,
             'allowed_closeout': 'Local verification, summary, evidence index, and scoped GitHub publication only.'}
    for name in names[:2]:
        value = json.loads((ROOT / name).read_bytes())
        value.update(as_of=now, status='PAUSED_BY_USER_AFTER_L22_COMPLETE', objective_complete=False,
            current_blocker=None, current_blocker_scope='No external blocker; paused explicitly by user.',
            next_action='Wait for explicit user resume. Do not launch another model/profile/candidate or rollback automatically.',
            next_checkpoint='GitHub publication of L22 pause summary; then remain paused.')
        value['user_pause'] = pause
        value['L22_loop'] = {'state': 'COMPLETE_DIAGNOSTIC_ONLY_PAUSED', 'primary_acceptance': str(acceptance_path.relative_to(ROOT)),
            'primary_acceptance_sha256': sha(acceptance_path), 'summary': summary_path, 'cohorts': cohort_summaries,
            'new_candidate_deployed': False, 'performance_improvement_confirmed': False}
        value['active_execution'] = {'phase': 'NONE_USER_PAUSED', 'model_or_formal_client_active': False,
            'profile_control_terminal': True, 'collector_active': False, 'external_executor_active': False}
        value.setdefault('active', {}).update(phase='PAUSED_BY_USER_AFTER_L22_COMPLETE', metadata_only_executor_active=False,
            external_executor_active=False, new_experiments_allowed=False, optimization_authorized=False,
            optimization_authorization_scope='PAUSED_BY_LATEST_HUMAN_REQUEST; prior authorization remains historical only.')
        value['model_http_attempts_by_this_task'] = totals['backend_HTTP_attempts']
        value['successful_model_inferences'] = totals['qualified_successful_backend_inferences']
        value['pending_native_model_inferences'] = totals['pending_native_backend_inferences']
        value['model_request_ledger'] = str(ledger_path.relative_to(ROOT))
        value['model_request_ledger_sha256'] = sha(ledger_path)
        value['model_request_ledger_totals'] = totals
        for framework in ['vllm', 'sglang']:
            value.setdefault('framework_activity', {}).setdefault(framework, {}).update(new_model_experiment_active=False,
                new_model_HTTP_experiment_active=False, current_work='L22 complete; user paused; see current primary acceptance and pause summary.')
        value['GitHub_publication_receipt_for_current_snapshot'] = 'PENDING_CURRENT_L22_PUBLICATION; historical receipts do not certify this snapshot.'
        write(ROOT / name, value)
    summary = '''# GLM-5.3 C32：L22 完成，按用户要求暂停

本次仅普通 W8A8、固定两套 2P2D、P→D SDMA、C32、真实 80K 输入/600 输出。最新两边请求、profile 与采集均已结束；不再自动开始候选、回滚、模型或 profile。保留当前服务与缓存。原优化目标尚未达成。

## 结果与证据边界

| L22 诊断 | vLLM | SGLang |
| --- | ---: | ---: |
| 逻辑请求 / 后端实际尝试 | 32 / 64 | 32 / 64 |
| 计数与终态通过 / 输出 token | 64 / 19232 | 64 / 19232 |
| 前缀缓存通过 | 32/32 | 32/32 |
| profile 原始控制结果 | 两节点 start/stop 均 200 | start 200、stop 500，控制失败原样保留 |
| 原生文件证据 | 32 rank 原始采集完整；仅解析两代表 rank 副本 | 32 rank 的六类 CSV 完整稳定；两代表 trace 已采集 |

vLLM 的 daemon worker 未自动解析原始数据；另用已安装 torch_npu 官方离线 analyse 接口在副本上解析两个代表 rank，未重跑请求或 profile。SGLang 的 stop 500 原因仍未知，原生文件完成性另行验证，不把 500 改写为成功。

这两次是带 profile 的诊断，不替代正式性能基线；没有本轮新增优化收益。D 的采样门槛为原生 DP4、DP10 活跃且总 running≥8，实际触发 12；不是全 D32 或全 16DP 同时活跃。实际 batch/context 尚未逐 replay 完全对齐。

## 差距归纳

两边代表 rank 均为每 target replay 75 次 MoE Dispatch、每周期 5 次 eager draft Dispatch。用同一硬件进程的核心任务和重复首任务切分，按 Decimal 在每台主机内计算时间差：

| 原生 DP | 首周期差 SG−VL | target Graph 核心跨度差 | Graph 之间间隔差 |
| --- | ---: | ---: | ---: |
| 4 | 78.552540 ms | 8.912508 ms | 69.640032 ms |
| 10 | 78.901106 ms | 9.401834 ms | 69.499272 ms |

SGLang 两个代表的 Graph 间隔约 90 ms，vLLM 约 20.5 ms。SGLang eager MoE Dispatch 的设备跨度中位约 5.7–6.0 ms，vLLM 约 50–55 μs；但 target Graph 中同名算子的中位均约 43–47 μs。长跨度可能包含通信等待或到达不齐，不能宣称计算效率相差百倍。

实际 trace 中，vLLM 有异步任务队列流，workspace 查询在主线程、执行 API 在后端线程；SGLang 的对应 API 同在主线程。vLLM target 使用 MlaPrologV3，SGLang 为拆分路径。这些是恢复后的候选方向，未隔离因果、未部署 TASK_QUEUE_ENABLE=1、未验证收益。SG 每代表只有一个完整周期，VL 有七个，profile 扰动与请求上下文仍限制比较。

## 正式性能结论保持不变

| 历史正式 C32（128/128） | tokens/s | TPOT P50 | TTFT P50 |
| --- | ---: | ---: | ---: |
| vLLM 参考 | 530.8125 | 19.1527 ms | 21.1620 s |
| SGLang L12D 历史参考 | 389.8741 | 36.6586 ms | 17.7572 s |
| 当前 SGLang L17 NoCopy | 411.9606 | 37.9931 ms | 16.6567 s |

L17 对 L12D 的 TPS 增加 5.665%，TPOT 中位却增加 3.640%；冻结的解码改善目标未达到，NoCopy 解码优化未验收，固定 SLA 未通过。当前 D36/41 仍为 L17 NoCopy；L12D 是历史参考。没有本轮部署或回滚。

## MemCache、记账和恢复入口

按 Ascend/框架对应部署文档与实际 SDK 源码核对。L20 结束后元数据检查发现既有 32×625=20000 个 key 全部存在；这不倒推解释 L20 请求时的 miss。L21、L22 使用真实 tokenizer 生成新后缀后，32/32 前缀命中通过。未通过清缓存或重复预热掩盖 miss。

SG 实际累计后端尝试为 6793；完整资格成功 6492 保持不变。L17 旧 64 个 pending 不变；L22 新 64 个计数/终态/缓存通过，因未重做完整逐请求 SDMA 传输资格，另列 native pending。vLLM L20/L21/L22 仅本段合计 192 后端 /57696 输出，独立记账，不混入 SG。

- 终态独立验收：[C32-L22-terminal-primary-acceptance-v1.json](../../results/C32-L22-terminal-primary-acceptance-v1.json)
- 可复算 trace 分析：[v2 结果](../../results/C32-L22-paired-trace-primary-analysis-v2.json)、[分析脚本](../../scripts/analyze_C32_L22_paired_traces_local_v2.py)。v1 API 过滤漏掉 AscendCL@ 前缀，v2 已修正；图周期结果不变。
- 证据按路径、大小、SHA256 索引；原始 trace、业务请求/输出文本、凭证、权重、镜像及 SDK 二进制不上传。
- 恢复后先核对现场身份/负载、batch/context、全 rank eager Dispatch 分布、队列与 NoCopy/SDMA 生命周期兼容性，再冻结单变量候选、当前运行时回退方案，做最小正确性和无 profile C32 验证。有可信收益后再考虑正式 128 请求。

官方部署/解析参考：[vLLM Ascend KV Pool](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md)、[原生 profile 离线解析](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/service_profiling_guide.html)、[TASK_QUEUE_ENABLE](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/TASK_QUEUE_ENABLE.md)。实际运行结论以本项目原生证据为准。
'''
    (ROOT / summary_path).write_text(summary)
    current = f'# 当前恢复点 {now} — 用户要求暂停\n\nL22 两框架诊断、profile 与采集均已结束；当前无模型/benchmark/profile/collector 执行。最新用户要求“最新的这个跑完就先停一下 总结归纳 然后到github”。只完成本地总结和 GitHub 发布，然后保持暂停；禁止旧 launcher 重放或自动开始下一候选。服务、缓存和当前 L17 D 运行时保留；objective_complete=false。\n\n归纳见 [{summary_path}]({summary_path})；验收见 results/C32-L22-terminal-primary-acceptance-v1.json。SG stop HTTP500 与控制失败保留，32 rank 文件完成另证；VL 32 原始 rank 完整但仅两个代表解析。不构成新正式性能结果。旧 SSH 阻塞及下方“继续执行”指令均为历史。\n\nSG 实际6793/完整资格6492，native pending128（旧64+新64）；vLLM 三诊断192/57696另列。所有原始报告和历史 ledger 保持原字节。GitHub 当前发布收据另存，旧收据不代表本快照。\n\n下方为历史记录；仅此当前头与 STATE.user_pause 决定是否执行。\n\n'
    for name in ['HANDOFF.md', 'RECOVERY_LATEST.md']:
        (ROOT / name).write_text(current + (before / name).read_text())
    (ROOT / 'README.md').write_text('## 当前状态：2026-10-19 L22 完成，用户要求暂停\n\n两边 C32 诊断和原生 trace 采集结束；当前不再启动实验。详见 [本轮总结]('+summary_path+') 与 [当前交接](HANDOFF.md)。正式性能仍为历史 vLLM 530.8125 TPS /19.1527 ms、当前 SGLang L17 411.9606 TPS /37.9931 ms；没有新候选收益。下方旧状态为历史。\n\n' + (before / 'README.md').read_text())
    (ROOT / 'PLAN.md').write_text('# 当前计划：按用户要求暂停\n\n最新 L22 两边请求/profile/采集均结束。完成本地总结、证据索引和仅本项目 GitHub 发布后，等待用户明确恢复。不得自动启动新模型请求、profile、候选、回滚、预热或定时任务。保留当前服务与缓存；目标尚未达成。\n\n恢复后的候选审查和最小验证顺序见 [本轮总结]('+summary_path+')。既往广泛持续授权在本次暂停期间不允许自动执行。\n')
    (ROOT / 'PERFORMANCE_CHECKPOINT.md').write_text('# L22 诊断已结束，按用户要求暂停\n\n本轮不是正式性能测量；下列 L17/L12D/vLLM 正式数字不变。Graph 与 Graph 间隔差的证据边界见 [L22 总结]('+summary_path+')；当前无新增候选收益。\n\n' + (before / 'PERFORMANCE_CHECKPOINT.md').read_text())
    receipt.update(as_of=now, acceptance=str(acceptance_path.relative_to(ROOT)), acceptance_sha256=sha(acceptance_path), ledger=str(ledger_path.relative_to(ROOT)), ledger_sha256=sha(ledger_path), user_pause=pause)
    for name in names:
        receipt['after'][name] = sha(ROOT / name)
    assert sha(previous_ledger) == 'b86b1b11d582522df50ed15835f29d72373c8e015662cf413e61f418d8a48cfb'
    write(RUN / 'receipt.json', receipt)
    print(json.dumps({'status': 'PAUSED_BY_USER_AFTER_L22_COMPLETE', 'acceptance_sha256': sha(acceptance_path), 'ledger_sha256': sha(ledger_path), 'actual_SG_backend': totals['backend_HTTP_attempts'], 'qualified_SG_backend': totals['qualified_successful_backend_inferences']}))


if __name__ == '__main__':
    main()
