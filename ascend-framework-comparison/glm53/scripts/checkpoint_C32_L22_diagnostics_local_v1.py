"""Local ledger/state checkpoint from three actual terminal vLLM diagnostics."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    run = ROOT / 'runs/20261019T-L22-diagnostic-local-checkpoint-v1'
    run.mkdir(exist_ok=False)
    before = run / 'before'
    before.mkdir()
    names = ['STATE.json', 'STATUS_BRIEF.json', 'HANDOFF.md', 'PLAN.md']
    receipt = {'project_id': 'glm53', 'before': {}, 'after': {}, 'model_requests': 0,
               'SDK_init_calls': 0, 'remote_operations': 0}
    for name in names:
        raw = (ROOT / name).read_bytes()
        (before / name).write_bytes(raw)
        receipt['before'][name] = digest(before / name)
    records = []
    all_ids = set()
    for loop, version in [(20, 2), (21, 1), (22, 1)]:
        rid = f'20261019T-L{loop}-vllm-C32-paired-diagnostic-v{version}'
        folder = ROOT / 'runs' / rid
        report_path = folder / 'diagnostic-result.remote.json'
        report = json.loads(report_path.read_bytes())
        rpc = json.loads((folder / 'controller-rpc-state.json').read_bytes())
        assert rpc['phase'] == 'DONE' and rpc['client_rc'] == 0 and rpc['cpu_connection_rc'] == 0
        assert rpc['report_sha256'] == digest(report_path)
        assert report['framework'] == 'vllm' and report['diagnostic_only']
        assert not report['performance_certificate_allowed'] and not report['pending_terminal_ids']
        rows = report['rows']
        assert len(rows) == 32 and all(row['count_and_terminal_pass'] for row in rows)
        ids = {row['request_id'] for row in rows}
        assert len(ids) == 32 and not ids.intersection(all_ids)
        all_ids.update(ids)
        assert report['actual']['backend_HTTP_attempts'] == 64
        assert report['actual']['all_64_backend_successes']
        assert report['actual']['all_successful_backend_output_tokens'] == 19232
        records.append({'run_id': rid, 'report_sha256': digest(report_path),
                        'actual_backend_HTTP_attempts': 64, 'count_terminal_qualified_backend': 64,
                        'successful_backend_output_tokens': 19232,
                        'P_cache_qualified_logical': report['raw_cache_observation_passes'],
                        'native_profile_control': report['profile_completion'],
                        'profile_file_completion': 'PENDING_SEPARATE_NATIVE_EVIDENCE' if loop == 22 else 'NO_PROFILE_STARTED'})
    ledger_path = ROOT / 'results/model-request-ledger-vllm-L20-L21-L22-diagnostics-v1.json'
    ledger = {'project_id': 'glm53', 'framework': 'vllm', 'scope': 'These three diagnostic cohorts only; no historical cumulative total inferred.',
              'runs': records, 'totals_this_continuation': {'backend_HTTP_attempts': 192,
              'count_terminal_qualified_backend': 192, 'successful_backend_output_tokens': 57696},
              'historical_vllm_cumulative': None, 'no_formal_performance_promotion': True,
              'new_SGLang_requests_booked': 0}
    assert not ledger_path.exists()
    write(ledger_path, ledger)
    sg = ROOT / 'results/model-request-ledger-through-L17-formal-v2.json'
    assert digest(sg) == 'b86b1b11d582522df50ed15835f29d72373c8e015662cf413e61f418d8a48cfb'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    latest = {'run_id': records[-1]['run_id'], 'actual_backend_attempts': 64, 'actual_outputs': 19232,
              'true_coordinator_terminal': True, 'client_rc': 0, 'all32_P_cached74368': True,
              'profile_control_both_scopes_start_stop_HTTP200': True,
              'native_completion': 'PENDING_NATIVE_FILE_EVIDENCE',
              'native_profile_scope': 'C32client/80K/600, DP4+DP10 active andtotalD>=8; actual trigger12. No full-D32/all16active claim.',
              'collector_entry': 'scripts/collect_C32_L22_paired_native_evidence_v3.py',
              'collector_review': 'results/C32-L22-native-evidence-collect-review-v3.json',
              'SGLang_L22_status': 'NOT_STARTED_AT_CHECKPOINT_CREATION; re-read its own coordinator before any dispatch',
              'vllm_continuation_ledger': str(ledger_path.relative_to(ROOT)),
              'vllm_continuation_ledger_sha256': digest(ledger_path),
              'no_new_human_or_parent_approval_required': True}
    for name in names[:2]:
        value = json.loads((ROOT / name).read_bytes())
        value.update(as_of=now, status='L22_NATIVE_PROFILE_EVIDENCE_AND_PAIRED_DIAGNOSTIC_IN_PROGRESS', objective_complete=False)
        value['L22_loop'] = latest
        value['active_execution'] = {'phase': 'VLLM_MODEL_CLIENT_TERMINAL_NATIVE_READONLY_COLLECT_PENDING_OR_RUNNING',
                                     'model_or_formal_client_active': False, 'profile_control_terminal': True,
                                     'collector_actual_state': 'READ_OWN_COLLECT_RUN_RECEIPT',
                                     'current_vllm_run_id': records[-1]['run_id']}
        value['current_blocker'] = 'No external SSH block. Native trace evidence and SGLang paired diagnostic remain required.'
        value['next_action'] = 'Accept native vLLM file evidence plus D-idle; dispatch frozen L22 SGLang once; collect paired evidence, match actual batch/context, rank critical-path deltas, continue one evidence-based optimization. Do not stop at this checkpoint.'
        write(ROOT / name, value)
    previous = (ROOT / 'HANDOFF.md').read_text()
    header = f"""# 当前恢复点 {now}

L22 vLLM真实终态：32logical/64backend/19232output，全部计数终态与前缀命中通过。两D scope start/stop均200，原生文件完成性待独立采集。客户端C32/真实80K/600不变，采样按DP4/10与总D>=8，实际触发12；不能称全D32或性能提升。

L20/L21/L22本轮vLLM实际合计192backend/57696output，独立ledger见 results/model-request-ledger-vllm-L20-L21-L22-diagnostics-v1.json。原SG6729/6492/1432790与64pending字节不变。无新增SG记账；启动前读其独立run目录防重放。

下一步：接受L22 vLLM原生trace和D-idle，执行已冻结L22 SG一次，再采集匹配batch/context的trace。持续授权有效，无需父/用户审批；仍需继续优化，objective_complete=false。旧SSH阻塞已历史化。实际资源经ZCode；无跨对话消息/Git/浏览器/系统更改。

下方为保存的历史记录。

"""
    (ROOT / 'HANDOFF.md').write_text(header + previous)
    (ROOT / 'PLAN.md').write_text("# 当前计划\n\n接受L22 vLLM原生文件与D-idle证据后执行冻结的L22 SG一次；真实终态后只读采集两D节点trace。匹配实际batch/context，区分工作量、设备效率、通信与等待的绝对关键路径差距，再实施单一候选和最小正确性/无profile收益验证。可信整机收益后再跑正式128请求。\n\n客户端C32/80K/600、普通W8A8、固定2P2D、SDMA不变；代表DP4/10与总D>=8仅为诊断采样门槛，不称满D32。持续授权有效，无需用户或父对话审批。禁止旧launcher重放、其它项目/对话/Git/浏览器/系统SDK驱动BIOS时钟改动。\n\n详情与精确下一命令见本目录L22 review、各framework独立coordinator和HANDOFF当前头；不能把计划记实际或在本地准备后停工。\n")
    for name in names:
        receipt['after'][name] = digest(ROOT / name)
    receipt['vllm_actual_diagnostic_totals'] = ledger['totals_this_continuation']
    receipt['SG_ledger_unchanged_sha256'] = digest(sg)
    write(run / 'receipt.json', receipt)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
