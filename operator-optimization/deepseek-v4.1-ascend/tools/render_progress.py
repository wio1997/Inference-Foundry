#!/usr/bin/env python3
"""Render the small, reviewable candidate progress pages; --check is read-only."""
import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def render():
    register = json.loads((ROOT / 'progress/candidates.json').read_text())
    events = [json.loads(line) for line in (ROOT / 'progress/events.jsonl').read_text().splitlines() if line.strip()]
    candidates = register['candidates']
    expected = [f'F{i:02}' for i in range(1, 36)] + [f'H{i:02}' for i in range(1, 8)]
    if [item['id'] for item in candidates] != expected:
        raise ValueError('The register must retain exactly the existing 35+7 stable IDs, in order.')
    focus = register.get('current_focus', {'candidate': 'F01', 'variant': 'EARLY-Y'})
    if focus['candidate'] not in expected:
        raise ValueError('Unknown current focus candidate.')
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('Event IDs must be unique; append corrections instead of replacing past events.')
    known_ids = set(expected)
    for event in events:
        if not set(event['candidates']).issubset(known_ids):
            raise ValueError('Unknown candidate in event.')
    outputs = {}
    for item in candidates:
        cid = item['id']
        if item['implemented'] and not item.get('implementation_evidence'):
            raise ValueError(f'{cid}: implementation promotion requires implementation evidence.')
        if item['correctness_code'] == 'PASS' and not item.get('correctness_evidence'):
            raise ValueError(f'{cid}: correctness promotion requires actual candidate-path evidence.')
        if item['performance_status'] != 'NOT_MEASURED' and not item.get('performance_evidence'):
            raise ValueError(f'{cid}: a performance conclusion requires comparison evidence, including negative results.')
        if item['measured_speedup'] and (not item['implemented'] or item['correctness_code'] != 'PASS' or item['performance_status'] != 'MEASURED_GAIN'):
            raise ValueError(f'{cid}: speedup requires an implementation, correctness acceptance, and a measured gain.')
        if item['independent_review_code'] == 'COMPLETED' and not item.get('independent_review_evidence'):
            raise ValueError(f'{cid}: an independent review request is not a completed review.')
        lines = [f"# {cid} · {item['name']}", '', '[返回进展总览](README.md)', '',
                 f"最近有意义的更新：{item['updated_at']}（Asia/Shanghai）", '', '## 当前状态', '',
                 '|项目|状态|', '|---|---|', f"|优化类型|{item['kind']}|",
                 f"|历史分析|{item['historical_status']}|", f"|本轮复核|{item['review_status']}|",
                 '|已登记的实现/试验|本台账未登记实现|' if not item['implemented'] else '|已登记的实现/试验|见具体变体的实现证据；含引入的已有实验|',
                 f"|已登记的正确性|{item['correctness_status']}|",
                 f"|优化后同条件性能|{item['performance_summary']}|",
                 f"|独立复核|{item['independent_review']}|", '', '## 算子与原有流程', '',
                 item['operator_binding'], '', item['baseline_flow'], '', '## 候选改动及其意义', '',
                 item['proposal'], '', item['mechanism'], '', '## 成立条件与证据边界', '', item['conditions'], '']
        priority = item.get('current_priority')
        if priority:
            order = priority.get('current_action_order')
            groups = {'CURRENT': '历史当前行动', 'QUALIFIED_BASELINE': '已验证组合基线',
                      'ALGORITHM_CANDIDATE': '具体算法候选', 'JOINT_MECHANISM_PRICING': '联合机制待定价',
                      'FRONTEND_RESERVE': '前端条件后备', 'LONG_OR_CAPACITY': '长行/容量后备',
                      'STOP_UNCHANGED': '停止当前不变变体',
                      'STOP_CURRENT_WITH_CONDITIONAL_RESERVE': '当前变体停止，保留其他条件',
                      'DEFERRED_CURRENT_LOAD': '当前不独立开发'}
            order_text = groups[priority['group']] + (f'（行动层级{order}）' if order else '')
            lines[6:6] = ['## 当前优先级', '',
                          f"{order_text}：{priority['reason']}", '',
                          '[当前优先级、证据和停止条件](../PRIORITY-PLAN.md)。行动顺序不等于已证明的生产收益大小；未选变体继续保留其条件。', '']
        if item.get('details_markdown'):
            lines += [item['details_markdown'], '']
        lines += ['## 可能要做的事', '', '下列已完成记录和条件计划受上方当前优先级约束；停止变体不因保留待办而重开。没有执行证据不能勾选，复核完成不表示实现或性能通过。', '']
        for action in item['next_steps']:
            done = action.get('status') == 'DONE'
            if done and not action.get('evidence'):
                raise ValueError(f"{cid}/{action['id']}: a completed action requires evidence.")
            mark = 'x' if done else ' '
            refs = ' '.join(f'[记录{i+1}]({path})' for i, path in enumerate(action.get('evidence', [])))
            lines.append(f"- [{mark}] **{action['id']} · {action['title']}**：{action['description']} 接受/停止条件：{action['decision_rule']}" + (f' {refs}' if refs else ''))
        lines += ['', '## 证据入口', '']
        for evidence in item['evidence']:
            lines.append(f"- [{evidence['label']}]({evidence['path']})：{evidence['scope']}")
        lines += ['', '## 进展与纠正记录', '']
        related = [e for e in events if cid in e['candidates']]
        for event in related:
            lines += [f"### {event['date']} · {event['event_id']}", '',
                      f"类型：{event['kind']}；依据：{event['basis']}", '', event['decision_delta'], '']
        outputs[ROOT / 'progress' / f'{cid}.md'] = '\n'.join(lines).rstrip() + '\n'

    counts = Counter(c['review_code'] for c in candidates)
    implementation_count = sum(c['implemented'] for c in candidates)
    correctness_count = sum(c['correctness_code'] == 'PASS' for c in candidates)
    performance_count = sum(c['performance_status'] != 'NOT_MEASURED' for c in candidates)
    independent_count = sum(c['independent_review_code'] == 'COMPLETED' for c in candidates)
    index = ['# 算子优化逐条进展', '',
             '这是正在逐条讨论和复核的工作看板。历史 v180 分析归档保留；本看板不把候选登记、报告成稿或重复讲解当作实现完成。', '',
             f"当前记录：{register['updated_at']}（Asia/Shanghai）。", '',
             f"- 42/42 个既有候选族已建立独立记录入口；这是记录覆盖，不是优化验证完成率。",
             f"- 已记录子方案讨论、复核或已有证据引入：{len(candidates)-counts['HISTORICAL_ANALYSIS_PENDING_CURRENT_REVIEW']}/42。范围见各页；引入已有结果不是本轮新增实验，子方案不关闭整个候选族。",
             f"- 其余 {counts['HISTORICAL_ANALYSIS_PENDING_CURRENT_REVIEW']}/42 的详细技术页保留历史条件记录；当前42族行动筛选见优先级页，不自动重开逐条调查。",
             f'- 本台账已登记实现/试验的族：{implementation_count}/42；有具体变体限定正确性通过：{correctness_count}/42；有同条件性能结论（含无已建立收益）：{performance_count}/42；已完成独立复核：{independent_count}/42。',
             '- 上述是已登记证据计数，包含外部已有实验的引入，不是全量实现/全族验收率。native局部收益、真实推理收益和失败分别记录。', '',
             '## 从这里看', '',
             f"- 当前讨论：[{focus['candidate']} · {focus['variant']}]({focus['candidate']}.md)。",
             '- 更新规则：[记录规范](PROCESS.md)；新记录使用[模板](TEMPLATE.md)。',
             '- 所有进展：[变更记录](CHANGELOG.md)；结构化状态：[candidates.json](candidates.json)；追加事件：[events.jsonl](events.jsonl)。',
             '- 已登记 F01 Task 的机器恢复入口：[Task README](tasks/f01-early-y/README.md)和[恢复包](tasks/f01-early-y/resume-pack.json)。转向下一候选不代表关闭 F01。', '',
             '## 候选看板', '', '|编号|对象|优化类型|本轮状态|下一步|', '|---|---|---|---|---|']
    for c in candidates:
        pending = next((a for a in c['next_steps'] if a.get('status') != 'DONE'), None)
        next_title = pending['title'] if pending else '无已登记待办；不据此关闭候选'
        priority_group = c.get('current_priority', {}).get('group')
        current_actions = {
            'QUALIFIED_BASELINE': '保留组合；采用入口时验资格',
            'STOP_UNCHANGED': '停止不变版；新机制才重开',
            'STOP_CURRENT_WITH_CONDITIONAL_RESERVE': '停止当前版；其他目标条件留档',
            'FRONTEND_RESERVE': '仅核三项组合后增量',
            'LONG_OR_CAPACITY': '仅长行/容量条件准入',
            'DEFERRED_CURRENT_LOAD': '当前不独立开发；保留条件',
        }
        next_title = current_actions.get(priority_group, next_title)
        index.append(f"|[{c['id']}]({c['id']}.md)|{c['name']}|{c['kind']}|{c['review_status']}|{next_title}|")
    priority_plan = register.get('priority_plan')
    if priority_plan:
        cp = priority_plan['counts']
        index[5:5] = ['', '## 优先级与当前结果入口', '',
                      f"**[当前优先级与证据](../PRIORITY-PLAN.md)**：已验证组合{cp['qualified_baseline_changes']}项改动；具体算法候选{cp['algorithm_candidates']}项，联合机制待定价{cp['joint_pricing_mechanisms']}项；前端后备{cp['frontend_reserves']}项，长行/容量后备{cp['long_or_capacity_reserves']}项。停止的具体变体单列，历史11项不是当前开发顺序。", '',
                      register.get('current_snapshot', {}).get('current_action_summary', '历史排序见优先级页；当前结果按追加证据解释。'), '',
                      '[当前分层与42族行动](priority-plan.json) · [本次优先级复核](evidence/priority-review-20261106.json) · [冻结运行结果](evidence/runtime-results-through-20261106.json)', '']
    index += ['', '## 读表时的边界', '',
              '42 个是候选族，包含融合、拆分、复用、状态重组和调度。不是 42 个新增融合算子。相互重叠的方案不能重复计算收益。', '',
              '源码逻辑成立、当前二进制实际执行、可实现性、数值正确性、性能收益分别记录。旧 166 时序、新 171 图回放、不同采集批次的工作量不能混成同一次实验。', '',
              '本次只更新分析与计划，没有采集新 profile、修改服务器或启动独立子代理。原始 profile 不上传。后续执行仍遵循当时的任务范围。', '']
    outputs[ROOT / 'progress/README.md'] = '\n'.join(index)
    changelog = ['# 逐条进展变更记录', '',
                 '由 events.jsonl 生成；新的纠正通过追加事件记录。Git 提交保存每次记录变更，事件不替代技术证据。', '']
    for e in reversed(events):
        links = '、'.join(f"[{cid}]({cid}.md)" for cid in e['candidates'])
        changelog += [f"## {e['date']} · {e['event_id']}", '', f"涉及：{links}", '',
                      f"类型：{e['kind']}；依据：{e['basis']}", '', e['decision_delta'], '']
    outputs[ROOT / 'progress/CHANGELOG.md'] = '\n'.join(changelog)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare generated pages without writing.')
    args = parser.parse_args()
    outputs = render()
    stale = [str(path.relative_to(ROOT)) for path, value in outputs.items()
             if not path.exists() or path.read_text() != value]
    if args.check:
        print(json.dumps({'status': 'PASS' if not stale else 'STALE', 'pages': len(outputs), 'stale': stale}, ensure_ascii=False))
        raise SystemExit(bool(stale))
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    print(json.dumps({'status': 'RENDERED', 'pages': len(outputs), 'changed': len(stale)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

