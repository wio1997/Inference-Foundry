#!/usr/bin/env python3
"""为算子开发提供确定性的 Task/Loop/Run 状态管理。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TASK_FILES = (
    "README.md",
    "index.json",
    "status.md",
    "cases.json",
    "baselines.json",
    "knowledge-state.json",
    "knowledge-events.jsonl",
    "task-events.jsonl",
    "loop-index.jsonl",
    "resume-pack.json",
)
LOOP_FILES = (
    "status.md",
    "goal.md",
    "state.json",
    "hypotheses.md",
    "decision.json",
    "report.md",
    "journal.md",
    "cases.jsonl",
    "measurements.jsonl",
    "commands.jsonl",
    "artifacts.jsonl",
    "comparisons.jsonl",
    "knowledge-delta.json",
)
LOOP_MODES = (
    "spec",
    "design",
    "implementation",
    "correctness",
    "baseline",
    "optimization",
    "integration",
)
VERDICTS = ("accepted", "rejected", "inconclusive", "pivoted")
KNOWLEDGE_KINDS = (
    "fact",
    "constraint",
    "hypothesis",
    "strategy",
    "exclusion",
    "risk",
)
KNOWLEDGE_STATUSES = (
    "candidate",
    "accepted",
    "limited",
    "contradicted",
    "deprecated",
)
CONFIDENCE_LEVELS = ("low", "medium", "high")
DEFAULT_CONTEXT_LIMITS = {"knowledge": 20, "loops": 5, "cases": 12}


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 自动生成的固定界面文字改为中文。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置参数"
        self._optionals.title = "可选参数"
        for action in self._actions:
            if action.dest == "help":
                action.help = "显示帮助信息并退出"

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法：", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：", 1)

    def error(self, message: str) -> None:
        translations = (
            ("the following arguments are required:", "缺少必需参数："),
            ("invalid choice:", "选项值无效："),
            ("unrecognized arguments:", "无法识别参数："),
            ("expected one argument", "需要一个参数值"),
            ("argument ", "参数 "),
        )
        for source, target in translations:
            message = message.replace(source, target)
        if " (choose from " in message:
            message = message.replace(" (choose from ", "（可选值：")
            if message.endswith(")"):
                message = f"{message[:-1]}）"
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 参数错误：{message}\n")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fail(message: str) -> None:
    raise SystemExit(f"错误：{message}")


def check_id(value: str, label: str) -> str:
    if not ID_RE.fullmatch(value):
        fail(f"{label} 必须符合 {ID_RE.pattern}：{value!r}")
    return value


def resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"缺少 JSON 文件：{path}")
    except json.JSONDecodeError as exc:
        fail(f"JSON 格式无效：{path}：{exc}")
    if not isinstance(value, dict):
        fail(f"文件内容应为 JSON 对象：{path}")
    return value


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"JSONL 格式无效 {path}:{line_no}：{exc}")
        if not isinstance(value, dict):
            fail(f"JSONL 每行必须是对象 {path}:{line_no}")
        values.append(value)
    return values


def parse_json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        fail(f"{label} 不是有效 JSON：{exc}")
    if not isinstance(parsed, dict):
        fail(f"{label} 必须是 JSON 对象")
    return parsed


def parse_shape(value: str) -> tuple[list[Any], int | None]:
    try:
        shape = json.loads(value)
    except json.JSONDecodeError as exc:
        fail(f"--shape 不是有效 JSON 数组：{exc}")
    if not isinstance(shape, list) or not shape:
        fail("--shape 必须是非空 JSON 数组")
    numel: int | None = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            numel = None
            break
        numel *= dim
    return shape, numel


def append_task_event(task_dir: Path, event: str, payload: dict[str, Any]) -> None:
    append_jsonl(task_dir / "task-events.jsonl", {
        "event": event,
        "payload": payload,
        "recorded_at": now(),
    })


def append_journal(loop_dir: Path, message: str) -> None:
    with (loop_dir / "journal.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n- `{now()}` {message}\n")


def task_dir_from_loop(loop_dir: Path) -> Path:
    if loop_dir.parent.name != "loops":
        fail(f"Loop 目录必须位于 <task>/loops/ 下：{loop_dir}")
    task_dir = loop_dir.parent.parent
    if not (task_dir / "index.json").is_file():
        fail(f"未找到该 Loop 所属 Task 的 index.json：{loop_dir}")
    return task_dir


def find_loop(task_dir: Path, loop_id: str) -> Path:
    loop_dir = task_dir / "loops" / loop_id
    if not loop_dir.is_dir():
        fail(f"Loop 不存在：{loop_dir}")
    return loop_dir


def phase_for_mode(mode: str) -> str:
    return {
        "spec": "DEFINED",
        "design": "DESIGNING",
        "implementation": "IMPLEMENTING",
        "correctness": "VALIDATING",
        "baseline": "BASELINING",
        "optimization": "OPTIMIZING",
        "integration": "INTEGRATING",
    }[mode]


def render_loop_status(loop_dir: Path) -> None:
    state = read_json(loop_dir / "state.json")
    decision = read_json(loop_dir / "decision.json")
    knowledge_delta = read_json(loop_dir / "knowledge-delta.json") if (loop_dir / "knowledge-delta.json").is_file() else {"changes": [], "baseline_updates": []}
    text = f"""# Loop 状态

- Loop：`{state['loop_id']}`
- 标题：{state['title']}
- 模式：`{state['mode']}`
- 目标用例：{', '.join(f'`{item}`' for item in state.get('cases', [])) or 'Task 级'}
- 执行 Skill：`{state.get('execution_skill') or 'NONE'}`
- 状态：`{state['status']}`
- 结论：`{decision['result']}`
- 下一步：{state.get('next_action') or 'NONE'}
- 更新时间：`{state['updated_at']}`

## Run 记录

{chr(10).join(f'- `{run_id}`' for run_id in state.get('run_order', [])) or '- 暂无'}

## 阻塞项

{chr(10).join(f'- {item}' for item in state.get('blockers', [])) or '- 暂无'}

## 待归约知识

- 知识变化：`{len(knowledge_delta.get('changes', []))}`
- 基线变化：`{len(knowledge_delta.get('baseline_updates', []))}`
"""
    atomic_text(loop_dir / "status.md", text)


def loop_summary(loop_dir: Path) -> dict[str, Any]:
    state = read_json(loop_dir / "state.json")
    decision = read_json(loop_dir / "decision.json")
    return {
        "loop_id": state["loop_id"],
        "title": state["title"],
        "mode": state["mode"],
        "status": state["status"],
        "verdict": decision["result"],
        "reason": decision.get("reason", ""),
        "next_action": state.get("next_action", ""),
        "cases": state.get("cases", []),
        "execution_skill": state.get("execution_skill"),
        "updated_at": state.get("updated_at"),
    }


def load_cases(task_dir: Path) -> dict[str, Any]:
    return read_json(task_dir / "cases.json")


def load_baselines(task_dir: Path) -> dict[str, Any]:
    return read_json(task_dir / "baselines.json")


def load_knowledge(task_dir: Path) -> dict[str, Any]:
    return read_json(task_dir / "knowledge-state.json")


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 0)


def select_cases(
    cases_doc: dict[str, Any],
    limit: int,
    dtype: str | None = None,
    group: str | None = None,
) -> list[dict[str, Any]]:
    values = list(cases_doc.get("cases", {}).values())
    if dtype:
        values = [item for item in values if item.get("dtype") == dtype]
    if group:
        values = [item for item in values if item.get("group") == group]
    values.sort(key=lambda item: (not item.get("protected", False), -float(item.get("weight", 1.0)), item.get("id", "")))
    return values[:limit]


def select_knowledge(knowledge_doc: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    values = [
        item for item in knowledge_doc.get("entries", {}).values()
        if item.get("status") not in ("deprecated",)
    ]
    status_rank = {"accepted": 5, "limited": 4, "contradicted": 3, "candidate": 2, "deprecated": 1}
    values.sort(key=lambda item: (
        -status_rank.get(item.get("status", ""), 0),
        -confidence_rank(item.get("confidence", "")),
        item.get("id", ""),
    ))
    return values[:limit]


def loop_index_items(task_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(task_dir / "loop-index.jsonl")


def render_resume_pack(
    task_dir: Path,
    max_knowledge: int = DEFAULT_CONTEXT_LIMITS["knowledge"],
    max_loops: int = DEFAULT_CONTEXT_LIMITS["loops"],
    max_cases: int = DEFAULT_CONTEXT_LIMITS["cases"],
    dtype: str | None = None,
    case_group: str | None = None,
) -> dict[str, Any]:
    index = read_json(task_dir / "index.json")
    cases_doc = load_cases(task_dir)
    baselines_doc = load_baselines(task_dir)
    knowledge_doc = load_knowledge(task_dir)
    selected_cases = select_cases(cases_doc, max_cases, dtype=dtype, group=case_group)
    selected_case_ids = {item["id"] for item in selected_cases}
    selected_baselines = [
        item for case_id, item in baselines_doc.get("cases", {}).items()
        if case_id in selected_case_ids
    ]
    recent_loops = loop_index_items(task_dir)[-max_loops:] if max_loops else []
    read_first = [
        str(task_dir / "README.md"),
        str(task_dir / "index.json"),
        str(task_dir / "status.md"),
        str(task_dir / "knowledge-state.json"),
    ]
    active_summary = None
    if index.get("active_loop"):
        loop_dir = find_loop(task_dir, index["active_loop"])
        active_summary = loop_summary(loop_dir)
        read_first.extend(str(loop_dir / name) for name in (
            "goal.md",
            "state.json",
            "hypotheses.md",
            "decision.json",
            "status.md",
            "knowledge-delta.json",
        ))
    pack = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now(),
        "context_policy": {
            "max_knowledge": max_knowledge,
            "max_loops": max_loops,
            "max_cases": max_cases,
            "dtype_filter": dtype,
            "case_group_filter": case_group,
            "raw_runs_loaded": False,
        },
        "task": {
            "task_dir": str(task_dir),
            "task_id": index["task_id"],
            "title": index["title"],
            "goal": index.get("goal"),
            "operator": index["operator"],
            "framework": index["framework"],
            "source_root": index["source_root"],
            "status": index["status"],
            "phase": index["phase"],
            "evidence_level": index.get("evidence_level", "E0_SPEC_ONLY"),
            "active_loop": index.get("active_loop"),
            "next_action": index.get("next_action"),
            "blockers": index.get("blockers", []),
            "skill_route": index.get("skill_route", {}),
        },
        "case_summary": {
            "total": len(cases_doc.get("cases", {})),
            "groups": {name: len(ids) for name, ids in cases_doc.get("groups", {}).items()},
            "selected": selected_cases,
        },
        "baseline_summary": selected_baselines,
        "knowledge_summary": {
            "total": len(knowledge_doc.get("entries", {})),
            "selected": select_knowledge(knowledge_doc, max_knowledge),
        },
        "recent_loops": recent_loops,
        "active_loop_summary": active_summary,
        "read_first": read_first,
        "history_indexes": {
            "loops": str(task_dir / "loop-index.jsonl"),
            "knowledge": str(task_dir / "knowledge-events.jsonl"),
            "task_events": str(task_dir / "task-events.jsonl"),
        },
    }
    atomic_json(task_dir / "resume-pack.json", pack)
    return pack


def render_task_status(task_dir: Path) -> None:
    index = read_json(task_dir / "index.json")
    cases_doc = load_cases(task_dir)
    knowledge_doc = load_knowledge(task_dir)
    rows = []
    loop_order = index.get("loop_order", [])
    visible_loop_ids = loop_order[-10:]
    for loop_id in visible_loop_ids:
        loop_dir = task_dir / "loops" / loop_id
        if not loop_dir.is_dir():
            rows.append(f"| `{loop_id}` | MISSING | UNKNOWN | Loop 目录缺失 |")
            continue
        item = loop_summary(loop_dir)
        summary = (item["reason"] or item["next_action"] or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{loop_id}` | `{item['status']}` | `{item['verdict']}` | {summary} |")
    text = f"""# Task 状态

- Task：`{index['task_id']}`
- 标题：{index['title']}
- 算子：`{index['operator']}`
- 开发框架：`{index['framework']}`
- 状态：`{index['status']}`
- 阶段：`{index['phase']}`
- 活动 Loop：`{index.get('active_loop') or 'NONE'}`
- 已接受基线：`{index.get('accepted_baseline') or 'NONE'}`
- 证据成熟度：`{index.get('evidence_level', 'E0_SPEC_ONLY')}`
- 用例数：`{len(cases_doc.get('cases', {}))}`
- 当前知识条目：`{len(knowledge_doc.get('entries', {}))}`
- 下一步：{index.get('next_action') or 'NONE'}
- 更新时间：`{index['updated_at']}`

## 最近 Loop

共 `{len(loop_order)}` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
{chr(10).join(rows) or '| - | - | - | 暂无 Loop |'}

## 阻塞项

{chr(10).join(f'- {item}' for item in index.get('blockers', [])) or '- 暂无'}
"""
    atomic_text(task_dir / "status.md", text)


def cmd_init_task(args: argparse.Namespace) -> None:
    task_id = check_id(args.task_id, "task-id")
    tasks_root = resolve(args.tasks_root)
    task_dir = tasks_root / task_id
    if task_dir.exists():
        fail(f"Task 已存在：{task_dir}")
    source_root = resolve(args.source_root)
    created = now()
    for name in ("doc", "loops", "scripts", "reports"):
        (task_dir / name).mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "title": args.title,
        "goal": args.goal,
        "operator": args.operator,
        "framework": args.framework,
        "source_root": str(source_root),
        "status": "ACTIVE",
        "phase": "DEFINED",
        "evidence_level": "E0_SPEC_ONLY",
        "active_loop": None,
        "accepted_baseline": None,
        "next_action": "冻结第一个 Loop",
        "blockers": [],
        "loop_order": [],
        "skill_route": {
            "coordinator": "operator-task-loop",
            "decision": None,
            "execution": None,
        },
        "completion_gate": None,
        "created_at": created,
        "updated_at": created,
    }
    atomic_json(task_dir / "index.json", index)
    atomic_json(task_dir / "cases.json", {
        "schema_version": SCHEMA_VERSION,
        "cases": {},
        "groups": {},
        "updated_at": created,
    })
    atomic_json(task_dir / "baselines.json", {
        "schema_version": SCHEMA_VERSION,
        "cases": {},
        "updated_at": created,
    })
    atomic_json(task_dir / "knowledge-state.json", {
        "schema_version": SCHEMA_VERSION,
        "entries": {},
        "updated_at": created,
    })
    for name in ("knowledge-events.jsonl", "task-events.jsonl", "loop-index.jsonl"):
        atomic_text(task_dir / name, "")
    append_task_event(task_dir, "task_initialized", {
        "task_id": task_id,
        "goal": args.goal,
        "source_root": str(source_root),
    })
    allowed = "\n".join(f"- `{item}`" for item in args.allowed_path) or "- UNKNOWN"
    non_goals = "\n".join(f"- {item}" for item in args.non_goal) or "- 暂无记录"
    readme = f"""# {args.title}

## Task 契约

- Task ID：`{task_id}`
- 算子：`{args.operator}`
- 开发框架：`{args.framework}`
- 源码根目录：`{source_root}`
- 创建时间：`{created}`

## 任务目标

{args.goal}

## 允许修改的路径

{allowed}

## 非目标

{non_goals}

## 正确性与性能契约

实现前在此记录已经冻结的算子语义、用例、正确性判据、性能目标、硬件环境和交付边界。

## 恢复入口

执行 `taskctl.py resume --task-dir {task_dir}`，并继续其中记录的下一步动作。
"""
    atomic_text(task_dir / "README.md", readme)
    render_task_status(task_dir)
    render_resume_pack(task_dir)
    print(task_dir)


def cmd_add_case(args: argparse.Namespace) -> None:
    task_dir = resolve(args.task_dir)
    index = read_json(task_dir / "index.json")
    if index.get("status") == "COMPLETED":
        fail("已完成的 Task 不能增加用例")
    case_id = check_id(args.case_id, "case-id")
    cases_doc = load_cases(task_dir)
    if case_id in cases_doc.get("cases", {}):
        fail(f"用例已存在：{case_id}")
    shape, numel = parse_shape(args.shape)
    attributes = parse_json_object(args.attributes_json, "--attributes-json")
    recorded = now()
    case = {
        "id": case_id,
        "shape": shape,
        "numel": numel,
        "dtype": args.dtype,
        "layout": args.layout,
        "attributes": attributes,
        "weight": args.weight,
        "group": args.group,
        "protected": args.protected == "yes",
        "reason": args.reason,
        "created_at": recorded,
        "updated_at": recorded,
    }
    cases_doc.setdefault("cases", {})[case_id] = case
    cases_doc.setdefault("groups", {}).setdefault(args.group, []).append(case_id)
    cases_doc["updated_at"] = recorded
    atomic_json(task_dir / "cases.json", cases_doc)
    append_task_event(task_dir, "case_added", case)
    index["updated_at"] = recorded
    atomic_json(task_dir / "index.json", index)
    render_task_status(task_dir)
    render_resume_pack(task_dir)
    print(json.dumps(case, ensure_ascii=False, indent=2))


def next_loop_id(index: dict[str, Any]) -> str:
    numbers = []
    for item in index.get("loop_order", []):
        match = re.fullmatch(r"loop-(\d+)", item)
        if match:
            numbers.append(int(match.group(1)))
    return f"loop-{max(numbers, default=0) + 1:03d}"


def cmd_new_loop(args: argparse.Namespace) -> None:
    task_dir = resolve(args.task_dir)
    index = read_json(task_dir / "index.json")
    if index["status"] == "COMPLETED":
        fail("已完成的 Task 不能创建新 Loop")
    if index.get("active_loop"):
        fail(f"Task 已存在活动 Loop：{index['active_loop']}")
    loop_id = check_id(args.loop_id or next_loop_id(index), "loop-id")
    loop_dir = task_dir / "loops" / loop_id
    if loop_dir.exists():
        fail(f"Loop 已存在：{loop_dir}")
    created = now()
    cases_doc = load_cases(task_dir)
    for case_id in args.case:
        if case_id not in cases_doc.get("cases", {}):
            fail(f"Loop 引用了未登记用例：{case_id}；先执行 add-case")
    (loop_dir / "doc").mkdir(parents=True)
    (loop_dir / "runs").mkdir()
    state = {
        "schema_version": SCHEMA_VERSION,
        "loop_id": loop_id,
        "title": args.title,
        "mode": args.mode,
        "cases": args.case,
        "execution_skill": args.execution_skill,
        "status": "FROZEN",
        "next_action": args.next_action,
        "blockers": [],
        "run_order": [],
        "created_at": created,
        "updated_at": created,
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "loop_id": loop_id,
        "result": "PENDING",
        "reason": "",
        "evidence_paths": [],
        "decided_at": None,
    }
    atomic_json(loop_dir / "state.json", state)
    atomic_json(loop_dir / "decision.json", decision)
    atomic_json(loop_dir / "knowledge-delta.json", {
        "schema_version": SCHEMA_VERSION,
        "loop_id": loop_id,
        "changes": [],
        "baseline_updates": [],
        "updated_at": created,
    })
    allowed = "\n".join(f"- `{item}`" for item in args.allowed_path) or "- UNKNOWN"
    cases = "\n".join(f"- {item}" for item in args.case) or "- UNKNOWN"
    goal = f"""# Loop 目标：{args.title}

- Loop ID：`{loop_id}`
- 模式：`{args.mode}`
- 所属 Task：`{index['task_id']}`
- 执行 Skill：`{args.execution_skill or 'NONE'}`
- 冻结时间：`{created}`

## 目标/假设

{args.hypothesis}

## 允许修改的路径

{allowed}

## 目标用例

{cases}

## 接受条件

{args.success}

## 否定条件

{args.falsification}

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
"""
    atomic_text(loop_dir / "goal.md", goal)
    atomic_text(loop_dir / "hypotheses.md", f"# 假设记录\n\n## H001 - ACTIVE\n\n{args.hypothesis}\n")
    atomic_text(loop_dir / "journal.md", f"# 执行日志\n\n- `{created}` Loop 已冻结。下一步：{args.next_action}\n")
    atomic_text(loop_dir / "report.md", f"# Loop 报告：{loop_id}\n\n尚未作出结论。\n")
    for name in ("cases.jsonl", "measurements.jsonl", "commands.jsonl", "artifacts.jsonl", "comparisons.jsonl"):
        atomic_text(loop_dir / name, "")
    for case in args.case:
        append_jsonl(loop_dir / "cases.jsonl", {"case_key": case, "recorded_at": created})
    index["active_loop"] = loop_id
    index["phase"] = phase_for_mode(args.mode)
    index["next_action"] = args.next_action
    index["loop_order"].append(loop_id)
    index["skill_route"]["decision"] = "kernel-optimization" if args.mode in ("baseline", "optimization") else None
    index["skill_route"]["execution"] = args.execution_skill
    index["updated_at"] = created
    atomic_json(task_dir / "index.json", index)
    render_loop_status(loop_dir)
    render_task_status(task_dir)
    append_task_event(task_dir, "loop_frozen", {
        "loop_id": loop_id,
        "mode": args.mode,
        "cases": args.case,
        "execution_skill": args.execution_skill,
    })
    render_resume_pack(task_dir)
    print(loop_dir)


def cmd_new_run(args: argparse.Namespace) -> None:
    loop_dir = resolve(args.loop_dir)
    state = read_json(loop_dir / "state.json")
    if state["status"] in ("ACCEPTED", "REJECTED", "PIVOTED"):
        fail(f"不能向已结束的 Loop 添加 Run：{state['status']}")
    case_key = check_id(args.case_key, "case-key")
    task_dir = task_dir_from_loop(loop_dir)
    cases_doc = load_cases(task_dir)
    case_snapshot = cases_doc.get("cases", {}).get(case_key)
    evidence_kinds = ("test", "simulation", "benchmark", "profile")
    if args.kind in evidence_kinds and case_snapshot is None:
        fail(f"{args.kind} Run 必须引用已登记用例：{case_key}")
    if state.get("cases") and case_key not in state["cases"]:
        fail(f"Run 用例不在冻结的 Loop 用例范围内：{case_key}")
    run_id = check_id(args.run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "run-id")
    run_dir = loop_dir / "runs" / case_key / run_id
    if run_dir.exists():
        fail(f"Run 已存在：{run_dir}")
    run_dir.mkdir(parents=True)
    created = now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "loop_id": state["loop_id"],
        "case_key": case_key,
        "case_snapshot": case_snapshot,
        "kind": args.kind,
        "command": args.command,
        "code_ref": args.code_ref,
        "working_directory": str(resolve(args.working_directory)) if args.working_directory else None,
        "status": "PLANNED",
        "created_at": created,
        "updated_at": created,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_text(run_dir / "run.log", "")
    run_rel = str(run_dir.relative_to(loop_dir))
    state["status"] = "RUNNING"
    state["run_order"].append(run_rel)
    state["next_action"] = f"执行 Run {run_id}：{args.command}"
    state["updated_at"] = created
    atomic_json(loop_dir / "state.json", state)
    append_jsonl(loop_dir / "commands.jsonl", {
        "run_id": run_id,
        "case_key": case_key,
        "kind": args.kind,
        "command": args.command,
        "code_ref": args.code_ref,
        "run_dir": str(run_dir),
        "recorded_at": created,
    })
    append_journal(loop_dir, f"为用例 `{case_key}` 创建 Run `{run_id}`（{args.kind}）。")
    render_loop_status(loop_dir)
    render_task_status(task_dir)
    append_task_event(task_dir, "run_created", {
        "loop_id": state["loop_id"],
        "run_id": run_id,
        "case_key": case_key,
        "kind": args.kind,
    })
    render_resume_pack(task_dir)
    print(run_dir)


def cmd_finish_run(args: argparse.Namespace) -> None:
    run_dir = resolve(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    loop_dir = run_dir.parents[2]
    state = read_json(loop_dir / "state.json")
    if manifest["status"] not in ("PLANNED", "EXECUTING"):
        fail(f"Run 已结束：{manifest['status']}")
    if args.metric_value is not None and not args.metric_name:
        fail("使用 --metric-value 时必须同时提供 --metric-name")
    finished = now()
    manifest["status"] = "RECORDED"
    manifest["updated_at"] = finished
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": args.status,
        "correctness": args.correctness,
        "summary": args.summary,
        "metric": None,
        "artifacts": [str(resolve(item)) for item in args.artifact],
        "finished_at": finished,
    }
    if args.metric_name:
        result["metric"] = {
            "name": args.metric_name,
            "value": args.metric_value,
            "unit": args.metric_unit or "",
        }
        append_jsonl(loop_dir / "measurements.jsonl", {
            "run_id": manifest["run_id"],
            "case_key": manifest["case_key"],
            "case_snapshot": manifest.get("case_snapshot"),
            **result["metric"],
            "correctness": args.correctness,
            "recorded_at": finished,
        })
    for artifact in result["artifacts"]:
        append_jsonl(loop_dir / "artifacts.jsonl", {
            "run_id": manifest["run_id"],
            "path": artifact,
            "recorded_at": finished,
        })
    atomic_json(manifest_path, manifest)
    atomic_json(run_dir / "result.json", result)
    state["status"] = "EVALUATING"
    state["next_action"] = f"审查 Run {manifest['run_id']} 的证据，并判断是否需要更多 Run"
    state["updated_at"] = finished
    atomic_json(loop_dir / "state.json", state)
    append_journal(loop_dir, f"Run `{manifest['run_id']}` 记录为 `{args.status}`；正确性为 `{args.correctness}`。{args.summary}")
    render_loop_status(loop_dir)
    task_dir = task_dir_from_loop(loop_dir)
    render_task_status(task_dir)
    append_task_event(task_dir, "run_recorded", {
        "loop_id": manifest["loop_id"],
        "run_id": manifest["run_id"],
        "case_key": manifest["case_key"],
        "status": args.status,
        "correctness": args.correctness,
        "metric": result.get("metric"),
    })
    render_resume_pack(task_dir)
    print(run_dir / "result.json")


def cmd_compare(args: argparse.Namespace) -> None:
    loop_dir = resolve(args.loop_dir)
    before = resolve(args.before_run)
    after = resolve(args.after_run)
    for run_dir in (before, after):
        if not (run_dir / "result.json").is_file():
            fail(f"缺少已完成 Run 的结果文件：{run_dir / 'result.json'}")
    before_manifest = read_json(before / "manifest.json")
    after_manifest = read_json(after / "manifest.json")
    before_result = read_json(before / "result.json")
    after_result = read_json(after / "result.json")
    if args.comparable == "yes":
        if before_manifest.get("case_key") != after_manifest.get("case_key"):
            fail("同条件比较要求 before/after Run 使用相同 case-key")
        before_metric = before_result.get("metric") or {}
        after_metric = after_result.get("metric") or {}
        if not before_metric or not after_metric:
            fail("同条件性能比较要求两个 Run 都包含性能指标")
        if before_metric.get("name") != after_metric.get("name") or before_metric.get("unit") != after_metric.get("unit"):
            fail("同条件比较要求性能指标名称和单位一致")
    recorded = now()
    value = {
        "before_run": str(before),
        "after_run": str(after),
        "comparable": args.comparable == "yes",
        "summary": args.summary,
        "recorded_at": recorded,
    }
    append_jsonl(loop_dir / "comparisons.jsonl", value)
    append_journal(loop_dir, f"已记录对比（`comparable={args.comparable}`）：{args.summary}")
    task_dir = task_dir_from_loop(loop_dir)
    append_task_event(task_dir, "comparison_recorded", {
        "loop_id": read_json(loop_dir / "state.json")["loop_id"],
        **value,
    })
    render_resume_pack(task_dir)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def completed_run_results(loop_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((loop_dir / "runs").glob("*/*/result.json")):
        value = read_json(path)
        value["_path"] = str(path)
        results.append(value)
    return results


def cmd_stage_knowledge(args: argparse.Namespace) -> None:
    loop_dir = resolve(args.loop_dir)
    state = read_json(loop_dir / "state.json")
    if state["status"] in ("ACCEPTED", "REJECTED", "PIVOTED"):
        fail("已结束的 Loop 不能增加知识变化")
    knowledge_id = check_id(args.knowledge_id, "knowledge-id")
    delta_path = loop_dir / "knowledge-delta.json"
    delta = read_json(delta_path)
    if any(item.get("id") == knowledge_id for item in delta.get("changes", [])):
        fail(f"当前 Loop 已存在知识变化：{knowledge_id}")
    scope = parse_json_object(args.scope_json, "--scope-json")
    evidence_paths = [str(resolve(item)) for item in args.evidence_path]
    if not evidence_paths:
        fail("知识变化至少需要一个 --evidence-path")
    for path in evidence_paths:
        if not Path(path).exists():
            fail(f"知识证据路径不存在：{path}")
    replaces = list(dict.fromkeys(check_id(item, "replaces") for item in args.replaces))
    if knowledge_id in replaces:
        fail(f"知识变化不能替代自身：{knowledge_id}")
    task_dir = task_dir_from_loop(loop_dir)
    existing_ids = set(load_knowledge(task_dir).get("entries", {}))
    staged_ids = {
        item.get("id")
        for item in read_json(delta_path).get("changes", [])
        if item.get("id")
    }
    missing_replacements = [
        item for item in replaces
        if item not in existing_ids and item not in staged_ids
    ]
    if missing_replacements:
        fail(f"被替代知识不存在：{', '.join(missing_replacements)}")
    change = {
        "id": knowledge_id,
        "kind": args.kind,
        "statement": args.statement,
        "status": args.status,
        "confidence": args.confidence,
        "scope": scope,
        "supporting_evidence": evidence_paths if args.status not in ("contradicted",) else [],
        "contradicting_evidence": evidence_paths if args.status == "contradicted" else [],
        "replaces": replaces,
        "global_candidate": args.global_candidate == "yes",
        "source_loop": state["loop_id"],
        "staged_at": now(),
    }
    delta.setdefault("changes", []).append(change)
    delta["updated_at"] = now()
    atomic_json(delta_path, delta)
    append_journal(loop_dir, f"暂存知识变化 `{knowledge_id}`：{args.statement}")
    render_loop_status(loop_dir)
    render_resume_pack(task_dir_from_loop(loop_dir))
    print(json.dumps(change, ensure_ascii=False, indent=2))


def cmd_stage_baseline(args: argparse.Namespace) -> None:
    loop_dir = resolve(args.loop_dir)
    state = read_json(loop_dir / "state.json")
    if state["status"] in ("ACCEPTED", "REJECTED", "PIVOTED"):
        fail("已结束的 Loop 不能增加基线变化")
    run_dir = resolve(args.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    result = read_json(run_dir / "result.json")
    if manifest.get("loop_id") != state["loop_id"]:
        fail("基线 Run 不属于当前 Loop")
    if result.get("status") != "pass" or result.get("correctness") != "pass":
        fail("基线 Run 必须同时满足 status=pass 和 correctness=pass")
    if not result.get("metric"):
        fail("基线 Run 缺少性能指标")
    task_dir = task_dir_from_loop(loop_dir)
    case_id = manifest["case_key"]
    if case_id not in load_cases(task_dir).get("cases", {}):
        fail(f"基线 Run 引用了未登记用例：{case_id}")
    update = {
        "case_id": case_id,
        "run_dir": str(run_dir),
        "code_ref": manifest.get("code_ref"),
        "metric": result["metric"],
        "reason": args.reason,
        "source_loop": state["loop_id"],
        "staged_at": now(),
    }
    delta_path = loop_dir / "knowledge-delta.json"
    delta = read_json(delta_path)
    updates = [item for item in delta.get("baseline_updates", []) if item.get("case_id") != case_id]
    updates.append(update)
    delta["baseline_updates"] = updates
    delta["updated_at"] = now()
    atomic_json(delta_path, delta)
    append_journal(loop_dir, f"暂存用例 `{case_id}` 的基线变化，来源 Run `{manifest['run_id']}`。")
    render_loop_status(loop_dir)
    render_resume_pack(task_dir)
    print(json.dumps(update, ensure_ascii=False, indent=2))


def apply_knowledge_delta(task_dir: Path, loop_dir: Path, verdict_result: str) -> dict[str, int]:
    delta = read_json(loop_dir / "knowledge-delta.json")
    knowledge = load_knowledge(task_dir)
    baselines = load_baselines(task_dir)
    loop_id = read_json(loop_dir / "state.json")["loop_id"]
    applied_knowledge = 0
    deprecated_knowledge = 0
    applied_baselines = 0
    global_candidates_path = task_dir / "reports" / "knowledge-candidates.jsonl"
    for change in delta.get("changes", []):
        value = dict(change)
        value["verdict"] = verdict_result.upper()
        value["updated_at"] = now()
        knowledge.setdefault("entries", {})[value["id"]] = value
        append_jsonl(task_dir / "knowledge-events.jsonl", {
            "event": "knowledge_updated",
            "knowledge": value,
            "recorded_at": now(),
        })
        if value.get("global_candidate"):
            append_jsonl(global_candidates_path, value)
        applied_knowledge += 1
    for change in delta.get("changes", []):
        replacement_id = change["id"]
        for replaced_id in change.get("replaces", []):
            if replaced_id == replacement_id:
                fail(f"知识变化不能替代自身：{replacement_id}")
            old = knowledge.setdefault("entries", {}).get(replaced_id)
            if old is None:
                fail(f"被替代知识不存在：{replaced_id}")
            old = dict(old)
            old["status"] = "deprecated"
            old["replaced_by"] = replacement_id
            old["updated_at"] = now()
            knowledge["entries"][replaced_id] = old
            append_jsonl(task_dir / "knowledge-events.jsonl", {
                "event": "knowledge_deprecated",
                "knowledge_id": replaced_id,
                "replaced_by": replacement_id,
                "source_loop": loop_id,
                "recorded_at": now(),
            })
            deprecated_knowledge += 1
    if verdict_result == "accepted":
        for update in delta.get("baseline_updates", []):
            value = dict(update)
            value["accepted_at"] = now()
            baselines.setdefault("cases", {})[value["case_id"]] = value
            applied_baselines += 1
    knowledge["updated_at"] = now()
    baselines["updated_at"] = now()
    atomic_json(task_dir / "knowledge-state.json", knowledge)
    atomic_json(task_dir / "baselines.json", baselines)
    append_task_event(task_dir, "loop_knowledge_reduced", {
        "loop_id": loop_id,
        "verdict": verdict_result.upper(),
        "knowledge_changes": applied_knowledge,
        "deprecated_knowledge": deprecated_knowledge,
        "baseline_updates": applied_baselines,
    })
    return {
        "knowledge": applied_knowledge,
        "deprecated_knowledge": deprecated_knowledge,
        "baselines": applied_baselines,
    }


def cmd_verdict(args: argparse.Namespace) -> None:
    loop_dir = resolve(args.loop_dir)
    task_dir = task_dir_from_loop(loop_dir)
    state = read_json(loop_dir / "state.json")
    decision = read_json(loop_dir / "decision.json")
    if decision["result"] != "PENDING":
        fail(f"Loop 已有最终结论：{decision['result']}")
    results = completed_run_results(loop_dir)
    if not results:
        fail("至少需要一个已完成的 Run 才能作出 Loop 结论")
    evidence_paths = [str(resolve(item)) for item in args.evidence_path]
    for path in evidence_paths:
        if not Path(path).exists():
            fail(f"结论证据路径不存在：{path}")
    if args.result == "accepted":
        if args.goal_satisfied != "yes":
            fail("accepted 结论要求 --goal-satisfied=yes")
        invalid = [item for item in results if item.get("correctness") in ("fail", "invalid")]
        if invalid:
            fail("存在正确性为 fail/invalid 的 Run，不能接受当前 Loop")
        if not any(item.get("status") == "pass" for item in results):
            fail("accepted 结论至少需要一个状态为 pass 的 Run")
        if state["mode"] == "correctness" and not any(item.get("correctness") == "pass" for item in results):
            fail("正确性 Loop 的 accepted 结论至少需要一个 correctness=pass 的 Run")
        if state["mode"] in ("baseline", "optimization"):
            metric_results = [item for item in results if item.get("metric") and item.get("correctness") == "pass" and item.get("status") == "pass"]
            if not metric_results:
                fail("性能 Loop 的 accepted 结论至少需要一个正确且有效的性能 Run")
        if state["mode"] == "optimization":
            comparable = [item for item in read_jsonl(loop_dir / "comparisons.jsonl") if item.get("comparable")]
            if args.comparability != "valid" or not comparable:
                fail("优化 Loop 的 accepted 结论要求有效的同条件 before/after 对比")
    if args.result == "rejected" and args.goal_satisfied == "yes":
        fail("rejected 结论不能同时声明目标已满足")
    decided = now()
    decision.update({
        "result": args.result.upper(),
        "reason": args.reason,
        "goal_satisfied": args.goal_satisfied,
        "comparability": args.comparability,
        "causal_conclusion": args.causal_conclusion,
        "evidence_paths": evidence_paths,
        "decided_at": decided,
    })
    state["status"] = args.result.upper()
    state["next_action"] = args.next_action
    state["updated_at"] = decided
    atomic_json(loop_dir / "decision.json", decision)
    atomic_json(loop_dir / "state.json", state)
    report = f"""# Loop 报告：{state['loop_id']}

- 结论：`{decision['result']}`
- 决定时间：`{decided}`
- 模式：`{state['mode']}`
- 冻结目标是否满足：`{args.goal_satisfied}`
- 可比性：`{args.comparability}`
- 因果结论：`{args.causal_conclusion}`

## 决定依据

{args.reason}

## 证据

{chr(10).join(f'- `{item}`' for item in decision['evidence_paths']) or '- 参见 `runs/` 下的 Run 结果包。'}

## 下一步

{args.next_action}
"""
    atomic_text(loop_dir / "report.md", report)
    append_journal(loop_dir, f"主控结论为 `{decision['result']}`。{args.reason}")
    index = read_json(task_dir / "index.json")
    if index.get("active_loop") != state["loop_id"]:
        fail(f"Task 的 active_loop 不匹配：{index.get('active_loop')} != {state['loop_id']}")
    index["active_loop"] = None
    index["next_action"] = args.next_action
    if args.update_baseline:
        if args.result != "accepted":
            fail("只有 accepted 结论才能使用 --update-baseline")
        index["accepted_baseline"] = args.update_baseline
    if args.result == "accepted":
        if state["mode"] in ("implementation", "correctness"):
            index["evidence_level"] = "E1_RUNNABLE"
        elif state["mode"] == "baseline":
            index["evidence_level"] = "E2_BENCHMARKED"
        elif state["mode"] == "optimization":
            run_kinds = {
                read_json(path.parent / "manifest.json").get("kind")
                for path in (loop_dir / "runs").glob("*/*/result.json")
            }
            if "profile" in run_kinds and index.get("evidence_level") in ("E1_RUNNABLE", "E2_BENCHMARKED"):
                index["evidence_level"] = "E3_PROFILED"
            elif index.get("evidence_level") in ("E3_PROFILED", "E4_ITERATED"):
                index["evidence_level"] = "E4_ITERATED"
    reduced = apply_knowledge_delta(task_dir, loop_dir, args.result)
    loop_index_value = {
        "loop_id": state["loop_id"],
        "title": state["title"],
        "mode": state["mode"],
        "cases": state.get("cases", []),
        "hypothesis": (loop_dir / "hypotheses.md").read_text(encoding="utf-8").strip().splitlines()[-1],
        "verdict": decision["result"],
        "reason": args.reason,
        "knowledge_changes": reduced["knowledge"],
        "baseline_updates": reduced["baselines"],
        "evidence_paths": evidence_paths,
        "decided_at": decided,
    }
    append_jsonl(task_dir / "loop-index.jsonl", loop_index_value)
    append_task_event(task_dir, "verdict_issued", loop_index_value)
    index["skill_route"]["execution"] = None
    index["skill_route"]["decision"] = None
    index["updated_at"] = decided
    atomic_json(task_dir / "index.json", index)
    render_loop_status(loop_dir)
    render_task_status(task_dir)
    render_resume_pack(task_dir)
    print(loop_dir / "decision.json")


def validate_jsonl(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"缺少文件：{path}")
        return
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"JSONL 格式无效 {path}:{line_no}：{exc}")


def cmd_validate(args: argparse.Namespace) -> None:
    task_dir = resolve(args.task_dir)
    errors: list[str] = []
    for name in TASK_FILES:
        if not (task_dir / name).is_file():
            errors.append(f"缺少文件：{task_dir / name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    index = read_json(task_dir / "index.json")
    if index.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"不支持的 Task 数据版本：{index.get('schema_version')}")
    if index.get("active_loop") and index["active_loop"] not in index.get("loop_order", []):
        errors.append("active_loop 未出现在 loop_order 中")
    cases_doc = load_cases(task_dir)
    case_ids = set(cases_doc.get("cases", {}))
    for group, members in cases_doc.get("groups", {}).items():
        missing_members = [case_id for case_id in members if case_id not in case_ids]
        if missing_members:
            errors.append(f"用例组 {group} 引用了不存在的用例：{missing_members}")
    baselines = load_baselines(task_dir)
    for case_id, baseline in baselines.get("cases", {}).items():
        if case_id not in case_ids:
            errors.append(f"基线引用了不存在的用例：{case_id}")
        if not Path(baseline.get("run_dir", "")).is_dir():
            errors.append(f"基线 Run 目录不存在：{baseline.get('run_dir')}")
    knowledge = load_knowledge(task_dir)
    for knowledge_id, entry in knowledge.get("entries", {}).items():
        if knowledge_id != entry.get("id"):
            errors.append(f"知识键与 id 不一致：{knowledge_id}")
        if entry.get("kind") not in KNOWLEDGE_KINDS:
            errors.append(f"知识类型无效：{knowledge_id} -> {entry.get('kind')}")
        if entry.get("status") not in KNOWLEDGE_STATUSES:
            errors.append(f"知识状态无效：{knowledge_id} -> {entry.get('status')}")
    for loop_id in index.get("loop_order", []):
        loop_dir = task_dir / "loops" / loop_id
        if not loop_dir.is_dir():
            errors.append(f"缺少 Loop 目录：{loop_dir}")
            continue
        for name in LOOP_FILES:
            if not (loop_dir / name).is_file():
                errors.append(f"缺少文件：{loop_dir / name}")
        for name in ("cases.jsonl", "measurements.jsonl", "commands.jsonl", "artifacts.jsonl", "comparisons.jsonl"):
            validate_jsonl(loop_dir / name, errors)
        state_path = loop_dir / "state.json"
        decision_path = loop_dir / "decision.json"
        if state_path.is_file() and decision_path.is_file():
            state = read_json(state_path)
            decision = read_json(decision_path)
            if state.get("status") in ("ACCEPTED", "REJECTED", "INCONCLUSIVE", "PIVOTED") and decision.get("result") != state.get("status"):
                errors.append(f"state 与 verdict 不一致：{loop_dir}")
            for case_id in state.get("cases", []):
                if case_id not in case_ids:
                    errors.append(f"Loop {loop_id} 引用了不存在的用例：{case_id}")
        for manifest in (loop_dir / "runs").glob("*/*/manifest.json"):
            run_dir = manifest.parent
            value = read_json(manifest)
            if value.get("status") == "RECORDED" and not (run_dir / "result.json").is_file():
                errors.append(f"已记录的 Run 缺少 result.json：{run_dir}")
    if index.get("status") == "COMPLETED" and index.get("active_loop"):
        errors.append("已完成的 Task 仍存在 active_loop")
    if index.get("status") == "COMPLETED" and not index.get("completion_gate"):
        errors.append("已完成的 Task 缺少 completion_gate")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"验证通过：{task_dir}")


def cmd_resume(args: argparse.Namespace) -> None:
    task_dir = resolve(args.task_dir)
    output = render_resume_pack(
        task_dir,
        max_knowledge=args.max_knowledge,
        max_loops=args.max_loops,
        max_cases=args.max_cases,
        dtype=args.dtype,
        case_group=args.case_group,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def cmd_compact(args: argparse.Namespace) -> None:
    task_dir = resolve(args.task_dir)
    index = read_json(task_dir / "index.json")
    summaries: list[dict[str, Any]] = []
    for loop_id in index.get("loop_order", []):
        loop_dir = task_dir / "loops" / loop_id
        if not loop_dir.is_dir():
            continue
        state = read_json(loop_dir / "state.json")
        decision = read_json(loop_dir / "decision.json")
        if decision.get("result") == "PENDING":
            continue
        summaries.append({
            "loop_id": loop_id,
            "title": state.get("title"),
            "mode": state.get("mode"),
            "cases": state.get("cases", []),
            "hypothesis": (loop_dir / "hypotheses.md").read_text(encoding="utf-8").strip().splitlines()[-1],
            "verdict": decision.get("result"),
            "reason": decision.get("reason"),
            "evidence_paths": decision.get("evidence_paths", []),
            "decided_at": decision.get("decided_at"),
        })
    atomic_text(task_dir / "loop-index.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in summaries))
    render_task_status(task_dir)
    pack = render_resume_pack(
        task_dir,
        max_knowledge=args.max_knowledge,
        max_loops=args.max_loops,
        max_cases=args.max_cases,
    )
    append_task_event(task_dir, "context_compacted", pack["context_policy"])
    print(task_dir / "resume-pack.json")


def cmd_close_task(args: argparse.Namespace) -> None:
    task_dir = resolve(args.task_dir)
    index = read_json(task_dir / "index.json")
    if index.get("active_loop"):
        fail(f"存在活动 Loop，不能结束 Task：{index['active_loop']}")
    if args.correctness_gate != "pass":
        fail("结束 Task 要求 --correctness-gate=pass")
    if args.regression_gate != "pass":
        fail("结束 Task 要求 --regression-gate=pass")
    if args.delivery_gate != "pass":
        fail("结束 Task 要求 --delivery-gate=pass")
    if args.performance_gate not in ("pass", "waived", "not-applicable"):
        fail("性能门槛必须为 pass、waived 或 not-applicable")
    closed = now()
    index["status"] = "COMPLETED"
    index["phase"] = "COMPLETED"
    index["next_action"] = "NONE"
    index["completion_gate"] = {
        "correctness": args.correctness_gate,
        "performance": args.performance_gate,
        "regression": args.regression_gate,
        "delivery": args.delivery_gate,
        "stop_reason": args.stop_reason,
        "closed_at": closed,
    }
    index["updated_at"] = closed
    atomic_json(task_dir / "index.json", index)
    atomic_text(task_dir / "reports" / "final-report.md", f"# 最终报告\n\n- Task：`{index['task_id']}`\n- 结束时间：`{closed}`\n- 正确性门槛：`{args.correctness_gate}`\n- 性能门槛：`{args.performance_gate}`\n- 回归门槛：`{args.regression_gate}`\n- 交付门槛：`{args.delivery_gate}`\n- 停止原因：{args.stop_reason}\n\n{args.summary}\n")
    append_task_event(task_dir, "task_completed", index["completion_gate"])
    render_task_status(task_dir)
    render_resume_pack(task_dir)
    cmd_validate(argparse.Namespace(task_dir=str(task_dir)))
    print(task_dir / "reports" / "final-report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init_task = sub.add_parser("init-task", help="初始化 Task 目录")
    init_task.add_argument("--tasks-root", required=True, help="用于存放 Task 的根目录")
    init_task.add_argument("--task-id", required=True, help="稳定且唯一的 Task 标识")
    init_task.add_argument("--title", required=True, help="任务标题")
    init_task.add_argument("--operator", required=True, help="算子名称")
    init_task.add_argument("--framework", required=True, help="开发框架")
    init_task.add_argument("--source-root", required=True, help="源码根目录")
    init_task.add_argument("--goal", required=True, help="Task 完成目标")
    init_task.add_argument("--allowed-path", action="append", default=[], help="允许修改的路径，可重复指定")
    init_task.add_argument("--non-goal", action="append", default=[], help="明确不处理的事项，可重复指定")
    init_task.set_defaults(func=cmd_init_task)

    add_case = sub.add_parser("add-case", help="登记一个结构化 Shape/Dtype 用例")
    add_case.add_argument("--task-dir", required=True, help="Task 目录")
    add_case.add_argument("--case-id", required=True, help="稳定且唯一的用例标识")
    add_case.add_argument("--shape", required=True, help="JSON 数组，例如 [1,4096]")
    add_case.add_argument("--dtype", required=True, help="数据类型，例如 fp16")
    add_case.add_argument("--layout", default="UNKNOWN", help="数据布局")
    add_case.add_argument("--attributes-json", default="{}", help="算子属性 JSON 对象")
    add_case.add_argument("--weight", type=float, default=1.0, help="目标函数中的用例权重")
    add_case.add_argument("--group", default="default", help="性能机制用例组")
    add_case.add_argument("--protected", choices=("yes", "no"), default="no", help="是否为禁止回归用例")
    add_case.add_argument("--reason", required=True, help="登记该用例的原因")
    add_case.set_defaults(func=cmd_add_case)

    new_loop = sub.add_parser("new-loop", help="创建并冻结一个新 Loop")
    new_loop.add_argument("--task-dir", required=True, help="所属 Task 目录")
    new_loop.add_argument("--loop-id", help="可选的 Loop 标识，默认自动生成")
    new_loop.add_argument("--title", required=True, help="本轮标题")
    new_loop.add_argument("--mode", choices=LOOP_MODES, required=True, help="本轮开发模式")
    new_loop.add_argument("--hypothesis", required=True, help="本轮要建立的结论或机制假设")
    new_loop.add_argument("--success", required=True, help="接受条件")
    new_loop.add_argument("--falsification", required=True, help="否定条件")
    new_loop.add_argument("--allowed-path", action="append", default=[], help="允许修改的路径，可重复指定")
    new_loop.add_argument("--case", action="append", default=[], help="目标用例，可重复指定")
    new_loop.add_argument("--execution-skill", help="本轮唯一的专项执行 Skill")
    new_loop.add_argument("--next-action", required=True, help="第一个可执行动作")
    new_loop.set_defaults(func=cmd_new_loop)

    new_run = sub.add_parser("new-run", help="创建一个 Run 证据包")
    new_run.add_argument("--loop-dir", required=True, help="所属 Loop 目录")
    new_run.add_argument("--run-id", help="可选的 Run 标识，默认按时间生成")
    new_run.add_argument("--case-key", required=True, help="稳定的用例标识")
    new_run.add_argument("--kind", choices=("design-check", "build", "test", "simulation", "benchmark", "profile", "review"), required=True, help="执行类型")
    new_run.add_argument("--command", required=True, help="原样记录的执行命令")
    new_run.add_argument("--code-ref", required=True, help="提交、哈希或工作区状态说明")
    new_run.add_argument("--working-directory", help="命令工作目录")
    new_run.set_defaults(func=cmd_new_run)

    finish_run = sub.add_parser("finish-run", help="记录一个 Run 的执行结果")
    finish_run.add_argument("--run-dir", required=True, help="Run 目录")
    finish_run.add_argument("--status", choices=("pass", "fail", "error", "invalid"), required=True, help="执行状态")
    finish_run.add_argument("--correctness", choices=("pass", "fail", "not-applicable", "invalid"), required=True, help="正确性状态")
    finish_run.add_argument("--summary", required=True, help="实际观察结果摘要")
    finish_run.add_argument("--metric-name", help="可选的性能指标名称")
    finish_run.add_argument("--metric-value", type=float, help="可选的性能指标数值")
    finish_run.add_argument("--metric-unit", help="可选的性能指标单位")
    finish_run.add_argument("--artifact", action="append", default=[], help="证据产物路径，可重复指定")
    finish_run.set_defaults(func=cmd_finish_run)

    compare = sub.add_parser("compare", help="记录优化前后的可比性与变化")
    compare.add_argument("--loop-dir", required=True, help="所属 Loop 目录")
    compare.add_argument("--before-run", required=True, help="优化前 Run 目录")
    compare.add_argument("--after-run", required=True, help="优化后 Run 目录")
    compare.add_argument("--comparable", choices=("yes", "no"), required=True, help="是否满足同条件比较要求")
    compare.add_argument("--summary", required=True, help="对比判断和变化量")
    compare.set_defaults(func=cmd_compare)

    stage_knowledge = sub.add_parser("stage-knowledge", help="为当前 Loop 暂存一个 Task 知识变化")
    stage_knowledge.add_argument("--loop-dir", required=True, help="Loop 目录")
    stage_knowledge.add_argument("--knowledge-id", required=True, help="稳定的知识标识")
    stage_knowledge.add_argument("--kind", choices=KNOWLEDGE_KINDS, required=True, help="知识类型")
    stage_knowledge.add_argument("--statement", required=True, help="可独立理解的知识陈述")
    stage_knowledge.add_argument("--status", choices=KNOWLEDGE_STATUSES, required=True, help="知识状态")
    stage_knowledge.add_argument("--confidence", choices=CONFIDENCE_LEVELS, required=True, help="置信度")
    stage_knowledge.add_argument("--scope-json", default="{}", help="适用范围 JSON 对象")
    stage_knowledge.add_argument("--evidence-path", action="append", default=[], help="证据路径，可重复指定")
    stage_knowledge.add_argument("--replaces", action="append", default=[], help="被替代知识标识，可重复指定")
    stage_knowledge.add_argument("--global-candidate", choices=("yes", "no"), default="no", help="是否进入全局知识候选队列")
    stage_knowledge.set_defaults(func=cmd_stage_knowledge)

    stage_baseline = sub.add_parser("stage-baseline", help="从一个有效 Run 暂存用例基线变化")
    stage_baseline.add_argument("--loop-dir", required=True, help="Loop 目录")
    stage_baseline.add_argument("--run-dir", required=True, help="作为新基线的 Run 目录")
    stage_baseline.add_argument("--reason", required=True, help="更新基线的原因")
    stage_baseline.set_defaults(func=cmd_stage_baseline)

    verdict = sub.add_parser("verdict", help="记录主控对当前 Loop 的结论")
    verdict.add_argument("--loop-dir", required=True, help="Loop 目录")
    verdict.add_argument("--result", choices=VERDICTS, required=True, help="结论类型")
    verdict.add_argument("--reason", required=True, help="有证据支撑的决定依据")
    verdict.add_argument("--goal-satisfied", choices=("yes", "no", "unknown"), required=True, help="冻结目标是否得到证据支持")
    verdict.add_argument("--comparability", choices=("valid", "invalid", "not-applicable"), required=True, help="证据是否满足同条件比较")
    verdict.add_argument("--causal-conclusion", choices=("supported", "partially-supported", "falsified", "not-identifiable", "not-applicable"), required=True, help="机制因果结论")
    verdict.add_argument("--next-action", required=True, help="Task 的下一步动作")
    verdict.add_argument("--evidence-path", action="append", default=[], help="结论证据路径，可重复指定")
    verdict.add_argument("--update-baseline", help="可选：更新已接受基线")
    verdict.set_defaults(func=cmd_verdict)

    resume = sub.add_parser("resume", help="输出用于恢复任务的最小上下文")
    resume.add_argument("--task-dir", required=True, help="Task 目录")
    resume.add_argument("--max-knowledge", type=int, default=DEFAULT_CONTEXT_LIMITS["knowledge"], help="最多返回的知识条目")
    resume.add_argument("--max-loops", type=int, default=DEFAULT_CONTEXT_LIMITS["loops"], help="最多返回的历史 Loop 摘要")
    resume.add_argument("--max-cases", type=int, default=DEFAULT_CONTEXT_LIMITS["cases"], help="最多返回的用例摘要")
    resume.add_argument("--dtype", help="只返回指定 Dtype 的用例")
    resume.add_argument("--case-group", help="只返回指定用例组")
    resume.set_defaults(func=cmd_resume)

    compact = sub.add_parser("compact", help="重建历史索引和固定大小恢复包")
    compact.add_argument("--task-dir", required=True, help="Task 目录")
    compact.add_argument("--max-knowledge", type=int, default=DEFAULT_CONTEXT_LIMITS["knowledge"])
    compact.add_argument("--max-loops", type=int, default=DEFAULT_CONTEXT_LIMITS["loops"])
    compact.add_argument("--max-cases", type=int, default=DEFAULT_CONTEXT_LIMITS["cases"])
    compact.set_defaults(func=cmd_compact)

    validate = sub.add_parser("validate", help="校验 Task/Loop/Run 结构与不变量")
    validate.add_argument("--task-dir", required=True, help="Task 目录")
    validate.set_defaults(func=cmd_validate)

    close = sub.add_parser("close-task", help="结束已满足交付条件的 Task")
    close.add_argument("--task-dir", required=True, help="Task 目录")
    close.add_argument("--summary", required=True, help="最终结果、证据、风险和交付路径")
    close.add_argument("--correctness-gate", choices=("pass", "fail"), required=True)
    close.add_argument("--performance-gate", choices=("pass", "waived", "not-applicable", "fail"), required=True)
    close.add_argument("--regression-gate", choices=("pass", "fail"), required=True)
    close.add_argument("--delivery-gate", choices=("pass", "fail"), required=True)
    close.add_argument("--stop-reason", required=True, help="满足目标、空间收敛、预算或用户决定等停止理由")
    close.set_defaults(func=cmd_close_task)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
