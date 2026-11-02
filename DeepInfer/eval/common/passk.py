from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def estimator(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased Pass@k estimator: 1 - C(n-c, k) / C(n, k).

    (The Codex/HumanEval estimator.)
      * for k=1 it degenerates to c/n, the mean per-sample accuracy, i.e. the usual avg@n / Pass@1;
      * when n-c < k (too few wrong samples for a k-subset to be all-correct) it returns 1.0.
    """
    assert 0 <= num_correct <= num_samples, f"invalid num_correct={num_correct} num_samples={num_samples}"
    assert 0 <= k <= num_samples, f"invalid k={k} num_samples={num_samples}"
    if num_samples - num_correct < k:
        return 1.0
    return float(1.0 - math.prod(1.0 - k / i for i in range(num_samples - num_correct + 1, num_samples + 1)))


def pass_at_k_by_task(
    results: list[dict[str, Any]],
    k_list: list[int],
    *,
    task_key: str = "task_id",
    correct_key: str = "correct",
) -> dict[str, float]:
    """Mean Pass@k over tasks, with tasks grouped by `task_key`.

    When a task has fewer than k samples (judging failure, truncated sampling), approximate with
    min(k, n) instead of dropping the task: dropping it would compute Pass@k only over the "clean"
    tasks, and those are usually the easier ones, which biases the result upward.
    """
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in results:
        grouped[row[task_key]].append(bool(row[correct_key]))

    metrics: dict[str, float] = {}
    for k in k_list:
        per_task = [estimator(len(flags), sum(flags), min(k, len(flags))) for flags in grouped.values() if flags]
        metrics[f"Pass@{k}"] = sum(per_task) / len(per_task) if per_task else 0.0
    return metrics


def default_k_list(n_sample: int) -> list[int]:
    """Default k values: 1, 2, 4, ... (powers of two), with n_sample appended."""
    k_list = [2**i for i in range(max(math.ceil(math.log2(n_sample)), 1))]
    if n_sample not in k_list:
        k_list.append(n_sample)
    return k_list


def summarize(
    results: list[dict[str, Any]],
    *,
    n_sample: int,
    k_list: list[int] | None = None,
    task_key: str = "task_id",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One rollout per row: {task_id, correct, ...}."""
    k_list = k_list or default_k_list(n_sample)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row[task_key]].append(row)

    summary: dict[str, Any] = {
        "num_tasks": len(grouped),
        "num_rollouts": len(results),
        "n_sample": n_sample,
        "score": sum(1 for row in results if row["correct"]) / len(results) if results else 0.0,
        "task_accuracy": {
            task_id: sum(1 for row in task_rows if row["correct"]) / len(task_rows) for task_id, task_rows in sorted(grouped.items())
        },
    }
    summary.update(pass_at_k_by_task(results, k_list, task_key=task_key))
    if extra:
        summary.update(extra)
    return summary


def print_report(summary: dict[str, Any], *, title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
    print(f"tasks={summary['num_tasks']}  rollouts={summary['num_rollouts']}  n_sample={summary['n_sample']}")
    pass_keys = sorted(
        (metric for metric in summary if metric.startswith("Pass@")),
        key=lambda k: int(k.split("@")[1]),
    )
    for key in pass_keys:
        print(f"  {key:<10s} = {summary[key] * 100:6.2f}")
    note = "" if abs(summary["score"] - summary.get("Pass@1", 0.0)) < 1e-9 else "  (ragged sample counts, differs from Pass@1)"
    print(
        f"  {'score':<10s} = {summary['score'] * 100:6.2f}   (= mean per-sample accuracy, equals Pass@1 when sample counts are even){note}"
    )
    print("-" * 72)
    print("per-task accuracy:")
    for tid, acc in summary["task_accuracy"].items():
        bar = "#" * int(round(acc * 20))
        print(f"  {tid:<20s} {acc * 100:6.2f}  {bar}")
    print("=" * 72)
