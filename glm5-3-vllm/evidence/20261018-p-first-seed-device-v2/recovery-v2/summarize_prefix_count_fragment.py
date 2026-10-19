from typing import Dict, List, Any
import math


def _cmp_valid(c: Any) -> bool:
    if not isinstance(c, dict):
        return False
    if c.get("bitwise_equal") is not True:
        return False
    de = c.get("different_elements")
    if not isinstance(de, int) or isinstance(de, bool):
        return False
    if de != 0:
        return False
    ma = c.get("max_abs")
    if isinstance(ma, bool) or not isinstance(ma, (int, float)) or ma != 0 or not math.isfinite(ma):
        return False
    return True


def summarize_prefix_count_fragment(report: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")
    if report.get("completed") is not True:
        raise ValueError("completed must be True")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("rows must be a list of exactly 8")

    pairs: List[tuple] = []
    replays = 0

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be a dict")
        N = row.get("N")
        if not (isinstance(N, int) and not isinstance(N, bool) and N in (6, 24)):
            raise ValueError("N must be strict int 6 or 24")
        pattern = row.get("pattern")
        if not isinstance(pattern, str) or pattern not in ("scatter", "head", "tail", "mixed"):
            raise ValueError("invalid pattern")
        pair = (N, pattern)
        if pair in pairs:
            raise ValueError("duplicate (N, pattern) pair")
        pairs.append(pair)

        cmp = row.get("eager_native_vs_prefix")
        if not _cmp_valid(cmp):
            raise ValueError("eager_native_vs_prefix comparison invalid for %r" % (pair,))
        if row.get("routing_histogram_after_consumers_correct") is not True:
            raise ValueError("routing histogram not correct for %r" % (pair,))

        captured = row.get("captured")
        if not isinstance(captured, list) or len(captured) != 12:
            raise ValueError("captured must be a list of 12 for %r" % (pair,))
        expected_modes = ["native", "prefix", "prefix", "native"] * 3
        for i, rec in enumerate(captured):
            if not isinstance(rec, dict):
                raise ValueError("captured item must be dict")
            it = rec.get("iteration")
            if not (isinstance(it, int) and not isinstance(it, bool)):
                raise ValueError("iteration must be strict int")
            if it not in (0, 1, 2):
                raise ValueError("iteration out of range for %r" % (pair,))
            if it != i // 4:
                raise ValueError("iteration group/order mismatch for %r" % (pair,))
            mode = rec.get("mode")
            if mode != expected_modes[i]:
                raise ValueError("captured order mismatch for %r" % (pair,))
            if not _cmp_valid(rec.get("comparison_to_current_native_eager")):
                raise ValueError("comparison invalid for %r iter %d" % (pair, it))
            if rec.get("recovered_histogram_correct") is not True:
                raise ValueError("replayed recovered count was not checked")
            replays += 1

    if len(pairs) != 8:
        raise ValueError("expected exactly 8 distinct (N, pattern) pairs")

    return {
        "passed": True,
        "cases": 8,
        "replays": replays,
        "case_pairs": sorted(pairs),
        "scope": "Synthetic original NZ MoE chain; metadata/equality evidence only, "
                 "no full model/TP/KV/service claim.",
    }


