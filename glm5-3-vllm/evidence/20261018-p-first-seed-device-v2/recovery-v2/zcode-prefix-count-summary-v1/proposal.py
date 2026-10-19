from typing import Dict, List, Any


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
    if not isinstance(ma, float) or ma != 0.0:
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
        if pattern not in ("scatter", "head", "tail", "mixed"):
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
        seen_iters = set()
        for i, rec in enumerate(captured):
            if not isinstance(rec, dict):
                raise ValueError("captured item must be dict")
            it = rec.get("iteration")
            if not (isinstance(it, int) and not isinstance(it, bool)):
                raise ValueError("iteration must be strict int")
            if it not in (0, 1, 2):
                raise ValueError("iteration out of range for %r" % (pair,))
            if it in seen_iters:
                raise ValueError("duplicate iteration for %r" % (pair,))
            seen_iters.add(it)
            mode = rec.get("mode")
            if mode != expected_modes[i]:
                raise ValueError("captured order mismatch for %r" % (pair,))
            if not _cmp_valid(rec.get("comparison_to_current_native_eager")):
                raise ValueError("comparison invalid for %r iter %d" % (pair, it))
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


# ---------- tests ----------

def _replay(i: int) -> Dict[str, Any]:
    return {
        "iteration": i,
        "mode": ["native", "prefix", "prefix", "native"][i % 4],
        "comparison_to_current_native_eager": {
            "bitwise_equal": True,
            "different_elements": 0,
            "max_abs": 0.0,
        },
    }


def _build_fixture() -> Dict[str, Any]:
    rows = []
    for N in (6, 24):
        for pattern in ("scatter", "head", "tail", "mixed"):
            rows.append({
                "N": N,
                "pattern": pattern,
                "eager_native_vs_prefix": {
                    "bitwise_equal": True,
                    "different_elements": 0,
                    "max_abs": 0.0,
                },
                "routing_histogram_after_consumers_correct": True,
                "captured": [_replay(i) for i in range(12)],
            })
    return {"completed": True, "rows": rows}


def test_positive():
    out = summarize_prefix_count_fragment(_build_fixture())
    assert out["passed"] is True
    assert out["cases"] == 8
    assert out["replays"] == 96
    assert out["case_pairs"] == sorted((N, p) for N in (6, 24) for p in ("scatter", "head", "tail", "mixed"))
    assert "scope" in out and out["scope"]


def _expect_raise(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError for %s" % label)


def test_malformed():
    fx = _build_fixture()

    # 1: wrong completed
    bad1 = dict(fx, completed=False)
    _expect_raise(lambda: summarize_prefix_count_fragment(bad1), "completed false")

    # 2: rows length not 8
    bad2 = dict(fx, rows=fx["rows"][:7])
    _expect_raise(lambda: summarize_prefix_count_fragment(bad2), "rows <8")

    # 3: N is bool/float
    bad3 = _build_fixture()
    bad3["rows"][0]["N"] = True
    _expect_raise(lambda: summarize_prefix_count_fragment(bad3), "N bool")

    # 4: duplicate (N, pattern)
    bad4 = _build_fixture()
    bad4["rows"].append(dict(bad4["rows"][0]))
    _expect_raise(lambda: summarize_prefix_count_fragment(bad4), "duplicate pair")

    # 5: max_abs non-zero (reject float mismatch)
    bad5 = _build_fixture()
    bad5["rows"][2]["eager_native_vs_prefix"]["max_abs"] = 0.5
    _expect_raise(lambda: summarize_prefix_count_fragment(bad5), "nonzero max_abs")

    # 6: captured order wrong (prefix and native swapped)
    bad6 = _build_fixture()
    bad6["rows"][0]["captured"][0]["mode"] = "prefix"
    _expect_raise(lambda: summarize_prefix_count_fragment(bad6), "mode order")

    # 7: histogram flag not strict True
    bad7 = _build_fixture()
    bad7["rows"][1]["routing_histogram_after_consumers_correct"] = 1
    _expect_raise(lambda: summarize_prefix_count_fragment(bad7), "histogram int")

    # 8: missing captured (partial)
    bad8 = _build_fixture()
    del bad8["rows"][0]["captured"]
    _expect_raise(lambda: summarize_prefix_count_fragment(bad8), "missing captured")


if __name__ == "__main__":
    test_positive()
    test_malformed()
    print("all fragment parser tests passed")
