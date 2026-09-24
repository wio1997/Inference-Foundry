#!/usr/bin/env python3
"""Release large post-write copies before repeated target executions."""
from pathlib import Path

path=Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
s=path.read_text()
pairs=[
(
"""                _rc = _restore_stock()
                _repeat = None
""",
"""                _rc = _restore_stock()
                _post_ab_report = _compare_post_writes(_post_a, _post_b)
                _post_ac_report = _compare_post_writes(_post_a, _post_c)
                del _post_a, _post_b, _post_c
                _repeat = None
"""
),
(
"""                    _post_b2 = _capture_post_writes()
                    _rb2 = _restore_stock()
                    _pc2, _tc2, _cc2, _c2_top2_ids, _c2_top2_scores = _stock_target_once()
                    _post_c2 = _capture_post_writes()
                    _rc2 = _restore_stock()
""",
"""                    _rb2 = _restore_stock()
                    _pc2, _tc2, _cc2, _c2_top2_ids, _c2_top2_scores = _stock_target_once()
                    _rc2 = _restore_stock()
"""
),
(
"""                        "post_write_ab2": _compare_post_writes(_post_a, _post_b2),
                        "post_write_ac2": _compare_post_writes(_post_a, _post_c2),
""",
""""""
),
(
"""                    "post_write_ab": _compare_post_writes(_post_a, _post_b),
                    "post_write_ac": _compare_post_writes(_post_a, _post_c),
""",
"""                    "post_write_ab": _post_ab_report,
                    "post_write_ac": _post_ac_report,
"""
),
]
for old,new in pairs:
    if s.count(old)!=1: raise RuntimeError(f"memory patch anchor count={s.count(old)}")
    s=s.replace(old,new)
path.write_text(s)
print("applied ABA memory bound")
