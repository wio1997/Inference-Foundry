#!/usr/bin/env python3
"""Tighten the opt-in Stock A/Product B/Stock C diagnostic snapshot."""
from pathlib import Path

path = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
source = path.read_text()
old = """                _kv = _assets.snapshot_slots(
                    self._extreme_all_slot_mappings, num_tokens=96
                )
                def _stock_target_once():
"""
new = """                if _extreme_runtime.target_page_audit is not None:
                    # The source-backed manifest unions actual aliases for
                    # all 67 cache views; group slot specs alone are incomplete.
                    _page_audit = _extreme_runtime.target_page_audit
                    _page_audit.observe(0)
                    _kv = _page_audit.pending_snapshot
                    if _kv is None or _page_audit.pending_metadata is None:
                        raise RuntimeError("strict Stock ABA snapshot missing")
                    _meta_tensors = [
                        (_tensor, _old)
                        for _, _tensor, _old in _page_audit.pending_metadata
                    ]
                    _strict_coverage = _kv.coverage()
                else:
                    _kv = _assets.snapshot_slots(
                        self._extreme_all_slot_mappings, num_tokens=96
                    )
                    _strict_coverage = None
                def _stock_target_once():
"""
if source.count(old) != 1:
    if new in source:
        print("strict ABA diagnostic hook already applied")
        raise SystemExit(0)
    raise RuntimeError("Stock ABA snapshot anchor changed")
source = source.replace(old, new)
old_row = """                    "restores": [_ra, _rb, _rc],
                    "post_write_ab": _compare_post_writes(_post_a, _post_b),
"""
new_row = """                    "restores": [_ra, _rb, _rc],
                    "strict_coverage": _strict_coverage,
                    "metadata_tensor_count": len(_meta_tensors),
                    "post_write_ab": _compare_post_writes(_post_a, _post_b),
"""
if source.count(old_row) != 1:
    raise RuntimeError("Stock ABA evidence anchor changed")
source = source.replace(old_row, new_row)
path.write_text(source)
print("applied strict ABA diagnostic hook")
