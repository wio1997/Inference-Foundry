#!/usr/bin/env python3
"""Add one same-state repeat of each source to the opt-in strict ABA diagnostic."""
from pathlib import Path

path = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
source = path.read_text()
pairs = [
(
"""                    _tokens, _counts = greedy_accept(
                        _st.draft_tokens, _pred[:, :7], _pred[:, 7]
                    )
                    return _pred, _tokens.clone(), _counts.clone()
""",
"""                    _tokens, _counts = greedy_accept(
                        _st.draft_tokens, _pred[:, :7], _pred[:, 7]
                    )
                    _top2 = torch.topk(_logits[:96], k=2, dim=-1)
                    return (_pred, _tokens.clone(), _counts.clone(),
                            _top2.indices.clone(), _top2.values.clone())
"""
),
(
"""                _pa, _ta, _ca = _stock_target_once()
""",
"""                _pa, _ta, _ca, _a_top2_ids, _a_top2_scores = _stock_target_once()
"""
),
(
"""                _tb_tokens, _cb = greedy_accept(
                    _st.draft_tokens, _pb[:, :7], _pb[:, 7]
                )
                _post_b = _capture_post_writes()
""",
"""                _tb_tokens, _cb = greedy_accept(
                    _st.draft_tokens, _pb[:, :7], _pb[:, 7]
                )
                _b_top2 = torch.topk(_tb.logits[:96], k=2, dim=-1)
                _b_top2_ids = _b_top2.indices.clone()
                _b_top2_scores = _b_top2.values.clone()
                _post_b = _capture_post_writes()
"""
),
(
"""                _pc, _tc, _cc = _stock_target_once()
                _post_c = _capture_post_writes()
                _rc = _restore_stock()
                _first = torch.arange(12, device=input_ids.device) * 8
""",
"""                _pc, _tc, _cc, _c_top2_ids, _c_top2_scores = _stock_target_once()
                _post_c = _capture_post_writes()
                _rc = _restore_stock()
                _repeat = None
                if os.getenv("EXTREME_STOCK_TARGET_ABA_REPEAT") == "1":
                    if _extreme_runtime.target_metadata is not None:
                        _extreme_runtime.target_metadata.update(_st)
                    _tb2 = _extreme_runtime.target.execute(_st)
                    _pb2 = _tb2.logits.argmax(dim=-1).view(12, 8).clone()
                    _tb2_tokens, _cb2 = greedy_accept(
                        _st.draft_tokens, _pb2[:, :7], _pb2[:, 7]
                    )
                    _b2_top2 = torch.topk(_tb2.logits[:96], k=2, dim=-1)
                    _post_b2 = _capture_post_writes()
                    _rb2 = _restore_stock()
                    _pc2, _tc2, _cc2, _c2_top2_ids, _c2_top2_scores = _stock_target_once()
                    _post_c2 = _capture_post_writes()
                    _rc2 = _restore_stock()
                    _repeat = {
                        "argmax_b2": _pb2.cpu().tolist(),
                        "argmax_c2": _pc2.cpu().tolist(),
                        "counts_b2": _cb2.cpu().tolist(),
                        "counts_c2": _cc2.cpu().tolist(),
                        "argmax_equal_bb2": int((_pb == _pb2).sum().item()),
                        "argmax_equal_cc2": int((_pc == _pc2).sum().item()),
                        "accepted_equal_bb2": int((_tb_tokens == _tb2_tokens).sum().item()),
                        "accepted_equal_cc2": int((_tc == _tc2).sum().item()),
                        "accepted_equal_ab2": int((_ta == _tb2_tokens).sum().item()),
                        "accepted_equal_ac2": int((_ta == _tc2).sum().item()),
                        "post_write_ab2": _compare_post_writes(_post_a, _post_b2),
                        "post_write_ac2": _compare_post_writes(_post_a, _post_c2),
                        "restores": [_rb2, _rc2],
                        "top2_b2_ids": _b2_top2.indices.cpu().tolist(),
                        "top2_b2_scores": _b2_top2.values.float().cpu().tolist(),
                        "top2_c2_ids": _c2_top2_ids.cpu().tolist(),
                        "top2_c2_scores": _c2_top2_scores.float().cpu().tolist(),
                    }
                _first = torch.arange(12, device=input_ids.device) * 8
"""
),
(
"""                    "argmax_c": _pc.cpu().tolist(),
                }
""",
"""                    "argmax_c": _pc.cpu().tolist(),
                    "top2_a_ids": _a_top2_ids.cpu().tolist(),
                    "top2_a_scores": _a_top2_scores.float().cpu().tolist(),
                    "top2_b_ids": _b_top2_ids.cpu().tolist(),
                    "top2_b_scores": _b_top2_scores.float().cpu().tolist(),
                    "top2_c_ids": _c_top2_ids.cpu().tolist(),
                    "top2_c_scores": _c_top2_scores.float().cpu().tolist(),
                    "repeat": _repeat,
                }
"""
),
]
for old,new in pairs:
    if source.count(old) != 1:
        if new in source:
            print("repeat ABA diagnostic hook already applied")
            raise SystemExit(0)
        raise RuntimeError("repeat ABA anchor changed")
    source=source.replace(old,new)
old="""                _row["pass"] = bool(all(x["exact"] for x in _row["restores"])
                    and _row["metadata_tensors_restored"])
"""
new="""                _row["pass"] = bool(all(x["exact"] for x in _row["restores"])
                    and (_repeat is None or all(x["exact"] for x in _repeat["restores"]))
                    and _row["metadata_tensors_restored"])
"""
if source.count(old)!=1: raise RuntimeError("pass gate anchor changed")
source=source.replace(old,new)
path.write_text(source)
print("applied repeat ABA diagnostic hook")
