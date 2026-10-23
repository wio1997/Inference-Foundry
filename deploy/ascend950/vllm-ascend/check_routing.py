#!/usr/bin/env python3
"""ABI and numerical smoke for the 20-argument aclnnMoeGatingTopKHash provider.

Cases:
  A  vision fusion: nonzero additional_bias, mixed additional_token_mask, no hash
  B  text-only fusion: same op, mask all False (text bias only)
  C  legacy hash semantics: input_ids + tid2eid, no additional bias/mask
Expected values are computed independently on CPU from the checkpoint Gate
semantics (sqrtsoftplus scores, bias picks experts, weights from raw scores,
norm_topk_prob=True, route_scale=1.5).
"""
from __future__ import annotations

import json
import sys

import torch
import torch_npu

K = 6
E = 384
VOCAB = 4096
SCALING = 1.5
EPS = 1e-20


def maps():
    keys = ("libcust_opapi", "custom_ops_lib", "libnnopbase")
    out = []
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if any(k in line for k in keys):
                out.append(line.strip())
    return out


def reference(logits, bias_text, bias_add, mask, ids=None, tid2eid=None):
    """CPU reference mirroring model.py Gate (sqrtsoftplus, norm_topk_prob, scale)."""
    x = logits.float()
    scores = torch.nn.functional.softplus(x).sqrt()
    if ids is not None:
        topk_ids = tid2eid[ids.long()].long()
    else:
        if bias_add is not None:
            bias = torch.where(mask.unsqueeze(-1), bias_add.unsqueeze(0), bias_text.unsqueeze(0))
        else:
            bias = bias_text.unsqueeze(0)
        topk_ids = torch.topk(scores + bias, k=K, dim=-1, sorted=True).indices
    weights = scores.gather(1, topk_ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + EPS)
    weights = weights * SCALING
    return topk_ids.to(torch.int32), weights


def compare(name, got_idx, got_w, exp_idx, exp_w, tol):
    idx_ok = bool(torch.equal(got_idx, exp_idx))
    diff = (got_w.float() - exp_w.float()).abs()
    max_abs = float(diff.max())
    denom = exp_w.float().abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max())
    ok = idx_ok and max_abs <= tol
    print(json.dumps({"case": name, "idx_equal": idx_ok,
                      "max_abs": max_abs, "max_rel": max_rel, "tol": tol,
                      "ok": ok}), flush=True)
    if not idx_ok:
        bad = (got_idx != exp_idx).nonzero()
        print(json.dumps({"case": name, "first_idx_mismatch": bad[:5].tolist()}), flush=True)
    return ok


def main() -> int:
    import custom_ops  # noqa: F401  registers torch.ops.custom

    op = torch.ops.custom.npu_moe_gating_top_k
    schemas = [str(getattr(v, "_schema", None)) for v in (getattr(op, "_schemas", None) or {}).values()]
    schema = schemas or [str(getattr(op, "default", None))]
    print(json.dumps({"stage": "register", "schema": schema}), flush=True)

    torch_npu.npu.set_device(0)
    torch.manual_seed(20261022)
    device = "npu"
    results = []

    def run_case(name, logits, bias_text, bias_add, mask, ids, tid2eid, tol):
        kw = dict(k=K, bias=bias_text, k_group=1, group_count=1,
                  routed_scaling_factor=SCALING, eps=EPS, group_select_mode=1,
                  renorm=0, norm_type=2, out_flag=False)
        if bias_add is not None:
            kw["additional_bias"] = bias_add
            kw["additional_token_mask"] = mask
        if ids is not None:
            kw["input_ids"] = ids
            kw["tid2eid"] = tid2eid
        with torch.no_grad():
            y, idx, out = op(logits, **kw)
            torch_npu.npu.synchronize()
        ci = idx.int().cpu()
        yc = y.float().cpu()
        oc = out.float().cpu()
        fin = bool(torch.isfinite(yc).all() and torch.isfinite(oc).all())
        print(json.dumps({"case": name, "shapes": [list(y.shape), list(idx.shape), list(out.shape)],
                          "dtypes": [str(y.dtype), str(idx.dtype), str(out.dtype)],
                          "finite": fin, "idx_min": int(ci.min()), "idx_max": int(ci.max())}),
              flush=True)
        exp_idx, exp_w = reference(logits.cpu(), bias_text.cpu() if bias_text is not None else None,
                                   bias_add.cpu() if bias_add is not None else None,
                                   mask.cpu() if mask is not None else None,
                                   ids.cpu() if ids is not None else None,
                                   tid2eid.cpu() if tid2eid is not None else None)
        results.append(compare(name, ci, yc, exp_idx, exp_w, tol) and fin)

    # A + B: vision fusion on non-hash layers (bias_vl present, tid2eid absent).
    logits = torch.randn(6, E, dtype=torch.float32)
    text_bias = torch.randn(E, dtype=torch.float32) * 0.25
    add_bias = torch.randn(E, dtype=torch.float32) * 0.25
    mask_mixed = torch.tensor([True, False, True, False, False, True])
    mask_none = torch.zeros(6, dtype=torch.bool)
    run_case("A_mixed_mask_nonzero_add_bias",
             logits.npu(), text_bias.npu(), add_bias.npu(), mask_mixed.npu(), None, None, 2e-3)
    run_case("B_text_only_mask_false",
             logits.npu(), text_bias.npu(), add_bias.npu(), mask_none.npu(), None, None, 2e-3)

    # C: legacy hash semantics preserved.
    ids = torch.randint(0, VOCAB, (6,), dtype=torch.int64)
    tid2eid = torch.randint(0, E, (VOCAB, K), dtype=torch.int32)
    run_case("C_hash_tid2eid_legacy",
             logits.npu(), text_bias.npu(), None, None, ids.npu(), tid2eid.npu(), 2e-3)

    # D: bf16 storage dtype smoke (same math, looser tolerance).
    logits_bf = torch.randn(6, E, dtype=torch.bfloat16).float()
    run_case("D_bf16_output_dtype",
             logits_bf.to(torch.bfloat16).npu(), text_bias.to(torch.bfloat16).npu(),
             add_bias.to(torch.bfloat16).npu(), mask_mixed.npu(), None, None, 3e-2)

    ok = all(results)
    print(json.dumps({"stage": "final", "cases": len(results), "ok": ok,
                      "libs": maps()}, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
