#!/usr/bin/env python3
"""Run497: CPU-only first-position lemma, with separate evidence gates."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "runtime/greedy_accept.py": "9399cad7d76056dbb59db6140227e4379df8f366d7c441b973f9e423775cffbc",
    "runtime/fixed_acceptance.py": "824289bae3bd31676e3a697cc59b4169a345c70e10063c5660763e3e3557c6ae",
    "runtime/fixed_decode.py": "7d74f4cbba380ff0a9296cc06d86e0cb9c9c921b6213b7965700831a29a6bafa",
    "runtime/fixed_serving.py": "137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a",
    "runtime/target_adapter.py": "c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5",
}


def source_gate() -> dict[str, str]:
    checked = {}
    for name, expected in PINS.items():
        raw = (ROOT / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == expected, (name, digest)
        ast.parse(raw, filename=name)
        checked[name] = digest
    # Bind the tested greedy function and its callers to exact source snapshots.
    src = (ROOT / "runtime/greedy_accept.py").read_text()
    for anchor in ("leading_match = torch.cumprod(",
                   "copy_mask = positions < accepted_drafts.unsqueeze(1)",
                   "output[:, :width].copy_", "output.scatter_(1, accepted_drafts.unsqueeze(1)"):
        assert src.count(anchor) == 1, anchor
    assert "predicted[:, : cfg.speculative_tokens]" in (
        ROOT / "runtime/fixed_acceptance.py").read_text()
    assert "torch.where(state.active_mask, acceptance.num_sampled, 0)" in (
        ROOT / "runtime/fixed_decode.py").read_text()
    return checked


def load_greedy():
    path = ROOT / "runtime/greedy_accept.py"
    spec = importlib.util.spec_from_file_location("run497_pinned_greedy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.greedy_accept


def main() -> None:
    checked = source_gate()
    greedy = load_greedy()
    lengths = set()
    negatives = {"wrong_column": 0, "inactive_count": 0,
                 "runtime_clip": 0, "scheduler_clip": 0}
    for bits in range(128):
        target = torch.tensor([[101, 103, 107, 109, 113, 127, 131]], dtype=torch.int64)
        draft = target.clone()
        for j in range(7):
            if not bits & (1 << j):
                draft[0, j] = 1001 + j
        bonus = torch.tensor([997], dtype=torch.int64)
        sampled, counts = greedy(draft, target, bonus)
        assert sampled.device.type == "cpu" and counts.device.type == "cpu"
        assert sampled.shape == (1, 8) and int(sampled[0, 0]) == int(target[0, 0])
        a = 0
        while a < 7 and bits & (1 << a):
            a += 1
        assert int(counts[0]) == a + 1
        lengths.add(a)
        if int(sampled[0, 0]) != int(target[0, 1]):
            negatives["wrong_column"] += 1
        active = torch.tensor([False])
        masked_count = torch.where(active, counts, 0)
        assert int(masked_count[0]) == 0
        negatives["inactive_count"] += 1
    assert lengths == set(range(8))
    assert negatives["wrong_column"] == negatives["inactive_count"] == 128

    # These prefix predicates are conditional source consequences, not observed
    # Scheduler commitment or client publication.
    R = 1024
    for q, retained in ((0, True), (R - 1, True), (R, False), (R + 1, False)):
        assert (q < R) is retained
        if not retained:
            negatives["runtime_clip"] += 1
    for G, q, admitted in ((0, 1023, True), (0, 1024, False),
                            (584, 439, True), (584, 440, False)):
        assert (q < 1024 - G) is admitted
        if not admitted:
            negatives["scheduler_clip"] += 1
    assert negatives["runtime_clip"] == negatives["scheduler_clip"] == 2
    print(json.dumps({
        "run": "run497", "scope": "CPU source lemma only",
        "source_sha256": checked,
        "equality_patterns": 128, "leading_accept_lengths": sorted(lengths),
        "negative_controls": negatives,
        "source_identity": "pass",
        "runtime_retention": "conditional predicate checked; actual lineage unobserved",
        "scheduler_admission": "conditional predicate checked; actual G and request join unobserved",
        "external_publication": "unobserved", "fresh_in_formal_window": "unobserved",
        "ordinary_dense_work_class": "assumption only",
        "hardware_capacity_C_plus": "unavailable", "finite_bound_endpoint": None,
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
