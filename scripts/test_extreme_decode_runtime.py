#!/usr/bin/env python3
"""Standalone CPU/NPU test of the product-owned decode driver."""

from __future__ import annotations

import argparse
import sys

import torch

from runtime.assets import OwnedCacheTensor, RuntimeAssets
from runtime.extreme_decode import ExtremeDecodeRuntime
from runtime.fixed_acceptance import FixedGreedyAcceptance
from runtime.fixed_decode import FixedDecodeConfig, FixedDecodeState
from runtime.target_adapter import FixedTargetAdapter, FixedTargetBinding


class FixedProposer:
    def __init__(self, config: FixedDecodeConfig, device: torch.device) -> None:
        self.config = config
        self.calls = torch.zeros((), dtype=torch.int32, device=device)

    def execute(self, state, target, acceptance):
        self.calls.add_(1)
        offsets = torch.arange(
            1,
            self.config.speculative_tokens + 1,
            dtype=state.draft_tokens.dtype,
            device=state.draft_tokens.device,
        )
        return state.last_sampled_tokens.unsqueeze(1) + offsets

    def state_fingerprint(self):
        return {"calls": self.calls.clone()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = FixedDecodeConfig()
    max_blocks = cfg.max_model_len // cfg.block_size
    block_table = torch.arange(
        cfg.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(cfg.batch_size, max_blocks)
    seq = torch.full((cfg.batch_size,), 32_768, dtype=torch.int32, device=device)
    last = torch.arange(cfg.batch_size, dtype=torch.int64, device=device) + 100
    draft = last.unsqueeze(1) + torch.arange(
        1,
        cfg.speculative_tokens + 1,
        dtype=torch.int64,
        device=device,
    )
    state = FixedDecodeState.allocate(cfg, block_table, seq, last, draft)
    target_cache = torch.arange(256, dtype=torch.int32, device=device)
    proposer_cache = torch.arange(128, dtype=torch.int32, device=device)
    assets = RuntimeAssets(
        [
            OwnedCacheTensor.take("target.cache", target_cache),
            OwnedCacheTensor.take("proposer.cache", proposer_cache),
        ]
    )
    cache_fingerprint = assets.exact_fingerprint(
        {
            "target.cache": torch.tensor([0, 31, 255], device=device),
            "proposer.cache": torch.tensor([1, 63, 127], device=device),
        }
    )
    assert cache_fingerprint["target.cache"].tolist() == [0, 31, 255]
    assert cache_fingerprint["proposer.cache"].tolist() == [1, 63, 127]

    target_calls = torch.zeros((), dtype=torch.int32, device=device)

    def target_forward(ids: torch.Tensor, positions: torch.Tensor):
        target_calls.add_(1)
        # Every draft position predicts the draft token itself, so the fixed
        # greedy path accepts all seven and emits the bonus token.
        hidden = torch.stack((ids.to(torch.int64), positions), dim=1)
        return hidden

    def compute_logits(hidden: torch.Tensor) -> torch.Tensor:
        ids = hidden[:, 0].view(cfg.batch_size, cfg.target_tokens_per_request)
        predicted = torch.cat((ids[:, 1:], ids[:, -1:] + 1), dim=1).reshape(-1)
        logits = torch.full((predicted.numel(), 256), -1.0, device=device)
        logits.scatter_(1, predicted.unsqueeze(1), 1.0)
        return logits

    target = FixedTargetAdapter(
        cfg,
        FixedTargetBinding(
            forward=target_forward,
            compute_logits=compute_logits,
            state_fingerprint=lambda: {"calls": target_calls.clone()},
        ),
    )
    proposer = FixedProposer(cfg, device)
    runtime = ExtremeDecodeRuntime(
        cfg,
        state,
        target,
        FixedGreedyAcceptance(cfg),
        proposer,
    )
    results = runtime.run(args.cycles)

    assert state.cycle_index == args.cycles
    assert len(results) == args.cycles
    assert all(int(result.acceptance.num_sampled[0]) == 8 for result in results)
    assert int(target_calls.cpu()) == args.cycles
    assert int(proposer.calls.cpu()) == args.cycles
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)
    print(
        {
            "pass": True,
            "cycles": args.cycles,
            "stage_order": runtime.stage_order,
            "target_calls": int(target_calls.cpu()),
            "proposer_calls": int(proposer.calls.cpu()),
            "vllm_imported": False,
            "owned_cache_tensors": len(assets.caches),
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
