#!/usr/bin/env python3
"""Standalone CPU/NPU test of the product-owned decode driver."""

from __future__ import annotations

import argparse
import sys

import torch

from runtime.extreme_decode import ExtremeDecodeRuntime
from runtime.fixed_decode import AcceptanceOutput, FixedDecodeConfig, FixedDecodeState
from runtime.greedy_accept import greedy_accept
from runtime.target_adapter import FixedTargetAdapter, FixedTargetBinding


class GreedyAcceptance:
    def __init__(self, config: FixedDecodeConfig) -> None:
        self.config = config

    def execute(self, state, target):
        cfg = self.config
        target_argmax = target.logits[:, 0].view(
            cfg.batch_size, cfg.target_tokens_per_request
        )[:, : cfg.speculative_tokens]
        bonus = target.logits[:, 0].view(
            cfg.batch_size, cfg.target_tokens_per_request
        )[:, -1]
        tokens, counts = greedy_accept(state.draft_tokens, target_argmax, bonus)
        return AcceptanceOutput(tokens, counts)


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

    target_calls = torch.zeros((), dtype=torch.int32, device=device)

    def target_forward(ids: torch.Tensor, positions: torch.Tensor):
        target_calls.add_(1)
        # Every draft position predicts the draft token itself, so the fixed
        # greedy path accepts all seven and emits the bonus token.
        hidden = torch.stack((ids.to(torch.int64), positions), dim=1)
        return hidden

    target = FixedTargetAdapter(
        cfg,
        FixedTargetBinding(
            forward=target_forward,
            compute_logits=lambda hidden: torch.cat(
                (
                    hidden[:, 0]
                    .view(cfg.batch_size, cfg.target_tokens_per_request)[:, 1:],
                    (
                        hidden[:, 0]
                        .view(cfg.batch_size, cfg.target_tokens_per_request)[:, -1:]
                        + 1
                    ),
                ),
                dim=1,
            ).reshape(-1, 1),
            state_fingerprint=lambda: {"calls": target_calls.clone()},
        ),
    )
    proposer = FixedProposer(cfg, device)
    runtime = ExtremeDecodeRuntime(
        cfg,
        state,
        target,
        GreedyAcceptance(cfg),
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
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
