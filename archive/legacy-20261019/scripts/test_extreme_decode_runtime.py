#!/usr/bin/env python3
"""Standalone CPU/NPU test of the product-owned decode driver."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import torch

from runtime.assets import OwnedCacheTensor, RuntimeAssets
from runtime.extreme_decode import ExtremeDecodeRuntime
from runtime.fixed_acceptance import FixedGreedyAcceptance
from runtime.fixed_decode import AcceptanceOutput, FixedDecodeConfig, FixedDecodeState
from runtime.fixed_serving import FixedCohortServing
from runtime.target_adapter import FixedTargetAdapter, FixedTargetBinding


class FixedProposer:
    def __init__(self, config: FixedDecodeConfig, device: torch.device) -> None:
        self.config = config
        self.calls = torch.zeros((), dtype=torch.int32, device=device)
        self._committed = torch.zeros(config.batch_size, dtype=torch.int64)

    def execute(self, state, target, acceptance):
        self.calls.add_(1)
        self._committed.add_(acceptance.num_sampled.cpu())
        offsets = torch.arange(
            1,
            self.config.speculative_tokens + 1,
            dtype=state.draft_tokens.dtype,
            device=state.draft_tokens.device,
        )
        return state.last_sampled_tokens.unsqueeze(1) + offsets

    def state_fingerprint(self):
        return {"calls": self.calls.clone()}

    def committed_emitted_token_count(self):
        return self._committed


class LaggedProgressProposer:
    def __init__(self, config: FixedDecodeConfig) -> None:
        self._committed = torch.zeros(config.batch_size, dtype=torch.int64)
        self._pending = None

    def advance(self) -> None:
        if self._pending is not None:
            self._committed.add_(self._pending)
            self._pending = None

    def stage(self, counts: torch.Tensor) -> None:
        self._pending = counts.to(torch.int64).clone()

    def committed_emitted_token_count(self):
        return self._committed


class PatternServingRuntime:
    """CPU-only shell double with the same one-cycle Host progress lag."""

    def __init__(self, config: FixedDecodeConfig) -> None:
        self.config = config
        self.proposer = LaggedProgressProposer(config)
        self.state = SimpleNamespace(
            accepted_tokens=torch.empty(
                config.batch_size,
                config.target_tokens_per_request,
                dtype=torch.int64,
            ),
            emitted_token_count=torch.zeros(config.batch_size, dtype=torch.int32),
        )
        self.cycle = 0

    def step(self):
        self.proposer.advance()
        width = self.config.target_tokens_per_request
        counts = (
            torch.arange(self.config.batch_size, dtype=torch.int32)
            + self.cycle
        ).remainder(width) + 1
        tokens = torch.full_like(self.state.accepted_tokens, -1)
        for slot, count in enumerate(counts.tolist()):
            tokens[slot, :count] = 1000 * self.cycle + 10 * slot + torch.arange(
                count, dtype=torch.int64
            )
        self.proposer.stage(counts)
        self.state.emitted_token_count.add_(counts)
        self.cycle += 1
        return SimpleNamespace(acceptance=AcceptanceOutput(tokens, counts))


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

    serving = FixedCohortServing(
        runtime,
        initial_output_counts=[3] * cfg.batch_size,
        max_output_tokens=19,
    ).run()
    assert serving.generated_output_counts == [16] * cfg.batch_size
    assert all(len(tokens) == 16 for tokens in serving.token_ids)
    assert serving.cycles == 2
    assert serving.overshoot_tokens == 0

    varied_remaining = [17 + slot for slot in range(cfg.batch_size)]
    varied = FixedCohortServing(
        PatternServingRuntime(cfg),
        initial_output_counts=[5] * cfg.batch_size,
        max_output_tokens=5 + max(varied_remaining),
    ).run()
    # All slots use the same absolute limit; exact output trimming must hold
    # even though their acceptance counts and completion cycles differ.
    assert varied.generated_output_counts == [max(varied_remaining)] * 12
    assert varied.overshoot_tokens > 0
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)
    print(
        {
            "pass": True,
            "cycles": args.cycles,
            "stage_order": runtime.stage_order,
            "target_calls": int(target_calls.cpu()),
            "proposer_calls": int(proposer.calls.cpu()),
            "serving_cycles": serving.cycles,
            "serving_output_tokens": sum(serving.generated_output_counts),
            "varied_serving_cycles": varied.cycles,
            "varied_serving_overshoot": varied.overshoot_tokens,
            "vllm_imported": False,
            "owned_cache_tensors": len(assets.caches),
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
