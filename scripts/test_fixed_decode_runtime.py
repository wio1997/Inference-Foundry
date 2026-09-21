#!/usr/bin/env python3
"""CPU semantic smoke for the fixed-buffer multi-cycle runtime.

This does not claim model correctness.  It proves that the runtime ABI is
causal across cycles and requires no SchedulerOutput/request/InputBatch object.
Real-weight parity is the next integration gate.
"""

from __future__ import annotations

import argparse

import torch

from runtime.fixed_decode import (
    AcceptanceOutput,
    FixedDecodeConfig,
    FixedDecodeRuntime,
    FixedDecodeState,
    TargetOutput,
)
from runtime.oracle_shadow import FixedDecodeOracleShadow
from runtime.greedy_accept import greedy_accept
from runtime.target_adapter import FixedTargetAdapter, FixedTargetBinding


class DeterministicOperators:
    def __init__(self, config: FixedDecodeConfig) -> None:
        self.config = config

    def target_forward(self, state: FixedDecodeState) -> TargetOutput:
        hidden = state.target_input_ids.to(torch.float32).unsqueeze(1)
        return TargetOutput(
            logits=hidden,
            hidden_states=hidden,
            aux_hidden_states=(),
        )

    def verify(
        self, state: FixedDecodeState, target: TargetOutput
    ) -> AcceptanceOutput:
        width = self.config.target_tokens_per_request
        count = 2 + (state.cycle_index % (width - 1))
        sampled = torch.full(
            (self.config.batch_size, width),
            -1,
            dtype=torch.int64,
            device=state.block_table.device,
        )
        base = state.last_sampled_tokens.to(torch.int64).unsqueeze(1)
        sampled[:, :count] = base + torch.arange(
            1, count + 1, dtype=torch.int64, device=base.device
        )
        num_sampled = torch.full(
            (self.config.batch_size,),
            count,
            dtype=torch.int32,
            device=base.device,
        )
        return AcceptanceOutput(sampled, num_sampled)

    def propose(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
        acceptance: AcceptanceOutput,
    ) -> torch.Tensor:
        offsets = torch.arange(
            1,
            self.config.speculative_tokens + 1,
            dtype=torch.int64,
            device=state.block_table.device,
        )
        return state.last_sampled_tokens.to(torch.int64).unsqueeze(1) + offsets

    def state_fingerprint(self, state: FixedDecodeState) -> dict[str, torch.Tensor]:
        return {
            "num_computed_tokens": state.num_computed_tokens.clone(),
            "last_sampled_tokens": state.last_sampled_tokens.clone(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = FixedDecodeConfig()
    max_blocks = cfg.max_model_len // cfg.block_size
    block_table = torch.arange(
        cfg.batch_size * max_blocks, dtype=torch.int32, device=device
    ).view(cfg.batch_size, max_blocks)
    initial_num_computed = torch.full(
        (cfg.batch_size,), 32_768, dtype=torch.int32, device=device
    )
    initial_last = torch.arange(cfg.batch_size, dtype=torch.int64, device=device) + 100
    initial_draft = initial_last.unsqueeze(1) + torch.arange(
        1, cfg.speculative_tokens + 1, dtype=torch.int64, device=device
    )
    state = FixedDecodeState.allocate(
        cfg,
        block_table,
        initial_num_computed.clone(),
        initial_last.clone(),
        initial_draft,
    )
    runtime = FixedDecodeRuntime(cfg, state, DeterministicOperators(cfg))
    outputs = runtime.run(8)

    expected_advance = sum(int(x.num_sampled[0]) for x in outputs)
    assert state.cycle_index == 8
    assert torch.equal(
        state.num_computed_tokens,
        initial_num_computed + expected_advance,
    )
    assert torch.equal(
        state.emitted_token_count,
        torch.full(
            (cfg.batch_size,), expected_advance, dtype=torch.int32, device=device
        ),
    )
    assert state.target_input_ids.shape == (96,)
    assert state.target_positions.shape == (96,)
    assert state.target_query_start_loc.tolist() == list(range(0, 97, 8))
    assert bool(torch.all(state.target_slot_mapping >= 0))

    shadow = FixedDecodeOracleShadow(cfg)
    first_ids = torch.cat(
        (initial_last.unsqueeze(1), initial_draft), dim=1
    ).to(torch.int32).flatten()
    first_positions = (
        initial_num_computed.to(torch.int64).unsqueeze(1)
        + torch.arange(
            cfg.target_tokens_per_request, dtype=torch.int64, device=device
        )
    ).flatten()
    assert (
        shadow.observe_target_inputs(
            block_table=block_table,
            input_ids=first_ids,
            positions=first_positions,
            query_start_loc=torch.arange(
                0, 97, 8, dtype=torch.int32, device=device
            ),
        )
        is None
    )
    first_acceptance = outputs[0]
    next_draft = (
        first_acceptance.sampled_token_ids[:, 1:]
        if first_acceptance.sampled_token_ids.shape[1] == cfg.target_tokens_per_request
        else initial_draft
    )
    shadow.observe_cycle_outputs(first_acceptance.sampled_token_ids, next_draft)
    predicted = shadow.runtime.state
    comparison = shadow.observe_target_inputs(
        block_table=block_table,
        input_ids=predicted.target_input_ids.clone(),
        positions=predicted.target_positions.clone(),
        query_start_loc=predicted.target_query_start_loc.clone(),
        seq_lens=predicted.target_seq_lens.clone(),
        slot_mapping=predicted.target_slot_mapping.clone(),
    )
    assert comparison is not None and comparison.exact

    drafts = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
        dtype=torch.int32,
        device=device,
    )
    target = torch.tensor(
        [[10, 11, 99, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
        dtype=torch.int32,
        device=device,
    )
    accepted, counts = greedy_accept(
        drafts,
        target,
        torch.tensor([77, 88], dtype=torch.int64, device=device),
    )
    assert accepted.tolist() == [
        [10, 11, 99, -1, -1, -1, -1, -1],
        [20, 21, 22, 23, 24, 25, 26, 88],
    ]
    assert counts.tolist() == [3, 8]

    target_forward_calls = 0

    def target_forward(input_ids: torch.Tensor, positions: torch.Tensor):
        nonlocal target_forward_calls
        target_forward_calls += 1
        hidden = torch.stack((input_ids, positions.to(input_ids.dtype)), dim=1)
        return hidden, [hidden + 1]

    target_adapter = FixedTargetAdapter(
        cfg,
        FixedTargetBinding(
            forward=target_forward,
            compute_logits=lambda hidden: hidden.to(torch.float32),
            state_fingerprint=lambda: {
                "cycle": torch.tensor(target_forward_calls, device=device)
            },
        ),
    )
    target_output = target_adapter.execute(runtime.state)
    assert target_output.hidden_states.shape == (cfg.target_token_count, 2)
    assert target_output.logits.shape == (cfg.target_token_count, 2)
    assert len(target_output.aux_hidden_states) == 1
    first_input_ptr = runtime.state.target_input_ids.data_ptr()
    first_position_ptr = runtime.state.target_positions.data_ptr()
    target_adapter.execute(runtime.state)
    assert runtime.state.target_input_ids.data_ptr() == first_input_ptr
    assert runtime.state.target_positions.data_ptr() == first_position_ptr
    assert target_forward_calls == 2
    print(
        {
            "pass": True,
            "cycles": state.cycle_index,
            "batch_size": cfg.batch_size,
            "target_tokens_per_cycle": cfg.target_token_count,
            "advanced_per_request": expected_advance,
            "shadow_next_cycle_exact": comparison.exact,
            "greedy_accept_exact": True,
            "fixed_target_adapter_calls": target_forward_calls,
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
