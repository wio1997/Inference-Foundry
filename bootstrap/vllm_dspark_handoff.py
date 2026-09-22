"""One-time extraction of the working DSpark7 operator from vLLM-Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from vllm.config import set_current_vllm_config

from runtime.fixed_decode import AcceptanceOutput, FixedDecodeConfig, FixedDecodeState, TargetOutput


def refresh_fixed_common(
    state: FixedDecodeState,
    common: Any,
) -> None:
    """Refresh the fixed common-attention buffers without a metadata builder."""

    batch = state.target_seq_lens.shape[0]
    total = state.target_input_ids.shape[0]
    common.query_start_loc[: batch + 1].copy_(state.target_query_start_loc)
    common.seq_lens[:batch].copy_(state.target_seq_lens)
    common.slot_mapping[:total].copy_(state.target_slot_mapping)
    if common.slot_mapping.shape[0] > total:
        common.slot_mapping[total:].fill_(-1)
    if common.positions is not None:
        common.positions[:total].copy_(state.target_positions)
    common.block_table_tensor = state.block_table
    common.num_reqs = batch
    common.num_actual_tokens = total
    common.num_input_tokens = total
    common.max_query_len = total // batch

    # M2 correctness bridge: current DSA builders still consume pinned host
    # mirrors.  Keeping these mirrors here makes the dependency explicit and
    # removable; the product runtime state itself remains device-owned.
    query_cpu = state.target_query_start_loc.cpu()
    seq_cpu = state.target_seq_lens.cpu()
    computed_cpu = state.num_computed_tokens.cpu()
    if common.query_start_loc_cpu is not None:
        common.query_start_loc_cpu[: batch + 1].copy_(query_cpu)
    for name in ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
        value = getattr(common, name, None)
        if value is not None:
            value[:batch].copy_(seq_cpu)
    for name in ("num_computed_tokens_cpu", "_num_computed_tokens_cpu"):
        value = getattr(common, name, None)
        if value is not None:
            value[:batch].copy_(computed_cpu)


@dataclass(frozen=True)
class DSparkHandoffInputs:
    proposer: Any
    common_attn_metadata: Any
    sampling_metadata: Any
    target_model_batch_desc: Any
    actual_seq_lengths_q: Any
    attn_state: Any
    decode_token_per_req: Any
    refresh_common: Callable[[FixedDecodeState, Any], None]


class _DP1RunnerShim:
    """Only the fixed eager DSpark fields still read by the borrowed operator."""

    def __init__(self, inputs: DSparkHandoffInputs) -> None:
        self.dcp_manager = None
        self.input_batch = SimpleNamespace(lora_id_to_lora_request={})
        self.dynamic_eplb = False
        self.eplb_heat_collection_status = False
        self.num_rejected_tokens_event = None
        self.actual_seq_lengths_q = inputs.actual_seq_lengths_q
        self.attn_state = inputs.attn_state
        self.decode_token_per_req = inputs.decode_token_per_req

    @staticmethod
    def _sync_metadata_across_dp(num_tokens: int, **_: Any):
        return num_tokens, None, False


class DirectDSparkHandoff:
    """Request-free adapter around the already-loaded fixed DSpark model.

    The proposer/model and its device buffers are retained, but its reference
    to the generic ModelRunner is replaced during handoff by a DP1 fixed shim.
    """

    def __init__(
        self,
        config: FixedDecodeConfig,
        inputs: DSparkHandoffInputs,
    ) -> None:
        if getattr(inputs.proposer, "use_cuda_graph", False):
            raise ValueError("initial Extreme DSpark handoff requires eager proposer")
        self.config = config
        self.proposer = inputs.proposer
        self.common_attn_metadata = inputs.common_attn_metadata
        self.sampling_metadata = inputs.sampling_metadata
        self.target_model_batch_desc = inputs.target_model_batch_desc
        self.refresh_common = inputs.refresh_common
        self.proposer.runner = _DP1RunnerShim(inputs)
        draft_counts = torch.full(
            (config.batch_size,),
            config.speculative_tokens,
            dtype=torch.int32,
            device=self.common_attn_metadata.query_start_loc.device,
        )
        self._spec_metadata = SimpleNamespace(
            cu_num_draft_tokens=draft_counts.cumsum(dim=0)
        )

    def execute(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
        acceptance: AcceptanceOutput,
    ) -> torch.Tensor:
        with set_current_vllm_config(self.proposer.vllm_config):
            return self._execute(state, target, acceptance)

    def _execute(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
        acceptance: AcceptanceOutput,
    ) -> torch.Tensor:
        self.refresh_common(state, self.common_attn_metadata)
        (
            common,
            token_indices,
            token_indices_to_sample,
            num_rejected,
        ) = self.proposer.prepare_inputs_padded(
            self.common_attn_metadata,
            self._spec_metadata,
            acceptance.num_sampled,
        )
        hidden = (
            torch.cat(target.aux_hidden_states, dim=-1)
            if target.aux_hidden_states
            else target.hidden_states
        )
        return self.proposer._propose(
            target_token_ids=state.target_input_ids[token_indices],
            target_positions=state.target_positions[token_indices],
            target_hidden_states=hidden[token_indices],
            next_token_ids=state.last_sampled_tokens,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common,
            target_model_batch_desc=self.target_model_batch_desc,
            sampling_metadata=self.sampling_metadata,
            num_scheduled_tokens=self.config.target_token_count,
            num_rejected_tokens_gpu=num_rejected,
            num_draft_tokens_cpu=[
                self.config.speculative_tokens
            ] * self.config.batch_size,
        )

    @staticmethod
    def state_fingerprint() -> dict[str, torch.Tensor]:
        return {}
