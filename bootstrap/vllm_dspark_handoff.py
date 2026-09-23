"""One-time extraction of the working DSpark7 operator from vLLM-Ascend."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch.profiler import record_function
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

    # Host mirrors are advanced by DirectDSparkHandoff from the tiny
    # acceptance-count vector.  Pulling query, sequence and computed tensors
    # back from the device here serialized every cycle before DSpark.


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
        self._profile_scopes = os.getenv("EXTREME_RUNTIME_PROFILE_SCOPES") == "1"
        self._profile_dag = os.getenv("EXTREME_RUNTIME_PROFILE_DAG") == "1"
        self.dag_events: list[list[tuple[str, Any]]] = []
        self._draft_group_slot_refresh = os.getenv("EXTREME_DRAFT_GROUP_SLOT_REFRESH") == "1"
        self.draft_group_slot_audit: list[dict[str, Any]] = []
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
        self._profile_hooks: list[Any] = []
        self._host_count_copy: torch.Tensor | None = None
        self._host_copy_stream: Any = None
        self._host_copy_event: Any = None
        self._host_copy_pending = False
        self._committed_emitted_count = torch.zeros(
            config.batch_size, dtype=torch.int64
        )
        if self.common_attn_metadata.query_start_loc_cpu is None:
            raise ValueError("fixed DSpark requires a bootstrap query-start host mirror")
        if getattr(self.common_attn_metadata, "_seq_lens_cpu", None) is None:
            raise ValueError("fixed DSpark requires a bootstrap sequence-length host mirror")
        if self.common_attn_metadata.query_start_loc_cpu.numel() < config.batch_size + 1:
            raise ValueError("bootstrap query-start host mirror is too small")
        if self.common_attn_metadata._seq_lens_cpu.numel() < config.batch_size:
            raise ValueError("bootstrap sequence-length host mirror is too small")
        device = self.common_attn_metadata.query_start_loc.device
        if device.type == "npu":
            self._host_count_copy = torch.empty(
                config.batch_size, dtype=torch.int32, pin_memory=True
            )
            self._host_copy_stream = torch.npu.Stream(device=device)
            self._host_copy_event = torch.npu.Event()
        if self._profile_scopes:
            self._install_layer_profile_hooks()

    def _scope(self, name: str):
        if self._profile_scopes:
            return record_function(name)
        return nullcontext()

    def _install_layer_profile_hooks(self) -> None:
        draft_model = getattr(getattr(self.proposer, "model", None), "model", None)
        layers = getattr(draft_model, "layers", None)
        if layers is None:
            return
        items = layers.items() if hasattr(layers, "items") else enumerate(layers)
        for name, layer in items:
            active: list[Any] = []

            def before(_module, _inputs, label=str(name), stack=active):
                scope = record_function(f"extreme::dspark_layer::{label}")
                scope.__enter__()
                stack.append(scope)

            def after(_module, _inputs, output, stack=active):
                stack.pop().__exit__(None, None, None)
                return output

            self._profile_hooks.append(layer.register_forward_pre_hook(before))
            self._profile_hooks.append(layer.register_forward_hook(after))

    @staticmethod
    def _unique_host_mirrors(common: Any, names: tuple[str, ...]) -> list[torch.Tensor]:
        mirrors: list[torch.Tensor] = []
        seen: set[int] = set()
        for name in names:
            value = getattr(common, name, None)
            if value is None:
                continue
            identity = value.data_ptr()
            if identity not in seen:
                mirrors.append(value)
                seen.add(identity)
        return mirrors

    def _launch_host_count_copy(self, counts: torch.Tensor) -> None:
        if self._host_count_copy is None:
            return
        if self._host_copy_pending:
            raise RuntimeError("previous DSpark host mirror copy was not committed")
        current = torch.npu.current_stream(counts.device)
        self._host_copy_stream.wait_stream(current)
        with torch.npu.stream(self._host_copy_stream):
            self._host_count_copy.copy_(counts, non_blocking=True)
            self._host_copy_event.record()
        self._host_copy_pending = True

    def _commit_host_mirrors(self) -> None:
        if self._host_count_copy is None or not self._host_copy_pending:
            return
        # The copy only depends on acceptance.  It runs on a side stream across
        # the rest of this cycle and the next target pass, then advances the
        # fixed mirrors immediately before their next DSpark consumer.
        self._host_copy_event.synchronize()
        batch = self.config.batch_size
        counts = self._host_count_copy[:batch]
        self._committed_emitted_count[:batch].add_(counts)
        for mirror in self._unique_host_mirrors(
            self.common_attn_metadata,
            ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"),
        ):
            mirror[:batch].add_(counts)
        self._host_copy_pending = False
        for mirror in self._unique_host_mirrors(
            self.common_attn_metadata,
            ("num_computed_tokens_cpu", "_num_computed_tokens_cpu"),
        ):
            mirror[:batch].add_(counts)

    def committed_emitted_token_count(self) -> torch.Tensor:
        """Return lagged Host progress without synchronizing the proposer."""

        return self._committed_emitted_count

    def validate_host_mirrors(self, state: FixedDecodeState) -> bool:
        """Post-run correctness gate; never called on the timed hot path."""

        self._commit_host_mirrors()
        batch = self.config.batch_size
        width = self.config.target_tokens_per_request
        query = state.target_query_start_loc.cpu()
        next_seq = state.num_computed_tokens.cpu() + width
        computed = state.num_computed_tokens.cpu()
        checks = [
            torch.equal(
                self.common_attn_metadata.query_start_loc_cpu[: batch + 1],
                query,
            )
        ]
        checks.append(
            torch.equal(
                self._committed_emitted_count[:batch],
                state.emitted_token_count.cpu().to(torch.int64),
            )
        )
        for mirror in self._unique_host_mirrors(
            self.common_attn_metadata,
            ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"),
        ):
            checks.append(torch.equal(mirror[:batch], next_seq))
        for mirror in self._unique_host_mirrors(
            self.common_attn_metadata,
            ("num_computed_tokens_cpu", "_num_computed_tokens_cpu"),
        ):
            checks.append(torch.equal(mirror[:batch], computed))
        return all(checks)

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
        markers: list[tuple[str, Any]] = []
        def mark(label: str) -> None:
            if self._profile_dag:
                event = torch.npu.Event(enable_timing=True)
                event.record()
                markers.append((label, event))
        mark("begin")
        # Consume the previous cycle's count copy only after the next target
        # pass has given the side stream hundreds of milliseconds to finish.
        # The current cycle's copy remains pending until the following cycle.
        with self._scope("extreme::dspark_host_mirror_commit"):
            self._commit_host_mirrors()
        with self._scope("extreme::dspark_host_mirror_launch"):
            self._launch_host_count_copy(acceptance.num_sampled)
        mark("host_mirror")
        with self._scope("extreme::dspark_refresh_common"):
            self.refresh_common(state, self.common_attn_metadata)
        mark("refresh_common")
        if self._draft_group_slot_refresh:
            positions = state.target_positions.view(self.config.batch_size, -1)
            request_index = torch.arange(
                self.config.batch_size, device=positions.device
            ).unsqueeze(1)
            for gid in sorted(self.proposer._per_group_slot_mappings):
                if gid not in {
                    group.kv_cache_group_id
                    for group in self.proposer.draft_attn_groups
                }:
                    continue
                block_size = self.proposer._per_group_kernel_block_sizes[gid]
                block_table = self.proposer._per_group_block_tables[gid]
                block_index = torch.div(
                    positions, block_size, rounding_mode="floor"
                )
                if bool((block_index >= block_table.shape[1]).any()):
                    raise RuntimeError(f"draft group {gid} block table too narrow")
                physical = block_table[request_index, block_index]
                expected = (
                    physical * block_size + positions.remainder(block_size)
                ).flatten().to(torch.int32)
                actual = self.proposer._per_group_slot_mappings[gid]
                equal = torch.equal(actual[:expected.numel()], expected)
                if len(self.draft_group_slot_audit) < 24:
                    self.draft_group_slot_audit.append(
                        {
                            "cycle": state.cycle_index,
                            "gid": gid,
                            "equal_before": bool(equal),
                            "mismatch_count": int(
                                (actual[:expected.numel()] != expected).sum().item()
                            ),
                        }
                    )
                if state.cycle_index == 0 and not equal:
                    raise RuntimeError(
                        f"draft group {gid} initial slot mapping disagrees with oracle"
                    )
                actual[:expected.numel()].copy_(expected)
        with self._scope("extreme::dspark_prepare_inputs"):
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
        mark("prepare_inputs")
        if os.getenv("EXTREME_PROPOSER_PARITY") == "1":
            self.parity_prepare = {
                "token_indices": token_indices.clone(),
                "sample_indices": token_indices_to_sample.clone(),
                "num_rejected": num_rejected.clone(),
                "query_start_loc": common.query_start_loc.clone(),
                "seq_lens": common.seq_lens.clone(),
            }
        with self._scope("extreme::dspark_pack_hidden"):
            hidden = (
                torch.cat(target.aux_hidden_states, dim=-1)
                if target.aux_hidden_states
                else target.hidden_states
            )
            target_token_ids = state.target_input_ids[token_indices]
            target_positions = state.target_positions[token_indices]
            target_hidden_states = hidden[token_indices]
        mark("pack_hidden")
        if os.getenv("EXTREME_PROPOSER_PARITY") == "1":
            self.parity_inputs = {
                "target_token_ids": target_token_ids.clone(),
                "target_positions": target_positions.clone(),
                "target_hidden_states": target_hidden_states.clone(),
                "next_token_ids": state.last_sampled_tokens.clone(),
            }
        with self._scope("extreme::dspark_model"):
            next_draft = self.proposer._propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
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
        mark("model")
        if markers:
            self.dag_events.append(markers)
        return next_draft

    @staticmethod
    def state_fingerprint() -> dict[str, torch.Tensor]:
        return {}
