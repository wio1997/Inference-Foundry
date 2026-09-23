"""One-time vLLM-Ascend target handoff.

This module is intentionally outside ``runtime``.  It may import the oracle to
borrow loaded weights, process groups, bound attention operators, and physical
cache tensors.  The returned target binding calls the model directly and does
not retain or invoke ModelRunner/Scheduler objects per decode cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.config import set_current_vllm_config

from runtime.assets import OwnedCacheTensor, RuntimeAssets
from runtime.target_adapter import FixedTargetBinding
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.acl_graph import update_full_graph_params
from vllm_ascend.ops.rotary_embedding import update_cos_sin


def bind_graph_update(
    attn_backend: Any,
    update_stream: Any,
    vllm_config: Any,
) -> Callable[[Any, int], None]:
    """Bind graph metadata updates without retaining a ModelRunner."""

    def update(context: Any, num_tokens: int) -> None:
        update_full_graph_params(
            attn_backend,
            update_stream,
            context,
            num_tokens,
            vllm_config,
            vllm_config.speculative_config,
        )

    return update


def _gather_hidden(hidden: torch.Tensor) -> torch.Tensor:
    context = get_forward_context()
    pad_size = context.pad_size
    num_tokens = context.num_tokens
    padded = getattr(context, "padded_length", num_tokens + pad_size)
    if hidden.shape[0] != padded:
        hidden = tensor_model_parallel_all_gather(hidden, 0)
    if pad_size:
        hidden = hidden[:-pad_size]
    return hidden


def _gather_output(output):
    if isinstance(output, tuple):
        return _gather_hidden(output[0]), [
            _gather_hidden(hidden) for hidden in output[1]
        ]
    return _gather_hidden(output)


def _flatten_cache_tree(
    tree: list[Any],
    slot_specs: list[tuple[torch.Tensor, int]] | None = None,
) -> tuple[list[OwnedCacheTensor], dict[str, tuple[torch.Tensor, int]]]:
    owned: list[OwnedCacheTensor] = []
    owned_specs: dict[str, tuple[torch.Tensor, int]] = {}
    if slot_specs is not None and len(slot_specs) != len(tree):
        raise ValueError(
            f"cache slot spec count {len(slot_specs)} != cache tree {len(tree)}"
        )
    seen: set[int] = set()

    def visit(
        value: Any,
        name: str,
        slot_spec: tuple[torch.Tensor, int] | None,
    ) -> None:
        if torch.is_tensor(value):
            pointer = value.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                owned.append(OwnedCacheTensor.take(name, value))
                if slot_spec is not None:
                    owned_specs[name] = slot_spec
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{name}.{index}", slot_spec)

    for index, value in enumerate(tree):
        slot_spec = None if slot_specs is None else slot_specs[index]
        visit(value, f"kv_cache.{index}", slot_spec)
    return owned, owned_specs


@dataclass(frozen=True)
class TargetHandoffInputs:
    model: torch.nn.Module
    vllm_config: Any
    attn_metadata: Any
    batch_descriptor: Any
    has_sinks: bool
    kv_cache_tree: list[Any]
    num_tokens_across_dp: torch.Tensor | None = None
    aclgraph_runtime_mode: Any = None
    num_actual_tokens: int | None = None
    skip_compiled: bool = False
    eplb_heat_collection_status: bool = False
    graph_update: Callable[[Any, int], None] | None = None
    graph_update_before: bool = False
    model_kwargs: dict[str, Any] | None = None
    mutable_tensors: list[tuple[str, torch.Tensor]] | None = None
    cache_slot_specs: list[tuple[torch.Tensor, int]] | None = None


class DirectTargetHandoff:
    """Direct real-weight target callable after one-time oracle bootstrap."""

    def __init__(self, inputs: TargetHandoffInputs) -> None:
        self.model = inputs.model
        self.vllm_config = inputs.vllm_config
        self.attn_metadata = inputs.attn_metadata
        self.batch_descriptor = inputs.batch_descriptor
        self.has_sinks = inputs.has_sinks
        self.num_tokens_across_dp = inputs.num_tokens_across_dp
        self.aclgraph_runtime_mode = inputs.aclgraph_runtime_mode
        self.num_actual_tokens = inputs.num_actual_tokens
        self.skip_compiled = inputs.skip_compiled
        self.eplb_heat_collection_status = inputs.eplb_heat_collection_status
        self.graph_update = inputs.graph_update
        self.graph_update_before = inputs.graph_update_before
        self.model_kwargs = dict(inputs.model_kwargs or {})
        caches, cache_slot_specs = _flatten_cache_tree(
            inputs.kv_cache_tree, inputs.cache_slot_specs
        )
        self.assets = RuntimeAssets(
            caches,
            [
                OwnedCacheTensor.take(name, tensor)
                for name, tensor in (inputs.mutable_tensors or [])
            ],
            cache_slot_specs,
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        # Optional bootstrap-only diagnostic. Product runs leave this unset.
        diagnostic_refresh = getattr(self, "diagnostic_metadata_refresh", None)
        if diagnostic_refresh is not None:
            diagnostic_refresh(positions)
        num_tokens = input_ids.shape[0]
        update_cos_sin(positions)
        context_kwargs = {}
        if self.aclgraph_runtime_mode is not None:
            context_kwargs["aclgraph_runtime_mode"] = self.aclgraph_runtime_mode
        with set_current_vllm_config(self.vllm_config), set_ascend_forward_context(
                self.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=self.num_tokens_across_dp,
                num_actual_tokens=self.num_actual_tokens or num_tokens,
                batch_descriptor=self.batch_descriptor,
                model_instance=self.model,
                skip_compiled=self.skip_compiled,
                has_sinks=self.has_sinks,
                input_ids=input_ids,
                eplb_heat_collection_status=self.eplb_heat_collection_status,
                **context_kwargs,
            ):
            context = get_forward_context()
            if self.graph_update is not None and self.graph_update_before:
                self.graph_update(context, num_tokens)
            output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                **self.model_kwargs,
            )
            if self.graph_update is not None and not self.graph_update_before:
                self.graph_update(context, num_tokens)
            if context.flash_comm_v1_enabled:
                output = _gather_output(output)
        return output

    def binding(self) -> FixedTargetBinding:
        return FixedTargetBinding(
            forward=self.forward,
            compute_logits=self.model.compute_logits,
            state_fingerprint=lambda: {},
        )
