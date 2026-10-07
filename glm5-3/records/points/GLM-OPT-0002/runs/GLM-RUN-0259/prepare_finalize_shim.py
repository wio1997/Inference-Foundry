# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.lora.fused_moe import prepare_lora_indices
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg


class PrepareAndFinalize(ABC):
    """
    Abstract base class for MoE (Mixture-of-Experts) tensor preparation and finalization
    in distributed environments. Subclasses implement specific communication strategies
    (e.g., AllGather, All2All, MC2) to handle tensor padding, slicing,
    broadcasting, and reduction across TP/DP/EP groups.

    Attributes:
        moe_config (FusedMoEConfig): Configuration object containing TP/DP/EP group info,
                                     sizes, ranks, and communication settings.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config
        self.lora_context = None
        self.dynamic_mx_quant_scale_alg = get_dynamic_mx_quant_scale_alg()

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Prepare tensors before MoE computation. May involve:
          - Padding to align communication boundaries
          - Slicing across tensor-parallel ranks
          - Broadcasting across data-parallel ranks

        Args:
            hidden_states (torch.Tensor): Input features, shape [num_tokens, hidden_size]
            router_logits (torch.Tensor): Router outputs, shape [num_tokens, num_experts]
            replace_allreduce (bool): Bypass default all-reduce behavior
            quant_type: none, w8a8, w4a8, mxfp8, or mxfp4

        Returns:
            MoEPrepareOutput:
                - processed hidden_states (may be padded/sliced/broadcasted)
                - processed router_logits (may be recomputed or broadcasted)
                - optional communication mask (e.g., mc2_mask for sparse ops)
                - optional padded hidden state shape for finalization
                - optional per-token scale for quantized path
        """
        raise NotImplementedError("Prepare not implemented.")

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalize MoE output. May involve:
          - Gathering sliced tensors across TP ranks
          - Reducing or scattering across DP ranks
          - Unpadding to original token count
          - Applying all-reduce across TP/EP if requested

        Args:
            hidden_states (torch.Tensor): MoE layer output, possibly padded or sliced
            reduce_results (bool): Whether to apply all-reduce across TP/EP groups

        Returns:
            torch.Tensor: Final output with shape [original_num_tokens, hidden_size]
        """
        raise NotImplementedError("Finalize function not implemented.")


class PrepareAndFinalizeWithAll2All(PrepareAndFinalize):
    """
    MoE communication strategy using All-to-All style slicing.
    Similar to MC2 but does not use mc2_mask; instead pads to TP size for uniform slicing.
    Will be used when num_tokens exceed mc2's limitation (512 tokens/rank).
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """Restore original TP configuration (same as MC2)."""
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Pad hidden_states and router_logits to next multiple of TP size.
          2. If TP > 1, split along token dim and select current TP rank's slice.
          3. Save splits for later all-gather in finalize.

        Skips if `replace_allreduce` is True.

        Returns:
            MoEPrepareOutput where `mc2_mask` is None for All2All path.
        """
        self.replace_allreduce = replace_allreduce

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)
            if self.lora_context is not None:
                prepare_lora_indices(
                    self.lora_context,
                    num_tokens=self.num_tokens,
                    pad_size=pad_size,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                )

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)

                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def pad_and_split_input_ids(
        self,
        input_ids,
    ):
        if not self.replace_allreduce:
            pad_size = self.tp_size - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `replace_allreduce` is True.
        """

        if not self.replace_allreduce:
            if self.tp_size > 1:
                assert padded_hidden_states_shape is not None
                # Cannot reuse `split_hidden_states` from prepare phase as it
                # may share memory with original hidden_states. Since shared
                # experts may use the original tensor, reusing it would cause
                # in-place modification during all_gather, corrupting the data.
                gathered_hidden_states = torch.empty(
                    padded_hidden_states_shape, device=hidden_states.device, dtype=hidden_states.dtype
                )
                split_hidden_states = torch.tensor_split(gathered_hidden_states, self.tp_size, dim=0)
                dist.all_gather(list(split_hidden_states), hidden_states, self.moe_config.tp_group.device_group)
                hidden_states = gathered_hidden_states

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[: self.num_tokens]

        return hidden_states


class PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All):
    """
    MoE communication strategy using MC2, based on All2All with additional
    DP-wide padding and unpadding for sequence-parallel inputs.
    Designed for Ascend or environments requiring explicit padding and slicing control.
    Relies on `mc2_mask` and `padded_num_tokens` from forward_context for alignment.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """
        Restore original TP configuration.
        vLLM flattens TP and DP into a single dimension; this method recovers
        the true TP world size and rank for correct tensor slicing.
        """
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch `mc2_mask` and target padding length from forward context.
          2. Pad `hidden_states` and `router_logits` to target length if needed.
          3. If TP > 1, split tensors along token dimension and select current TP rank's slice.
          4. Split and return corresponding `mc2_mask`.

        With `replace_allreduce`, inputs are already TP-sharded. Pad only the
        local shard to the DP-wide MC2 length, preserving its original mask.

        Returns:
            MoEPrepareOutput, possibly sliced/padded.
        """
        self.replace_allreduce = replace_allreduce
        mc2_mask = _EXTRA_CTX.mc2_mask
        if self.replace_allreduce:
            # SP shards use the local token count, not the largest DP batch.
            # Select valid bits before adding padding for uniform MC2 batches.
            self.num_tokens = hidden_states.shape[0]
            start = self.tp_rank * self.num_tokens
            mc2_mask = mc2_mask[start : start + self.num_tokens]
            pad_size = _EXTRA_CTX.padded_num_tokens // self.tp_size - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                mc2_mask = nn.functional.pad(mc2_mask, (0, pad_size), value=False)
        elif self.tp_size > 1:
            # Also slice mc2_mask
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            # Slice across TP ranks
            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=mc2_mask,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.replace_allreduce:
            # Return the original SP shard to the residual/shared-expert path.
            return hidden_states[: self.num_tokens]
        return super().finalize(hidden_states, reduce_results, padded_hidden_states_shape)

    def pad_and_split_input_ids(
        self,
        input_ids,
    ):
        if self.replace_allreduce:
            # MoE-only SP retains full token IDs, while model-level SP may
            # already shard them. Align to the local hidden states first.
            if input_ids.numel() != self.num_tokens:
                input_ids = sequence_parallel_chunk(input_ids.reshape(-1, 1)).reshape(-1)
            pad_size = _EXTRA_CTX.padded_num_tokens // self.tp_size - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))
        else:
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids


class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
    """
    MoE communication strategy using All-Gather + Reduce-Scatter on EP group.
    There are two sets of prepare and finalize:
    1. _prepare_with_dp_group/_finalize_with_dp_group: When sequence parallelism is not enabled,
    we gather inputs across DP ranks before MoE, scatter outputs after.
    The communication and calculation process is as follows (AG, AR and RS
    are abbreviations for All-Gather, All-Reduce and Reduce-Scatter, respectively):

    Attn → TP AR → DP AG → MoE → DP RS → TP AR

    2. _prepare_with_ep_group/_finalize_with_ep_group: When sequence parallelism is enabled,
    the above process becomes:

    TP AG → Attn → TP RS → TP AG → DP AG → MoE → DP RS → TP RS

    This strategy further combines TP AG + DP AG into EP All-Gather and TP RS + DP RS
    into EP Reduce-Scatter to improve communication performance. The optimized process is as follows:

    TP AG → Attn → TP RS → EP AG → MoE → EP RS
    """

    def _use_ep_sequence_parallel(self) -> bool:
        """Whether MoE itself must use the EP sequence-parallel path.

        The MoE configuration owns sequence-parallel tokens.  Selecting the
        EP path from any other flag would gather tokens a second time before
        routing.
        """
        return self.moe_config.is_sequence_parallel

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          AllGather hidden_states and router_logits to form global tensors.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        if self._use_ep_sequence_parallel():
            return self._prepare_with_ep_group(hidden_states, router_logits, quant_type)

        return self._prepare_with_dp_group(hidden_states, router_logits, replace_allreduce)

    def _prepare_with_ep_group(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, quant_type=QuantType.NONE
    ) -> MoEPrepareOutput:
        pertoken_scale = None
        if quant_type == QuantType.W8A8:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        elif quant_type in (QuantType.W8A8MXFP, QuantType.W4A8MXFP):
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
                scale_alg=self.dynamic_mx_quant_scale_alg,
            )
        elif quant_type == QuantType.W4A4MXFP:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch_npu.float4_e2m1fn_x2,
                round_mode="round",
            )

        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states)
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(router_logits)

        self.num_tokens = hidden_states.shape[0]

        if pertoken_scale is not None:
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(pertoken_scale)

        if self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                if pertoken_scale is not None:
                    pertoken_scale = (
                        nn.functional.pad(pertoken_scale, (0, pad_size))
                        if pertoken_scale.dim() == 1
                        else nn.functional.pad(pertoken_scale, (0, 0, 0, pad_size))
                    )

            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)
            if pertoken_scale is not None:
                pertoken_scale = get_pcp_group().all_gather(pertoken_scale, dim=0)

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=pertoken_scale,
        )

    def _prepare_with_dp_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch max token count across DP group from forward context.
          2. Pad local tensors to that size.
          3. All-gather across DP group to form global input tensor.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            # All-gather across DP group
            hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)

        if self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    def all_gather_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._use_ep_sequence_parallel():
            return self.all_gather_input_id_with_ep_group(input_ids)
        return self.all_gather_input_id_with_dp_group(input_ids)

    def all_gather_input_id_with_ep_group(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_ids.reshape(-1, 1)).reshape(-1)
        if self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp
            pad_size = max_tokens_across_pcp - input_ids.numel()
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))
            input_ids = get_pcp_group().all_gather(input_ids, dim=0)
        return input_ids

    def all_gather_input_id_with_dp_group(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            input_ids = self.moe_config.dp_group.all_gather(input_ids, 0)
        return input_ids

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          Reduce Scatter hidden states.

        Returns:
            Tensor with shape [local_num_tokens, hidden_size]
        """
        if self._use_ep_sequence_parallel():
            return self._finalize_with_ep_group(hidden_states)

        return self._finalize_with_dp_group(hidden_states, reduce_results)

    def _finalize_with_ep_group(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Argument `reduce_results` is not needed in this func. Given sequence parallelism is enabled:
        1. Reduce_results is False usually happens when models have shared experts and need to
        allreduce hidden states after results of shared experts and routed experts are added in FusedMoe.
        We do reduce scatter for hidden states here, then skip allreudce in FusedMoe and add it to the
        result of shared experts.
        2 Reduce_results is True usually happens when model has no shared experts. We still do reduce scatter
        here, then skip allreudce in FusedMoe.
        """
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
            hidden_states = hidden_states[: self.num_tokens_pcp]

        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)

        return hidden_states

    def _finalize_with_dp_group(self, hidden_states: torch.Tensor, reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert, reduce-scatter output across DP group.
          2. Slice to original local token count.
          3. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [original_local_num_tokens, hidden_size]
        """
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
            hidden_states = hidden_states[: self.num_tokens_pcp]

        if self.moe_config.dp_size > 1:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[: self.num_tokens]
        return hidden_states

def _glm53_direct_finalize(self, hidden_states: torch.Tensor, reduce_results: bool, padded_hidden_states_shape: torch.Size | None=None) -> torch.Tensor:
    """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `replace_allreduce` is True.
        """
    if not self.replace_allreduce:
        if self.tp_size > 1:
            assert padded_hidden_states_shape is not None
            gathered_hidden_states = torch.empty(padded_hidden_states_shape, device=hidden_states.device, dtype=hidden_states.dtype)
            local_numel = gathered_hidden_states.numel() // self.tp_size
            if padded_hidden_states_shape[0] > 0 and padded_hidden_states_shape[0] % self.tp_size == 0 and (local_numel * hidden_states.element_size() % 512 == 0):
                assert hidden_states.shape == (padded_hidden_states_shape[0] // self.tp_size, *padded_hidden_states_shape[1:])
                dist.all_gather_into_tensor(gathered_hidden_states, hidden_states.contiguous(), group=self.moe_config.tp_group.device_group)
            else:
                split_hidden_states = torch.tensor_split(gathered_hidden_states, self.tp_size, dim=0)
                dist.all_gather(list(split_hidden_states), hidden_states, self.moe_config.tp_group.device_group)
            hidden_states = gathered_hidden_states
        if self.num_tokens < hidden_states.shape[0]:
            hidden_states = hidden_states[:self.num_tokens]
    return hidden_states
# Diagnostic-only same-process selector. Mode changes only at controller idle.
import mmap as _glm53_mmap, os as _glm53_os, json as _glm53_json
from pathlib import Path as _glm53_Path
_glm53_root = _glm53_Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0259')
_glm53_fd = _glm53_os.open(_glm53_root / 'gather_mode.bin', _glm53_os.O_RDONLY)
_glm53_mode = _glm53_mmap.mmap(_glm53_fd, 1, access=_glm53_mmap.ACCESS_READ)
_glm53_os.close(_glm53_fd)
_glm53_last_mode = None
_glm53_stock_finalize = PrepareAndFinalizeWithAll2All.finalize

def _glm53_matched_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape=None):
    global _glm53_last_mode
    mode = _glm53_mode[0]
    assert mode in (0,1,2,3), 'invalid diagnostic mode'
    method = _glm53_direct_finalize if mode % 2 else _glm53_stock_finalize
    if mode != _glm53_last_mode:
        assert self.tp_size == 16 and not self.replace_allreduce
        assert padded_hidden_states_shape is not None and hidden_states.shape[-1] == 6144
        equal = None
        if mode % 2:
            before_cpu = hidden_states.detach().cpu()
            before_bits = before_cpu.contiguous().view(torch.uint8).clone()
            original_num_tokens = self.num_tokens
            try:
                self.num_tokens = padded_hidden_states_shape[0]
                expected = _glm53_stock_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape)
                result = _glm53_direct_finalize(self, hidden_states, reduce_results, padded_hidden_states_shape)
            finally:
                self.num_tokens = original_num_tokens
            expected_cpu = expected.detach().cpu()
            result_cpu = result.detach().cpu()
            equal = bool(torch.equal(expected_cpu.contiguous().view(torch.uint8), result_cpu.contiguous().view(torch.uint8)))
            input_equal = bool(torch.equal(before_bits, hidden_states.detach().cpu().contiguous().view(torch.uint8)))
            output_private = result.data_ptr() != hidden_states.data_ptr()
            flag = torch.tensor([int(equal and input_equal and output_private)], dtype=torch.int32, device=hidden_states.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.moe_config.tp_group.device_group)
            all_equal = bool(flag.item())
            result = result[:original_num_tokens]
        else:
            result = method(self, hidden_states, reduce_results, padded_hidden_states_shape)
        rank = dist.get_rank()
        row = dict(rank=rank, pid=_glm53_os.getpid(), mode=mode,
                   actual_hidden_bitwise_equal=equal, input_bytes_equal=input_equal if mode%2 else None, all_ranks_bytes_equal=all_equal if mode%2 else None, input_nan_count=int(torch.isnan(before_cpu).sum()) if mode%2 else None, input_format=torch_npu.get_npu_format(hidden_states), shape=list(hidden_states.shape),
                   padded_shape=list(padded_hidden_states_shape), dtype=str(hidden_states.dtype),
                   local_bytes=hidden_states.numel()*hidden_states.element_size(),
                   input_preserved=input_equal if mode%2 else None,
                   artifact='candidate:'+'26e52eef2714f9e3246d001b9ffc4e05d453a79cf95c3c38c0a2ba860498f3f9')
        (_glm53_root/'witnesses'/('mode%d_rank%d.json'%(mode,rank))).write_text(_glm53_json.dumps(row)+'\n')
        _glm53_last_mode = mode
        if mode%2 and not all_equal: raise RuntimeError('all-rank full-gather byte gate failed; witnesses retained')
        return result
    return method(self, hidden_states, reduce_results, padded_hidden_states_shape)

PrepareAndFinalizeWithAll2All.finalize = _glm53_matched_finalize
