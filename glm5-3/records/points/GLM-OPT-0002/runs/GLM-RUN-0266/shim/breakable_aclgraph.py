#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    weak_ref_workspaces,
)



_H9_MAP = None
_H9_LAST_MODE = None
_H9_TRANSITIONS = 0

def _h9_mode():
    # Research selector only; production diff has no mode file or witness.
    import mmap, os
    global _H9_MAP
    root = os.environ.get("GLM_STREAM_ORDER_DIAGNOSTIC_ROOT")
    if root is None:
        return 0
    if _H9_MAP is None:
        with open(root + "/stream_order_mode.bin", "rb") as source:
            _H9_MAP = mmap.mmap(source.fileno(), 1, access=mmap.ACCESS_READ)
    mode = _H9_MAP[0]
    assert mode in (0, 1)
    return mode

def _h9_record(mode, eligible, wrapper, entry):
    # One small CPU witness per mode transition, emitted on warmup only.
    import json, os
    from pathlib import Path
    global _H9_LAST_MODE, _H9_TRANSITIONS
    root = os.environ.get("GLM_STREAM_ORDER_DIAGNOSTIC_ROOT")
    if root is None or mode == _H9_LAST_MODE:
        return
    from vllm.distributed import get_tp_group
    rank = get_tp_group().rank_in_group
    config = wrapper.vllm_config
    compilation = config.compilation_config
    row = dict(rank=rank, pid=os.getpid(), mode=mode, eligible=bool(eligible),
        skipped_host_sync=bool(mode and eligible and not wrapper.enable_enpu),
        target=True, runtime_mode=str(get_forward_context().cudagraph_runtime_mode),
        capture_num_graphs=entry.capture.num_graphs, eager_breaks=entry.capture.num_eager_breaks,
        num_tokens=entry.batch_descriptor.num_tokens,
        target_enforce_eager=config.model_config.enforce_eager,
        MTP_enforce_eager=config.speculative_config.enforce_eager,
        compilation_mode=str(compilation.mode), graph_mode=str(compilation.cudagraph_mode),
        capture_sizes=list(compilation.cudagraph_capture_sizes),
        candidate_graph_sha256='90630b608a612fdb2ce777f8411f374c675d7143f39c1689c8b38675c8ff83a6', candidate_runner_sha256='b54800adaafcb357856cc1ce76cd845d5d825c675745518e1bade41619f81615',
        transition_count=_H9_TRANSITIONS + 1, logging_scope="mode transition warmup")
    path = Path(root) / "witnesses" / ("mode%d_rank%d.json" % (mode, rank))
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(row, sort_keys=True) + "\n")
    temp.replace(path)
    _H9_TRANSITIONS += 1
    _H9_LAST_MODE = mode


class BreakableACLGraphWrapper(BreakableCUDAGraphWrapper):
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ) -> None:
        super().__init__(
            runnable=runnable,
            vllm_config=vllm_config,
        )

        self.use_eagle = use_eagle
        self.enable_enpu = enable_enpu

    def _capture(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        is_full_capture = forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        if is_full_capture:
            # Ascend FULL graph attention creates task groups and records the
            # mutable graph parameters only while this flag is set.
            forward_context.capturing = True

        output = super()._capture(entry, args, kwargs)

        if is_full_capture:
            # Keep the same workspace lifetime contract as ACLGraphWrapper.
            weak_ref_workspaces(get_graph_params())
            weak_ref_workspaces(get_draft_graph_params())
            weak_ref_workspaces(get_draft_graph_prefill_params())

        return output

    def _replay(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        mode = 0
        stream_ordered_sfa_target = False
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # Match ACLGraphWrapper's ordering between async attention
            # parameter updates and the previous/current FULL graph replay.
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
            stream_ordered_sfa_target = (
                not _EXTRA_CTX.is_draft_model
                and getattr(forward_context, "_ascend_sfa_stream_ordered_replay", False)
            )
            mode = _h9_mode() if not _EXTRA_CTX.is_draft_model else 0
            if not self.enable_enpu and not is_draft_eagle and not (mode and stream_ordered_sfa_target):
                torch.npu.current_stream().synchronize()
        super()._replay(entry, args, kwargs)
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.is_draft_model:
            _h9_record(mode, stream_ordered_sfa_target, self, entry)
        return entry.output
