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



# Run263 bounded observation, outside capture/replay internals. No math change.
import json as _glm_json
import os as _glm_os
from pathlib import Path as _glm_Path

def _glm_tensor_snapshot(t):
    if not isinstance(t, torch.Tensor):
        return None
    # Diagnostic-only synchronization; these requests cannot claim latency gain.
    sample = t.detach().reshape(-1)[:32].to("cpu").tolist()
    return {"ptr": t.data_ptr(), "shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "device": str(t.device), "first32": sample}

def _glm_graph_observe(wrapper, kind, args, kwargs, error=None):
    root = _glm_Path(_glm_os.environ["GLM_GRAPH_DIAGNOSTIC_ROOT"])
    scope = (root / "scope.txt").read_text().strip()
    from vllm.distributed import get_tp_group
    rank = get_tp_group().rank_in_group
    fc = get_forward_context()
    draft = bool(_EXTRA_CTX.is_draft_model)
    mode = str(fc.cudagraph_runtime_mode)
    bd = fc.batch_descriptor
    desc = {name: getattr(bd, name, None) for name in
            ("num_tokens", "num_reqs", "uniform", "has_lora")} if bd is not None else None
    state = wrapper.__dict__.setdefault("_glm_diag", {"counts": {}, "snapshots": [], "errors": []})
    key = scope + ":" + ("draft" if draft else "target") + ":" + mode + ":" + kind
    state["counts"][key] = state["counts"].get(key, 0) + 1
    if error is not None:
        state["errors"].append({"scope": scope, "kind": kind, "error": repr(error)})
    snapshots = [row for row in state["snapshots"] if row["scope"] == scope]
    # Startup capture snapshot plus two real FULL replay inputs per PD request.
    take = kind == "capture_before" or (kind == "replay_before" and scope != "startup"
            and not draft and len(snapshots) < 2)
    if take:
        metadata = fc.attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        groups = []
        seen = set()
        for name, md in sorted((metadata or {}).items()):
            if id(md) in seen:
                continue
            seen.add(id(md))
            fields = {field: _glm_tensor_snapshot(getattr(md, field, None)) for field in
                      ("seq_lens", "seq_lens_cpu", "cum_query_lens", "slot_mapping",
                       "block_table", "sin", "cos")}
            fields["scalar"] = {field: str(getattr(md, field, None)) for field in
                      ("attn_state", "num_input_tokens", "num_actual_tokens", "num_decodes", "num_prefills", "block_size")}
            dc = getattr(md, "dcp_context", None)
            fields["dcp"] = {field: _glm_tensor_snapshot(getattr(dc, field, None)) for field in
                        ("seq_lens", "slot_mapping", "block_table")} if dc else None
            groups.append({"first_layer": name, "metadata_type": type(md).__name__, "fields": fields})
        state["snapshots"].append({"scope": scope, "kind": kind, "draft": draft, "mode": mode,
            "descriptor": desc, "inputs": {name: _glm_tensor_snapshot(t) for name,t in kwargs.items()
                if isinstance(t, torch.Tensor)}, "metadata": groups})
    entries = [{"descriptor": str(entry.batch_descriptor),
                "captured": entry.capture is not None,
                "graphs": entry.capture.num_graphs if entry.capture is not None else None,
                "eager_breaks": entry.capture.num_eager_breaks if entry.capture is not None else None,
                "input_addresses": entry.input_addresses} for entry in wrapper.entries.values()]
    config = wrapper.vllm_config
    cc = config.compilation_config
    ac = config.additional_config or {}
    record = {"rank": rank, "pid": _glm_os.getpid(), "draft": draft, "runnable": type(wrapper.runnable).__name__,
        "counts": state["counts"], "snapshots": state["snapshots"], "errors": state["errors"], "entries": entries,
        "effective": {"compilation_mode": str(cc.mode), "cudagraph_mode": str(cc.cudagraph_mode),
            "capture_sizes": cc.cudagraph_capture_sizes, "max_capture": cc.max_cudagraph_capture_size,
            "target_enforce_eager": config.model_config.enforce_eager,
            "speculative_enforce_eager": config.speculative_config.enforce_eager,
            "additional": ac, "blocking_wait_environment": {key: _glm_os.environ.get(key) for key in
                ("TORCH_HCCL_BLOCKING_WAIT", "HCCL_BLOCKING_WAIT", "ASCEND_LAUNCH_BLOCKING")}}}
    path = root / "witnesses" / ("%s_rank%d.json" % ("draft" if draft else "target", rank))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_glm_json.dumps(record, indent=2) + "\n");tmp.replace(path)


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not _glm_os.environ.get("GLM_GRAPH_DIAGNOSTIC_ROOT"):
            return super().__call__(*args, **kwargs)
        from vllm.forward_context import is_forward_context_available
        if not is_forward_context_available():
            return super().__call__(*args, **kwargs)
        fc = get_forward_context()
        entry = self.entries.get(fc.batch_descriptor) if fc.batch_descriptor is not None else None
        kind = ("none" if fc.cudagraph_runtime_mode == CUDAGraphMode.NONE else
                "capture" if entry is None or entry.capture is None else "replay")
        _glm_graph_observe(self, kind + "_before", args, kwargs)
        try:
            output = super().__call__(*args, **kwargs)
        except BaseException as error:
            _glm_graph_observe(self, kind + "_error", args, kwargs, error)
            raise
        _glm_graph_observe(self, kind + "_after", args, kwargs)
        return output

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
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # Match ACLGraphWrapper's ordering between async attention
            # parameter updates and the previous/current FULL graph replay.
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
            if not self.enable_enpu and not is_draft_eagle:
                torch.npu.current_stream().synchronize()
        super()._replay(entry, args, kwargs)
        return entry.output
