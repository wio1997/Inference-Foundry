"""Sparse, conditional cohort5->6 frontier observation.

This module records Host predicates and current-stream events only.  It never
claims that a caller-stream event completes a private stream, HCCL operation,
KV side effect, or client publication.  The guarded controller installs its
call sites temporarily, and the runner's existing post-cohort synchronize is
the only synchronization used before event reduction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

_ROWS: list[dict] = []
_EVENTS: list[tuple[str, int, str, object]] = []
_CALL = 0


def _enabled() -> bool:
    return bool(os.getenv("EXTREME_FRONTIER_DIR"))


def _pending(runner) -> bool:
    return _enabled() and len(getattr(runner, "_extreme_served_cohorts", ())) == 5


def _rank() -> int:
    return int(torch.distributed.get_rank())


def _row(event: str, **fields) -> None:
    if _enabled():
        _ROWS.append(dict(event=event, monotonic_ns=time.monotonic_ns(), **fields))


def _mark(group: str, generation: int, label: str) -> None:
    event = torch.npu.Event(enable_timing=True)
    event.record()
    _EVENTS.append((group, generation, label, event))


def prepare_ids_before(runner, scheduler_output, num_reqs, total, cu_num_tokens):
    """Mirror the pinned parent's Host branch predicates without device reads."""
    global _CALL
    if not _pending(runner):
        return
    _CALL += 1
    call = _CALL
    req_ids = list(runner.input_batch.req_ids[:num_reqs])
    previous = runner.input_batch.prev_sampled_token_ids
    spec = scheduler_output.scheduled_spec_decode_tokens
    prev_positions = runner.prev_positions.np[:num_reqs]
    common = 0
    total_spec = 0
    same = True
    max_flat = -1
    for index, request_id in enumerate(req_ids):
        prev = int(prev_positions[index])
        if prev < 0:
            continue
        draft_len = len(spec.get(request_id, ()))
        flat = int(cu_num_tokens[index]) - 1
        total_spec += draft_len
        common += 1
        same &= prev == flat
        max_flat = max(max_flat, flat)
    prompt_upload = previous is None or common < int(total) - total_spec
    if previous is None:
        branch = "normal_upload"
    elif common == 0:
        branch = "async_no_common"
    elif same and max_flat == common - 1:
        branch = "async_direct_slice"
    else:
        branch = "async_sample_scatter"
    _row("prepare_ids_before", call=call, req_ids=req_ids,
         scheduled={rid: int(scheduler_output.num_scheduled_tokens[rid])
                    for rid in req_ids}, num_reqs=int(num_reqs),
         total_scheduled=int(total), previous_sampled_present=previous is not None,
         previous_draft_present=torch.is_tensor(getattr(runner, "_draft_token_ids", None)),
         common=int(common), scheduled_spec_tokens=int(total_spec),
         prompt_upload=bool(prompt_upload), branch=branch,
         draft_scatter_possible=(branch == "async_sample_scatter" and
                                 total_spec > 0 and
                                 torch.is_tensor(getattr(runner, "_draft_token_ids", None))),
         input_ids_gpu_ptr=int(runner.input_ids.gpu.data_ptr()))
    _mark("ordinary_input", call, "before_prepare_input_ids")


def prepare_ids_after(runner):
    if _pending(runner):
        _mark("ordinary_input", _CALL, "after_prepare_input_ids")
        _row("prepare_ids_after", call=_CALL,
             input_ids_gpu_ptr=int(runner.input_ids.gpu.data_ptr()))


def dcp_result(runner, result, with_prefill):
    if not _pending(runner):
        return
    _mark("ordinary_input", _CALL, "after_dcp_rebuild")
    _row("dcp_result", call=_CALL,
         use_async_spec_decode=bool(runner.use_async_spec_decode),
         with_prefill=bool(with_prefill),
         previous_mapping=bool(runner.input_batch.prev_req_id_to_index),
         valid_count_gpu_present=runner.valid_sampled_token_count_gpu is not None,
         valid_count_event_present=runner.valid_sampled_token_count_event is not None,
         rebuilt=bool(result.rebuilt),
         positions_ready_on_device=bool(result.positions_ready_on_device),
         input_ids_gpu_ptr=int(runner.input_ids.gpu.data_ptr()),
         positions_gpu_ptr=int(runner.positions.data_ptr()))


def forward_before(runner, scheduler_output, num_tokens_padded):
    if not _pending(runner):
        return
    _mark("ordinary_forward", _CALL, "before_model_forward")
    _row("forward_before", call=_CALL,
         req_ids=list(runner.input_batch.req_ids[:runner.input_batch.num_reqs]),
         actual_tokens=int(scheduler_output.total_num_scheduled_tokens),
         padded_tokens=int(num_tokens_padded))


def forward_after(runner):
    if _pending(runner):
        _mark("ordinary_forward", _CALL, "after_model_forward_current_stream")
        _row("forward_after", call=_CALL)


def draft_copy_result(runner, scheduler_output):
    if not _pending(runner):
        return
    gates = dict(num_spec_tokens=int(runner.num_spec_tokens),
                 async_scheduling=bool(runner.use_async_scheduling),
                 structured=bool(scheduler_output.has_structured_output_requests),
                 output_token_ids=bool(runner.input_batch.sampling_metadata.output_token_ids),
                 draft_tensor=torch.is_tensor(getattr(runner, "_draft_token_ids", None)))
    from vllm.distributed import get_pp_group
    gates["pp_world_size"] = int(get_pp_group().world_size)
    gates["copy_branch_source_predicate"] = bool(
        gates["num_spec_tokens"] and gates["draft_tensor"] and
        (not gates["async_scheduling"] or gates["structured"] or
         gates["output_token_ids"] or gates["pp_world_size"] > 1))
    _row("draft_copy_return", call=_CALL, **gates)


def handoff(runner, runtime, req_ids):
    if not _enabled():
        return
    cohort = len(runner._extreme_served_cohorts)
    if cohort not in (5, 6):
        return
    runtime._frontier_cohort = cohort
    _row("frontier_handoff", cohort=cohort, req_ids=list(req_ids),
         rank=_rank(), first_target_input_ptr=int(runtime.state.target_input_ids.data_ptr()),
         first_target_positions_ptr=int(runtime.state.target_positions.data_ptr()))


def history_begin(serving, token_history, count_history):
    if getattr(serving.runtime, "_frontier_cohort", None) != 5:
        return
    serving.runtime._frontier_history_storage = dict(
        token_ptr=int(token_history.data_ptr()), count_ptr=int(count_history.data_ptr()),
        token_shape=list(token_history.shape), count_shape=list(count_history.shape))
    _mark("history", 5, "before_cycle_1")


def history_after_copy(serving, index):
    if getattr(serving.runtime, "_frontier_cohort", None) == 5 and index % 8 == 7:
        _mark("history", 5, f"after_cycle_{index + 1}")


def history_after_loop(serving, cycles):
    if getattr(serving.runtime, "_frontier_cohort", None) != 5:
        return
    if cycles % 8:
        _mark("history", 5, f"after_cycle_{cycles}")
    _row("history_after_loop", cohort=5, cycles=int(cycles),
         storage=serving.runtime._frontier_history_storage,
         note="history copies are current-stream ordered; event is not publication or KV lifetime")


def first_target_mark(runtime, label):
    if getattr(runtime, "_frontier_cohort", None) == 6 and runtime.state.cycle_index == 0:
        _mark("first_target", 6, label)
        _row("first_target_mark", cohort=6, label=label,
             cycle_index=int(runtime.state.cycle_index),
             kv_slot_audit=runtime.kv_slot_audit is not None,
             target_page_audit=runtime.target_page_audit is not None,
             target_self_replay=bool(getattr(runtime.target_page_audit, "self_replay", False)),
             input_ptr=int(runtime.state.target_input_ids.data_ptr()),
             positions_ptr=int(runtime.state.target_positions.data_ptr()))


def cohort_done(runner, runtime):
    """Called after the existing post-run torch.npu.synchronize()."""
    if not _enabled():
        return
    cohort = getattr(runtime, "_frontier_cohort", None)
    if cohort not in (5, 6):
        return
    groups = {}
    for group, generation, label, event in _EVENTS:
        groups.setdefault((group, generation), []).append((label, event))
    elapsed = []
    for (group, generation), events in groups.items():
        origin = events[0][1]
        elapsed.append(dict(group=group, generation=generation,
                            scope="same-rank current-stream event relation only",
                            points=[dict(label=label,
                                         elapsed_ms=float(origin.elapsed_time(event)))
                                    for label, event in events]))
    path = Path(os.environ["EXTREME_FRONTIER_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    output = dict(rank=_rank(), cohort=cohort, rows=list(_ROWS),
                  event_groups=elapsed,
                  device_ready="unknown_if_private_stream_or_HCCL_unjoined",
                  safe_publication="unknown", client_permit_parent="unknown")
    destination = path / f"rank{_rank()}_cohort{cohort}.json"
    if destination.exists():
        raise RuntimeError(f"frontier output collision: {destination}")
    destination.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    _ROWS.clear()
    _EVENTS.clear()
