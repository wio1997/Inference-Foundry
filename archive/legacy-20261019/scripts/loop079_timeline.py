"""Host-only request timeline for a frozen output-accounting diagnostic.

All events use CLOCK_MONOTONIC in one Linux host. This log is deliberately
outside timing evidence: JSONL file I/O can perturb admission/publication.
"""
import json
import os
import time
import atexit
import hashlib
from pathlib import Path

ENV = "EXTREME_TIMELINE_DIR"
ROWS = []


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def flush():
    if not ROWS:
        return
    root = os.getenv(ENV)
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    with (path / f"pid{pid}.jsonl").open("a") as out:
        out.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in ROWS)
    ROWS.clear()


atexit.register(flush)


def emit(event, **fields):
    root = os.getenv(ENV)
    if not root:
        return
    pid = os.getpid()
    if not ROWS:
        clock = time.get_clock_info("monotonic")
        def read(path):
            try:
                return Path(path).read_text().strip()
            except OSError:
                return None
        try:
            time_ns = os.readlink("/proc/self/ns/time")
        except OSError:
            time_ns = None
        ROWS.append(dict(event="clock_origin", pid=pid,
                         monotonic_ns=time.monotonic_ns(),
                         run_id=os.getenv("EXTREME_TIMELINE_RUN_ID"),
                         boot_id=read("/proc/sys/kernel/random/boot_id"),
                         time_namespace=time_ns,
                         time_namespace_offsets=read("/proc/self/timens_offsets"),
                         clock_implementation=clock.implementation,
                         clock_resolution_s=clock.resolution))
    row = dict(event=event, pid=pid, monotonic_ns=time.monotonic_ns(),
               run_id=os.getenv("EXTREME_TIMELINE_RUN_ID"), **fields)
    ROWS.append(row)


def scheduler_before(scheduler, request, new_token_ids, is_bulk, is_stale):
    if not os.getenv(ENV) or not new_token_ids:
        return None
    return dict(
        scheduler_class=type(scheduler).__module__ + "." + type(scheduler).__qualname__,
        request_id=request.request_id,
        request_object_id=id(request),
        before_ns=time.monotonic_ns(),
        g_before=len(request._output_token_ids),
        incoming_len=len(new_token_ids),
        bulk=bool(is_bulk), stale=bool(is_stale),
        placeholders_before=int(request.num_output_placeholders),
        status_before=str(request.status),
    )


def scheduler_after(row, request, admitted_ids, stopped):
    if row is None:
        return
    emit("scheduler_append", **row, after_ns=time.monotonic_ns(),
         g_after=len(request._output_token_ids), admitted_len=len(admitted_ids),
         admitted_token_sha256=token_digest(admitted_ids),
         stopped=bool(stopped), status_after=str(request.status))
    if row["bulk"]:
        flush()


def handoff(runner, scheduler_output, req_ids):
    if not os.getenv(ENV):
        return
    entry_ns = time.monotonic_ns()
    cached = scheduler_output.scheduled_cached_reqs
    idx = {r: i for i, r in enumerate(cached.req_ids)}
    rows = []
    for req_id in req_ids:
        state = runner.requests[req_id]
        j = idx[req_id]
        vals = state.output_token_ids
        rows.append(dict(request_id=req_id,
                         model_cache_output_len=len(vals),
                         model_cache_placeholder_count=sum(x == -1 for x in vals),
                         scheduled_output_including_placeholders=int(cached.num_output_tokens[j]),
                         scheduled_num_computed_tokens=int(cached.num_computed_tokens[j]),
                         scheduled_target_tokens=int(scheduler_output.num_scheduled_tokens[req_id])))
    from vllm.distributed import get_tp_group
    emit("runtime_handoff", handoff_entry_ns=entry_ns,
         tp_rank=int(get_tp_group().rank_in_group),
         runner_class=type(runner).__module__ + "." + type(runner).__qualname__,
         request_rows=rows, total_scheduled=int(scheduler_output.total_num_scheduled_tokens))


def output_processor(req_state, engine_core_output):
    if not os.getenv(ENV):
        return
    emit("output_processor_receive", request_id=engine_core_output.request_id,
         external_req_id=req_state.external_req_id,
         new_token_len=len(engine_core_output.new_token_ids),
         new_token_sha256=token_digest(engine_core_output.new_token_ids),
         finished=bool(engine_core_output.finished),
         finish_reason=str(engine_core_output.finish_reason))
    if engine_core_output.finished:
        flush()


def output_processor_queue(req_state, request_output):
    if not os.getenv(ENV):
        return
    emit("output_processor_queue_submitted",
         request_id=req_state.request_id,
         external_req_id=req_state.external_req_id,
         request_output_id=request_output.request_id,
         choice_token_lens=[len(x.token_ids) for x in request_output.outputs],
         choice_token_sha256=[token_digest(x.token_ids) for x in request_output.outputs],
         finished=bool(request_output.finished))
    if request_output.finished:
        flush()


def api_consume(internal_res_id, external_request_id, choice, new_ids, cumulative):
    emit("api_consume", request_id=internal_res_id,
         external_req_id=external_request_id, choice=int(choice),
         new_token_len=len(new_ids), new_token_sha256=token_digest(new_ids),
         raw_consumed_cumulative=int(cumulative))


def api_yield(internal_res_id, external_request_id, choice, cumulative, delta, finished):
    emit("api_generator_yield", request_id=internal_res_id,
         external_req_id=external_request_id, choice=int(choice),
         raw_consumed_cumulative=int(cumulative),
         has_content=bool(getattr(delta, "content", None)),
         has_reasoning=bool(getattr(delta, "reasoning", None)),
         has_tool_calls=bool(getattr(delta, "tool_calls", None)),
         finished=bool(finished))
    if finished:
        flush()
