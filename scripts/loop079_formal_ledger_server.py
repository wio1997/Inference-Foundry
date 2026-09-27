"""Diagnostic Host ledger. Never used to change serving decisions.

Every record has a process sequence and clock identity.  Cross-process joins
are by request ID or the rank/cohort/slot tuple, never by timestamp ordering.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

ENV = "EXTREME_FORMAL_LEDGER_DIR"
RUN = "EXTREME_FORMAL_LEDGER_RUN_ID"
PHASE_FILE = "EXTREME_FORMAL_LEDGER_PHASE_FILE"
_ROWS: list[dict] = []
_SEQ = 0
_FLUSH_INDEX = 0
_COHORT = None


def _ids(values):
    return [int(v) for v in values]


def _sha(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _clock():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    try:
        namespace = os.readlink("/proc/self/ns/time")
    except OSError:
        namespace = None
    info = time.get_clock_info("monotonic")
    return dict(boot_id=read("/proc/sys/kernel/random/boot_id"),
                time_namespace=namespace,
                time_namespace_offsets=read("/proc/self/timens_offsets"),
                clock_implementation=info.implementation,
                clock_resolution_s=info.resolution)


def emit(event, **fields):
    global _SEQ
    if not os.getenv(ENV):
        return
    if _SEQ == 0:
        _ROWS.append(dict(event="clock_origin", run_id=os.getenv(RUN),
                          pid=os.getpid(), seq=0, monotonic_ns=time.monotonic_ns(),
                          **_clock()))
    _SEQ += 1
    phase_path = os.getenv(PHASE_FILE)
    if phase_path:
        raw_phase = Path(phase_path).read_bytes()
        phase = json.loads(raw_phase)
        phase_fields = dict(phase=phase["phase"],
                            phase_generation=int(phase["generation"]),
                            phase_marker_sha256=hashlib.sha256(raw_phase).hexdigest(),
                            phase_run_id_matches=(phase["run_id"] == os.getenv(RUN)))
    else:
        phase_fields = dict(phase="unknown", phase_generation=None,
                            phase_marker_sha256=None, phase_run_id_matches=False)
    _ROWS.append(dict(event=event, run_id=os.getenv(RUN), pid=os.getpid(),
                      seq=_SEQ, monotonic_ns=time.monotonic_ns(),
                      **phase_fields, **fields))
    if len(_ROWS) >= 4096:
        flush("buffer_limit")


def flush(reason):
    global _FLUSH_INDEX
    if not _ROWS or not os.getenv(ENV):
        return
    root = Path(os.environ[ENV])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"pid{os.getpid()}.jsonl"
    blob = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False)
                   + "\n" for row in _ROWS).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        start = os.lseek(fd, 0, os.SEEK_END)
        view = memoryview(blob)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short ledger write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    footer = dict(run_id=os.getenv(RUN), pid=os.getpid(),
                  flush_index=_FLUSH_INDEX, reason=reason,
                  committed_records=len(_ROWS), first_seq=_ROWS[0]["seq"],
                  last_seq=_ROWS[-1]["seq"], byte_offset_begin=start,
                  byte_offset_end=start + len(blob),
                  blob_sha256=hashlib.sha256(blob).hexdigest())
    footer_blob = (json.dumps(footer, separators=(",", ":")) + "\n").encode()
    footer_path = root / f"pid{os.getpid()}.flush.jsonl"
    footer_fd = os.open(footer_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(footer_blob)
        while view:
            count = os.write(footer_fd, view)
            if count <= 0:
                raise OSError("short footer write")
            view = view[count:]
        os.fsync(footer_fd)
    finally:
        os.close(footer_fd)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    _FLUSH_INDEX += 1
    _ROWS.clear()


def verify_flushes(root, pid):
    """CPU-only integrity check; live validator should also check expected IDs."""
    root = Path(root)
    blob = (root / f"pid{pid}.jsonl").read_bytes()
    footers = [json.loads(line) for line in
               (root / f"pid{pid}.flush.jsonl").read_text().splitlines()]
    if not footers:
        raise ValueError("missing flush footer")
    offset = 0
    last_seq = -1
    total = 0
    for index, footer in enumerate(footers):
        if footer["flush_index"] != index or footer["byte_offset_begin"] != offset:
            raise ValueError("flush offset/index gap")
        stop = footer["byte_offset_end"]
        segment = blob[offset:stop]
        if len(segment) != stop - offset or _sha_bytes(segment) != footer["blob_sha256"]:
            raise ValueError("ledger segment truncated or changed")
        rows = [json.loads(line) for line in segment.splitlines()]
        if len(rows) != footer["committed_records"]:
            raise ValueError("flush record count mismatch")
        if rows[0]["seq"] != footer["first_seq"] or rows[-1]["seq"] != footer["last_seq"]:
            raise ValueError("flush sequence range mismatch")
        if rows[0]["seq"] != last_seq + 1 or any(
                row["seq"] != rows[0]["seq"] + j for j, row in enumerate(rows)):
            raise ValueError("ledger sequence gap")
        last_seq = rows[-1]["seq"]
        total += len(rows)
        offset = stop
    if offset != len(blob):
        raise ValueError("uncommitted trailing ledger bytes")
    return dict(flushes=len(footers), records=total, last_seq=last_seq,
                ledger_sha256=_sha_bytes(blob))


def _sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def scheduler_before(scheduler, request, incoming, bulk, stale):
    if not os.getenv(ENV):
        return None
    return dict(request_id=request.request_id, generation_object_id=id(request),
                scheduler_class=type(scheduler).__module__ + "." + type(scheduler).__qualname__,
                before_ns=time.monotonic_ns(), g_before=len(request._output_token_ids),
                incoming_raw_ids=_ids(incoming), bulk=bool(bulk), stale=bool(stale),
                max_tokens=int(request.max_tokens), num_prompt_tokens=int(request.num_prompt_tokens),
                placeholders=int(request.num_output_placeholders),
                in_flight=int(request.num_in_flight_tokens),
                stale_tokens=int(getattr(request, "num_stale_output_tokens", 0)),
                resumable=bool(request.resumable), status_before=str(request.status))


def scheduler_after(row, request, admitted, stopped):
    if row is None:
        return
    raw = _ids(admitted)
    emit("scheduler_append", **row, after_ns=time.monotonic_ns(),
         g_after=len(request._output_token_ids), admitted_raw_ids=raw,
         append_delta=len(request._output_token_ids) - row["g_before"],
         prefix_equal=(raw == row["incoming_raw_ids"][:len(raw)]),
         stopped=bool(stopped), status_after=str(request.status))
    if row["bulk"] or stopped:
        flush("scheduler_terminal")


def handoff(runner, scheduler_output, req_ids):
    global _COHORT
    if not os.getenv(ENV):
        return
    from vllm.distributed import get_tp_group
    rank = int(get_tp_group().rank_in_group)
    # Match the runner's own rank{rank}_cohort{index}.json index: insertion
    # occurs before bootstrap, and the runner uses this 1-based set cardinality.
    cohort = len(runner._extreme_served_cohorts)
    cached = scheduler_output.scheduled_cached_reqs
    index = {request_id: i for i, request_id in enumerate(cached.req_ids)}
    rows = []
    for slot, request_id in enumerate(req_ids):
        state = runner.requests[request_id]
        j = index[request_id]
        vals = _ids(state.output_token_ids)
        rows.append(dict(slot=slot, request_id=request_id,
                         runner_output_raw_ids=vals,
                         runner_output_sha256=_sha(vals),
                         runner_placeholders=sum(v == -1 for v in vals),
                         scheduled_output_including_placeholders=int(cached.num_output_tokens[j]),
                         scheduled_num_computed_tokens=int(cached.num_computed_tokens[j]),
                         scheduled_target_tokens=int(scheduler_output.num_scheduled_tokens[request_id])))
    _COHORT = dict(rank=rank, cohort=cohort, req_ids=list(req_ids))
    emit("runtime_handoff", **_COHORT, slots=rows,
         runner_class=type(runner).__module__ + "." + type(runner).__qualname__,
         total_scheduled=int(scheduler_output.total_num_scheduled_tokens))


def serving_cycle(index, progress, parked_before, parked_after):
    if _COHORT is None or not os.getenv(ENV):
        return
    emit("serving_host_cycle", rank=_COHORT["rank"], cohort=_COHORT["cohort"],
         cycle=int(index), progress=_ids(progress),
         parked_before=[bool(x) for x in parked_before],
         parked_after=[bool(x) for x in parked_after])


def serving_history(serving, cycles, tokens_cpu, counts_cpu, output, staged):
    if _COHORT is None or not os.getenv(ENV):
        return
    # These tensors have already crossed to CPU in the uninstrumented path.
    emit("runtime_post_drain", rank=_COHORT["rank"], cohort=_COHORT["cohort"],
         req_ids=_COHORT["req_ids"], cycles=int(cycles),
         width=int(serving.config.target_tokens_per_request),
         initial_positions=list(serving._initial_positions),
         progress_baseline=list(serving._progress_baseline),
         initial_output_counts=list(serving.initial_output_counts),
         remaining=list(serving.remaining),
         token_history=tokens_cpu.tolist(), count_history=counts_cpu.tolist(),
         retained_raw_ids=[_ids(row) for row in output], staged_counts=list(staged),
         device_ready="unknown", prefill_device_complete="unknown",
         seed_device_ready="unknown")
    flush("runtime_drain")


def runner_done(row):
    if not os.getenv(ENV):
        return
    emit("runner_done", rank=row["rank"], cohort=row["cohort"],
         req_ids=row["req_ids"], cycles=int(row["cycles"]),
         host_mirror_exact=bool(row["host_mirror_exact"]),
         generated_output_counts=row["generated_output_counts"],
         target_graph_mode=row["target_graph_mode"])
    flush("runner_done")


def output_add(request, req_state):
    if not os.getenv(ENV):
        return
    prompt = getattr(request, "prompt_token_ids", None)
    prompt = _ids(prompt) if prompt is not None else None
    emit("output_add_request", request_id=request.request_id,
         external_req_id=req_state.external_req_id,
         request_object_id=id(request), arrival_time=getattr(request, "arrival_time", None),
         prompt_raw_ids=prompt, prompt_sha256=_sha(prompt),
         output_processor_class=type(req_state).__module__ + "." + type(req_state).__qualname__)


def output_receive(req_state, engine_core_output):
    if not os.getenv(ENV):
        return
    emit("output_receive", request_id=engine_core_output.request_id,
         external_req_id=req_state.external_req_id,
         raw_ids=_ids(engine_core_output.new_token_ids),
         finished=bool(engine_core_output.finished),
         finish_reason=str(engine_core_output.finish_reason),
         prefill_stats=str(engine_core_output.prefill_stats))


def output_queue(req_state, request_output):
    if not os.getenv(ENV):
        return
    emit("output_queue", request_id=req_state.request_id,
         external_req_id=req_state.external_req_id,
         output_request_id=request_output.request_id,
         choices=[dict(index=int(x.index), raw_ids=_ids(x.token_ids), text=x.text,
                       finish_reason=str(x.finish_reason))
                  for x in request_output.outputs], finished=bool(request_output.finished),
         num_cached_tokens=request_output.num_cached_tokens)
    if request_output.finished:
        flush("output_finished")


def api_consume(output_request_id, api_request_id, choice, raw_ids, cumulative,
                delta_text, delta_message, parser, finish_reason):
    if not os.getenv(ENV):
        return
    # RequestOutput.request_id is constructed from the external request ID in
    # RequestState._new_request_output. The internal EngineCore ID is joined
    # separately through output_add_request/output_receive; do not relabel it.
    emit("api_consume", output_request_id=output_request_id,
         api_request_id=api_request_id,
         choice=int(choice), raw_ids=_ids(raw_ids), cumulative=int(cumulative),
         delta_text=delta_text, parser_class=(None if parser is None else
         type(parser).__module__ + "." + type(parser).__qualname__),
         parser_suppressed=delta_message is None,
         parsed_delta=(None if delta_message is None else
                       delta_message.model_dump(exclude_unset=True)),
         finish_reason=str(finish_reason))


def api_yield(external_id, payload):
    if os.getenv(ENV):
        emit("api_serialized_yield", external_req_id=external_id,
             payload=payload, payload_sha256=hashlib.sha256(payload.encode()).hexdigest())
        if payload == "data: [DONE]\n\n":
            flush("api_done")
    return payload
