"""Host-only Scheduler bulk ledger for a frozen diagnostic run."""
import json
import os
import time
from pathlib import Path

ENV = "EXTREME_BULK_LEDGER_DIR"


def before(scheduler, request, token_ids, output_is_stale):
    root = os.getenv(ENV)
    if not root:
        return None
    return {
        "run_id": os.getenv("EXTREME_BULK_LEDGER_RUN_ID"),
        "pid": os.getpid(),
        "scheduler_class": type(scheduler).__module__ + "." + type(scheduler).__qualname__,
        "request_id": request.request_id,
        "request_object_id": id(request),
        "timestamp_before_ns": time.monotonic_ns(),
        "g_before": len(request._output_token_ids),
        "incoming_bulk_len": len(token_ids),
        "max_tokens": int(request.max_tokens),
        "num_prompt_tokens": int(request.num_prompt_tokens),
        "resumable": bool(request.resumable),
        "num_output_placeholders": int(request.num_output_placeholders),
        "num_in_flight_tokens": int(request.num_in_flight_tokens),
        "num_stale_output_tokens": int(getattr(request, "num_stale_output_tokens", 0)),
        "drop_stale_output": bool(getattr(request, "drop_stale_output", False)),
        "output_is_stale": bool(output_is_stale),
        "status_before": str(request.status),
        "ledger_root": root,
    }


def after(row, request, admitted_ids, stopped):
    if row is None:
        return
    row = dict(row)
    root = Path(row.pop("ledger_root"))
    root.mkdir(parents=True, exist_ok=True)
    row.update(
        timestamp_after_ns=time.monotonic_ns(),
        g_after=len(request._output_token_ids),
        admitted_len=len(admitted_ids),
        stopped=bool(stopped),
        status_after=str(request.status),
    )
    row["append_delta"] = row["g_after"] - row["g_before"]
    # Every Scheduler process writes its own append-only file.
    with (root / ("pid%d.jsonl" % row["pid"])).open("a") as out:
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
