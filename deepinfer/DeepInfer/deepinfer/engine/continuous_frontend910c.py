"""Keep HTTP intake open while an original Scheduler session is executing."""
import json
import queue
import time


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")))
    temporary.replace(path)


def serve(frontend, directory, batch, body):
    readers = {}
    live = dict(enumerate(batch))
    done = set()
    deferred = []
    next_index, transaction = len(batch), 1
    pending_ack = None
    closing = False
    started = time.monotonic()
    (directory / "admissions").mkdir()
    try:
        while True:
            errors = list(directory.glob("error-rank*.txt"))
            if errors:
                raise RuntimeError(errors[0].read_text()[-5000:])
            if time.monotonic() - started > frontend.args.request_timeout:
                raise TimeoutError("continuous HTTP session timed out")
            if pending_ack is not None:
                number, indices = pending_ack
                path = directory / f"admission-result-{number:08d}.json"
                if path.exists():
                    response = json.loads(path.read_text())
                    accepted = {row["index"] for row in response["accepted"]}
                    rejected = set(response["deferred"])
                    if response["number"] != number or accepted & rejected or accepted | rejected != set(indices):
                        raise ValueError("invalid continuous admission response")
                    for index in indices:
                        if index in rejected:
                            deferred.append(live.pop(index))
                    pending_ack = None
            ready = len(list(directory.glob("admission-ready-dp*.json"))) == frontend.args.dp_size
            if ready and not closing and pending_ack is None:
                incoming = []
                while len(incoming) < 8:
                    try:
                        incoming.append(frontend.pending.get_nowait())
                    except queue.Empty:
                        break
                if incoming:
                    rows = []
                    now = time.monotonic_ns()
                    for request in incoming:
                        index = next_index
                        next_index += 1
                        live[index] = request
                        if request.ttft_trace is not None:
                            request.ttft_trace.update(dequeued_ns=now, batch_ready_ns=now,
                                job_id=directory.name, batch_size=len(incoming))
                        rows.append(dict(index=index, request=frontend.wire_request(request, directory)))
                    atomic(directory / "admissions" / f"{transaction:08d}.json", dict(number=transaction, requests=rows))
                    for request in incoming:
                        if request.ttft_trace is not None:
                            request.ttft_trace["ipc_published_ns"] = time.monotonic_ns()
                    pending_ack = transaction, [row["index"] for row in rows]
                    transaction += 1
            for dp in range(frontend.args.dp_size):
                path = directory / f"events-dp{dp}.jsonl"
                if dp not in readers and path.exists():
                    readers[dp] = path.open()
                if dp not in readers:
                    continue
                stream = readers[dp]
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        stream.seek(offset)
                        break
                    event = json.loads(line)
                    index = event["index"]
                    if index not in live:
                        raise ValueError("worker emitted an unowned HTTP request")
                    request = live[index]
                    if request.ttft_trace is not None and event["type"] == "token" and event.get("step") == 0:
                        request.ttft_trace["worker_token_received_ns"] = time.monotonic_ns()
                        request.ttft_trace["worker_token_elapsed_s"] = event.get("elapsed")
                    if event["type"] in ("done", "error") and index in done:
                        raise RuntimeError("worker completed an HTTP request twice")
                    request.events.put(event)
                    if event["type"] in ("done", "error"):
                        done.add(index)
                        with frontend.lock:
                            if event["type"] == "error":
                                frontend.counts["failed_requests"] += 1
                            else:
                                frontend.counts["requests"] += 1
                                for key in ("prompt_tokens", "cached_tokens", "output_tokens"):
                                    frontend.counts[key] += event[key]
            if not closing and pending_ack is None and done == set(live):
                atomic(directory / "admission-close.json", dict(closed=True, requests=len(live)))
                closing = True
            if closing and len(list(directory.glob("result-dp*.json"))) == frontend.args.dp_size:
                break
            time.sleep(.001)
    except Exception as exc:
        unfinished = set(live) - done
        for index in unfinished:
            live[index].events.put(dict(type="error", message=str(exc)))
        with frontend.lock:
            frontend.counts["failed_requests"] += len(unfinished)
        raise
    finally:
        for stream in readers.values():
            stream.close()
        frontend.deferred.extend(deferred)
        frontend.retire_requests(live.values())
