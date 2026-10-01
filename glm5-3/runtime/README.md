# GLM characterization runtime

`controller.py` holds task-global controller and legacy formal-test locks, detaches from SSH, persists PID/boot/start ticks and 5-second heartbeat, and schedules immutable stages with source hashes. Unknown interrupted stages require reconciliation rather than automatic replay. This serializes callers using these locks; direct legacy lifecycle calls can still bypass ownership and are not claimed protected.

`phase_runner.py` records real argv/exit/signal/timeout and raw log identity. Its source derives from the existing, previously untested task fixture. Killing a container-exec client does not certify an in-container process is gone; no such recovery claim is made.

`stream_probe.py` observes actual streamed content, usage, finish reason and DONE. Effective counts require all these checks. Requests use a bounded client pool; this initial implementation is a closed-loop diagnostic, not an open-loop stable-capacity certificate. TPOT is wall after first content divided by effective output minus one; timestamped SSE frames stay on the server. Resident-cache condition is explicit. No performance KEEP is implied.
