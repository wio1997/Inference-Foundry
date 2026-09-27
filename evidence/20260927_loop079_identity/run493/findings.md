# Run491–493 — installed NPUGraph debug JSON method gate

Run491 failed before container/NPU action because its host log directory did not exist; TaskCtl records it invalid. Run492 and Run493 are isolated, one-card **measurement-method tests**, not Extreme timing or necessary-work evidence. The dedicated service was stopped and no VLLM worker occupied the devices before and after.

Installed torch_npu is 2.10.0.post4. Its Python `NPUGraph.debug_dump(path)` calls the installed native graph debug API; the archived installed NPUGraph.cpp shows `aclmdlRIDebugJsonPrint(model_ri_, path, 1)` after successful capture. Run492 captured/replayed a tiny single-stream graph, verified output, and wrote a three-task JSON with stream/task IDs, task types and kernel argument address text. Run493 captured/replayed a correct tiny two-stream graph and wrote a ten-task JSON across stream IDs12/13. It includes `EVENT_RECORD_0/1`, matching `EVENT_WAIT_0/1`, `EVENT_RESET_0/1`, kernels and terminal notify. Thus this installed exporter can reveal explicit child-stream joins when they exist.

The JSON is grouped by stream; list order alone is not execution order. Kernel argument addresses are untyped text and do not by themselves label a tensor as input or output. A production output-producer certificate still needs the **same acquisition** graph entry/generation, output leaf addresses, relevant kernel signatures and actual event links. The small graphs prove exporter capability only; they do not certify Run487's four hidden/aux last writers, the after-update backend, uninstrumented timing, or a Bound endpoint.

Artifacts: `run492/simple_graph.json`, `run492/probe_result.json`, `run493/multistream_graph.json`, `run493/probe_result.json`.
