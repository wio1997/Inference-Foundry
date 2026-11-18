"""Profile-only CPU annotations around the unmodified native NPU worker.

No tensors, sampled tokens, scheduler decisions, or profiler lifecycle are
changed. The native function is called once and its return value is untouched.
Scheduled work describes the worker input, not physical padding or accepted
draft tokens. Those require trace/metrics evidence from the same run.
"""
import json


def scheduler_description(scheduled, rank, iteration):
    """Read only CPU metadata; omit token IDs, block tables and prompt text."""
    cached = scheduled.scheduled_cached_reqs
    computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
    emitted = dict(zip(cached.req_ids, cached.num_output_tokens))
    new = {item.req_id: item for item in scheduled.scheduled_new_reqs}
    proposals = scheduled.scheduled_spec_decode_tokens
    rows = []
    for request_id, query_tokens in scheduled.num_scheduled_tokens.items():
        is_new = request_id in new
        row = {
            "request_id": request_id,
            "scheduled_query_tokens": int(query_tokens),
            "new_request": is_new,
            "resumed": request_id in cached.resumed_req_ids,
            "draft_proposals": len(proposals.get(request_id, [])),
        }
        if is_new:
            item = new[request_id]
            row.update(
                computed_tokens=int(item.num_computed_tokens),
                output_tokens_so_far=0,
                prompt_tokens=len(item.prompt_token_ids)
                if item.prompt_token_ids is not None else None,
                context_phase=True,
            )
        else:
            row.update(
                computed_tokens=int(computed[request_id])
                if request_id in computed else None,
                output_tokens_so_far=int(emitted[request_id])
                if request_id in emitted else None,
                context_phase=emitted[request_id] == 0
                if request_id in emitted else None,
            )
        rows.append(row)
    return {
        "schema": 1,
        "rank": int(rank),
        "iteration": int(iteration),
        "scheduled_query_tokens": int(scheduled.total_num_scheduled_tokens),
        "requests": rows,
        "scope": "scheduler_input_not_physical_padding_or_draft_acceptance",
    }


def _native_worker_class():
    # Delay runtime imports: the metadata encoder can be checked on CPU
    # without importing torch, vLLM, torch_npu, or initializing a device.
    import hashlib
    from pathlib import Path
    import vllm_ascend.worker.worker as worker
    pins = json.loads((Path(__file__).parent / "profile_runtime_source_pins.json").read_text())
    for item in pins:
        path = Path(item["runtime_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], str(path)
    return worker.NPUWorker


# vLLM imports this class only in the dedicated, owned profiling epoch.
# CPU contract checks extract scheduler_description from its AST instead.
class LogicalProfileNPUWorker(_native_worker_class()):
    def execute_model(self, scheduler_output):
        profiler = self.profiler
        if profiler is None or not profiler.is_running:
            return super().execute_model(scheduler_output)
        import torch
        iteration = getattr(self, "_logical_profile_iteration", 0) + 1
        self._logical_profile_iteration = iteration
        metadata = scheduler_description(scheduler_output, self.rank, iteration)
        annotation = "ds41.scheduler:" + json.dumps(metadata, separators=(",", ":"))
        with torch.profiler.record_function(annotation):
            return super().execute_model(scheduler_output)
