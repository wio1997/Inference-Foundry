"""Reuse native two-batch dispatch and native CPU MTP metadata refresh."""
from .sdk_async_dispatch import SDKAsyncBatchWorkerAdapter
from .sdk_mtp_dispatch import SDKMTPBatchWorkerAdapter


class SDKAsyncMTPBatchWorkerAdapter(SDKAsyncBatchWorkerAdapter, SDKMTPBatchWorkerAdapter):
    allow_speculative_metadata = True

    def __init__(self, runtime, controller, group):
        super().__init__(runtime, controller, group)
        self.expected_nextn = runtime.num_speculative_tokens

    def _can_defer(self, metadata, is_prompt):
        return super()._can_defer(metadata, is_prompt) and all(
            self.controller.requests[m.request_id].request.num_computed_tokens + 1 + self.expected_nextn
            <= self.controller.max_context for m in metadata)

    def _resolve_step(self, worker, step, *, driver):
        import numpy as np
        output = self.runtime.resolve_output(step.output)
        if not driver:
            return np.empty(0, dtype=np.uint64)
        emitted = self.controller.commit_completed(output)
        return self._process_completed(worker, step.metadata, step.scheduler_output,
            output, emitted, is_prompt=step.is_prompt, end_event=step.end_event,
            prepare_native=False, refresh_metadata=True)
