"""Carry real SDK draft IDs to the native scheduler without fake placeholders.

This bounded initial bridge resolves MTP batches inline. SDK asynchronous
sampling remains enabled, but native cross-batch host overlap is not implemented
here and must not be reported as equivalent to the ordinary async bridge.
"""
from dataclasses import dataclass


@dataclass
class ResolvedMTPOutput:
    output: object
    draft_token_ids: dict

    def __getattr__(self, name):
        return getattr(self.output, name)


@dataclass
class PendingMTPOutput:
    output: object
    request_ids: list
    cpu_drafts: object
    copied_event: object
    gpu_drafts: object

    def resolve(self):
        from vllm.v1.outputs import AsyncModelRunnerOutput
        output = self.output.get_output() if isinstance(self.output, AsyncModelRunnerOutput) else self.output
        if self.cpu_drafts is None:
            return ResolvedMTPOutput(output, {})
        self.copied_event.synchronize()
        rows = self.cpu_drafts.tolist()
        if len(rows) != len(self.request_ids) or any(len(row) != 1 for row in rows):
            raise ValueError('one real MTP draft per SDK input row required')
        return ResolvedMTPOutput(output, dict(zip(self.request_ids, rows, strict=True)))


def capture_mtp_output(runtime, output, descriptor):
    """Snapshot actual GPU drafts before a later execute can overwrite them."""
    import torch
    import torch.distributed as dist
    if not descriptor.total_num_scheduled_tokens or dist.get_rank(runtime.native_group) != 0:
        return output
    runner = runtime.compute_worker.model_runner
    drafts = runner._draft_token_ids
    request_ids = list(runner.input_batch.req_ids)
    if drafts is None:
        # A non-final prefill chunk may not propose anything. Commit rejects
        # this absence if that batch actually emits a ready output token.
        return PendingMTPOutput(output, request_ids, None, None, None)
    if (not isinstance(drafts, torch.Tensor) or drafts.device.type != 'npu'
            or tuple(drafts.shape) != (len(request_ids), 1)):
        raise ValueError('SDK did not produce the expected real device draft tensor')
    # Own the snapshot; the model runner reuses its draft buffers. This copy is
    # an explicit bridge cost, not an assumed performance improvement.
    owned = drafts.clone()
    cpu = torch.empty(owned.shape, dtype=owned.dtype, device='cpu', pin_memory=True)
    if getattr(runtime, '_mtp_copy_stream', None) is None:
        runtime._mtp_copy_stream = torch.npu.Stream(device=runtime.compute_worker.device)
    stream = runtime._mtp_copy_stream
    stream.wait_stream(torch.npu.current_stream())
    event = torch.npu.Event()
    with torch.npu.stream(stream):
        cpu.copy_(owned, non_blocking=True)
        event.record()
    owned.record_stream(stream)
    return PendingMTPOutput(output, request_ids, cpu, event, owned)
