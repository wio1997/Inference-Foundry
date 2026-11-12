"""Carry real SDK draft IDs to the native scheduler without fake placeholders.

Each result owns its draft snapshot and completion event, so later device
execution cannot overwrite it. Whether a caller overlaps batches is determined
by the dispatch path and must be verified by its actual execution counters.
"""
from dataclasses import dataclass


class DraftOutputContractError(ValueError):
    """Driver-only metadata rejection after all ranks completed SDK dispatch."""


def verify_draft_output_sources():
    import hashlib
    from pathlib import Path
    import vllm.v1.worker.gpu_model_runner as core
    import vllm_ascend.spec_decode.llm_base_proposer as ascend
    for module, expected in (
        (core, '87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51'),
        (ascend, '458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31'),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise DraftOutputContractError('SDK active-prefix draft output contract changed')


def active_draft_rows(shape, active_requests, width, *, captured_shapes=()):
    """Mirror the pinned SDK CPU consumer's [:len(req_ids), :K] boundary.

    Eager exact shapes stay exact. Extra rows are accepted only when this
    runnable owns an already captured output with that actual tensor shape.
    This function handles metadata only, never generates or changes a token.
    """
    if (len(shape) != 2 or type(active_requests) is not int or active_requests < 1
            or shape[1] != width or shape[0] < active_requests):
        raise DraftOutputContractError(f'invalid SDK draft shape {shape};active={active_requests},width={width}')
    if shape[0] != active_requests and tuple(shape) not in captured_shapes:
        raise DraftOutputContractError(f'unowned SDK draft padding {shape};active={active_requests}')
    return active_requests


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
        if len(rows) != len(self.request_ids) or any(len(row) != self.gpu_drafts.shape[1] for row in rows):
            raise ValueError('real MTP drafts must retain the captured device width')
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
    if not isinstance(drafts, torch.Tensor) or drafts.device.type != 'npu':
        raise DraftOutputContractError('SDK did not produce a real NPU draft tensor')
    captured_shapes = ()
    if drafts.ndim == 2 and drafts.shape[0] != len(request_ids):
        drafter = runner.drafter
        if drafter.use_cuda_graph:
            entries = getattr(getattr(drafter, '_runnable', None), 'concrete_aclgraph_entries', {})
            # Match the actual captured buffer, not arbitrary permissive row
            # padding. The fixed Ascend proposer returns this entry directly.
            captured_shapes = tuple(tuple(entry.output.shape) for entry in entries.values()
                if isinstance(entry.output, torch.Tensor)
                and entry.aclgraph is not None
                and tuple(entry.output.shape) == tuple(drafts.shape)
                and entry.output.stride() == drafts.stride()
                and entry.output.data_ptr() == drafts.data_ptr())
    count = active_draft_rows(tuple(drafts.shape), len(request_ids),
                              runtime.num_speculative_tokens, captured_shapes=captured_shapes)
    stats = getattr(runtime, 'mtp_output_stats', None)
    if stats is None:
        stats = runtime.mtp_output_stats = {'calls': 0, 'padded_graph_calls': 0, 'shapes': []}
    stats['calls'] += 1
    stats['padded_graph_calls'] += int(drafts.shape[0] != count)
    shape = {'raw': list(drafts.shape), 'active_requests': count}
    if shape not in stats['shapes']:
        stats['shapes'].append(shape)
    # Own the snapshot; the model runner reuses its draft buffers. This copy is
    # an explicit bridge cost, not an assumed performance improvement.
    owned = drafts[:count].clone()
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
