"""Native graph resources for the original DFlashCudaGraphRunner.

Only device APIs and stable metadata storage live here. Graph size selection,
input padding, proposal sampling, output trimming and graph release belong to
the original runner/module interfaces.
"""
import inspect
import math
from types import SimpleNamespace

import torch
import torch.distributed as dist

from deepinfer.worker.dflash_cuda_graph_runner import DFlashCudaGraphRunner


def native_graph_api():
    parameters = inspect.signature(torch.npu.graph).parameters
    if not {'pool', 'stream'} <= parameters.keys():
        raise RuntimeError('native graph context must support a pool and a capture stream')

    def graph(value, *, pool, stream, capture_error_mode):
        kwargs = dict(pool=pool, stream=stream)
        if 'capture_error_mode' in parameters:
            kwargs['capture_error_mode'] = capture_error_mode
        return torch.npu.graph(value, **kwargs)

    return SimpleNamespace(current_stream=torch.npu.current_stream, stream=torch.npu.stream,
                           CUDAGraph=torch.npu.NPUGraph, graph=graph)


class OriginalDraftGraphs:
    def __init__(self, module, model, worker):
        self.module, self.model = module, model
        self.maximum = worker.scheduler_config.max_num_seqs
        self.runner = None
        self.context = None
        self.replays = 0
        error = None
        try:
            if self.maximum < 1 or self.maximum >= model.requests:
                raise ValueError('draft graphs exceed the original real-state capacity')
            api = native_graph_api()
            self.stream = torch.npu.Stream(device=model.device)
            self.pool = torch.npu.graph_pool_handle()
            self.context = SimpleNamespace(
                positions=torch.zeros(self.maximum * (1 + module.max_MTP_value), dtype=torch.int32, device=model.device),
                # Capture has no admitted state owners. Distinct dummy slots
                # avoid scratch collisions while all capture rows are padding.
                slots=torch.arange(self.maximum, dtype=torch.int32, device=model.device),
                valid=torch.zeros(self.maximum, dtype=torch.bool, device=model.device),
                global_bs=dist.get_world_size() * math.ceil(self.maximum * module.max_MTP_value / module.draft.tp_size),
                scratch=model.requests - 1)
            self.runner = DFlashCudaGraphRunner(self.maximum, worker.is_driver_worker,
                                                graph_api=api, capture_vote=module._vote)
            snapshots = [plane.cache.clone() for plane in module.draft.planes]
        except Exception as exc:
            error = str(exc)
        module._vote(error)
        try:
            self.runner.capture(module, model.cache, self.context, self.pool, self.stream)
        finally:
            torch.npu.current_stream(model.device).wait_stream(self.stream)
            for plane, saved in zip(module.draft.planes, snapshots, strict=True):
                plane.cache.copy_(saved)
        error = None
        try:
            if worker.is_driver_worker:
                module.sampling_validation.check_now()
        except Exception as exc:
            error = str(exc)
        module._vote(error)
        self.batch_sizes = self.runner.batch_sizes

    def stage(self, positions, slots, *, real_requests):
        count = len(slots)
        if (not 1 <= count <= self.maximum or positions.ndim != 1
                or positions.dtype != torch.int32 or positions.device != self.model.device
                or positions.numel() > self.context.positions.numel()
                or len(set(slots)) != count
                or any(type(slot) is not int or not 0 <= slot < self.model.requests for slot in slots)):
            raise ValueError('original draft graph input exceeds its stable metadata capacity')
        self.context.positions.zero_()
        self.module.sampling_validation.reset()
        self.context.positions[:positions.numel()].copy_(positions)
        self.context.slots.fill_(self.context.scratch)
        self.context.slots[:count].copy_(torch.tensor(slots, dtype=torch.int32, device=self.model.device))
        self.context.valid.zero_()
        if real_requests:
            self.context.valid[:count].fill_(True)

    def replay(self, decode_input):
        if self.runner is None:
            raise RuntimeError('original draft graphs have been released')
        result = self.runner.replay(decode_input)
        if self.module.worker.is_driver_worker:
            self.module.sampling_validation.enqueue()
        self.replays += 1
        return result

    def destroy(self):
        work = getattr(self.model, '_original_device_work', None)
        if work is not None:
            work.drain()
        else:
            torch.npu.current_stream(self.model.device).synchronize()
        if self.runner is not None:
            self.runner.destroy()
        self.runner = self.context = self.stream = self.pool = None
