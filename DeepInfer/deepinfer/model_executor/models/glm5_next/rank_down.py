# SPDX-License-Identifier: Apache-2.0
"""Explicit input-sharded W8A8 dense down projection with BF16 reduction.

This opt-in seam expects identical replicated BF16 activations and a matching
initialized group. It does not establish whole-model tensor-parallel support.
"""
import hashlib
import torch
import torch.distributed as dist
from torch import nn


class RankW8A8InputLinear(nn.Module):
    def __init__(self, reader, prefix, device, *, rank, world_size, group=None,
                 max_input_tokens=8):
        super().__init__()
        if torch.device(device).type != 'npu':
            raise ValueError('explicit NPU device required')
        if type(world_size) is not int or world_size < 2 or type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError('explicit multi-rank allocation required')
        if not dist.is_initialized() or dist.get_world_size(group) != world_size or dist.get_rank(group) != rank:
            raise ValueError('explicit initialized matching process group required')
        self.rank, self.world_size, self.group = rank, world_size, group
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        status = torch.zeros((1,), dtype=torch.int32, device=self.device)
        failure = None
        try:
            if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= 40:
                raise ValueError('input token bound must be 1..40')
            if not isinstance(prefix, str) or not prefix:
                raise ValueError('nonempty projection prefix required')
            self.prefix_fingerprint = int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:8], 'big', signed=True)
            desc = reader.descriptor(prefix + '.weight')[2]
            if reader.quant.get(prefix + '.weight') != 'W8A8_DYNAMIC' or desc['dtype'] != 'I8' or len(desc['shape']) != 2:
                raise ValueError('expected W8A8_DYNAMIC matrix')
            self.output_size, self.input_size = desc['shape']
            if self.input_size % world_size or (self.input_size // world_size) % 16:
                raise ValueError('equal 16-column aligned shards required')
            width = self.input_size // world_size
            self.local_start, self.local_stop = rank * width, (rank + 1) * width
            weight, scale = reader.linear_payload(prefix)
            if tuple(weight.shape) != (self.output_size, self.input_size) or tuple(scale.shape) != (self.output_size,):
                raise ValueError('invalid down projection payload geometry')
            from deepinfer.npu_ops.w8a8 import AscendW8A8Linear
            self.local = AscendW8A8Linear(weight[:, self.local_start:self.local_stop].contiguous().to(device), scale.to(device))
            import torch_npu
            torch_npu.npu.synchronize()
        except Exception as exc:
            failure = exc
        status.fill_(int(failure is not None))
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
        if int(status.item()):
            raise RuntimeError('one or more ranks failed down projection initialization') from failure

    def forward(self, x):
        if not dist.is_initialized() or dist.get_world_size(self.group) != self.world_size or dist.get_rank(self.group) != self.rank:
            raise RuntimeError('rank group changed after construction')
        valid = (isinstance(x, torch.Tensor) and x.device.type == 'npu' and x.dtype == torch.bfloat16
                 and x.ndim == 2 and 1 <= x.shape[0] <= self.max_input_tokens and x.shape[1] == self.input_size)
        meta = torch.tensor([0, x.shape[0], x.shape[1], self.output_size, self.prefix_fingerprint, self.max_input_tokens] if valid else [1, 0, 0, 0, 0, 0], dtype=torch.int64, device=self.device)
        gathered = [torch.empty_like(meta) for _ in range(self.world_size)]
        dist.all_gather(gathered, meta, group=self.group)
        if any(int(v[0].item()) for v in gathered) or not all(torch.equal(gathered[0], v) for v in gathered[1:]):
            raise ValueError('rank down input preflight failed or geometry differs')
        status = torch.zeros((1,), dtype=torch.int32, device=self.device)
        failure = None
        try:
            value = x.contiguous()
            replicas = [torch.empty_like(value) for _ in range(self.world_size)]
            import torch_npu
            torch_npu.npu.synchronize()
        except Exception as exc:
            failure = exc
        status.fill_(int(failure is not None))
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self.group)
        if int(status.item()):
            raise RuntimeError('one or more ranks failed down input preparation') from failure
        dist.all_gather(replicas, value, group=self.group)
        if not all(torch.equal(replicas[0], item) for item in replicas[1:]):
            raise ValueError('rank down inputs differ')
        failure = None
        try:
            partial = self.local(value[:, self.local_start:self.local_stop].contiguous())
            if partial.shape != (value.shape[0], self.output_size) or partial.dtype != torch.bfloat16:
                raise ValueError('invalid local down output')
            torch_npu.npu.synchronize()
        except Exception as exc:
            failure = exc
        status.fill_(int(failure is not None))
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=self.group)
        if int(status.item()):
            raise RuntimeError('one or more ranks failed local down projection') from failure
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.group)
        return partial
