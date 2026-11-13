# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-ascend commit
# 77cf58266b084b517a658c19ba2d3c97fec27df9, engram/npu.py and
# ops/triton/engram_lookup.py. Copyright contributors to the vLLM project.
"""Isolated SGLang NPU Engram candidate: INT8 codes / E8M0 group32 scales.

No device work or global installation happens merely by importing this module.
Call install() before constructing the model. Not validated on an NPU yet.
"""
import ctypes
import math
import mmap
import os
from functools import cache

import torch
from torch import nn

CHUNK_ROWS = 1 << 22
ALIGNMENT = 1 << 16


@cache
def _acl():
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class AlignedSharedMapping:
    """Reserve our own address range, then map the memfd at 64 KiB alignment.

    MAP_FIXED replaces only the reservation owned here. Every rank maps offset
    zero, so differing virtual addresses never shift the shared physical data.
    Ownership lasts for the worker lifetime, as with native _HostTable.
    """
    def __init__(self, fd, nbytes):
        self.size = math.ceil(nbytes / ALIGNMENT) * ALIGNMENT
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        self.libc = libc
        total = self.size + ALIGNMENT
        base = libc.mmap(None, total, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
        failed = ctypes.c_void_p(-1).value
        if base == failed:
            raise OSError(ctypes.get_errno(), "reserve aligned mapping")
        self.base, self.total = base, total
        self.address = math.ceil(base / ALIGNMENT) * ALIGNMENT
        actual = libc.mmap(self.address, self.size, mmap.PROT_READ | mmap.PROT_WRITE,
                           mmap.MAP_SHARED | 0x10, fd, 0)  # MAP_FIXED within reservation
        if actual == failed:
            libc.munmap(base, total)
            raise OSError(ctypes.get_errno(), "map aligned shared memfd")
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise(self.address, self.size, mmap.MADV_HUGEPAGE)
        self.buffer = (ctypes.c_uint8 * nbytes).from_address(self.address)
        # frombuffer is intrinsically host-backed and has no device argument.
        # Pin its construction context explicitly; never migrate the shared
        # mapping because the outer model loader defaults factories to NPU.
        with torch.device("cpu"):
            self.bytes = torch.frombuffer(self.buffer, dtype=torch.uint8)
        if self.bytes.device.type != "cpu":
            raise RuntimeError("Engram shared mapping must remain a CPU tensor")


def register_host(bytes_tensor):
    """A3 standard CANN ABI: MAPPED | PINNED, not A5's MAPPED-only flags."""
    pointer = bytes_tensor.data_ptr()
    rc = _acl().aclrtHostRegisterV2(pointer, bytes_tensor.numel(), 0x10000002)
    if rc:
        raise RuntimeError(f"aclrtHostRegisterV2 rc={rc}, bytes={bytes_tensor.numel()}")
    address = ctypes.c_void_p()
    rc = _acl().aclrtHostGetDevicePointer(pointer, ctypes.byref(address), 0)
    if rc or not address.value:
        _acl().aclrtHostUnregister(pointer)
        raise RuntimeError(f"aclrtHostGetDevicePointer rc={rc}, address={address.value}")
    return address.value


def make_pointer_tables(address, rows, dim, device):
    starts = range(0, rows, CHUNK_ROWS)
    weight = torch.tensor([address + r * dim for r in starts], dtype=torch.int64, device=device)
    scale_base = address + rows * dim
    scales = torch.tensor([scale_base + r * (dim // 32) for r in starts], dtype=torch.int64, device=device)
    return weight, scales


def gather(weight_ptrs, scale_ptrs, indices, dim=256, *, total_rows):
    # Import lazily; importing this helper alone must not initialize Triton.
    from engram_npu_uva_kernel import _gather_int8_e8m0
    ids = indices.reshape(-1)
    output = torch.empty((*indices.shape, dim), dtype=torch.bfloat16, device=indices.device)
    if ids.numel():
        _gather_int8_e8m0[(ids.numel(),)](weight_ptrs, scale_ptrs, ids, output, total_rows,
                                        WIDTH=dim, GROUP=32, CHUNK=CHUNK_ROWS,
                                        num_warps=4)
    return output


_ROTATION = None


def load_rotation(path, hidden_size=None):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        rotation = f.get_tensor("global_rotation").float()
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("global_rotation must be square")
    dim = rotation.shape[0]
    if hidden_size is not None and dim != hidden_size:
        raise ValueError("global_rotation does not match hidden_size")
    if dim % 32:
        raise ValueError("global_rotation hidden size must be divisible by 32")
    n = dim // 32
    blocks = rotation.reshape(n, 32, n, 32).permute(0, 2, 1, 3)
    cpu_index = torch.arange(n, device="cpu")
    diag = blocks[cpu_index, cpu_index].contiguous()
    expected = torch.zeros_like(blocks, device="cpu")
    expected[cpu_index, cpu_index] = diag
    if not torch.equal(blocks, expected):
        raise ValueError("expected block32 diagonal rotation; dense rotation requires separate review")
    if not torch.equal(diag, diag[:1].expand_as(diag)):
        raise ValueError("target requires identical block32 rotations; heterogeneous blocks need separate review")
    return diag[0].contiguous()


def rotated_gate(x, kv, q_weight, k_weight, eps, clamp_value,
                 input_ids=None, image_token_id=None):
    # Native gate formula, except normalization and dot use the original basis;
    # value projection and residual stay in the checkpoint's rotated basis.
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    rotation = _ROTATION
    if rotation is None or rotation.device != x.device:
        raise RuntimeError("rotation must be loaded on NPU before graph capture")
    # The verified checkpoint repeats one exact 32x32 block. Flattening
    # avoids broadcast-expanding 160 rotation blocks over tokens/hc copies.
    original = (x.float().reshape(-1, 32) @ rotation.T).reshape(x.shape)
    rstd = torch.rsqrt(original.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    channel_weight = q_weight.float() * k_weight.float()
    dot = (original * channel_weight * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(clamp_value).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    out = (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    if input_ids is not None:
        out = torch.where((input_ids == image_token_id)[:, None, None], x, out)
    return out


def install(rotation_path=None):
    """Explicit target-only installation before SGLang model construction.

    Target must have verified INT8_SYM / E8M0 Engram metadata. Requires native
    --engram-config placement=cpu,replication=per_node. Native hash/history,
    DSpark acceptance and projection loaders remain the owners of their state.
    """
    global _ROTATION
    from sglang.srt.layers import engram as native
    if getattr(native, "_npu_engram_int8_e8m0_installed", False):
        return
    if not torch.npu.is_available():
        raise RuntimeError("NPU Engram compatibility requires NPU")
    if not rotation_path:
        raise ValueError("this rotated A3 checkpoint requires global_rotation path")
    _ROTATION = load_rotation(rotation_path).to(torch.device("npu", torch.npu.current_device()))

    class NpuHostTable(native._HostTable):
        def __init__(self, replication, nbytes, name, group):
            if replication != native.EngramReplication.PER_NODE:
                raise ValueError("NPU Engram candidate requires replication=per_node")
            self.replication, self.nbytes, self.group = replication, nbytes, group
            self.dirty = False
            self.fd = self._open_shared_fd(nbytes, name)
            self.mapping = AlignedSharedMapping(self.fd, nbytes)
            self.bytes = self.mapping.bytes
            group.barrier()
            self.device_ptr = register_host(self.bytes)

    class NpuEngramEmbedding(native.EngramEmbedding):
        def _init_host_table(self, num_embeddings, dim, layer_id, replication):
            if self.mxfp4 or dim != 256:
                raise ValueError("NPU candidate supports INT8/E8M0 width256 only")
            self.host_table = NpuHostTable(replication, num_embeddings * (dim + dim // 32),
                                           f"sglang_engram_{layer_id}", native.get_parallel().tp_group)
            raw = self.host_table.bytes
            boundary = num_embeddings * dim
            self.weight = nn.Parameter(raw[:boundary].view(torch.int8).view(num_embeddings, dim), requires_grad=False)
            self.scale = nn.Parameter(raw[boundary:].view(torch.float8_e8m0fnu).view(num_embeddings, dim // 32), requires_grad=False)
            self._npu_total_rows = num_embeddings
            self._npu_ptrs = make_pointer_tables(self.host_table.device_ptr, num_embeddings, dim,
                                                 torch.device("npu", torch.npu.current_device()))
            self._sort_host_reads = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.host_table is None or not self._shared:
                raise ValueError("NPU Engram candidate requires cpu/per_node placement")

        def _load_rows(self, param, loaded_weight):
            expected = torch.int8 if param is self.weight else torch.float8_e8m0fnu
            if loaded_weight.dtype != expected:
                raise ValueError(f"Engram checkpoint dtype {loaded_weight.dtype}, expected {expected}; no reinterpret/requantization allowed")
            rows = slice(self.row_start, self.row_start + self.rows)
            src, dst = loaded_weight[rows], param.data[rows]
            if src.shape != dst.shape:
                raise ValueError(f"Engram row shape mismatch {src.shape} vs {dst.shape}")
            # Copy exponent/code bytes exactly; no CPU float8 arithmetic required.
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
            self.host_table.dirty = True

        def forward(self, indices, forward_batch=None, *, cp_all_tokens=False):
            # Shared full table means each DP rank reads its own token ids.
            # This covers extend, decode and DSpark target verify uniformly.
            return gather(*self._npu_ptrs, indices, self.dim, total_rows=self._npu_total_rows)

    native.EngramEmbedding = NpuEngramEmbedding
    native.engram_gate = rotated_gate
    native._npu_engram_int8_e8m0_installed = True
