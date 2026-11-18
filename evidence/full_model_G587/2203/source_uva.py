# SPDX-License-Identifier: Apache-2.0
# Extracted unchanged declarations from the pinned vLLM-Ascend image; source provenance in source_manifest.json.
import ctypes
from functools import cache
import torch,triton
import triton.language as tl
SCALE_GROUP=32
CHUNK_ROWS=1<<22
class DeviceOperator:
    @staticmethod
    def host_register_flags(): return 2

@cache
def _host_library() -> ctypes.CDLL:
    """The CANN runtime entry points that publish host memory to the device."""

    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class HostUvaBuffer:
    """Host memory the device gathers from directly."""

    def __init__(self, shape, dtype, device):
        self.lib = _host_library()
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        try:
            rc = self.lib.aclrtHostRegisterV2(self.pointer, size, DeviceOperator.host_register_flags())
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception:
            # A half-built buffer must not leave the host range registered: the
            # caller only sees the exception, so nothing else can release it.
            self.lib.aclrtHostUnregister(self.pointer)
            self.lib.aclrtFreeHost(self.pointer)
            self.tensor = None
            self.buffer = None
            self.pointer = ctypes.c_void_p()
            raise

    def close(self):
        """Unregister and release the host range; safe to call more than once.

        The NPU gathers out of this range through the device address the
        registration published, so the mapping must not go away while work that
        reads it is still in flight.
        """
        if self.pointer is None or not self.pointer.value:
            return
        if self.ptrs is not None and self.ptrs.device.type == "npu":
            torch.npu.synchronize()
        rc = self.lib.aclrtHostUnregister(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")
        self.tensor = None
        self.buffer = None
        self.ptrs = None
        rc = self.lib.aclrtFreeHost(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtFreeHost failed: rc={rc}")
        self.pointer = ctypes.c_void_p()


@triton.jit
def _decode_e8m0(scale):
    # E8M0 has no sign or mantissa. 0 is 2^-127 (not FP32 zero),
    # and 255 is NaN (not FP32 infinity).
    # Ascend byte loads can sign-extend values >= 128 during widening.
    exponent = scale.to(tl.int32) & 0xFF
    bits = exponent << 23
    bits = tl.where(exponent == 0, 0x00400000, bits)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _engram_host_uva_gather_dequant_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
    MXFP8: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        token = row // LOCAL_HEADS
        head_local = row % LOCAL_HEADS
        index = tl.load(ids + token * ids_stride_t + HEAD_START + head_local).to(tl.int64)
        owned = (index >= vocab_start) & (index < vocab_end)
        local_row = tl.where(owned, index - vocab_start, 0)
        chunk = local_row // CHUNK
        local = local_row % CHUNK
        if MXFP8:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.float8e4nv))
        elif QUANTIZED:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        else:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.bfloat16))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        if QUANTIZED:
            if MXFP8:
                scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.uint8))
            else:
                scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
            if MXFP8:
                # Read and decode each group scale once, then broadcast to
                # its values instead of gathering WIDTH repeated bytes.
                groups = tl.arange(0, WIDTH // GROUP)
                scale = tl.load(scales + local * (WIDTH // GROUP) + groups)
                scale = _decode_e8m0(scale)
                value = tl.reshape(value, (WIDTH // GROUP, GROUP)) * scale[:, None]
                value = tl.reshape(value, (WIDTH,))
            else:
                scale = tl.load(scales + local * (WIDTH // GROUP) + col // GROUP)
                value = value * scale
        result = value.to(tl.bfloat16)
        tl.store(
            output + (token * PAD_HEADS + head_local) * WIDTH + col,
            tl.where(owned, result, tl.zeros_like(result)),
        )

