"""CPU-only PyTorch dispatch probe; does not exercise the NPU MC2 operator."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

seen = []
definition = torch.library.Library("wio_snapshot_cpu_probe", "DEF")
definition.define("probe(Tensor x, int capacity, Tensor(a!) out) -> Tensor(a!)")
implementation = torch.library.Library("wio_snapshot_cpu_probe", "IMPL", "CPU")


def cpu_op(x, capacity, out):
    out.copy_(x)
    return out


implementation.impl("probe", cpu_op)


class Probe(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name == "wio_snapshot_cpu_probe::probe":
            seen.append((str(func._schema), args[1], len(args)))
        return func(*args, **(kwargs or {}))


x = torch.zeros(2, dtype=torch.int32)
out = torch.empty_like(x)
with Probe():
    result = torch.ops.wio_snapshot_cpu_probe.probe(x=x, capacity=17, out=out)
assert len(seen) == 1 and seen[0][1] == 17 and result is out
print("CPU_CUSTOM_MUTABLE_OP_MODE_CAPTURED_CAPACITY_AND_RETURN_ALIAS", True)
print("SCHEMA", seen[0][0])
