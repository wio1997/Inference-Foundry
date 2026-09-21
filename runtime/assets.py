"""One-time transfer of physical model assets into Extreme Runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OwnedCacheTensor:
    name: str
    tensor: torch.Tensor
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @classmethod
    def take(cls, name: str, tensor: torch.Tensor) -> "OwnedCacheTensor":
        if not name:
            raise ValueError("cache tensor name must be non-empty")
        return cls(
            name=name,
            tensor=tensor,
            data_ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
        )

    def assert_stable(self) -> None:
        if self.tensor.data_ptr() != self.data_ptr:
            raise RuntimeError(f"cache tensor address changed: {self.name}")
        if tuple(self.tensor.shape) != self.shape:
            raise RuntimeError(f"cache tensor shape changed: {self.name}")
        if tuple(self.tensor.stride()) != self.stride:
            raise RuntimeError(f"cache tensor stride changed: {self.name}")


class RuntimeAssets:
    """Runtime-owned references after the oracle bootstrap handoff.

    Ownership here means the continuous decode loop addresses these tensors
    directly.  No scheduler or ModelRunner object is retained in this object.
    """

    def __init__(self, caches: list[OwnedCacheTensor]) -> None:
        if not caches:
            raise ValueError("at least one physical cache tensor is required")
        names = [cache.name for cache in caches]
        if len(names) != len(set(names)):
            raise ValueError("cache tensor names must be unique")
        self._caches = tuple(caches)
        self._by_name = {cache.name: cache for cache in caches}

    @property
    def caches(self) -> tuple[OwnedCacheTensor, ...]:
        return self._caches

    def assert_stable(self) -> None:
        for cache in self._caches:
            cache.assert_stable()

    def exact_fingerprint(
        self,
        indices: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Clone selected physical cache elements for exact parity checks.

        The integration binding supplies indices derived from the fixed slot
        mappings, so validation covers touched cache locations instead of a
        lossy scalar checksum or the entire multi-gigabyte cache.
        """

        self.assert_stable()
        result: dict[str, torch.Tensor] = {}
        for name, selected in indices.items():
            cache = self._by_name[name]
            flat = cache.tensor.view(-1)
            result[name] = flat.index_select(0, selected.to(torch.int64)).clone()
        return result
