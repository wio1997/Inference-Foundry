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


@dataclass(frozen=True)
class CacheSlotSnapshot:
    """Transactional copy of touched physical cache rows."""

    rows: tuple[
        tuple[
            OwnedCacheTensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
        ],
        ...,
    ]

    def restore(self) -> None:
        for cache, first, second, values in self.rows:
            if second is None:
                cache.tensor.index_copy_(0, first, values)
            else:
                cache.tensor[first, second] = values


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

    def snapshot_slots(
        self,
        slot_mapping: torch.Tensor,
        *,
        block_size: int = 32,
    ) -> CacheSlotSnapshot:
        """Snapshot only cache rows addressed by a fixed target cycle."""

        self.assert_stable()
        slots = torch.unique(slot_mapping.to(torch.int64))
        slots = slots[slots >= 0]
        rows: list[
            tuple[
                OwnedCacheTensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
            ]
        ] = []
        for cache in self._caches:
            tensor = cache.tensor
            if tensor.ndim == 0:
                continue
            if tensor.ndim >= 2 and tensor.shape[1] == block_size:
                block_indices = torch.div(
                    slots,
                    block_size,
                    rounding_mode="floor",
                )
                offsets = slots.remainder(block_size)
                valid_mask = block_indices < tensor.shape[0]
                block_indices = block_indices[valid_mask]
                offsets = offsets[valid_mask]
                if block_indices.numel() == slots.numel():
                    rows.append(
                        (
                            cache,
                            block_indices,
                            offsets,
                            tensor[block_indices, offsets].clone(),
                        )
                    )
            else:
                valid = slots[slots < tensor.shape[0]]
                if valid.numel() == slots.numel():
                    rows.append(
                        (cache, valid, None, tensor.index_select(0, valid).clone())
                    )
        if not rows:
            raise RuntimeError("no physical cache tensor accepted target slots")
        return CacheSlotSnapshot(tuple(rows))
