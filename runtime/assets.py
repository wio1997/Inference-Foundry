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
    skipped: tuple[dict[str, object], ...] = ()

    def restore(self) -> None:
        # DSA cache writes may still be in flight on its overlap stream when
        # target forward returns.  Fence before and after the transactional
        # restore so an older write cannot race past the copy and corrupt the
        # replay pre-state.  This path is bootstrap validation only.
        if self.rows and self.rows[0][0].tensor.device.type == "npu":
            torch.npu.synchronize()
        for cache, first, second, values in self.rows:
            if second is None:
                cache.tensor.index_copy_(0, first, values)
            else:
                cache.tensor[first, second] = values
        if self.rows and self.rows[0][0].tensor.device.type == "npu":
            torch.npu.synchronize()

    def verify_restored(self) -> dict[str, object]:
        """Verify that every captured row currently matches its pre-state."""

        if self.rows and self.rows[0][0].tensor.device.type == "npu":
            torch.npu.synchronize()
        mismatches: list[dict[str, object]] = []
        for cache, first, second, expected in self.rows:
            if second is None:
                current = cache.tensor.index_select(0, first)
            else:
                current = cache.tensor[first, second]
            equal = current == expected
            if current.is_floating_point():
                equal = equal | (torch.isnan(current) & torch.isnan(expected))
            if not bool(equal.all()):
                delta = (current.float() - expected.float()).abs()
                finite_delta = delta[torch.isfinite(delta)]
                mismatches.append(
                    {
                        "name": cache.name,
                        "selected": int(first.numel()),
                        "max_abs": (
                            float(finite_delta.max().item())
                            if finite_delta.numel()
                            else None
                        ),
                    }
                )
        return {
            "exact": not mismatches,
            "checked": len(self.rows),
            "mismatches": mismatches,
        }


class RuntimeAssets:
    """Runtime-owned references after the oracle bootstrap handoff.

    Ownership here means the continuous decode loop addresses these tensors
    directly.  No scheduler or ModelRunner object is retained in this object.
    """

    def __init__(
        self,
        caches: list[OwnedCacheTensor],
        mutable_tensors: list[OwnedCacheTensor] | None = None,
        cache_slot_specs: dict[str, tuple[torch.Tensor, int]] | None = None,
    ) -> None:
        if not caches:
            raise ValueError("at least one physical cache tensor is required")
        names = [cache.name for cache in caches]
        if len(names) != len(set(names)):
            raise ValueError("cache tensor names must be unique")
        self._caches = tuple(caches)
        self._by_name = {cache.name: cache for cache in caches}
        self._mutable_tensors = tuple(mutable_tensors or ())
        self._cache_slot_specs = dict(cache_slot_specs or {})

    @property
    def caches(self) -> tuple[OwnedCacheTensor, ...]:
        return self._caches

    def assert_stable(self) -> None:
        for cache in self._caches:
            cache.assert_stable()
        for tensor in self._mutable_tensors:
            tensor.assert_stable()

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
        slot_mappings: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        num_tokens: int | None = None,
    ) -> CacheSlotSnapshot:
        """Snapshot cache rows addressed by every KV-cache group.

        Hybrid DeepSeek execution uses different physical block sizes for
        different cache groups. Each slot mapping uses its group's block size,
        so a single common mapping cannot restore the complete target state.
        """

        self.assert_stable()
        if torch.is_tensor(slot_mappings):
            slot_mappings = (slot_mappings,)
        else:
            slot_mappings = tuple(slot_mappings)
        if not slot_mappings:
            raise ValueError("at least one slot mapping is required")
        rows: list[
            tuple[
                OwnedCacheTensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor,
            ]
        ] = []
        skipped: list[dict[str, object]] = []
        if num_tokens is not None:
            for mutable in self._mutable_tensors:
                count = min(num_tokens, mutable.tensor.shape[0])
                indices = torch.arange(
                    count, dtype=torch.int64, device=mutable.tensor.device
                )
                rows.append(
                    (
                        mutable,
                        indices,
                        None,
                        mutable.tensor.index_select(0, indices).clone(),
                    )
                )
        for cache in self._caches:
            tensor = cache.tensor
            if tensor.ndim == 0:
                continue
            cache_spec = self._cache_slot_specs.get(cache.name)
            mappings = (cache_spec[0],) if cache_spec is not None else slot_mappings
            slots = torch.unique(
                torch.cat([mapping.to(torch.int64).flatten() for mapping in mappings])
            )
            slots = slots[slots >= 0]
            if tensor.ndim >= 2:
                logical_block_size = (
                    cache_spec[1] if cache_spec is not None else tensor.shape[1]
                )
                physical_block_size = tensor.shape[1]
                logical_blocks = torch.div(
                    slots, logical_block_size, rounding_mode="floor"
                )
                logical_offsets = slots.remainder(logical_block_size)
                row_elements = tensor[0].numel()
                padded_page_layout = (
                    tensor.stride(0) > row_elements
                    and logical_blocks.numel() > 0
                    and bool((logical_blocks < tensor.shape[0]).all())
                )
                if padded_page_layout:
                    page_indices = torch.unique(logical_blocks)
                    rows.append(
                        (
                            cache,
                            page_indices,
                            None,
                            tensor.index_select(0, page_indices).clone(),
                        )
                    )
                    continue
                if logical_block_size % physical_block_size == 0:
                    chunks_per_block = logical_block_size // physical_block_size
                    block_indices = (
                        logical_blocks * chunks_per_block
                        + torch.div(
                            logical_offsets,
                            physical_block_size,
                            rounding_mode="floor",
                        )
                    )
                    offsets = logical_offsets.remainder(physical_block_size)
                else:
                    block_indices = logical_blocks
                    offsets = logical_offsets
                valid_mask = block_indices < tensor.shape[0]
                block_indices = block_indices[valid_mask]
                offsets = offsets[valid_mask]
                if block_indices.numel() > 0:
                    rows.append(
                        (
                            cache,
                            block_indices,
                            offsets,
                            tensor[block_indices, offsets].clone(),
                        )
                    )
                else:
                    skipped.append(
                        {
                            "name": cache.name,
                            "shape": list(cache.shape),
                            "stride": list(cache.stride),
                            "logical_block_size": int(logical_block_size),
                            "physical_block_size": int(physical_block_size),
                            "min_slot": int(slots.min().item()) if slots.numel() else None,
                            "max_slot": int(slots.max().item()) if slots.numel() else None,
                        }
                    )
            else:
                valid = slots[slots < tensor.shape[0]]
                if valid.numel() > 0:
                    rows.append(
                        (cache, valid, None, tensor.index_select(0, valid).clone())
                    )
        if not rows:
            raise RuntimeError("no physical cache tensor accepted target slots")
        return CacheSlotSnapshot(tuple(rows), tuple(skipped))
