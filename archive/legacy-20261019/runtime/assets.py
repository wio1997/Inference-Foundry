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
    no_write: tuple[str, ...] = ()

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

    def coverage(self) -> dict[str, object]:
        return {"captured_names": sorted({cache.name for cache, _, _, _ in self.rows}),
                "captured_rows": len(self.rows), "skipped": list(self.skipped),
                "certified_no_write": list(self.no_write)}

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

    def snapshot_pages(
        self,
        pages_by_cache: dict[str, torch.Tensor],
        *,
        strict: bool = True,
        no_write_caches: set[str] | None = None,
    ) -> CacheSlotSnapshot:
        """Clone complete physical pages selected by an external write-set proof.

        A cache tensor may alias layers in several KV groups with different
        logical block sizes. The caller must union the actual layer write pages
        for each cache view. This method deliberately does not infer pages from
        positional group slot specs.
        """
        self.assert_stable()
        if self._caches[0].tensor.device.type == "npu":
            torch.npu.synchronize()
        expected = set(self._by_name)
        no_write = set(no_write_caches or ())
        supplied = set(pages_by_cache) | no_write
        overlap = sorted(set(pages_by_cache) & no_write)
        unknown = sorted(supplied - expected)
        missing = sorted(expected - supplied)
        if overlap or unknown or (strict and missing):
            raise RuntimeError(
                f"incomplete physical page manifest: overlap={overlap}, unknown={unknown}, missing={missing}"
            )
        rows = []
        skipped = []
        for cache in self._caches:
            if cache.name not in pages_by_cache:
                continue
            tensor = cache.tensor
            if tensor.ndim == 0:
                skipped.append({"name": cache.name, "reason": "scalar_cache"})
                continue
            pages = torch.unique(
                pages_by_cache[cache.name].to(device=tensor.device, dtype=torch.int64).flatten()
            )
            if pages.numel() == 0:
                skipped.append({"name": cache.name, "reason": "empty_page_set"})
                continue
            valid = (pages >= 0) & (pages < tensor.shape[0])
            if not bool(valid.all()):
                skipped.append({
                    "name": cache.name,
                    "reason": "page_out_of_bounds",
                    "invalid_count": int((~valid).sum().item()),
                })
                pages = pages[valid]
            if pages.numel():
                rows.append((cache, pages, None, tensor.index_select(0, pages).clone()))
        for mutable in self._mutable_tensors:
            tensor = mutable.tensor
            if tensor.ndim == 0:
                raise RuntimeError(f"scalar mutable tensor cannot be page-restored: {mutable.name}")
            indices = torch.arange(tensor.shape[0], device=tensor.device, dtype=torch.int64)
            rows.append((mutable, indices, None, tensor.index_select(0, indices).clone()))
        if strict and skipped:
            raise RuntimeError(f"incomplete physical page snapshot: {skipped}")
        if not rows:
            raise RuntimeError("no physical pages captured")
        return CacheSlotSnapshot(tuple(rows), tuple(skipped), tuple(sorted(no_write)))

    def snapshot_slots(
        self,
        slot_mappings: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        num_tokens: int | None = None,
        strict: bool = False,
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
        if num_tokens is not None or strict:
            for mutable in self._mutable_tensors:
                count = (mutable.tensor.shape[0] if strict else
                         min(num_tokens, mutable.tensor.shape[0]))
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
            if strict and bool((slots < 0).any()):
                skipped.append({"name": cache.name, "reason": "negative_candidate_slot",
                                "invalid_count": int((slots < 0).sum().item())})
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
                valid_mask = (block_indices >= 0) & (block_indices < tensor.shape[0])
                if not bool(valid_mask.all()):
                    invalid = block_indices[~valid_mask]
                    skipped.append({
                        "name": cache.name,
                        "reason": "partial_block_index_out_of_bounds",
                        "invalid_count": int(invalid.numel()),
                        "min_invalid_block": int(invalid.min().item()),
                        "max_invalid_block": int(invalid.max().item()),
                        "cache_blocks": int(tensor.shape[0]),
                    })
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
                valid_mask = slots < tensor.shape[0]
                if not bool(valid_mask.all()):
                    invalid = slots[~valid_mask]
                    skipped.append({
                        "name": cache.name,
                        "reason": "partial_flat_index_out_of_bounds",
                        "invalid_count": int(invalid.numel()),
                        "min_invalid_slot": int(invalid.min().item()),
                        "max_invalid_slot": int(invalid.max().item()),
                        "cache_slots": int(tensor.shape[0]),
                    })
                valid = slots[valid_mask]
                if valid.numel() > 0:
                    rows.append(
                        (cache, valid, None, tensor.index_select(0, valid).clone())
                    )
        if not rows:
            raise RuntimeError("no physical cache tensor accepted target slots")
        if strict:
            captured = {cache.name for cache, _, _, _ in rows}
            missing = sorted(cache.name for cache in self._caches if cache.name not in captured)
            if skipped or missing:
                raise RuntimeError(f"incomplete cache snapshot: missing={missing}, skipped={skipped}")
        return CacheSlotSnapshot(tuple(rows), tuple(skipped))
