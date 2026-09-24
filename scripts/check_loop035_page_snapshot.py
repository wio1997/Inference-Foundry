"""CPU gate for page-level transactional KV snapshots and shared storage."""
from __future__ import annotations

import json

import torch

from runtime.assets import OwnedCacheTensor, RuntimeAssets


def main() -> None:
    base = torch.arange(16, dtype=torch.int64).reshape(4, 4)
    alias = base.view(4, 2, 2)
    mutable = torch.arange(6, dtype=torch.int64)
    assets = RuntimeAssets(
        [OwnedCacheTensor.take("cache.a", base),
         OwnedCacheTensor.take("cache.b", alias)],
        [OwnedCacheTensor.take("scratch", mutable)],
    )
    expected_base = base.clone()
    expected_mutable = mutable.clone()
    snapshot = assets.snapshot_pages({
        "cache.a": torch.tensor([1, 1]),
        "cache.b": torch.tensor([1]),
    })
    base[1].fill_(-9)
    mutable.fill_(-3)
    snapshot.restore()
    assert snapshot.verify_restored()["exact"]
    assert torch.equal(base, expected_base)
    assert torch.equal(mutable, expected_mutable)

    rejected = []
    for label, pages in (
        ("missing", {"cache.a": torch.tensor([1])}),
        ("out_of_bounds", {"cache.a": torch.tensor([1]),
                           "cache.b": torch.tensor([9])}),
        ("empty", {"cache.a": torch.tensor([1]),
                   "cache.b": torch.tensor([], dtype=torch.int64)}),
    ):
        try:
            assets.snapshot_pages(pages)
        except RuntimeError:
            rejected.append(label)
    assert rejected == ["missing", "out_of_bounds", "empty"]
    print(json.dumps({
        "shared_storage_restore_exact": True,
        "mutables_restored_exact": True,
        "strict_rejections": rejected,
        "scope": "synthetic CPU page snapshot gate, not live KV coverage",
    }))


if __name__ == "__main__":
    main()
