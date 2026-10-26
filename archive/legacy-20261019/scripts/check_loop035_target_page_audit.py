"""CPU gate for alias-union target page candidates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from diagnostics.target_page_audit import TargetPageAudit
from runtime.assets import OwnedCacheTensor, RuntimeAssets


def main() -> None:
    names = (
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.compressor.state_cache",
    )
    cache = torch.zeros((100, 4), dtype=torch.int32)
    assets = RuntimeAssets([OwnedCacheTensor.take("shared.cache", cache)])
    tables = [torch.arange(1, 1001, dtype=torch.int32).repeat(12, 1)
              for _ in range(6)]
    bindings = [(table, torch.zeros(96, dtype=torch.int32), 2) for table in tables]
    starts = torch.zeros(12, dtype=torch.int32)
    qsl = torch.arange(0, 97, 8, dtype=torch.int32)
    metadata = {}
    for name, gid, size in ((names[0], 0, 32), (names[1], 4, 2)):
        req = SimpleNamespace(
            num_reqs_actual=12, start_pos=starts, query_start_loc=qsl,
            block_size=size, block_table=tables[gid],
        )
        metadata[name] = SimpleNamespace(req_metadata=req, num_actual_tokens=96)
    context = {name: SimpleNamespace(kv_cache=[cache]) for name in names}
    sources = (SimpleNamespace(layer_name=names[0], ratio=4),
               SimpleNamespace(layer_name=names[1], ratio=1))
    out = Path("/tmp/loop035_page_audit_cpu")
    audit = TargetPageAudit(
        assets=assets, attn_metadata=metadata, static_context=context,
        group_bindings=(), metadata_sources=sources, output_dir=out,
        rank=0, limit=1,
    )
    audit.observe(0)
    row = audit.rows[0]
    assert row["source_layers"] == 2
    assert row["cache_views"] == 1
    assert row["cache_views_snapshotted"] == 1
    assert row["page_count_min"] == 4
    assert row["skipped"] == []
    print(json.dumps({
        "alias_union_exact": True,
        "physical_pages": row["page_count_min"],
        "snapshot_views": row["cache_views_snapshotted"],
        "scope": "synthetic CPU source mapping, not real-weight target writes",
    }))


if __name__ == "__main__":
    main()

