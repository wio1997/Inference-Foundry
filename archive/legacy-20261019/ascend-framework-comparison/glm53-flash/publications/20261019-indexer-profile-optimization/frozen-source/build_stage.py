#!/usr/bin/env python3
"""Reproduce the R247 B candidate stage from immutable R238/R246/R210 sources.

R247 does not claim the original 46-file B overlay is unchanged.  It copies
that overlay, then installs exactly two performance candidates:

* R246 runtime row-stride score (four changed declarations) as
  ``flash_kpool_score.py``;
* R210 generic ``query_parallel_indexer`` plus the R210 caller that passes the
  current native attention-TP group.

No R233/Graph/MTP/KDA/FP32-scale/topology/cache-grid change is copied.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import pathlib
import shutil
import sys

R = pathlib.Path(__file__).resolve().parent
PROJECT = R.parents[1]
R238 = PROJECT / "runs/20261019-r238-paired-cold-prefill-profile"
R246 = PROJECT / "runs/20261019-r246-dynamic-rowstride-tp16-score-component"
R210 = PROJECT / "runs/20261019-r210-native-topology-alignment"
BASE = R238 / "stage/B"
TARGET = R / "stage/B"
OPS = "sglang/srt/hardware_backend/npu/ops"
SCORE = OPS + "/flash_kpool_score.py"
CALLER = OPS + "/flash_kpool_indexer_npu.py"
QUERY = OPS + "/flash_indexer_query_parallel.py"

TOP_FILES = ("argv46.json", "argv47.json", "flash_pd_writer_first_cleanup.py",
             "native-environ.py", "native-source-pins.json", "preflight.py",
             "start_after_preflight.py")


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def unified(rel, left, right):
    return list(difflib.unified_diff(
        left.read_text().splitlines(keepends=True),
        right.read_text().splitlines(keepends=True),
        fromfile=str(left.relative_to(PROJECT)), tofile=str(right.relative_to(PROJECT))))


def changed(lines):
    removed = [line[1:].rstrip("\n") for line in lines
               if line.startswith("-") and not line.startswith("---")]
    added = [line[1:].rstrip("\n") for line in lines
             if line.startswith("+") and not line.startswith("+++")]
    return removed, added


def verify_preconditions():
    oracle = R246 / "score_ops/flash_kpool_score.py"
    stride = R246 / "score_ops/flash_kpool_score_stride.py"
    assert sha(BASE / "overlay" / SCORE) == sha(oracle), "R238 score is not the R246 oracle"
    score_diff = unified(SCORE, BASE / "overlay" / SCORE, stride)
    removed, added = changed(score_diff)
    assert removed == [
        '@triton.jit(do_not_specialize=["requests", "pools_count", "blocks_count"])',
        "    TABLE_SR: tl.constexpr,",
        '@triton.jit(do_not_specialize=["token_offset", "rows", "max_pool_seq_len", "num_reqs", "num_cache_blocks"])',
        "    table_stride_r: tl.constexpr,",
    ], removed
    assert added == [
        '@triton.jit(do_not_specialize=["requests", "pools_count", "blocks_count", "TABLE_SR"])',
        "    TABLE_SR,",
        '@triton.jit(do_not_specialize=["token_offset", "rows", "max_pool_seq_len", "num_reqs", "num_cache_blocks", "table_stride_r"])',
        "    table_stride_r,",
    ], added
    assert sum(1 for line in score_diff if line.startswith("@@")) == 4
    caller_diff = unified(CALLER, BASE / "overlay" / CALLER,
                          R210 / "score_ops/flash_kpool_indexer_npu.py")
    removed, added = changed(caller_diff)
    assert removed == [
        "    from sglang.srt.hardware_backend.npu.ops.flash_kpool_score import glm5_next_lightning_indexer_triton",
        "    output=glm5_next_lightning_indexer_triton(query,index_cache,weights,query_ends,",
    ], removed
    assert added == [
        "    from sglang.srt.hardware_backend.npu.ops.flash_indexer_query_parallel import query_parallel_indexer",
        "    from sglang.srt.runtime_context import get_parallel",
        "    parallel=get_parallel()",
        "    output=query_parallel_indexer(query,index_cache,weights,query_ends,",
        "        group=parallel.attn_tp_group,rank=parallel.attn_tp_rank,world_size=parallel.attn_tp_size,",
        "        prefill=forward_batch.forward_mode.is_extend(),",
    ], added
    assert sum(1 for line in caller_diff if line.startswith("@@")) == 2
    assert sha(R210 / "score_ops/flash_indexer_query_parallel.py") == (
        "b10d0b387e4fa283ead4eb842f851148ef49ceab99ae6ac89e326f9fed377a12")
    return score_diff, caller_diff


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    score_diff, caller_diff = verify_preconditions()

    # Copy the exact R238 B stage top files, then the 46-file overlay.
    for name in TOP_FILES:
        if name in ("preflight.py", "start_after_preflight.py"):
            continue  # R247-specific source checks are maintained in this unit.
        shutil.copyfile(BASE / name, TARGET / name)
    for source in sorted((BASE / "overlay").rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(BASE / "overlay")
        target = TARGET / "overlay" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(R246 / "score_ops/flash_kpool_score_stride.py",
                    TARGET / "overlay" / SCORE)
    shutil.copyfile(R210 / "score_ops/flash_indexer_query_parallel.py",
                    TARGET / "overlay" / QUERY)
    shutil.copyfile(R210 / "score_ops/flash_kpool_indexer_npu.py",
                    TARGET / "overlay" / CALLER)

    rows = []
    for path in sorted((TARGET / "overlay").rglob("*")):
        if not path.is_file():
            continue
        target = str(path.relative_to(TARGET / "overlay"))
        rows.append({"target": target, "sha256": sha(path), "bytes": path.stat().st_size})
    assert len(rows) == 47, len(rows)
    by_target = {row["target"]: row for row in rows}
    original = json.loads((BASE / "combined-manifest.json").read_text())
    original_rows = {row["target"]: row for row in original["files"]}
    assert len(original_rows) == 46
    unchanged = [target for target in original_rows
                 if target not in (SCORE, CALLER) and by_target[target]["sha256"] == original_rows[target]["sha256"]]
    assert len(unchanged) == 44, len(unchanged)
    assert by_target[SCORE]["sha256"] == sha(R246 / "score_ops/flash_kpool_score_stride.py")
    assert by_target[CALLER]["sha256"] == sha(R210 / "score_ops/flash_kpool_indexer_npu.py")
    assert by_target[QUERY]["sha256"] == sha(R210 / "score_ops/flash_indexer_query_parallel.py")
    manifest = {
        "project_id": "glm53-flash",
        "unit": "r247",
        "status": "R247_CANDIDATE_DELTA_NOT_ORIGINAL_46",
        "base": "R238 B original 46-file overlay, copied byte-for-byte except the listed delta",
        "candidate_delta": {
            "changed": [
                {"target": SCORE, "from": "R238 original", "source": "R246 stride candidate",
                 "original_sha256": original_rows[SCORE]["sha256"],
                 "installed_sha256": by_target[SCORE]["sha256"],
                 "change": "four runtime-scalar declarations (TABLE_SR, table_stride_r)"},
                {"target": CALLER, "from": "R238 original", "source": "R210 caller",
                 "original_sha256": original_rows[CALLER]["sha256"],
                 "installed_sha256": by_target[CALLER]["sha256"],
                 "change": "query-independent row split using current native attention TP group"},
            ],
            "added": [
                {"target": QUERY, "source": "R210 query_parallel_indexer",
                 "installed_sha256": by_target[QUERY]["sha256"],
                 "change": "local row scoring plus one int32 all_gather_into_tensor"},
            ],
            "unchanged_original_count": len(unchanged),
            "overlay_file_count": len(rows),
            "original_overlay_file_count": len(original_rows),
            "byte_identical_to_original_46": False,
        },
        "files": rows,
    }
    write_json(TARGET / "combined-manifest.json", manifest)

    stage = {"project_id": "glm53-flash", "unit": "r247", "kind": "R247_CANDIDATE_STAGE",
             "source_baseline": "runs/20261019-r238-paired-cold-prefill-profile/stage/B",
             "B": {}}
    for name in sorted(set(TOP_FILES) | {"combined-manifest.json"}):
        path = TARGET / name
        stage["B"][name] = {"source": str(path.relative_to(PROJECT)),
                            "sha256": sha(path), "bytes": path.stat().st_size}
    for row in rows:
        path = TARGET / "overlay" / row["target"]
        stage["B"]["overlay/" + row["target"]] = {
            "source": str(path.relative_to(PROJECT)),
            "sha256": row["sha256"], "bytes": row["bytes"]}
    assert len(stage["B"]) == 55, len(stage["B"])
    write_json(R / "stage/manifest.json", stage)
    write_json(R / "results/source-diff.json", {
        "project_id": "glm53-flash", "unit": "r247",
        "score": {"path": SCORE, "diff": "".join(score_diff),
                  "original_sha256": original_rows[SCORE]["sha256"],
                  "installed_sha256": by_target[SCORE]["sha256"]},
        "caller": {"path": CALLER, "diff": "".join(caller_diff),
                   "original_sha256": original_rows[CALLER]["sha256"],
                   "installed_sha256": by_target[CALLER]["sha256"]},
        "query_parallel": {"path": QUERY, "new_file": True,
                           "installed_sha256": by_target[QUERY]["sha256"]},
        "unchanged_original_count": len(unchanged),
        "overlay_file_count": len(rows),
        "byte_identical_to_original_46": False,
    })
    print(json.dumps({"B_files": len(stage["B"]), "overlay": len(rows),
                      "unchanged_original": len(unchanged),
                      "combined_manifest_sha256": sha(TARGET / "combined-manifest.json"),
                      "stage_manifest_sha256": sha(R / "stage/manifest.json")}))


if __name__ == "__main__":
    sys.exit(main())
