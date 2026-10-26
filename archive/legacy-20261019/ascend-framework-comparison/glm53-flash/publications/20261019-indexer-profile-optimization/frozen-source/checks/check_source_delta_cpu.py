#!/usr/bin/env python3
"""CPU proof of the exact R247 three-file candidate delta.

No torch, SGLang, Docker, SSH, HTTP or NPU import is performed.  The test reads
source text and hashes only.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
PROJECT = R.parents[1]
R238 = PROJECT / "runs/20261019-r238-paired-cold-prefill-profile"
R246 = PROJECT / "runs/20261019-r246-dynamic-rowstride-tp16-score-component"
R210 = PROJECT / "runs/20261019-r210-native-topology-alignment"
OPS = "sglang/srt/hardware_backend/npu/ops"
SCORE = R / "stage/B/overlay" / OPS / "flash_kpool_score.py"
CALLER = R / "stage/B/overlay" / OPS / "flash_kpool_indexer_npu.py"
QUERY = R / "stage/B/overlay" / OPS / "flash_indexer_query_parallel.py"
BASE_SCORE = R238 / "stage/B/overlay" / OPS / "flash_kpool_score.py"
BASE_CALLER = R238 / "stage/B/overlay" / OPS / "flash_kpool_indexer_npu.py"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def changed(left, right):
    lines = list(difflib.unified_diff(left.read_text().splitlines(keepends=True),
                                      right.read_text().splitlines(keepends=True)))
    removed = [line[1:].rstrip("\n") for line in lines
               if line.startswith("-") and not line.startswith("---")]
    added = [line[1:].rstrip("\n") for line in lines
             if line.startswith("+") and not line.startswith("+++")]
    return lines, removed, added


def test_score_four_declarations():
    assert sha(BASE_SCORE) == "44a6f449b24955ac406007a7ddf26409b862518a96079a1dec93fca3990db15c"
    assert sha(R246 / "score_ops/flash_kpool_score.py") == sha(BASE_SCORE)
    assert sha(R246 / "score_ops/flash_kpool_score_stride.py") == sha(SCORE)
    lines, removed, added = changed(BASE_SCORE, SCORE)
    assert sum(1 for line in lines if line.startswith("@@")) == 4, lines
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
    text = SCORE.read_text()
    assert "TABLE_SR: tl.constexpr" not in text and "table_stride_r: tl.constexpr" not in text


def test_query_parallel_identity():
    assert sha(QUERY) == sha(R210 / "score_ops/flash_indexer_query_parallel.py")
    assert sha(QUERY) == "b10d0b387e4fa283ead4eb842f851148ef49ceab99ae6ac89e326f9fed377a12"
    text = QUERY.read_text()
    assert "from sglang.srt.hardware_backend.npu.ops.flash_kpool_score import glm5_next_lightning_indexer_triton" in text
    assert "all_gather" in text and "torch.int32" in text
    # R246 tested the same logic with a one-line import alias for its isolated
    # oracle/stride modules.  R247 installs the stride file as the normal
    # flash_kpool_score module, so the R210 import is the correct composition.
    r246_query = R246 / "score_ops/flash_indexer_query_parallel.py"
    lines, removed, added = changed(QUERY, r246_query)
    assert removed == [
        "from sglang.srt.hardware_backend.npu.ops.flash_kpool_score import glm5_next_lightning_indexer_triton"
    ], removed
    assert added == [
        "from sglang.srt.hardware_backend.npu.ops.flash_kpool_score_stride import glm5_next_lightning_indexer_triton"
    ], added
    tree = ast.parse(text)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert {"query", "cache", "weights", "group", "rank", "world_size"} <= names


def test_caller_only_row_split():
    lines, removed, added = changed(BASE_CALLER, CALLER)
    assert sum(1 for line in lines if line.startswith("@@")) == 2, lines
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
    tree = ast.parse(CALLER.read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "query_parallel_indexer"]
    assert len(calls) == 1
    keywords = {row.arg: row.value for row in calls[0].keywords}
    assert isinstance(keywords["group"], ast.Attribute)
    assert keywords["group"].attr == "attn_tp_group"
    assert isinstance(keywords["rank"], ast.Attribute) and keywords["rank"].attr == "attn_tp_rank"
    assert isinstance(keywords["world_size"], ast.Attribute)
    assert keywords["world_size"].attr == "attn_tp_size"
    assert "prefill" in keywords


def test_overlay_and_manifests():
    rows = sorted((R / "stage/B/overlay").rglob("*"))
    files = [path for path in rows if path.is_file()]
    assert len(files) == 47, len(files)
    original = json.loads((R238 / "stage/B/combined-manifest.json").read_text())
    original_rows = {row["target"]: row for row in original["files"]}
    assert len(original_rows) == 46
    installed = json.loads((R / "stage/B/combined-manifest.json").read_text())
    assert installed["status"] == "R247_CANDIDATE_DELTA_NOT_ORIGINAL_46"
    assert len(installed["files"]) == 47
    assert installed["candidate_delta"]["byte_identical_to_original_46"] is False
    by_target = {row["target"]: row for row in installed["files"]}
    unchanged = [target for target in original_rows
                 if target not in (OPS + "/flash_kpool_score.py",
                                   OPS + "/flash_kpool_indexer_npu.py")
                 and by_target[target]["sha256"] == original_rows[target]["sha256"]]
    assert len(unchanged) == 44, len(unchanged)
    assert by_target[OPS + "/flash_kpool_score.py"]["sha256"] == sha(SCORE)
    assert by_target[OPS + "/flash_kpool_indexer_npu.py"]["sha256"] == sha(CALLER)
    assert by_target[OPS + "/flash_indexer_query_parallel.py"]["sha256"] == sha(QUERY)
    stage = json.loads((R / "stage/manifest.json").read_text())
    assert len(stage["B"]) == 55
    assert stage["B"]["overlay/" + OPS + "/flash_indexer_query_parallel.py"]["sha256"] == sha(QUERY)


def test_unchanged_configuration():
    for name in ("argv46.json", "argv47.json", "native-source-pins.json",
                 "flash_pd_writer_first_cleanup.py", "native-environ.py"):
        assert sha(R / "stage/B" / name) == sha(R238 / "stage/B" / name), name
    for name in ("r234_sglang_observer.py", "sitecustomize.py"):
        assert sha(R / "observers/sglang" / name) == sha(
            R238 / "observers/sglang" / name), name
    for name in ("r234_common.py", "r234_sitecustomize_chain.py"):
        assert sha(R / "observers/common" / name) == sha(
            R238 / "observers/common" / name), name
    assert sha(R / "client/r234_prefix_manifest.py") == sha(
        R238 / "client/r234_prefix_manifest.py")


def main():
    test_score_four_declarations()
    test_query_parallel_identity()
    test_caller_only_row_split()
    test_overlay_and_manifests()
    test_unchanged_configuration()
    print(json.dumps({"passed": True, "unit": "r247",
                      "score_sha256": sha(SCORE),
                      "query_parallel_sha256": sha(QUERY),
                      "caller_sha256": sha(CALLER),
                      "overlay_files": 47,
                      "unchanged_original": 44}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
