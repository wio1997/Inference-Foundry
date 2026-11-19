"""Exercise the Decode marker in a CPU-only PyTorch profiler."""

import ast
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import sqlite3
from types import SimpleNamespace
from typing import Any

import torch

from audit_decode_prefix_trace_g1389 import audit, marker_names
from decode_prefix_annotation_g1387 import PREFIX_SOURCE_SHA256, annotate


class Aborted:
    pass


def main():
    native = Path("/source/native_decode_hicache_mixin.py").read_bytes()
    assert sha256(native).hexdigest() == PREFIX_SOURCE_SHA256
    tree = ast.parse(native)
    definition, = (item for item in tree.body
                   if isinstance(item, ast.ClassDef) and item.name == "DecodePrefixMatch")
    namespace = {"Any": Any, "dataclass": dataclass, "torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "native_match", "exec"), namespace)
    match_type = namespace["DecodePrefixMatch"]
    torch.set_num_threads(1)
    matched = match_type(torch.arange(8), 4, 2, None)
    zero = match_type(torch.empty(0, dtype=torch.int64), 0, 0, None)

    def commit(queue, request):
        request.req.finished_reason = Aborted() if request.req.abort else None
        return request.req.bootstrap_room

    def transfer(room, match, *, rebootstrap=False, rank=0, fake=False, abort=False, enabled=True):
        marked = annotate(commit, lambda: enabled, torch.profiler.record_function,
                          lambda req: req.fake, Aborted)
        req = SimpleNamespace(bootstrap_room=room, origin_input_ids=range(64), fake=fake, abort=abort)
        item = SimpleNamespace(req=req, prefix_match=match, is_rebootstrap=rebootstrap)
        assert marked(SimpleNamespace(tp_rank=rank), item) == room

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        transfer(1, matched)
        transfer(2, zero)
        transfer(3, None)
        transfer(4, zero, rebootstrap=True)
        transfer(5, zero, rank=1)
        transfer(6, zero, fake=True)
        transfer(7, zero, abort=True)
        transfer(8, zero, enabled=False)
    chrome = Path("/output/cpu_trace.json")
    prof.export_chrome_trace(str(chrome))
    names = list(marker_names(chrome, "chrome"))
    report = audit(names, 3, 64)
    assert report["markers"] == 4 and report["known_rooms"] == 2
    assert report["zero_match_rooms"] == 1 and report["unavailable_match"] == 1
    assert report["rebootstrap"] == 1 and not report["full_coverage"]
    assert report["known_unique_fresh_room_token_totals"] == {"l1": 8, "l2": 4, "l3": 2, "declared": 14}
    assert audit([name for name in names if "match=known" in name], 2, 64)["full_coverage"]
    assert not audit(names + [names[0]], 3, 64)["full_coverage"]

    # Verify the exact supported schema using marker strings exported by the real CPU profiler.
    database = Path("/output/sqlite_schema_fixture.db")
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE STRING_IDS (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE PYTORCH_API (name INTEGER)")
        db.executemany("INSERT INTO STRING_IDS VALUES (?, ?)", enumerate(names, 1))
        db.executemany("INSERT INTO PYTORCH_API VALUES (?)", ((i,) for i in range(1, len(names) + 1)))
    assert audit(marker_names(database, "ascend-sqlite"), 3, 64) == report
    print(json.dumps({
        "native_match_source_SHA256": PREFIX_SOURCE_SHA256,
        "torch": str(torch.__version__),
        "architecture": platform.machine(),
        "profiler": "PyTorch CPU",
        "chrome_trace_SHA256": sha256(chrome.read_bytes()).hexdigest(),
        "sqlite_schema_fixture_SHA256": sha256(database.read_bytes()).hexdigest(),
        "sqlite_fixture_source": "real CPU-exported marker strings in generated tables",
        "exported_marker_audit": report,
        "filtered_transfers": ["nonzero_rank", "fake", "aborted", "disabled"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
