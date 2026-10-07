"""Actual observer AST versus production registration/lookup; CPU only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "shim/rotary_embedding.py"
PRODUCTION = ROOT / "candidate/rotary_embedding.py"
torch.set_num_threads(1)


def load(path, observer=False):
    names = {"get_cos_and_sin_mla", "_record_cos_and_sin_cache_interleaved"}
    if observer:
        names.update(("_h12_register", "_h12_lookup", "_h12_layout"))
    definitions = [n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(torch=torch, os=os, __file__=str(path), _cos_cache=None, _sin_cache=None,
              _cos_mla=torch.zeros(32, 1, 1, 64, dtype=torch.bfloat16),
              _sin_mla=torch.zeros(32, 1, 1, 64, dtype=torch.bfloat16),
              torch_npu=SimpleNamespace(get_npu_format=lambda t: 2),
              get_tp_group=lambda: SimpleNamespace(rank_in_group=0),
              _H12_OLD=None, _H12_DENSE=None, _H12_MAP=None,
              _H12_LAST=None, _H12_TRANSITION=0, _H12_ROWS=[])
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    if observer:
        ns["_H12_LOOKUP"] = ns["get_cos_and_sin_mla"]
    return ns


checks = 0
with tempfile.TemporaryDirectory(prefix="glm-H12-observer-") as temp:
    old_environment = os.environ.get("GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT")
    os.environ["GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT"] = temp
    try:
        (Path(temp) / "witnesses").mkdir()
        observed, production = load(SOURCE, True), load(PRODUCTION)
        raw = torch.arange(31 * 64).reshape(31, 64).to(torch.bfloat16)
        observed["_h12_register"](raw)
        production["_record_cos_and_sin_cache_interleaved"](raw)
        assert all(t.stride() == (128, 1) for t in observed["_H12_OLD"])
        assert all(t.stride() == (64, 1) for t in observed["_H12_DENSE"])
        persistent_ptrs = [observed[n].data_ptr() for n in ("_cos_mla", "_sin_mla")]
        for mode in (0, 1, 0, 1):
            observed["_H12_MAP"] = bytes([mode, 1])
            for values in ([0, 30], [-31, -1]):
                for use_cache in (False, True):
                    p = torch.tensor(values, dtype=torch.int64)
                    actual = observed["_h12_lookup"](p, use_cache)
                    expected = production["get_cos_and_sin_mla"](p, use_cache)
                    for a, b in zip(actual, expected):
                        assert torch.equal(a, b) and a.stride() == b.stride()
                    assert persistent_ptrs == [observed[n].data_ptr() for n in ("_cos_mla", "_sin_mla")]
                    checks += 1
            paths = list((Path(temp) / "witnesses").glob("*transition%d_*" % observed["_H12_TRANSITION"]))
            assert len(paths) == 1
            witness = json.loads(paths[0].read_text())
            assert len(witness["rows"]) == 4 and witness["mode"] == mode
            assert witness["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
            before = len(list((Path(temp) / "witnesses").glob("*")))
            observed["_H12_MAP"] = bytes([mode, 0])
            observed["_h12_lookup"](torch.tensor([2, 3]), True)
            assert len(list((Path(temp) / "witnesses").glob("*"))) == before
            checks += 1
        assert torch.equal(raw, torch.arange(31 * 64).reshape(31, 64).to(torch.bfloat16))
    finally:
        if old_environment is None:
            os.environ.pop("GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT", None)
        else:
            os.environ["GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT"] = old_environment

result = dict(passed=True, checks=checks,
              observer_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              production_sha256=hashlib.sha256(PRODUCTION.read_bytes()).hexdigest(),
              NPU_requests=0, NPU_initialized=False, format_getter_mocked=True,
              claims="CPU observer value/layout equivalence, prebuilt tables, output buffer identity, transition witnesses and observation-off behavior; not hardware proof")
destination = ROOT / "observer_CPU_result.json"
assert not destination.exists()
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
