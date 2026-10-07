"""CPU only: installed-source AST A/B selector wiring; no NPU calls."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import hashlib

ROOT = Path(__file__).resolve().parent
research = ROOT.parent.parent / "research/decode_path_20261006"
spec = importlib.util.spec_from_file_location("projection_CPU", research / "check_indexer_projection_CPU.py")
checker = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(checker)
torch = checker.torch
assert not torch.npu.is_initialized() if hasattr(torch, "npu") else True
cases = []
for mode in (0, 1):
    ns = checker.methods(ROOT / "shim/sfa_v1.py")
    ns.update(_glm_projection_enabled=lambda: bool(mode), _glm_projection_gate=[0], _glm_projection_mode=[mode])
    for dtype in (torch.bfloat16, torch.float16):
        wk = torch.randn(32, 192).to(dtype)
        qb = torch.randn(8, 8192).to(dtype)
        for tokens in (1, 2, 8):
            for preprocess, has, skip, state in [
                (checker.Preprocess.NATIVE, True, False, checker.Attention.DecodeOnly),
                (checker.Preprocess.NATIVE, True, False, checker.Attention.SpecDecoding),
                (checker.Preprocess.NATIVE, True, False, checker.Attention.Prefill),
                (checker.Preprocess.NATIVE, True, True, checker.Attention.DecodeOnly),
                (checker.Preprocess.NATIVE, False, True, checker.Attention.DecodeOnly),
                (checker.Preprocess.PROLOG_V3, True, False, checker.Attention.DecodeOnly),
                (checker.Preprocess.MLAPO, True, False, checker.Attention.SpecDecoding),
            ]:
                x = torch.randn(tokens, 32).to(dtype)
                a = checker.run(checker.namespaces[0], x, wk, qb, preprocess, has, skip, state)
                b = checker.run(ns, x, wk, qb, preprocess, has, skip, state)
                assert a[0].keys() == b[0].keys()
                assert all(torch.equal(a[0][name], b[0][name]) for name in a[0])
                assert torch.equal(a[2], b[2])
                removed = int(mode == 1 and preprocess == checker.Preprocess.NATIVE and has and not skip)
                assert len(a[1]) - len(b[1]) == removed
                cases.append(dict(mode=mode, dtype=str(dtype), tokens=tokens, preprocess=preprocess,
                                  has_indexer=has, skip_topk=skip, state=state,
                                  original_calls=len(a[1]), shim_calls=len(b[1]), bitwise=True))
print(json.dumps(dict(CPU_only=True, NPU_initialized=False, production_cases_reused=42,
                      selector_cases_passed=len(cases), cases=cases,
                      shim_sha256=hashlib.sha256((ROOT / "shim/sfa_v1.py").read_bytes()).hexdigest(),
                      limitations=["CPU doubles; native reference gates and model correctness still mandatory."])))
