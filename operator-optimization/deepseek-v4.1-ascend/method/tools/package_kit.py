"""Validate and zip this instruction package; no device or server operations."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

root = Path(__file__).resolve().parents[1]
required = ["AGENTS.md", "README.md", "START_HERE.txt", "PROTOCOL.md", "OPERATOR_MODEL.md", "REVIEW.md", "STATE.json", "templates/contract.json", "templates/records.json", "templates/bounds-input.json", "templates/REPORT.md", "examples/toy-input.json", "examples/toy-expected.json", "examples/tile-pipeline.md", "tools/calculate_bounds.py"]
required += ["FUSION_SCAN.md", "MODULE_DIAGNOSIS.md", "CHANGELOG.md",
             "PROFILING.md", "templates/profiling-plan.json",
             "templates/profile-index.json", "templates/profile-summary.json",
             "templates/fusion-scan.json", "templates/module-analysis.json",
             "templates/workflow-state.json", "reference/THEORY-v4.md",
             "reference/FRAMEWORK-v4.md", "reference/SOURCE_RESEARCH-v4.md",
             "tools/package_kit.py"]
for name in required:
    assert (root / name).is_file(), name
assert json.loads((root / "STATE.json").read_text())["state"] == "NOT_STARTED"
source_files = sorted(root / name for name in required)
for path in source_files:
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(path))
    elif path.suffix == ".json":
        json.loads(path.read_text())
for path in (p for p in source_files if p.suffix == ".md"):
    prose = "\n".join(line for line in path.read_text().splitlines() if not line.startswith(("    ", "\t")))
    for target in re.findall(r"(?<!\w)\[[^\]\n]+\]\(([^\s)]+)\)", prose):
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)
spec = importlib.util.spec_from_file_location("bounds_calc", root / "tools/calculate_bounds.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
actual = module.calculate(json.loads((root / "examples/toy-input.json").read_text()))
expected = json.loads((root / "examples/toy-expected.json").read_text())
assert [x["time_lower_from_known_terms"]["exact_fraction"] for x in actual["layers"]] == expected["time_lower_fractions"]
assert [x["throughput_upper_from_known_terms"]["exact_fraction"] for x in actual["layers"]] == expected["throughput_upper_fractions"]
assert actual["status"] == "ARITHMETIC_ONLY"
validation = {"status": "PASS", "scope": "File syntax, relative links, fictional-example exact arithmetic and ZIP integrity only", "theoretical_proof_checked": False, "hardware_limit_measured": False, "real_fusion_scan_performed": False, "real_module_diagnosis_performed": False, "server_operations": False}
(root / "VALIDATION.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
# Package only reusable sources; task runs, traces and credentials stay outside.
files = sorted(source_files + [root / "VALIDATION.json"])
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root / "SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
archive = root.with_name(root.name + "-v2.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files + [root / "SHA256.json"]):
        info = zipfile.ZipInfo(str(Path(root.name) / p.relative_to(root)), (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        z.writestr(info, p.read_bytes())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name, digest in manifest.items():
        assert hashlib.sha256(z.read(root.name + "/" + name)).hexdigest() == digest
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files_hashed": len(manifest), "validation": validation}, ensure_ascii=False, indent=2))
