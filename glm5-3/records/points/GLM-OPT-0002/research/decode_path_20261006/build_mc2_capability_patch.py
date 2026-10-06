"""Prepare a source-only MC2 availability candidate; never mutate installed code."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
archive = json.loads((ROOT / "mc2_op_sources.json").read_text())
parts, identities = [], []
for path, source in archive["files"].items():
    operation = "Dispatch" if "Dispatch" in path else "Combine"
    api = "aclnnMoeDistribute" + operation + "V4"
    before = source["text"]
    expected = '    if (check_aclnn_kernel_available("' + api + '")) {'
    replacement = (
        "    // Capability is fixed for this worker's loaded CANN/custom libraries.\n"
        '    static const bool has_v4 = check_aclnn_kernel_available("' + api + '");\n'
        "    if (has_v4) {"
    )
    assert before.count(expected) == 1
    after = before.replace(expected, replacement)
    assert after.replace(replacement, expected) == before
    # All allocation, argument, EXEC_NPU_CMD, fallback and output code is identical.
    parts.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                      fromfile="a/" + path, tofile="b/" + path))
    identities.append(dict(path=path, url=source["url"], api=api,
                           before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                           after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                           only_capability_branch_changed=True))
patch = "".join(parts)
(ROOT / "mc2_capability.patch").write_text(patch)
(ROOT / "mc2_capability_identity.json").write_text(json.dumps(dict(
    source_parent_commit=archive["parent_commit"],
    source_submodule_commit=archive["submodule_commit"], files=identities,
    patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
    installed_or_activated=False,
    contract="Libraries/capabilities remain immutable during worker lifetime; restart after changing them. Cached negative V4 must not be used to claim support for hot installation of a new V4 library.",
    validation="Exact reverse substitution equals original full source. Native wheel compilation/import and model correctness/A-B/E2E are not performed for this candidate."
), indent=2) + "\n")
print(patch)
