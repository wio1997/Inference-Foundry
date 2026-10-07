#!/usr/bin/env python3
"""Check archived patch bytes locally; never import or run model code."""

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe_path(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe path: {name}")
    require(path.parts and ".git" not in path.parts, f"Unsafe path: {name}")
    return Path(*path.parts)


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )
    require(result.returncode == 0, f"git {' '.join(args)}: {result.stderr}")


def syntax(data, path):
    if path.endswith(".py"):
        ast.parse(data, filename=path)


def function_ast(data, name):
    nodes = [
        node for node in ast.walk(ast.parse(data))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    require(len(nodes) == 1, f"Expected exactly one {name}")
    return ast.dump(nodes[0], include_attributes=False)


def verify(package):
    manifest = json.loads((package / "manifest.json").read_text())
    fixture = package / "fixtures.zip"
    require(sha256(fixture.read_bytes()) == manifest["fixtures_sha256"], "Fixture SHA")
    rows = []
    with zipfile.ZipFile(fixture) as archive:
        names = archive.namelist()
        require(len(set(names)) == len(names), "Duplicate fixture entries")
        for name in names:
            safe_path(name)
        for patch in manifest["patches"]:
            patch_path = package / safe_path(patch["patch"])
            require(sha256(patch_path.read_bytes()) == patch["patch_sha256"], patch["id"] + " patch SHA")
            with tempfile.TemporaryDirectory(prefix="glm53-patch-check-") as tmp:
                root = Path(tmp)
                git(root, "init", "-q")
                for file in patch["files"]:
                    target = root / safe_path(file["path"])
                    key = f"{patch['id']}/before/{file['path']}"
                    if file["before_sha256"] is not None:
                        before = archive.read(key)
                        require(sha256(before) == file["before_sha256"], key + " SHA")
                        require(len(before) == file["before_bytes"], key + " size")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(before)
                        syntax(before, file["path"])
                    else:
                        require(key not in names and file["before_bytes"] == 0, key + " should be absent")
                git(root, "apply", "--check", "--whitespace=nowarn", str(patch_path))
                git(root, "apply", "--whitespace=nowarn", str(patch_path))
                for file in patch["files"]:
                    after = (root / safe_path(file["path"])).read_bytes()
                    key = f"{patch['id']}/after/{file['path']}"
                    require(after == archive.read(key), key + " exact bytes")
                    require(sha256(after) == file["after_sha256"], key + " SHA")
                    require(len(after) == file["after_bytes"], key + " size")
                    syntax(after, file["path"])
                git(root, "apply", "--reverse", "--check", "--whitespace=nowarn", str(patch_path))
                git(root, "apply", "--reverse", "--whitespace=nowarn", str(patch_path))
                for file in patch["files"]:
                    target = root / safe_path(file["path"])
                    if file["before_sha256"] is None:
                        require(not target.exists(), file["path"] + " added file not removed")
                    else:
                        require(sha256(target.read_bytes()) == file["before_sha256"], file["path"] + " reverse SHA")
                rows.append({
                    "id": patch["id"], "files": len(patch["files"]),
                    "apply": "PASS", "reverse": "PASS", "exact_bytes": "PASS",
                    "python_AST": "PASS" if any(file["path"].endswith(".py") for file in patch["files"]) else "NOT_APPLICABLE",
                })

        runner = "vllm_ascend/worker/model_runner_v1.py"
        proposer = "vllm_ascend/spec_decode/llm_base_proposer.py"
        require(
            function_ast(archive.read("H11/after/" + runner), "_dummy_run")
            == function_ast(archive.read("H11-EXACT/after/" + runner), "_dummy_run"),
            "Clean H11 _dummy_run differs from device-tested source",
        )
        require(archive.read("H11/after/" + proposer) == archive.read("H11-EXACT/after/" + proposer), "H11 proposer exact bytes")
        prepare = "vllm_ascend/ops/fused_moe/prepare_finalize.py"
        require(
            ast.dump(ast.parse(archive.read("H13/after/" + prepare)), include_attributes=False)
            == ast.dump(ast.parse(archive.read("H13-EXACT/after/" + prepare)), include_attributes=False),
            "H13 line-ending packaging changes Python AST",
        )
    return {
        "scope": "LOCAL_CPU_PACKAGING_ONLY",
        "passed": True,
        "patch_count": len(rows),
        "file_instances": sum(row["files"] for row in rows),
        "patches": rows,
        "H11_changed_function_AST_matches_tested": True,
        "H11_proposer_matches_tested_bytes": True,
        "H13_module_AST_matches_exact_candidate": True,
        "native_compile": "NOT_RUN",
        "model_code_execution": "NOT_RUN",
        "new_NPU_requests": 0,
        "server_actions": 0,
        "new_device_correctness_or_performance_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional local JSON result path")
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parent)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
