#!/usr/bin/env python3
"""Build one gated worker overlay against the actual-image SHA, on Mac only."""
import argparse
import ast
import base64
import difflib
import hashlib
import json
from pathlib import Path
import zlib


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    base = args.base.read_bytes()
    assert sha(base) == "568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c"
    assert not args.output.exists(), "new output directory required"
    packet = json.loads((repo / "glm5-3-vllm/diagnostics/p-first-state-v1/review-plan.json").read_text())
    pins = {r["path"]: r["sha256"] for r in packet["source_identity"]
            if r["path"] != "vllm-ascend/vllm_ascend/worker/model_runner_v1.py"}
    prompts = {str(c["input_tokens"]): c["prompt_sha256"] for c in packet["cases"]}
    probe = (root / "probe.py").read_text()
    ast.parse(probe)
    # Appending leaves every native class/function body unchanged. Disabled
    # means only an environment lookup; no module/connector/device setup.
    bootstrap = ("\n\nif __import__('os').environ.get('GLM53_STATE_TAIL_PROBE') == '1':\n"
        "    import types as _glm53_types\n"
        "    _glm53_probe = _glm53_types.ModuleType('_glm53_state_tail_probe')\n"
        "    _glm53_probe.__file__ = __file__\n"
        "    exec(compile(" + repr(probe) + ", '<glm53-state-tail-probe-v1>', 'exec'), _glm53_probe.__dict__)\n"
        "    _glm53_probe.install(globals(), " + repr(pins) + ", " + repr(prompts) + ")\n")
    combined = base.decode() + bootstrap
    tree = ast.parse(combined)
    original_tree = ast.parse(base)
    native_equal = all(ast.dump(a) == ast.dump(b) for a, b in zip(original_tree.body, tree.body))
    assert native_equal, "native definitions changed"
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                   n.func.attr in ("synchronize", "barrier", "wait_event", "wait_stream")
                   for n in ast.walk(ast.parse(probe))), "new device synchronization is forbidden"
    args.output.mkdir(parents=True)
    (args.output / "model_runner_v1.py").write_text(combined)
    (args.output / "worker.patch").write_text("".join(difflib.unified_diff(
        base.decode().splitlines(True), combined.splitlines(True),
        fromfile="model_runner_v1.py", tofile="model_runner_v1.py")))
    remote = ("/data/wio/glm5-3-vllm-execution/20261017-official-pd/execution-patches/"
              "state-tail-probe-v1/" + sha(combined.encode())[:12] + "/model_runner_v1.py")
    stage = '''import base64,hashlib,json,socket,zlib
from pathlib import Path
assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
data=zlib.decompress(base64.b64decode(PAYLOAD))
assert len(data)<1048576 and hashlib.sha256(data).hexdigest()==EXPECTED
dest=Path(DESTINATION)
dest.parent.mkdir(parents=True,exist_ok=True)
if dest.exists():
 assert hashlib.sha256(dest.read_bytes()).hexdigest()==EXPECTED
else:
 with dest.open('xb') as f:f.write(data)
 dest.chmod(0o444)
print('GLM53_RESULT_JSON_BEGIN\\n'+json.dumps({'staged':True,'path':str(dest),'sha256':EXPECTED,'device_run':False,'framework_imported':False})+'\\nGLM53_RESULT_JSON_END')
'''.replace('PAYLOAD', repr(base64.b64encode(zlib.compress(combined.encode())).decode())).replace(
        'EXPECTED', repr(sha(combined.encode()))).replace('DESTINATION', repr(remote))
    ast.parse(stage)
    (args.output / "stage_overlay.py").write_text(stage)
    manifest = {"base_sha256": sha(base), "probe_sha256": sha(probe.encode()),
        "overlay_sha256": sha(combined.encode()), "bytes": len(combined.encode()),
        "gate": "GLM53_STATE_TAIL_PROBE=1", "default_enabled": False,
        "output_env": "GLM53_STATE_PROBE_OUTPUT=/out/state-tail-probe",
        "stage_script_sha256": sha(stage.encode()), "remote_overlay": remote,
        "target": "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
        "scope": "Mac syntax/build only. No image imports, device execution or performance acceptance.",
        "native_top_level_definitions_preserved": native_equal and len(tree.body) - len(ast.parse(bootstrap).body) == len(original_tree.body),
        "device_execution_authorized_by_build": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
