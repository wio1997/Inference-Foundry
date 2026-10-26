#!/usr/bin/env python3
"""CPU check of the actual generated R247 launcher paths.

It builds the exact docker argv for nodes 46/47, renders the host probe, and
asserts the frozen R201 flags, 47-file overlay, SDMA environment and B-only
scope.  Nothing is executed on a node.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "controller"))
sys.path.insert(0, str(R))

import controller  # noqa: E402
import dispatch  # noqa: E402
import launcher  # noqa: E402
import launch_command  # noqa: E402

RUN_ROOT = "/data/wio/inference-foundry/glm53-flash/runs/20261019-r247-b-indexer-optimized-profile/B"
IMAGE = "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263"


def env_map(args):
    values = {}
    for index, value in enumerate(args):
        if value == "-e":
            key, _, val = args[index + 1].partition("=")
            values[key] = val
    return values


def label_map(args):
    values = {}
    for index, value in enumerate(args):
        if value == "--label":
            key, _, val = args[index + 1].partition("=")
            values[key] = val
    return values


def flag(argv, name):
    assert name in argv, name
    return argv[argv.index(name) + 1]


def test_command():
    rows = json.loads((R / "stage/B/combined-manifest.json").read_text())["files"]
    argv46 = json.loads((R / "stage/B/argv46.json").read_text())
    argv47 = json.loads((R / "stage/B/argv47.json").read_text())
    args46 = launch_command.build(46, RUN_ROOT, rows, argv46, verify_host_paths=False)
    args47 = launch_command.build(47, RUN_ROOT, rows, argv47, verify_host_paths=False)
    assert args46[:3] == ["docker", "run", "-d"]
    assert IMAGE in args46 and IMAGE in args47
    assert "if-glm53-flash-r247-b46" in args46
    assert "if-glm53-flash-r247-b47" in args47
    for args, role, node in ((args46, "Prefill", 46), (args47, "Decode", 47)):
        labels = label_map(args)
        assert labels == {"project_id": "glm53-flash", "run_id": "r247",
                          "namespace": "glm53-flash-r247", "owner": "zcode"}
        env = env_map(args)
        assert env["ASCEND_MF_TRANSFER_PROTOCOL"] == "sdma"
        assert env["ASCEND_MF_STORE_URL"].startswith("tcp://172.16.12.46:")
        assert env["FLASH_SDMA_VERIFY_BYTES"] == "0"
        assert env["FLASH_PD_ROLE"] == role
        assert env["FLASH_PD_PERF_MODE"] == "1"
        assert env["FLASH_PD_COMPRESSION_ABI_AUDIT"] == "0"
        assert env["FLASH_PD_SCORE_ABI_AUDIT"] == "0"
        assert env["FLASH_MHC_FIXED_ROWS"] == "0"
        assert env["FLASH_PD_STATE_AUDIT"] == "0"
        assert env["FLASH_KDA_PREFIX_ONCE_AUDIT"] == "0"
        assert env["FLASH_PD_LAYOUT_ONCE"] == "0"
        assert env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] == "10800"
        assert env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] == "10800"
        assert "SGLANG_OPT_USE_TILELANG_MHC_PRE" in env
        if node == 46:
            assert env["FLASH_B103_PERSISTENT_OWNED_STORAGE"] == "0"
            assert env["R234_PROFILE_OBSERVER"] == "1"
            assert env["R234_PROFILE_ROOT"] == "/run-evidence/profile-B"
            assert env["R234_PROFILE_WINDOWS"] == "early:8192:16384,mid:32768:40960,late:61440:73728"
            assert env["R234_PROFILE_MAX_FORWARDS"] == "6"
            assert env["R234_PROFILE_EXPECT_ACTIVE"] == "1"
        else:
            assert env["FLASH_B103_PERSISTENT_OWNED_STORAGE"] == "1"
            assert "R234_PROFILE_OBSERVER" not in env
        joined = " ".join(args)
        assert "172.16.12.14" not in joined and "172.16.12.15" not in joined
        assert "sgl-workspace/sglang/python/" in joined
        assert joined.count(":/sgl-workspace/sglang/python/") == 47
        assert args.count("--device") == 16
        assert "--entrypoint" in args and "/bin/bash" in args
        assert "start_after_preflight.py" in joined
        assert "source /usr/local/Ascend/ascend-toolkit/set_env.sh" in joined
    assert "PYTHONPATH=/run-evidence/observer" in " ".join(args46)
    assert "PYTHONPATH=/run-evidence/observer" not in " ".join(args47)
    for argv in (argv46, argv47):
        assert flag(argv, "--tp-size") == "16"
        assert flag(argv, "--dp-size") == "1"
        assert flag(argv, "--ep-size") == "1"
        assert flag(argv, "--dtype") == "bfloat16"
        assert flag(argv, "--kv-cache-dtype") == "bfloat16"
        assert flag(argv, "--mamba-ssm-dtype") == "float32"
        assert flag(argv, "--quantization") == "modelslim"
        assert flag(argv, "--chunked-prefill-size") == "4096"
        assert flag(argv, "--max-prefill-tokens") == "4096"
        assert flag(argv, "--mamba-radix-cache-strategy") == "extra_buffer"
        assert flag(argv, "--max-mamba-cache-size") == "256"
        assert flag(argv, "--mamba-max-states-per-path") == "1"
        assert flag(argv, "--disaggregation-transfer-backend") == "ascend"
        assert "--disable-cuda-graph" in argv
        assert "--disable-overlap-schedule" in argv
        assert "--skip-server-warmup" in argv
        assert not any("mtp" in value.lower() or "speculative" in value.lower()
                       for value in argv)
    assert flag(argv46, "--disaggregation-mode") == "prefill"
    assert flag(argv47, "--disaggregation-mode") == "decode"
    assert flag(argv46, "--served-model-name") == "glm53-flash-r201"
    assert flag(argv47, "--served-model-name") == "glm53-flash-r201"


def test_rendered_probe():
    code = controller.render_host_probe(46, {})
    assert "__LAUNCH_COMMAND__" not in code and "__PROFILE_EXPORT__" not in code
    assert "def build(node, evidence_root, overlay_rows, argv" in code
    assert "def evaluate(files, windows=WINDOWS" in code
    assert "container_name(node)" in code
    compile(code, "rendered-host-probe", "exec")
    code47 = controller.render_host_probe(47, {})
    compile(code47, "rendered-host-probe-47", "exec")


def test_scope_constants():
    assert controller.RUN_ROOT == RUN_ROOT
    assert set(controller.PODS) == {46, 47}
    assert set(controller.HOSTS) == {46, 47}
    assert controller.SGLANG_IMAGE == IMAGE
    assert not hasattr(controller, "VLLM_IMAGE")
    plan = json.loads((R / "controller/profile_plan.json").read_text())
    assert plan["unit"] == "r247"
    assert plan["scope"]["formal_requests"] == 0
    assert plan["scope"]["cold_pairs"] == 2
    assert plan["scope"]["generation_POST"] == 4
    assert plan["scope"]["warm32"] is False
    assert plan["scope"]["dummy_requests"] == 0
    assert plan["scope"]["retry"] == 0
    assert plan["scope"]["flush"] is False
    assert plan["resources"]["nodes"] == [46, 47]
    assert 14 not in plan["resources"]["nodes"] and 15 not in plan["resources"]["nodes"]
    assert plan["budgets"]["single_controller_deadline_seconds"] == 1200
    assert plan["budgets"]["b_cleanup_bound_seconds"] == 180
    assert len(plan["windows"]) == 3
    assert all(window["forwards"] == 2 for window in plan["windows"])
    assert plan["requests"]["generation_POST_max"] == 4
    assert plan["requests"]["cold_pairs"] == 2
    assert plan["requests"]["formal_CLI"] is False
    assert plan["framework"]["P_nodes"] == [46] and plan["framework"]["D_nodes"] == [47]
    assert plan["r246_gate"]["required_before_dispatch"] is True
    assert plan["scope"]["candidate_delta"]["overlay_file_count"] == 47
    assert plan["scope"]["candidate_delta"]["byte_identical_to_original_46"] is False


def test_no_a_paths():
    for name in ("controller/controller.py", "controller/host_probe_sglang.py",
                 "controller/launch_command.py", "launcher.py", "dispatch.py",
                 "client/r234_client.py", "controller/profile_plan.json"):
        text = (R / name).read_text()
        assert "172.16.12.14" not in text and "172.16.12.15" not in text, name
        assert "flash_vllm_native_pd_client_v2" not in text, name
    client = (R / "client/r234_client.py").read_text()
    assert "choices=[\"sglang\"]" in client
    assert "generation_POST_expected\": 4" in client
    plan = (R / "controller/profile_plan.json").read_text()
    assert "\"vllm\"" not in plan
    assert dispatch.EXACT_COMMAND == "python3 dispatch.py"
    assert launcher.UNIT == "r247"
    assert launcher.GATEWAY_ROOT.endswith("20261019-r247-b-indexer-optimized-profile")
    assert launcher.SSH[2] == "/tmp/glm53-flash-r236-control"
    assert launcher.CONTROLLER_COMMAND == "python3 controller/controller.py"
    assert "--manifest" in (R / "launcher.py").read_text()
    gateway = launcher.gateway_code({"files": {}})
    compile(gateway, "rendered-gateway", "exec")
    assert "r246_host_receipt_sha256" in gateway
    assert "host-receipt.json" in gateway
    assert "No running processes found" in gateway
    assert "172.16.12.14" not in gateway and "172.16.12.15" not in gateway


def main():
    test_command()
    test_rendered_probe()
    test_scope_constants()
    test_no_a_paths()
    print(json.dumps({"passed": True, "unit": "r247",
                      "overlay_mounts": 47, "devices": 16,
                      "nodes": [46, 47], "generation_POST": 4,
                      "cold_pairs": 2, "budget_seconds": 1200}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
