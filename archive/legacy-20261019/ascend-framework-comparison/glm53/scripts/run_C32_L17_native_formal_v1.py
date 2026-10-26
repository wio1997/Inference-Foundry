"""DRAFT_PENDING_PRIMARY_REVIEW: exact-once L17 canonical formal C32 launcher.

The settings file must be frozen with actual readiness, startup, functional
decision/review and formal review SHA values.  This launcher performs one
controller execution only after those files and the controller source SHA match.
It creates no retry path, no profile and no other concurrency case.  The local
draft is intentionally not executable because the actual SHA fields are null.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = "scripts/glm53_C32_L17_formal_resource_controller_v1.py"
LAUNCH_RUN = "20261019T-L17-native-formal-launch-v1"


def load(name, relative, expected):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _actual(path_key, sha_key, settings):
    relative = settings.get(path_key)
    digest = settings.get(sha_key)
    assert isinstance(relative, str) and isinstance(digest, str) and len(digest) == 64, (
        path_key + " / " + sha_key + " must be frozen with actual evidence before execution"
    )
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    return relative, json.loads(path.read_text())


def main():
    settings_path = contract.FORMAL_SETTINGS
    settings = json.loads((ROOT / settings_path).read_text())
    assert settings["status"] == "FROZEN_READY_FOR_SINGLE_EXECUTION", "settings are still a draft"
    assert settings["run_id"] == contract.FORMAL_RUN and settings["namespace"] == contract.FORMAL_NAMESPACE
    assert settings["concurrency"] == 32 and settings["framework"] == "sglang"
    assert settings["profiling"] is False and settings["retries"] == 0
    assert settings["resource_controller_sha256"] == hashlib.sha256((ROOT / CONTROLLER).read_bytes()).hexdigest()
    controller = load("l17_formal_controller", CONTROLLER, settings["resource_controller_sha256"])
    plans = controller.plans(settings, settings_path)
    formal_review_relative, formal_review = _actual("actual_review_path", "actual_review_sha256", settings)
    assert str(formal_review.get("decision", "")).startswith("APPROVED")
    assert formal_review.get("host_codes_sha256") == plans["precheck_hashes"]
    assert formal_review.get("runtime_authorized") is True and formal_review.get("zero_retries") is True
    native_relative, native = _actual("native_qualification_path", "native_qualification_sha256", settings)
    assert native["pass"] is True and native["formal_ready"] is True
    assert native["all32_P_actual_native_packed_MTP_firstuse_verified"] is True
    assert native["native_cache_and_SDMA_verified"] is True
    startup_relative, startup = _actual("startup_qualification_path", "startup_qualification_sha256", settings)
    assert startup["pass"] is True and startup["runtime_CIDs"] == native["runtime_CIDs"]
    decision_relative, decision = _actual("functional_decision_path", "functional_decision_sha256", settings)
    assert decision["decision"] == "ACCEPTED" and decision["functional_ready"] is True
    functional_review_relative, functional_review = _actual(
        "functional_actual_review_path", "functional_actual_review_sha256", settings
    )
    assert str(functional_review.get("decision", "")).startswith("APPROVED")
    assert functional_review.get("formal_ready") is True
    run = ROOT / "runs" / LAUNCH_RUN
    run.mkdir(parents=True, exist_ok=False)
    with (run / "execution-once.lock").open("x") as file:
        file.write(str(os.getpid()))
    receipt = {
        "project_id": "glm53",
        "run_id": LAUNCH_RUN,
        "formal_run_id": contract.FORMAL_RUN,
        "settings": settings_path,
        "settings_sha256": hashlib.sha256((ROOT / settings_path).read_bytes()).hexdigest(),
        "controller_sha256": settings["resource_controller_sha256"],
        "actual_review": formal_review_relative,
        "native_qualification": native_relative,
        "startup_qualification": startup_relative,
        "functional_decision": decision_relative,
        "functional_actual_review": functional_review_relative,
        "host_codes_sha256": plans["precheck_hashes"],
        "formal_logical_requests": 128,
        "warmup_logical_requests": 32,
        "backend_HTTP_attempts_planned": contract.FORMAL_OFFERED_BACKEND,
        "successful_output_tokens_if_all_pass": contract.FORMAL_OUTPUT_IF_ALL,
        "formal_D_output_tokens_if_all_pass": 76800,
        "max_SSH": 2,
        "retries": 0,
        "profiling": False,
        "phase": "STARTED",
    }
    try:
        result = controller.execute(settings, settings_path)
        receipt["controller_result"] = result
        receipt["pass"] = True
        receipt["phase"] = "CONTROLLER_COMPLETE_POSTRUN_AUDIT_REQUIRED"
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"pass": True, "formal_run": contract.FORMAL_RUN, "launch_receipt": str(run / "receipt.json")}))


if __name__ == "__main__":
    main()
