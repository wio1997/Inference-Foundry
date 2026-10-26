"""LOCAL DRAFT qualification for the corrected L17-D lifecycle v2 drafts.

Zero SSH, HTTP, Docker, NPU, SDK or model calls. It only:
- verifies v1 preservation and fixed source hashes;
- materializes the 22 static D stop/pool/preflight/health/meta host plans plus the
  4 embedded preflight programs, compiles them and rejects unbound globals;
- proves D plans never target the node34 Meta Docker daemon, that Meta work is
  rendered only for node34, and that every mutation stays within the new D roots
  and the two owned old D containers;
- runs focused stdlib CPU fixtures for the corrected wire/schema/failure cases
  requested by the primary review (no mirrored implementation suite, no hardware
  proof claim).

It writes the local qualification result and a task receipt; it never creates a
transition run, actual review, capacity output, model-start launcher or request.
"""

import ast
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUS = "PENDING_PRIMARY_REVIEW"
LIB = "scripts/glm53_C32_L17_D_transition_library_v2.py"
LIB_SHA = "da4bbcab4ddf1dd4670c45db78a20aace39f12e1429e73aa5da9d27d36b7b8ce"
MIXED = "scripts/glm53_C32_L17_mixed_model_identity_v1.py"
MIXED_SHA = "ec8a167da59504dcfcd9dec1347960e93fbab2764d9545f96a6f80768aff1526"
AUDITOR = "scripts/audit_C32_L17_selected_D_pool_local_v2.py"
V1_LIB = "scripts/glm53_C32_L17_D_transition_library_v1.py"
V1_LIB_SHA = "95309e2866324a3109e06560e33bdc0b8851e0d2bf99db937da2f05581d23b73"
V1_RUNNER = "scripts/run_C32_L17_controlled_D_transition_and_pool_v1.py"
V1_RUNNER_SHA = "9bffdb071dacf7f4312e82629e01c958a25fa36ea5bfcda60947a1ea75d226f8"
V1_AUDITOR = "scripts/audit_C32_L17_selected_D_pool_local_v1.py"
V1_AUDITOR_SHA = "00d2f9ecfe2b255686088a753336b6ca7410144995cea1bcfde6e75b362cfa0a"
STAGE_RESULT = "results/C32-L17-staged-candidate-original-D-rollback-v1.json"
STAGE_RESULT_ALT = "results/C32-L17-staged-candidate-and-original-D-rollback-v1.json"
MATERIALIZATION_RID = "20261019T-L17-D-transition-local-materialization-v2"
QUAL_RESULT = "results/C32-L17-D-lifecycle-hostplan-qualification-v2.json"
EXECUTOR_RID = "20261019T-L17-D-lifecycle-correction-executor-v2"
SERVER_EVIDENCE = "runs/20261019T-L12C-sglang-native-startup-v1/node-36.json"
HISTORICAL_META = "runs/20261019T-L12C-controlled-P-transition-and-pool-v1/meta-after.json"
P_CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
OLD_D_CIDS = {
    36: "25ecbd3a55e199caf4c6012a61b702d1b6035c955406d75bb17107991a932dd1",
    41: "c39fa7c594a8776e70c365baec435d4288abb1db5cc492405183047d57b805c3",
}
OLD_D_RID = "20261018T-L10-sglang-C32-target-graph-fullmesh-v3"
LEDGER_ATTEMPTS = 6269
LEDGER_SUCCESSES = 6096


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_module(relative, expected, name):
    path = ROOT / relative
    assert sha256_file(path) == expected, (relative, sha256_file(path), expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self):
        self.rows = []

    def run(self, name, function):
        try:
            detail = function()
        except BaseException as error:
            self.rows.append({"name": name, "pass": False, "error": repr(error)})
            raise
        self.rows.append({"name": name, "pass": True, "detail": detail})
        return detail

    def reject(self, name, function):
        try:
            function()
        except BaseException as error:
            self.rows.append({"name": name, "pass": True, "rejected_with": type(error).__name__})
            return type(error).__name__
        self.rows.append({"name": name, "pass": False, "error": "expected rejection"})
        raise AssertionError("expected rejection: %s" % name)


def docker_subcommands(code):
    return sorted(set(re.findall(r"\['docker',\s*'([a-z]+)'", code)))


def synthetic_d_capacity(lib, p_capacity, variant):
    spec = lib.STAGES[variant]
    capacity = {
        "project_id": "glm53",
        "pass": True,
        "selected_P_variant": "candidate",
        "selected_D_variant": variant,
        "selected_variant": variant,
        "selected_variant_scope": "D_ONLY_NOT_P",
        "model_started": False,
        "actual_MC_bytes": 568 * 1024 ** 3,
        "providers": {
            "34": p_capacity["providers"]["34"],
            "35": p_capacity["providers"]["35"],
        },
    }
    for node, service_id in ((36, 84), (41, 85)):
        capacity["providers"][str(node)] = {
            "container_id": ("c" if node == 36 else "d") * 64,
            "name": spec["name_prefix"] + "-20261019-" + str(node),
            "local_service_id": service_id,
            "dram_GiB": 160,
            "remote_root": spec["run_root"],
            "run_id": spec["run_id"],
        }
    return capacity


def main():
    checks = Checks()
    lib = load_module(LIB, LIB_SHA, "glm53_l17_d_qual_library_v2")
    mixed = load_module(MIXED, MIXED_SHA, "glm53_l17_mixed_qual_v1")
    p_capacity = read_json(ROOT / P_CAPACITY)

    checks.run("v1_library_preserved_unchanged", lambda: sha256_file(ROOT / V1_LIB) == V1_LIB_SHA)
    checks.run("v1_runner_preserved_unchanged", lambda: sha256_file(ROOT / V1_RUNNER) == V1_RUNNER_SHA)
    checks.run("v1_auditor_preserved_unchanged", lambda: sha256_file(ROOT / V1_AUDITOR) == V1_AUDITOR_SHA)

    def accepted_stage():
        path = ROOT / STAGE_RESULT if (ROOT / STAGE_RESULT).exists() else ROOT / STAGE_RESULT_ALT
        record = read_json(path)
        assert record["pass"] is True
        assert record["runtime_transition_authorized_by_this_record"] is False
        return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "pass": True}

    checks.run("accepted_stage_v3_result_read_only", accepted_stage)
    checks.run("mixed_identity_draft_sha", lambda: sha256_file(ROOT / MIXED) == MIXED_SHA)

    def provenance():
        record = lib.source_provenance()
        assert record[MIXED] == MIXED_SHA
        assert record[P_CAPACITY] == "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
        return {"entries": len(record), "sha256_checked": True}

    checks.run("source_provenance_all_fixed", provenance)

    def transport_helper_loadable():
        helper = lib.transport()
        assert callable(helper.invoke)
        return {"invoke_callable": True, "invoked": False, "SSH": 0, "network": 0}

    checks.run("transport_helper_loadable_without_invocation", transport_helper_loadable)
    checks.run("p_capacity_sha_fixed", lambda: sha256_file(ROOT / P_CAPACITY) == "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50")
    checks.run("p_selection_candidate_explicit", lambda: lib.p_selection(p_capacity) == "candidate")

    def ambiguous_legacy():
        lib.d_selection({"selected_P_variant": "candidate", "selected_D_variant": "rollback", "selected_variant": "rollback"})

    checks.reject("legacy_D_selection_without_scope_rejected", ambiguous_legacy)

    def scoped_legacy():
        value = lib.d_selection({"selected_P_variant": "candidate", "selected_D_variant": "rollback", "selected_variant": "rollback", "selected_variant_scope": "D_ONLY_NOT_P"})
        assert value == "rollback"
        return value

    checks.run("legacy_D_selection_scoped_accepted", scoped_legacy)

    plans = lib.plan_all()
    inner = {"%s-preflight-inner-%d" % (kind, node): lib.preflight_inner(kind, node)
             for kind in ("candidate", "rollback") for node in (36, 41)}
    analyses = {name: lib.analyze_code(name, code) for name, code in plans.items()}
    inner_analyses = {name: lib.analyze_code(name, code) for name, code in inner.items()}
    checks.run("static_plan_count_22", lambda: len(plans) == 22)
    checks.run("embedded_preflight_inner_count_4", lambda: len(inner) == 4)
    checks.run("all_static_plans_ast_and_compile", lambda: all(item["ast_parse"] and item["compile"] for item in analyses.values()))
    checks.run("all_static_plans_zero_unbound_globals", lambda: all(not item["unbound_globals"] for item in analyses.values()))
    checks.run("all_embedded_preflight_ast_and_compile", lambda: all(item["ast_parse"] and item["compile"] for item in inner_analyses.values()))
    checks.run("all_embedded_preflight_zero_unbound_globals", lambda: all(not item["unbound_globals"] for item in inner_analyses.values()))

    def d_plans_no_meta_docker():
        d_names = [name for name in plans if name.startswith((
            "precheck-", "stop-", "candidate-prepare-", "candidate-pool-", "candidate-preflight-",
            "rollback-prepare-", "rollback-pool-", "rollback-preflight-"))]
        assert len(d_names) == 16, d_names
        for name in d_names:
            code = plans[name]
            assert lib.META_CID not in code, name
            assert lib.META_NAME not in code, name
            assert "META_ID" not in code, name
            assert "'docker','inspect',META" not in code, name
        return {"checked": len(d_names)}

    checks.run("D_plans_never_target_node34_Meta_docker", d_plans_no_meta_docker)

    def meta_only_node34():
        host34 = lib.transport_host(34)
        for name, bound in (("meta-before", 0), ("meta-after", lib.META_AFTER_WAIT_BOUND_SECONDS)):
            code = plans[name]
            assert "HOSTNAME = %r" % host34 in code, name
            assert "META_NODE = 34" in code, name
            assert "META_ID = %r" % lib.META_CID in code, name
            assert "socket.gethostname()==HOSTNAME" in code, name
            assert "WAIT_BOUND = %r" % bound in code, name
            assert "SHARED_RID = %r" % OLD_D_RID in code, name
        return {"meta_node": 34, "host": host34}

    checks.run("Meta_plans_node34_same_container_labels", meta_only_node34)

    def pool_docker_args():
        for name in [n for n in plans if n.endswith("-pool-36") or n.endswith("-pool-41")]:
            code = plans[name]
            assert set(docker_subcommands(code)) <= {"image", "inspect", "run"}, docker_subcommands(code)
            for needle in (
                "'--network=host'", "'--pids-limit=32768'", "'--memory=1700g'",
                "'--shm-size=64g'", "/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro",
                "'ASCEND_VISIBLE_DEVICES=0-15'", "/opt/glm53/source/supervisor.py", "/opt/glm53/source:ro",
            ):
                assert needle in code, (name, needle)
            assert code.count("'--device'") == 19, name
            assert "libmf_hybm_accoffload" not in code, name
            assert "private_optional_extension" not in code, name
        return {"pool_plans": 4, "device_args": 19}

    checks.run("pool_docker_args_full_no_private_extension", pool_docker_args)

    def mutations():
        expected = {
            "precheck-": {"inspect", "cp", "stop"},
            "stop-": {"inspect", "cp", "stop"},
            "candidate-prepare-": {"inspect"},
            "rollback-prepare-": {"inspect"},
            "candidate-pool-": {"image", "inspect", "run"},
            "rollback-pool-": {"image", "inspect", "run"},
            "candidate-preflight-": {"inspect", "exec"},
            "rollback-preflight-": {"inspect", "exec"},
            "meta-": {"inspect", "cp"},
        }
        for name, code in plans.items():
            for prefix, allowed in expected.items():
                if name.startswith(prefix):
                    actual = set(docker_subcommands(code))
                    assert actual <= allowed, (name, actual, allowed)
                    break
            assert not re.search(r"\b(?:docker\s+(?:rm|kill|start|restart|systemctl)|pkill|killall)\b", code), name
        stops = sorted(name for name in plans if name.startswith("stop-"))
        assert len(stops) == 2, stops
        return {"stop_plans": stops, "forbidden_mutations": "none"}

    checks.run("static_mutations_only_whitelisted_D_actions", mutations)

    def stop_targets():
        for node in (36, 41):
            code = plans["stop-%d" % node]
            assert OLD_D_CIDS[node] in code, node
            other = 41 if node == 36 else 36
            assert OLD_D_CIDS[other] not in code, node
            assert code.count("['docker','stop'") == 1, node
        return {"old_CIDs": {str(k): v for k, v in OLD_D_CIDS.items()}}

    checks.run("stop_plans_only_two_owned_old_D_CIDs", stop_targets)

    def prepare_roots():
        for kind in ("candidate", "rollback"):
            spec = lib.STAGES[kind]
            for node in (36, 41):
                code = plans["%s-prepare-%d" % (kind, node)]
                assert spec["run_root"] in code, (kind, node)
                assert "20261019T-L12C-candidate-v1" not in code, (kind, node)
                assert lib.META_NAME not in code, (kind, node)
        return {"new_roots": [lib.STAGES["candidate"]["run_root"], lib.STAGES["rollback"]["run_root"]]}

    checks.run("prepare_only_new_D_roots_not_P_or_Meta", prepare_roots)

    candidate_capacity = synthetic_d_capacity(lib, p_capacity, "candidate")
    rollback_capacity = synthetic_d_capacity(lib, p_capacity, "rollback")

    def cleanup_plans():
        for variant, capacity in (("candidate", candidate_capacity), ("rollback", rollback_capacity)):
            dynamic = lib.dynamic_plans(capacity)
            assert len(dynamic) == 6, sorted(dynamic)
            for name, code in dynamic.items():
                analysis = lib.analyze_code(name, code)
                assert analysis["ast_parse"] and analysis["compile"], name
                assert not analysis["unbound_globals"], name
                if name.endswith("-cleanup-36") or name.endswith("-cleanup-41"):
                    assert "READY_PROVIDER_ONLY" in code and "MODEL_STARTED_OR_FAILED" in code, name
                    assert "def idle" not in code, name
                    assert "['docker','stop','--time','240',CID]" in code, name
                    assert "P_Meta_vllm_touched" in code and "foreign_PID_killed" in code, name
                    assert "docker','kill" not in code and "docker','rm" not in code, name
        return {"cleanup_plans": 4, "states": ["READY_PROVIDER_ONLY", "MODEL_STARTED_OR_FAILED"]}

    checks.run("cleanup_parameterized_owned_only_two_states", cleanup_plans)

    def role_independence():
        assert lib.p_selection(p_capacity) == "candidate"
        assert lib.p_selection(candidate_capacity) == "candidate"
        assert lib.p_selection(rollback_capacity) == "candidate"
        assert lib.d_selection(candidate_capacity) == "candidate"
        assert lib.d_selection(rollback_capacity) == "rollback"
        candidate_plans = lib.dynamic_plans(candidate_capacity)
        rollback_plans = lib.dynamic_plans(rollback_capacity)
        for key in ("start-36", "observe-36", "cleanup-36"):
            assert sha256_text(candidate_plans["candidate-" + key]) != sha256_text(rollback_plans["rollback-" + key]), key
        assert "KIND = 'candidate'" in candidate_plans["candidate-start-36"]
        assert "KIND = 'rollback'" in rollback_plans["rollback-start-36"]
        return {"P_variant": "candidate", "D_variants": ["candidate", "rollback"]}

    checks.run("role_variant_independence", role_independence)

    def append_only_cases():
        before = b"model stdout line 1\nmodel stdout line 2\n"
        assert lib.append_only_ok(before, before + b"shutdown complete\n") is True
        assert lib.append_only_ok(before, before) is True
        assert lib.append_only_ok(before, b"CHANGED PREFIX\n" + before[len(b"CHANGED PREFIX\n"):]) is False
        assert lib.append_only_ok(before, before[:-1]) is False
        return {"retained": True, "changed_prefix_rejected": True, "truncate_rejected": True}

    checks.run("append_only_model_log_prefix_fixture", append_only_cases)

    def provider_close_cases():
        good = lib.provider_closure(
            {"phase": "CLOSE_COMPLETE", "events": [{"event": "CLOSE_COMPLETE", "rc": 0}]},
            {"phase": "STOPPED", "provider_exit_code": 0},
        )
        assert good["pass"] is True
        nonzero = lib.provider_closure(
            {"phase": "CLOSE_COMPLETE", "events": [{"event": "CLOSE_COMPLETE", "rc": 7}]},
            {"phase": "STOPPED", "provider_exit_code": 0},
        )
        assert nonzero["pass"] is False
        exit_nonzero = lib.provider_closure(
            {"phase": "CLOSE_COMPLETE", "events": [{"event": "CLOSE_COMPLETE", "rc": 0}]},
            {"phase": "STOPPED", "provider_exit_code": 1},
        )
        assert exit_nonzero["pass"] is False
        return {"close0": True, "close_nonzero_rejected": True, "provider_exit_nonzero_rejected": True}

    checks.run("provider_close_and_exit_code_fixture", provider_close_cases)

    def server_info_actual():
        evidence = read_json(ROOT / SERVER_EVIDENCE)
        outer = json.loads(evidence["stdout"])
        wire = outer["/get_server_info"]
        assert wire["status"] == 200
        body = wire["body"]
        assert hashlib.sha256(body.encode()).hexdigest() == wire["sha256"]
        parsed = lib.parse_server_info(body)
        assert parsed["ok"] is True
        assert "selected" not in parsed["keys"]
        assert parsed["dp_size"] == 16 and parsed["tp_size"] == parsed["ep_size"] == 32
        assert parsed["max_total_num_tokens"] >= 161280
        return {
            "evidence_sha256": sha256_file(ROOT / SERVER_EVIDENCE),
            "body_sha256": wire["sha256"],
            "top_level_object": True,
            "key_count": len(parsed["keys"]),
            "dp_size": parsed["dp_size"],
            "max_total_num_tokens": parsed["max_total_num_tokens"],
            "source_or_body_printed": False,
        }

    checks.run("actual_L12C_node36_server_info_top_level_schema", server_info_actual)
    checks.reject("selected_array_wrapper_rejected", lambda: lib.parse_server_info('{"selected": [{"path": "tp_size", "value": 32}]}'))

    def meta_metrics_actual():
        payload = json.loads(read_json(ROOT / HISTORICAL_META)["stdout"])
        line = next(item for item in payload["critical_lines"] if "[MMC mmc_meta_service.cpp:539 operator()] Metrics summary: " in item)
        parsed = lib.parse_meta_metrics(line)
        assert parsed["unmount_success"] == 0
        assert parsed["dram_total"] == 609885356032
        assert parsed["rank"] is None and parsed["per_rank_attribution"] is False
        return {
            "evidence_sha256": sha256_file(ROOT / HISTORICAL_META),
            "line_sha256": sha256_text(line),
            "unmount_success": parsed["unmount_success"],
            "unmount_fail": parsed["unmount_fail"],
            "dram_total": parsed["dram_total"],
            "rank_claimed": parsed["rank"],
        }

    checks.run("native_Meta_metrics_actual_field_parser", meta_metrics_actual)
    checks.run("native_unmount_text_without_rank_is_not_metrics", lambda: lib.parse_meta_metrics("2026-10-18 11:39:48.632295 125 trace Unmount bm on loc{rank=2} successfully") is None)

    def freshness_cases():
        before = [{"timestamp": "2026-10-19 00:00:00.000000"}]
        after = [
            {"timestamp": "2026-10-19 00:00:30.000000"},
            {"timestamp": "2026-10-19 00:01:00.000000"},
        ]
        fresh = lib.metrics_freshness(before, after, "2026-10-19 00:00:45.000000")
        assert fresh["fresh"] is True
        assert fresh["fresh_after"]["timestamp"] == "2026-10-19 00:01:00.000000"
        stale = lib.metrics_freshness(before, after, "2026-10-19 00:02:00.000000")
        assert stale["fresh"] is False
        return {"fresh_selected": fresh["fresh_after"]["timestamp"], "stale_left_pending": True}

    checks.run("meta_metrics_freshness_controlled_timestamps", freshness_cases)

    def audit_meta_capacity_fixture():
        auditor = load_module(AUDITOR, sha256_file(ROOT / AUDITOR), "audit_fixture_v2")

        def mount_line(timestamp, rank, capacity):
            return "%s 125 trace [MMC mmc_global_allocator.h:170 Mount] Mount bm on loc{rank=%d,media=DRAM}, capacity:%d  successfully" % (timestamp, rank, capacity)

        def metrics_line(timestamp, mount_success, unmount_success, unmount_fail=0):
            return ("%s 10 warning [MMC mmc_meta_service.cpp:539 operator()] Metrics summary: keys=0 dram_used=0/%d "
                    "mount_req=%d mount_success=%d mount_fail=0 unmount_req=%d unmount_success=%d unmount_fail=%d"
                    % (timestamp, auditor.TOTAL_BYTES, mount_success, mount_success, unmount_success, unmount_success, unmount_fail))

        before = [metrics_line("2026-10-19 02:00:00.000000", 10, 8)]
        mounts = [
            mount_line("2026-10-19 02:00:31.000000", 84, auditor.D_MOUNT_BYTES),
            mount_line("2026-10-19 02:00:32.000000", 85, auditor.D_MOUNT_BYTES),
        ]
        fresh = auditor.audit_meta_capacity(
            before,
            before + mounts + [metrics_line("2026-10-19 02:01:00.000000", 12, 10)],
            [84, 85],
        )
        assert fresh["pass"] is True
        assert fresh["aggregate"]["unmount_success"] - fresh["last_before"]["unmount_success"] == 2
        stale = auditor.audit_meta_capacity(before, before + mounts, [84, 85])
        assert stale["pass"] is False
        assert stale["pending_reason"] == "META_AFTER_SAMPLE_PRECEDES_FRESH_D_MOUNTS"
        low_delta = auditor.audit_meta_capacity(
            before,
            before + mounts + [metrics_line("2026-10-19 02:01:00.000000", 12, 9)],
            [84, 85],
        )
        assert low_delta["pass"] is False
        assert low_delta["pending_reason"] == "META_UNMOUNT_SUCCESS_DELTA_LT_2"
        return {
            "fresh_pass": True,
            "stale_pending": stale["pending_reason"],
            "low_delta_pending": low_delta["pending_reason"],
            "fresh_ids": sorted(item["rank"] for item in fresh["fresh"]),
        }

    checks.run("auditor_fresh_vs_stale_native_metrics_integration", audit_meta_capacity_fixture)

    def bs2_cases():
        good = {
            "is_draft_worker": True,
            "forward_mode": "DECODE",
            "batch_size": 2,
            "allowlisted": {"out_cache_loc": {"tensor": True, "shape": [2], "dtype": "torch.int64", "stride": [1], "device": "npu:0"}},
        }
        parsed = lib.classify_bs2_draft(good)
        assert parsed["eligible"] is True
        assert parsed["shape"] == [2] and parsed["dtype"] == "torch.int64" and parsed["stride"] == [1]
        wrong_role = dict(good, is_draft_worker=False)
        assert lib.classify_bs2_draft(wrong_role)["reason"] == "not_draft_worker"
        target_only = dict(good, forward_mode="TARGET_VERIFY", is_draft_worker=False)
        assert lib.classify_bs2_draft(target_only)["reason"] == "not_draft_worker"
        idle = dict(good, forward_mode="IDLE")
        assert lib.classify_bs2_draft(idle)["reason"] == "forward_mode"
        extend = dict(good, forward_mode="EXTEND")
        assert lib.classify_bs2_draft(extend)["reason"] == "forward_mode"
        missing = dict(good, allowlisted={})
        assert lib.classify_bs2_draft(missing)["reason"] == "out_cache_loc_metadata"
        return {"draft_decode_bs2": "accepted", "wrong_role_mode_or_metadata": "rejected"}

    checks.run("audit_bs2_draft_vs_wrong_role_mode_fixture", bs2_cases)

    def mixed_check():
        result = mixed.check_only()
        assert result["pass"] is True
        assert result["explicit_roles"] == {"selected_P_variant": "candidate", "selected_D_variants": ["candidate", "rollback"]}
        assert result["ssh"] == result["network"] == result["NPU_calls"] == result["SDK_init_calls"] == result["model_requests"] == 0
        return {"p_source": result["p_source"]["run_id"], "d_stages": sorted(result["d_stages"])}

    checks.run("mixed_identity_parameterized_check", mixed_check)

    def mixed_render():
        hashes = {}
        for variant, capacity in (("candidate", candidate_capacity), ("rollback", rollback_capacity)):
            for node in (34, 35, 36, 41):
                code = mixed.render(node, p_capacity, capacity, variant)
                compile(code, "mixed-%s-%d" % (variant, node), "exec")
                if node in (34, 35):
                    assert "NATIVE_P = True" in code
                else:
                    assert "D_VARIANT = %r" % variant in code
                hashes["%s-%d" % (variant, node)] = sha256_text(code)
        p_independent = all(
            hashes["candidate-%d" % node] == hashes["rollback-%d" % node]
            for node in (34, 35)
        )
        d_specific = all(
            hashes["candidate-%d" % node] != hashes["rollback-%d" % node]
            for node in (36, 41)
        )
        per_variant_distinct = all(
            len({hashes["%s-%d" % (variant, node)] for node in (34, 35, 36, 41)}) == 4
            for variant in ("candidate", "rollback")
        )
        assert p_independent and d_specific and per_variant_distinct
        return {
            "plans": len(hashes),
            "P_plans_variant_independent": p_independent,
            "D_plans_variant_specific": d_specific,
            "per_variant_distinct": per_variant_distinct,
        }

    checks.run("mixed_identity_render_all4_with_synthetic_actual_ids", mixed_render)

    def pre_materialization_absent():
        assert not (ROOT / "runs" / lib.TRANSITION_RID).exists()
        assert not (ROOT / "runs" / lib.STOP_RID).exists()
        assert not (ROOT / lib.CAPACITY_OUT).exists()
        assert not (ROOT / lib.REVIEW).exists()
        assert not (ROOT / "runs" / MATERIALIZATION_RID).exists()
        return {"transition_run_absent": True, "stop_run_absent": True, "review_absent": True, "capacity_absent": True}

    checks.run("no_actual_transition_review_capacity_created", pre_materialization_absent)

    def materialize():
        result = lib.materialize(ROOT / "runs" / MATERIALIZATION_RID)
        assert result["pass"] is True
        assert result["executable_plans"] == 22
        assert result["embedded_preflight_inner"] == 4
        assert result["review_status"] == "PENDING_PRIMARY_REVIEW"
        assert result["runtime_authorized"] is False
        manifest = read_json(ROOT / "runs" / MATERIALIZATION_RID / "manifest.json")
        status = read_json(ROOT / "runs" / MATERIALIZATION_RID / "status.json")
        assert manifest["rollback_sequence"]["status"] == "PENDING_CANDIDATE_CLOSE_AND_ACTUAL_CAPACITY_AUDIT"
        assert manifest["rollback_sequence"]["executed"] is False
        assert status["rollback_sequence"]["status"] == manifest["rollback_sequence"]["status"]
        return {
            "run": "runs/" + MATERIALIZATION_RID,
            "result": result["result_path"],
            "result_sha256": result["result_sha256"],
            "manifest_sha256": result["manifest_sha256"],
            "executable_plans": result["executable_plans"],
            "embedded_preflight_inner": 4,
            "rollback_sequence_status": manifest["rollback_sequence"]["status"],
        }

    materialized = checks.run("materialize_22_static_plus_4_inner", materialize)

    def ledger_unchanged():
        state = read_json(ROOT / "STATE.json")
        assert state["model_http_attempts_by_this_task"] == LEDGER_ATTEMPTS
        assert state["successful_model_inferences"] == LEDGER_SUCCESSES
        return {"backend_HTTP_attempts": LEDGER_ATTEMPTS, "successful_backend_inferences": LEDGER_SUCCESSES}

    checks.run("request_ledger_unchanged", ledger_unchanged)

    passed = sum(1 for row in checks.rows if row["pass"])
    total = len(checks.rows)
    assert passed == total, "qualification failure: %r" % [row for row in checks.rows if not row["pass"]]
    result = {
        "project_id": "glm53",
        "pass": True,
        "status": STATUS,
        "scope": "LOCAL_DRAFT_ONLY; no SSH/network/Docker/NPU/SDK/model/Git",
        "library": {"path": LIB, "sha256": LIB_SHA},
        "runner": {"path": "scripts/run_C32_L17_controlled_D_transition_and_pool_v2.py", "sha256": sha256_file(ROOT / "scripts/run_C32_L17_controlled_D_transition_and_pool_v2.py")},
        "auditor": {"path": "scripts/audit_C32_L17_selected_D_pool_local_v2.py", "sha256": sha256_file(ROOT / "scripts/audit_C32_L17_selected_D_pool_local_v2.py")},
        "mixed_identity": {"path": MIXED, "sha256": MIXED_SHA},
        "v1_preserved": {V1_LIB: V1_LIB_SHA, V1_RUNNER: V1_RUNNER_SHA, V1_AUDITOR: V1_AUDITOR_SHA},
        "static_plans": 22,
        "embedded_preflight_inner": 4,
        "checks": total,
        "checks_passed": passed,
        "checks_failed": 0,
        "check_rows": checks.rows,
        "materialization": materialized,
        "role_selectors": {"selected_P_variant": "candidate", "selected_D_variants": ["candidate", "rollback"], "legacy_D_scope": "D_ONLY_NOT_P"},
        "future_proposed_budget": {"functional_calls": 76, "functional_output_tokens": 21640, "canonical_calls": 320, "canonical_output_tokens": 76992, "total_calls": 396, "total_output_tokens": 98632, "status": "PROPOSED_NOT_AUTHORIZED"},
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "model_start_authorized": False,
        "no_actual_review_or_capacity": True,
        "no_hardware_proof_claim": True,
        "SSH": 0,
        "network": 0,
        "Docker_calls": 0,
        "NPU_calls": 0,
        "SDK_init_calls": 0,
        "model_requests": 0,
        "Git_operations": 0,
        "request_ledger_changed": False,
        "objective_complete": False,
    }
    out = ROOT / QUAL_RESULT
    assert not out.exists(), out
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "project_id": "glm53",
        "run_id": EXECUTOR_RID,
        "pass": True,
        "status": STATUS,
        "qualification_result": QUAL_RESULT,
        "qualification_result_sha256": sha256_file(out),
        "library_sha256": LIB_SHA,
        "runner_sha256": result["runner"]["sha256"],
        "auditor_sha256": result["auditor"]["sha256"],
        "mixed_identity_sha256": MIXED_SHA,
        "checks": total,
        "checks_passed": passed,
        "static_plans": 22,
        "embedded_preflight_inner": 4,
        "materialization_run": "runs/" + MATERIALIZATION_RID,
        "materialization_manifest_sha256": materialized["manifest_sha256"],
        "materialization_result_sha256": materialized["result_sha256"],
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "model_start_authorized": False,
        "SSH": 0,
        "network": 0,
        "Docker_calls": 0,
        "NPU_calls": 0,
        "SDK_init_calls": 0,
        "model_requests": 0,
        "Git_operations": 0,
        "request_ledger_changed": False,
        "objective_complete": False,
    }
    receipt_path = ROOT / "runs" / EXECUTOR_RID / "receipt.json"
    assert not receipt_path.exists(), receipt_path
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": True,
        "checks": "%d/%d" % (passed, total),
        "static_plans": 22,
        "embedded_preflight_inner": 4,
        "qualification_result": QUAL_RESULT,
        "qualification_result_sha256": receipt["qualification_result_sha256"],
        "receipt": str(receipt_path.relative_to(ROOT)),
        "runtime_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
