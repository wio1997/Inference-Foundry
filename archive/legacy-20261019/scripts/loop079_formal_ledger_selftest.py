#!/usr/bin/env python3
"""CPU-only fixture: patch install/restore on copies, phase, raw-ID records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from scripts import loop079_formal_ledger_patch as patcher
from scripts import loop079_formal_ledger_phase as phase
from scripts import loop079_formal_ledger_server as ledger


def invoke(action, state, record, offline=False):
    args = ["patcher", action, "--state-dir", str(state), "--record", str(record)]
    if offline:
        args.append("--offline-confirmed")
    sys.argv = args
    patcher.main()


def main():
    original_sources = patcher.SOURCES
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        copied = {}
        for key, (source, expected) in original_sources.items():
            path = root / (key + ".py")
            path.write_bytes(source.read_bytes())
            copied[key] = (path, expected)
        patcher.SOURCES = copied
        state = root / "state"
        invoke("check", state, root / "check.json")
        try:
            invoke("install", state, root / "denied.json")
        except ValueError as exc:
            assert "offline-confirmed" in str(exc)
        else:
            raise AssertionError("unguarded install accepted")
        assert not state.exists()

        # Mid-temp-file I/O failure must leave the destination's exact
        # original bytes in place; no short/truncated source is admitted.
        runner_path, runner_sha = copied["runner"]
        real_write = patcher.os.write
        writes = 0
        def interrupted_write(fd, data):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("injected temp write interruption")
            return real_write(fd, data)
        patcher.os.write = interrupted_write
        try:
            try:
                patcher.atomic_replace(runner_path, b"X" * 131072)
            except OSError as exc:
                assert "injected" in str(exc)
            else:
                raise AssertionError("injected write failure did not fire")
        finally:
            patcher.os.write = real_write
        assert hashlib.sha256(runner_path.read_bytes()).hexdigest() == runner_sha
        assert not list(root.glob(".runner.py.formal-ledger-*"))

        # A controller interruption after the first complete source rename
        # leaves a mixed known-hash state that the guarded restore can repair.
        real_atomic = patcher.atomic_replace
        def interrupted_install(path, data):
            if path == runner_path:
                raise OSError("injected second-source interruption")
            return real_atomic(path, data)
        patcher.atomic_replace = interrupted_install
        partial_state = root / "partial_state"
        try:
            try:
                invoke("install", partial_state, root / "partial_install.json", True)
            except OSError as exc:
                assert "injected" in str(exc)
            else:
                raise AssertionError("partial install injection did not fire")
        finally:
            patcher.atomic_replace = real_atomic
        partial_manifest = json.loads((partial_state / "manifest.json").read_text())
        assert hashlib.sha256(copied["scheduler"][0].read_bytes()).hexdigest() == partial_manifest["files"]["scheduler"]["patched"]
        assert hashlib.sha256(runner_path.read_bytes()).hexdigest() == runner_sha
        original_helper = patcher.HELPER
        changed_helper = root / "changed_helper.py"
        changed_helper.write_text("# helper changed after stopped service\n")
        patcher.HELPER = changed_helper
        try:
            invoke("restore", partial_state, root / "partial_restore.json", True)
        finally:
            patcher.HELPER = original_helper
        assert json.loads((root / "partial_restore.json").read_text())["helper_sha_drift"]
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected
                   for path, expected in copied.values())

        invoke("install", state, root / "install.json", True)
        first = json.loads((root / "install.json").read_text())
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == first["files"][key]["patched"]
                   for key, (path, _) in copied.items())
        invoke("install", state, root / "install_again.json", True)
        assert json.loads((root / "install_again.json").read_text())["idempotent"]
        manifest_path = state / "manifest.json"
        intact_manifest = manifest_path.read_bytes()
        for fault in ("missing", "extra"):
            corrupted = json.loads(intact_manifest)
            if fault == "missing":
                del corrupted["files"]["api"]
            else:
                corrupted["files"]["unexpected"] = dict(corrupted["files"]["api"])
            manifest_path.write_text(json.dumps(corrupted) + "\n")
            try:
                invoke("restore", state, root / f"{fault}_restore.json", True)
            except ValueError as exc:
                assert "source set differs" in str(exc)
            else:
                raise AssertionError(f"{fault} source key-set accepted")
            assert all(hashlib.sha256(path.read_bytes()).hexdigest() == first["files"][key]["patched"]
                       for key, (path, _) in copied.items())
        manifest_path.write_bytes(intact_manifest)
        # Simulate a controller stopping after only part of an install was
        # applied: restore must accept expected-patched or original per file.
        copied["serving"][0].write_bytes((state / "serving.original.py").read_bytes())
        invoke("restore", state, root / "restore.json", True)
        invoke("restore", state, root / "restore_again.json", True)
        assert json.loads((root / "restore_again.json").read_text())["idempotent"]
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected
                   for path, expected in copied.values())

        marker, transitions = root / "phase.json", root / "phase.jsonl"
        phase.set_phase(marker, transitions, "fixture", "warmup", None)
        os.environ[ledger.ENV] = str(root / "ledger")
        os.environ[ledger.RUN] = "fixture"
        os.environ[ledger.PHASE_FILE] = str(marker)
        class Request:
            request_id = "req-1"
            _output_token_ids = [12]
            max_tokens = 1024
            num_prompt_tokens = 32768
            num_output_placeholders = 0
            num_in_flight_tokens = 0
            resumable = False
            status = "running"
        request = Request()
        incoming = [21, 22, 23]
        row = ledger.scheduler_before(object(), request, incoming, True, False)
        incoming.pop()
        request._output_token_ids.extend([21, 22])
        ledger.scheduler_after(row, request, [21, 22], False)
        assert row["incoming_raw_ids"] == [21, 22, 23]
        ledger.api_consume("ext-1", "ext-1", 0, [91], 1, "x", None, None, None)
        assert ledger.api_yield("ext-1", "data: [DONE]\n\n") == "data: [DONE]\n\n"
        warmup_report = root / "warmup.json"
        warmup_report.write_text('{"completed":48}\n')
        phase.set_phase(marker, transitions, "fixture", "measured", warmup_report)
        ledger.emit("fixture_measured", request_id="req-2")
        ledger.flush("selftest")
        ledger_file = root / "ledger" / f"pid{os.getpid()}.jsonl"
        rows = [json.loads(line) for line in ledger_file.read_text().splitlines()]
        integrity = ledger.verify_flushes(root / "ledger", os.getpid())
        assert integrity["records"] == len(rows) and integrity["flushes"] >= 3
        intact = ledger_file.read_bytes()
        ledger_file.write_bytes(intact[:-1])
        try:
            ledger.verify_flushes(root / "ledger", os.getpid())
        except ValueError as exc:
            assert "truncated or changed" in str(exc)
        else:
            raise AssertionError("truncated ledger was accepted")
        ledger_file.write_bytes(intact)
        app = next(row for row in rows if row["event"] == "scheduler_append")
        measured = next(row for row in rows if row["event"] == "fixture_measured")
        assert app["incoming_raw_ids"] == [21, 22, 23]
        assert app["admitted_raw_ids"] == [21, 22]
        assert app["phase"] == "warmup" and app["phase_run_id_matches"]
        consume = next(row for row in rows if row["event"] == "api_consume")
        assert consume["output_request_id"] == "ext-1" and consume["api_request_id"] == "ext-1"
        assert "request_id" not in consume
        assert measured["phase"] == "measured" and measured["phase_generation"] == 1
        assert [json.loads(x)["phase"] for x in transitions.read_text().splitlines()] == ["warmup", "measured"]
        print(json.dumps(dict(status="pass", fixture_patch_files=len(copied),
                              ledger_rows=len(rows), flushes=integrity["flushes"],
                              transitions=2, interrupted_temp_write="rejected",
                              partial_install_restore="pass", helper_sha_drift="restored",
                              source_key_set_negatives=2, truncation="rejected")))


if __name__ == "__main__":
    main()
