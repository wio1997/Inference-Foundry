"""Non-operational, dependency-injected draft of an isolated diagnostic.

This file has NO command-line entrypoint, Docker, SSH, HTTP, device or real
filesystem adapter. It cannot start a service. Real adapters require separate
review; all checks here are necessary conditions, not a safety certificate.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Protocol

BASE_ID = 'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
IMAGE_ID = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
OWNER_LOCK = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic/owner-fence.lock'
CASE = {'prompt_tokens': 32768, 'completion_tokens': 1024, 'concurrency': 8,
        'prefix_queries': 262144, 'prefix_hits': 235904, 'preemptions': 0}


class Refusal(RuntimeError):
    pass


@dataclass(frozen=True)
class Baseline:
    container_id: str
    image_id: str
    running: bool
    health: int
    requests_running: int
    requests_waiting: int
    owned_host_pids: frozenset
    npu_host_pids: frozenset
    sixteen_chips_accounted: bool
    hardware_fingerprint_matches: bool
    hbm_reference_valid: bool
    service_argv: tuple
    config_hashes: tuple
    no_profiler_verified: bool
    no_old_injection_verified: bool


@dataclass(frozen=True)
class Diagnostic:
    container_id: str
    image_id: str
    name: str
    run_id: str
    owned_labels_verified: bool
    original_argv: tuple
    config_hashes: tuple
    baseline_mounts_reviewed: bool
    runtime_within_task_dir: bool
    writable_mounts_confined: bool
    original_profile_and_hooks_absent: bool
    observer_artifacts_reviewed: bool


class Port(Protocol):
    """Contracts that real adapters MUST implement; no implementation supplied."""
    def acquire_same_owner_fence(self, path: str): ...
    def assert_fence_live(self): ...
    def start_watchdog_and_verify_fresh_lease(self, run_id: str): ...
    def baseline_snapshot(self) -> Baseline: ...
    def prepare_new_owned_runtime_and_container(self, run_id: str, base: Baseline) -> Diagnostic: ...
    def save_and_reread_two_identities(self, base: Baseline, diag: Diagnostic): ...
    def verify_fresh_watchdog_lease(self, run_id: str): ...
    def stop_exact_baseline(self, baseline_id: str): ...
    def require_empirical_release_and_recheck(self): ...
    def start_exact_diagnostic(self, diagnostic_id: str): ...
    def wait_diag_health_and_recheck_owner(self, diag: Diagnostic): ...
    def record_runtime_installation_and_rank_coverage_or_unknown(self, diag: Diagnostic): ...
    def capture_no_profile_case(self, diag: Diagnostic): ...
    # Called with the existing fence held. A real adapter must not call the old
    # recovery helper, which acquires that same fence, without safely handing
    # it off; otherwise it deadlocks. Watchdog must coordinate with that handoff.
    # base/diag may be None after an ambiguous prepare result. Resolve only
    # exact run labels and durable identity; never stop ambiguous containers.
    def recover_and_verify_exact_baseline(self, base: Baseline | None, diag: Diagnostic | None, run_id: str) -> bool: ...
    def watchdog_done(self, run_id: str): ...
    def release_fence(self): ...


def _run_id(run_id):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise Refusal('fresh 32 lowercase hex run identity required')


def _baseline_ok(base):
    if (base.container_id != BASE_ID or base.image_id != IMAGE_ID or not base.running
        or base.health != 200 or base.requests_running != 0 or base.requests_waiting != 0
        or not base.owned_host_pids or not base.npu_host_pids
        or not base.npu_host_pids <= base.owned_host_pids
        or not base.sixteen_chips_accounted or not base.hardware_fingerprint_matches
        or not base.hbm_reference_valid or not base.no_profiler_verified
        or not base.no_old_injection_verified or not base.service_argv or not base.config_hashes):
        raise Refusal('baseline identity, occupancy, idle, configuration or ownership not established')


def _diagnostic_ok(diag, base, run_id):
    if (len(diag.container_id) != 64 or any(c not in '0123456789abcdef' for c in diag.container_id)
        or diag.container_id == BASE_ID or not diag.name or run_id not in diag.name
        or diag.run_id != run_id or diag.image_id != base.image_id
        or not diag.owned_labels_verified or diag.original_argv != base.service_argv
        or diag.config_hashes != base.config_hashes or not diag.baseline_mounts_reviewed
        or not diag.runtime_within_task_dir or not diag.writable_mounts_confined or not diag.original_profile_and_hooks_absent
        or not diag.observer_artifacts_reviewed):
        raise Refusal('diagnostic identity or exact original inference configuration not established')


def _same_frozen_baseline(first, second):
    _baseline_ok(second)
    # PIDs can legitimately change; fresh ownership is checked independently.
    if (first.container_id, first.image_id, first.service_argv, first.config_hashes) != (
        second.container_id, second.image_id, second.service_argv, second.config_hashes):
        raise Refusal('baseline configuration changed between guards')


def draft_run(port: Port, run_id: str):
    """Simulation-level ordering; NEVER pass an unreviewed real adapter."""
    _run_id(run_id)
    fence_held = False
    watchdog_started = False
    diag = None
    base = None
    pending = None
    result = None
    try:
        port.acquire_same_owner_fence(OWNER_LOCK); fence_held = True
        port.assert_fence_live()
        # Treat a timeout while starting the watchdog as an ambiguous side
        # effect; the adapter must look up this run's exact watchdog/lease.
        watchdog_started = True
        port.start_watchdog_and_verify_fresh_lease(run_id)
        base = port.baseline_snapshot(); _baseline_ok(base)
        diag = port.prepare_new_owned_runtime_and_container(run_id, base)
        _diagnostic_ok(diag, base, run_id)
        port.save_and_reread_two_identities(base, diag)
        port.assert_fence_live()
        second = port.baseline_snapshot(); _same_frozen_baseline(base, second)
        port.save_and_reread_two_identities(second, diag)
        port.verify_fresh_watchdog_lease(run_id)
        port.assert_fence_live()
        # Once this call is attempted, even a timeout is ambiguous. Finally
        # must recover: never assume that an exception means no stop occurred.
        port.stop_exact_baseline(BASE_ID)
        port.require_empirical_release_and_recheck()
        port.assert_fence_live()
        port.start_exact_diagnostic(diag.container_id)
        port.wait_diag_health_and_recheck_owner(diag)
        port.record_runtime_installation_and_rank_coverage_or_unknown(diag)
        result = port.capture_no_profile_case(diag)
    except BaseException as exc:
        pending = exc
    finally:
        if watchdog_started:
            try:
                port.assert_fence_live()
                healthy = port.recover_and_verify_exact_baseline(base, diag, run_id)
                if healthy:
                    port.watchdog_done(run_id)
                else:
                    raise Refusal('baseline recovery not independently verified; watchdog remains armed')
            except BaseException as exc:
                # Recovery failure takes precedence: stop the chain, never
                # claim completion. The production watchdog must own retry.
                pending = Refusal('recovery unresolved: ' + type(exc).__name__)
        if fence_held:
            try: port.release_fence()
            except BaseException:
                pending = Refusal('owner fence release unresolved')
    if pending is not None:
        raise pending
    return result


class ClientPort(Protocol):
    """A client has no profile endpoints; it is supplied by a reviewed adapter."""
    def seed(self, row): ...
    def verify_frozen_input_identity(self, seeds, requests): ...
    def metrics(self): ...
    def open_gate_atomic(self, run_id: str): ...
    # Remove only a marker owned by this run; refuse an ambiguous foreign one.
    def close_gate_and_record(self, run_id: str): ...
    def complete(self, row): ...
    def summarize_non_sensitive(self, responses, before, after): ...


def draft_case(client: ClientPort, run_id: str, seeds, requests):
    """Frozen inputs stay inside adapter memory; returns aggregate data only.

    The adapter must verify frozen file/content identity, implement HTTP and
    metrics handling without logging prompt or generated text, and validate the
    original sampling options. A barrier does not imply equal arrival times.
    """
    _run_id(run_id)
    if len(seeds) != 8 or len(requests) != 8:
        raise Refusal('exactly eight seed and eight measurement requests required')
    if not client.verify_frozen_input_identity(seeds, requests):
        raise Refusal('frozen request identity not established')
    def parallel(rows, fn):
        barrier=threading.Barrier(8)
        def worker(row):
            barrier.wait(timeout=30)
            return fn(row)
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(worker, rows))
    # Failure before successful opening must not leave a marker. Any exception
    # during open can occur after rename: always attempt idempotent close.
    seed_results=parallel(seeds, client.seed)
    if not all(x.get('prompt_tokens') == 29696 and x.get('completion_tokens') == 1 for x in seed_results):
        raise Refusal('prefix seeding usage mismatch')
    before = client.metrics()
    try:
        client.open_gate_atomic(run_id)
        responses=parallel(requests, client.complete)
    finally:
        client.close_gate_and_record(run_id)
    after = client.metrics()
    if len(responses) != 8 or not all(x.get('prompt_tokens') == 32768 and x.get('completion_tokens') == 1024 for x in responses):
        raise Refusal('measurement usage mismatch')
    summary = client.summarize_non_sensitive(responses, before, after)
    if (summary.get('successful_requests') != 8 or summary.get('prompt_tokens') != 262144
        or summary.get('generation_tokens') != 8192 or summary.get('prefix_queries') != CASE['prefix_queries']
        or summary.get('prefix_hits') != CASE['prefix_hits'] or summary.get('preemptions') != 0):
        raise Refusal('measurement counter mismatch')
    # Prevent a faulty adapter from persisting response text, request IDs, or
    # other unexpected content through the draft's return value.
    allowed=('successful_requests','prompt_tokens','generation_tokens','prefix_queries',
             'prefix_hits','preemptions','draft_tokens','accepted_tokens','exact_matches')
    if any(k in summary and (type(summary[k]) is not int or summary[k] < 0) for k in allowed):
        raise Refusal('unsafe or invalid aggregate field')
    return {k:summary[k] for k in allowed if k in summary}
