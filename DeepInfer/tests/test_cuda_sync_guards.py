from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Generator

import pytest
import torch

from deepinfer.utils import _sync_debug_arbiter, allow_cuda_sync, forbid_cuda_sync


@pytest.fixture
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> Generator[dict[str, str]]:
    """Replace the global sync debug mode with a plain variable to bypass CUDA initialization."""
    assert not _sync_debug_arbiter._stacks, "arbiter state left over; a previous test did not clean up"
    state = {"mode": "default"}
    monkeypatch.setattr(torch.cuda, "set_sync_debug_mode", lambda m: state.__setitem__("mode", m))
    monkeypatch.setattr(torch.cuda, "get_sync_debug_mode", lambda: state["mode"])
    yield state
    assert not _sync_debug_arbiter._stacks, "the arbiter still has an un-exited scope after the test"


def test_single_thread_nesting(fake_mode: dict[str, str]) -> None:
    assert fake_mode["mode"] == "default"
    with forbid_cuda_sync():
        assert fake_mode["mode"] == "error"
        with allow_cuda_sync():  # inner allow overrides outer forbid
            assert fake_mode["mode"] == "default"
            with forbid_cuda_sync():  # a further inner forbid overrides it back
                assert fake_mode["mode"] == "error"
            assert fake_mode["mode"] == "default"
        assert fake_mode["mode"] == "error"
    assert fake_mode["mode"] == "default"


def test_baseline_restored(fake_mode: dict[str, str]) -> None:
    fake_mode["mode"] = "warn"  # a baseline set externally (e.g. user debugging)
    with forbid_cuda_sync():
        assert fake_mode["mode"] == "error"
    assert fake_mode["mode"] == "warn"


def test_interleaved_exit_no_leak(fake_mode: dict[str, str]) -> None:
    """The old implementation (each save/restore of the global value) leaks error outside the scope under this interleaving:
    A enters forbid (saves default) -> B enters allow (saves error!) -> A exits (restores default) -> B exits (restores error).
    """
    steps = [threading.Event() for _ in range(3)]
    modes_seen = {}

    def thread_a():
        with forbid_cuda_sync():
            steps[0].set()  # A has entered
            steps[1].wait(10)  # wait for B to enter
        steps[2].set()  # A has exited

    def thread_b():
        steps[0].wait(10)
        with allow_cuda_sync():
            steps[1].set()
            modes_seen["b_inside"] = fake_mode["mode"]
            steps[2].wait(10)  # wait for A to exit, then exit
            modes_seen["b_inside_after_a_exit"] = fake_mode["mode"]

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start(), tb.start()
    ta.join(10), tb.join(10)
    assert not ta.is_alive() and not tb.is_alive()

    assert modes_seen["b_inside"] == "default"  # B never sees error inside allow
    assert modes_seen["b_inside_after_a_exit"] == "default"
    assert fake_mode["mode"] == "default"  # no leak after everything exits (the old implementation leaves error here)


def test_forbid_across_threads(fake_mode: dict[str, str]) -> None:
    """With only forbid across threads the global mode is error; it is restored after all exit."""
    n = 4
    entered = threading.Barrier(n + 1)
    release = threading.Event()

    def worker():
        with forbid_cuda_sync():
            entered.wait(10)
            release.wait(10)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    entered.wait(10)
    assert fake_mode["mode"] == "error"
    release.set()
    for t in threads:
        t.join(10)
    assert fake_mode["mode"] == "default"


def test_stress_allow_never_sees_error(fake_mode: dict[str, str]) -> None:
    """Random concurrent entry/exit: no thread may read global mode == error inside its own allow scope
    (the key invariant behind approximating the global switch as per-thread: legitimate syncs are never killed)."""
    stop = time.monotonic() + 3.0
    violations: list[str] = []
    barrier = threading.Barrier(8)

    def allow_worker(seed: int):
        rng = random.Random(seed)
        barrier.wait(10)
        while time.monotonic() < stop:
            with allow_cuda_sync():
                for _ in range(rng.randrange(1, 20)):
                    if fake_mode["mode"] == "error":
                        violations.append(f"allow_worker{seed} saw error")
                        return
                if rng.random() < 0.3:
                    with forbid_cuda_sync():  # also stir in nesting inside a thread
                        pass
            if rng.random() < 0.2:
                time.sleep(0)

    def forbid_worker(seed: int):
        rng = random.Random(seed)
        barrier.wait(10)
        while time.monotonic() < stop:
            with forbid_cuda_sync():
                mode = fake_mode["mode"]
                if mode not in ("error", "default"):  # may be default when overlapping allow, but must never take another value
                    violations.append(f"forbid_worker{seed} saw {mode}")
                    return
                if rng.random() < 0.3:
                    with allow_cuda_sync():
                        if fake_mode["mode"] == "error":
                            violations.append(f"forbid_worker{seed} inner allow saw error")
                            return
            if rng.random() < 0.2:
                time.sleep(0)

    threads = [threading.Thread(target=allow_worker, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=forbid_worker, args=(100 + i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive()
    assert not violations, violations
    assert fake_mode["mode"] == "default"  # no leak once everything settles


def test_decorator_concurrent(fake_mode: dict[str, str]) -> None:
    """Decorator usage (sharing one _GeneratorContextManager instance) called concurrently from multiple threads."""

    @allow_cuda_sync()
    def do_allow():
        assert fake_mode["mode"] == "default"

    @forbid_cuda_sync()
    def do_forbid():
        assert fake_mode["mode"] in ("error", "default")

    def worker(fn: Callable[[], None]) -> None:
        for _ in range(200):
            fn()

    threads = [threading.Thread(target=worker, args=(fn,)) for fn in (do_allow, do_forbid) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive()
    assert fake_mode["mode"] == "default"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_e2e_gpu() -> None:
    """Verify on a real GPU: syncs in forbid threads raise; syncs in concurrent allow threads never raise."""
    x = torch.ones(8, device="cuda")

    with forbid_cuda_sync(), pytest.raises(RuntimeError, match="synchroniz"):
        x.sum().item()
    assert x.sum().item() == 8.0  # normal again outside the scope

    in_allow = threading.Event()
    done_forbid = threading.Event()
    errors: list[Exception] = []

    def allow_thread():
        try:
            with allow_cuda_sync():
                in_allow.set()
                deadline = time.monotonic() + 10
                while not done_forbid.is_set() and time.monotonic() < deadline:
                    x.sum().item()  # must still be allowed while overlapping forbid
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def forbid_thread():
        try:
            with forbid_cuda_sync():
                in_allow.wait(10)
                x.sum().item()  # allowed globally while an allow thread is present (the missed report is the tradeoff)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            done_forbid.set()

    ta, tf = threading.Thread(target=allow_thread), threading.Thread(target=forbid_thread)
    ta.start(), tf.start()
    ta.join(30), tf.join(30)
    assert not ta.is_alive() and not tf.is_alive()
    assert not errors, errors

    with forbid_cuda_sync(), pytest.raises(RuntimeError, match="synchroniz"):
        x.sum().item()  # enforcement returns after the allow thread exits
