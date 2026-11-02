"""Run multi-rank regressions with a bounded lifetime and isolated environment."""

from __future__ import annotations

import os
import signal
from collections.abc import Callable
from contextlib import suppress
from multiprocessing import get_context

from examples._launch import launch_dp


def _run(entry: Callable[[int, int], None], dp_size: int) -> None:
    os.setsid()
    launch_dp(entry, dp_size)


def run_model_test(entry: Callable[[int, int], None], dp_size: int, timeout: int = 1800) -> None:
    process = get_context("spawn").Process(target=_run, args=(entry, dp_size))
    process.start()
    try:
        process.join(timeout)
        assert not process.is_alive(), f"Model regression timed out after {timeout}s"
        assert process.exitcode == 0, f"Model regression failed: exitcode={process.exitcode}"
    finally:
        # All DP workers inherit this process group, including on partial failure.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.is_alive():
            process.kill()
        process.join()
