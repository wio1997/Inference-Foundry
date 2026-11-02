"""Shared fixtures for tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _clean_env() -> Generator[None]:
    """Snapshot os.environ before each test and restore it afterwards.

    This prevents environment variables changed during one test from leaking
    into subsequent tests in the same pytest process.
    The env descriptors are re-armed so the next test re-resolves from the
    restored os.environ.
    """
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)
    if environ := sys.modules.get("deepinfer.environ"):
        environ.rearm_all()
