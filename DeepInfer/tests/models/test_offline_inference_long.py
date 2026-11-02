"""Long-context HF regression, separate from the short CI workload."""

import os

import pytest

from examples._config import DEFAULT_MODEL_NAME
from tests.models.test_offline_inference import _run_offline_inference

pytestmark = [
    pytest.mark.long_context,
]


def test_offline_inference_long_decode_ppl() -> None:
    _run_offline_inference(DEFAULT_MODEL_NAME, int(os.environ.get("DEVICE_OFFSET", "0")), long_context=True)
