from typing import Any

import torch

# noinspection PyUnresolvedReferences
import deep_ep._C as _C


def get_comm_stream(buffer: Any) -> torch.Stream:
    """Return DeepEP's communication stream."""
    stream: torch.Stream = _C.get_comm_stream()
    return torch.npu.Stream(stream_id=stream.stream_id, device_index=stream.device_index, device_type=stream.device_type)
