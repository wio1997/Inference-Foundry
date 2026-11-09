"""CUDA-compatible locality helpers; Ascend currently uses one logical domain."""

import torch


# TODO: Implement Ascend localization. Until then, keep the CUDA public API
# as no-ops and report one logical domain without localized allocations.
def get_num_locality_domains() -> int:
    return 1


def is_localization_available() -> bool:
    return False


def is_localized(t: torch.Tensor) -> bool:
    return False


def localize(t: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Return the input unchanged until Ascend localization is implemented."""
    return t


def destroy_localizer() -> None:
    pass
