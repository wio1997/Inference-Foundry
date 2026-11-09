import pytest
import torch

from examples.ascend910c_w8a8_semantics import quantization_contract


def test_exact_half_ties_both_nearest_integers_and_zero_row():
    x = torch.tensor([[3.625, 1.8125, -1.8125, 0.], [0., 0., 0., 0.]]).bfloat16()
    scales = x.float().abs().amax(-1)/127
    for half in (63, 64):
        q = torch.tensor([[127, half, -half, 0], [0, 0, 0, 0]], dtype=torch.int8)
        assert quantization_contract(x, q, scales)['max_rounding_error_code_units'] == .5


def test_wrong_nonboundary_code_is_rejected():
    x = torch.tensor([[127., 32., -32., 0.]]).bfloat16()
    with pytest.raises(AssertionError, match='non-nearest'):
        quantization_contract(x, torch.tensor([[127, 33, -32, 0]], dtype=torch.int8), torch.ones(1))


def test_wrong_scale_is_rejected():
    with pytest.raises(AssertionError):
        quantization_contract(torch.ones(1, 4).bfloat16(), torch.full((1, 4), 127, dtype=torch.int8), torch.ones(1))
