"""DSA causal/chunk coordinate regressions; safe without an NPU runtime."""
import pytest
import torch

from deepinfer.npu_ops.dsa910c import DSABatch, prepare_indices, quantize_query
from deepinfer.model_executor.models.w8a8_math910c import rotate_partial, rotate_positions


def test_c2_odd_chunk_and_multiple_requests():
    batch = DSABatch.build([2, 3], [32769, 7], 'cpu')
    torch.testing.assert_close(batch.cu_query, torch.tensor([0, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(batch.positions(), torch.tensor([32767, 32768, 4, 5, 6], dtype=torch.int32))
    lengths, residual = batch.compressed(2)
    torch.testing.assert_close(lengths, torch.tensor([16384, 3], dtype=torch.int32))
    torch.testing.assert_close(residual, torch.tensor([1, 1], dtype=torch.int32))


def test_query_uses_fp16_scale_before_rounding_and_zero_is_finite():
    query = torch.zeros((2, 1, 128))
    query[0, 0, :3] = torch.tensor([127*.12345, 64.5001*.12345, -64.5001*.12345])
    codes, scales = quantize_query(query)
    rounded_scale = torch.tensor(.12345).half().float()
    assert scales.dtype == torch.float16 and codes.dtype == torch.int8
    assert codes[0, 0, 1].item() == round(float(query[0, 0, 1]/rounded_scale))
    # This exact input distinguishes the required FP16 scale from FP32.
    assert codes[0, 0, 1].item() != round(float(query[0, 0, 1]/.12345))
    assert scales[1, 0].item() == 2.**-24
    assert not codes[1].any()


def test_future_and_incomplete_c2_keys_cannot_escape_filter():
    selected = torch.tensor([[4, 2, -1, 1, 3], [4, 2, -1, 1, 3]], dtype=torch.int32)
    result = prepare_indices(selected, torch.tensor([5, 6]), 2)
    # Both positions can see completed pairs 0,1,2; pair 3 is incomplete.
    torch.testing.assert_close(result, torch.tensor([[1, 2, -1, -1, -1]]*2, dtype=torch.int32))


@pytest.mark.parametrize('heads', [None, 32])
def test_packed_rope_matches_individual_positions_at_chunk_boundary(heads):
    shape = (4, 128) if heads is None else (4, heads, 128)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(73)).bfloat16()
    frequencies = 160000**(-torch.arange(0, 64, 2).float()/64)
    positions = torch.tensor([0, 127, 32767, 32768], dtype=torch.int32)
    for inverse in (False, True):
        actual = rotate_positions(x, positions, frequencies, inverse=inverse)
        expected = torch.stack([rotate_partial(value, int(position), frequencies, inverse=inverse)
                                for value, position in zip(x, positions)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('q,s', [([], []), ([0], [1]), ([2], [1]), ([1], [1, 2]), ([1.5], [2])])
def test_invalid_chunks_rejected(q, s):
    with pytest.raises(ValueError):
        DSABatch.build(q, s, 'cpu')
