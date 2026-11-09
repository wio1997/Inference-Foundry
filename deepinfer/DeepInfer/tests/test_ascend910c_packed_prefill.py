import copy

import pytest
import torch

from deepinfer.npu_ops.dsa910c import DSABatch
from deepinfer.model_executor.models.w8a8_compressor910c import compress_batch
from deepinfer.model_executor.models.w8a8_math910c import compress_step
from deepinfer.model_executor.models.w8a8_engram910c import CpuEngramHistory


@pytest.mark.parametrize('ratio', [1, 2])
def test_packed_compression_equals_scalar_across_odd_chunks_and_requests(ratio):
    g = torch.Generator().manual_seed(2921)
    tokens = {0: torch.randn((15, 16), generator=g).bfloat16(), 2: torch.randn((15, 16), generator=g).bfloat16()}
    wkv = torch.randn((8, 16), generator=g).bfloat16()
    wgate = torch.randn((8, 16), generator=g).bfloat16() if ratio == 2 else None
    norm = torch.ones(8, dtype=torch.bfloat16)
    pending, scalar_pending = {}, {0: [], 2: []}
    start = 0
    for length in (1, 3, 1, 6, 4):
        batch = DSABatch.build([length, length], [start+length]*2, 'cpu')
        x = torch.cat([tokens[slot][start:start+length] for slot in (2, 0)])
        actual, positions, slots = compress_batch(x, batch, [2, 0], ratio, wkv, wgate, norm, 1e-6, pending)
        expected, expected_positions, expected_slots = [], [], []
        for slot in (2, 0):
            for position in range(start, start+length):
                latent, source = compress_step(tokens[slot][position:position+1], position, ratio, wkv, wgate, norm, 1e-6, scalar_pending[slot])
                if latent is not None:
                    expected.append(latent); expected_positions.append(source); expected_slots.append(slot)
        reference = torch.cat(expected) if expected else x.new_empty((0, 8))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert positions == expected_positions and slots == expected_slots
        start += length


def fake_history():
    history = CpuEngramHistory.__new__(CpuEngramHistory)
    history.cfg = {'engram_max_ngram_size': 4}
    history.token_map = list(range(32))
    history.token_lookup = torch.tensor(history.token_map, dtype=torch.int32)
    history.pad_id = 2
    history.layers = [1, 14]
    history.multipliers = torch.tensor([[3, 5, 7, 11], [13, 17, 19, 23]], dtype=torch.int64)
    history.primes = torch.tensor([[[31, 37], [41, 43], [47, 53]], [[59, 61], [67, 71], [73, 79]]], dtype=torch.int32)
    history.offsets = torch.tensor([[0, 31, 68, 109, 152, 199], [0, 59, 120, 187, 258, 331]], dtype=torch.int32)
    history.image_ids = {30, 31}
    history.reset()
    return history


def test_engram_batch_equals_sequential_across_history_and_image_boundaries():
    chunked, scalar = fake_history(), fake_history()
    for chunk in ([0], [5, 7, 30], [8, 9], [31, 3, 4, 5, 6]):
        actual = chunked.batch(chunk)
        steps = [scalar.step(token) for token in chunk]
        for layer in chunked.layers:
            torch.testing.assert_close(actual[layer], torch.stack([s[layer] for s in steps]), rtol=0, atol=0)
        assert chunked.history == scalar.history
    before = copy.copy(chunked.history)
    with pytest.raises(ValueError):
        chunked.batch([4, 999])
    assert chunked.history == before
