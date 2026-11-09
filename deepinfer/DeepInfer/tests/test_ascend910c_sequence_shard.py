"""CPU contracts for TP token ownership, including an idle DP scratch row."""
import pytest
import torch

from deepinfer.npu_ops.sp910c import SequenceShard


@pytest.mark.parametrize('tokens', [1, 7, 9, 640])
def test_sequence_shards_preserve_tokens_and_mask_padding(tokens):
    values = torch.arange(tokens * 3, dtype=torch.int32).reshape(tokens, 3)
    slices = []
    for rank in range(8):
        shard = SequenceShard(None)
        shard.world, shard.rank = 8, rank
        slices.append(shard.split(values, fill=-1))
    combined = torch.cat(slices)
    torch.testing.assert_close(combined[:tokens], values, rtol=0, atol=0)
    assert (combined[tokens:] == -1).all()
    assert len({piece.shape for piece in slices}) == 1


def test_single_rank_preserves_mhc_axes():
    value = torch.arange(7 * 4 * 16).reshape(7, 4, 16)
    shard = SequenceShard(None)
    torch.testing.assert_close(shard.gather(shard.split(value), 7), value)
