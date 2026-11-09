import json
from pathlib import Path

import pytest
import torch

from deepinfer.npu_ops.dsa910c import DSABatch
from deepinfer.model_executor.models.w8a8_paged_cache910c import PagedPlane, W8A8PagedCache


def read_rows(plane, positions, slot):
    return torch.stack([plane.cache[plane.block_table[slot, p//plane.block_size], p%plane.block_size, 0]
                        for p in positions])


@pytest.mark.parametrize('capacity,lengths', [(257, (127, 257, 129, 257, 257, 257)),
                                            (2048, (127, 2048, 129, 2048, 2048))])
def test_swa_ring_retains_first_query_history_after_full_chunk_write(capacity, lengths):
    config = json.loads((Path(__file__).parent/'data/ascend_w8a8_config.json').read_text())
    config['text_config']['num_hidden_layers'] = 1
    config['text_config']['kv_source_layer_ids'] = []
    cache = W8A8PagedCache(config, 'cpu', max_tokens=10000, requests=3, max_query_tokens=capacity)
    plane = cache.swa[0]
    end = 0
    # Unaligned full chunks cross the physical ring boundary multiple times.
    for length in lengths:
        batch = DSABatch.build([length, length], [end+length, end+length], 'cpu')
        ids = list(range(end, end+length))
        values = torch.tensor(ids+ids, dtype=torch.bfloat16)[:, None, None].expand(-1, 1, 512).contiguous()
        values[length:] += 100
        plane.write(values, cache.swa_plan(0, batch, [2, 0]))
        for slot, offset in ((2, 0), (0, 100)):
            for position in (end, end+length-1):
                required = list(range(max(0, position-127), position+1))
                expected = torch.tensor(required, dtype=torch.bfloat16)+offset
                torch.testing.assert_close(read_rows(plane, required, slot)[:, 0], expected, rtol=0, atol=0)
        assert not plane.cache[plane.resident_pages:2*plane.resident_pages].any()
        end += length
    assert plane.cache.shape[0] < 3*plane.logical_pages


def test_c2_compressed_planes_share_coordinates_and_reject_half_group():
    config = json.loads((Path(__file__).parent/'data/ascend_w8a8_config.json').read_text())
    config['text_config']['num_hidden_layers'] = 1
    config['text_config']['kv_source_layer_ids'] = [2]
    cache = W8A8PagedCache(config, 'cpu', max_tokens=32769, requests=3, max_query_tokens=16)
    positions = [0, 126, 128, 32766]
    requests = [2, 0, 0, 2]
    # Each of the three planes must receive the same host coordinates even
    # when the caller supplies a one-shot iterator.
    plans = cache.compressed_plans(2, iter(positions), iter(requests))
    assert all(torch.equal(plans[0].physical_rows, p.physical_rows) for p in plans)
    for mapping, plan in zip((cache.compressed, cache.index_key, cache.index_scale), plans):
        plane = mapping[2]
        values = torch.ones((4, 1, plane.width), dtype=plane.cache.dtype)
        plane.write(values, plan)
        assert (read_rows(plane, [0, 16383], 2) == 1).all()
        assert (read_rows(plane, [63, 64], 0) == 1).all()
    with pytest.raises(ValueError, match='completed group'):
        cache.compressed_plans(2, [32767], [0])


def test_host_plans_reject_physical_aliases_and_out_of_range():
    plane = PagedPlane(requests=2, logical_rows=1024, block_size=128,
                       width=1, dtype=torch.int8, device='cpu', resident_pages=2)
    for positions, slots in (([0, 256], [0, 0]), ([1024], [0]), ([-1], [0]), ([0], [2])):
        with pytest.raises(ValueError):
            plane.plan(positions, slots)
    plane.write(torch.tensor([[[9]], [[-7]]], dtype=torch.int8), plane.plan([127, 128], [1, 0]))
    assert read_rows(plane, [127], 1).item() == 9
    assert read_rows(plane, [128], 0).item() == -7


def test_active_tables_preserve_slot_order_without_recreating_contiguous_views(monkeypatch):
    plane = PagedPlane(requests=5, logical_rows=512, block_size=128,
                       width=1, dtype=torch.int8, device='cpu')
    original = torch.tensor
    def reject_tensor(*args, **kwargs):
        raise AssertionError('contiguous table lookup must not upload slot IDs')
    monkeypatch.setattr(torch, 'tensor', reject_tensor)
    view = plane.table([1, 2, 3])
    monkeypatch.setattr(torch, 'tensor', original)
    assert view.data_ptr() == plane.block_table[1:].data_ptr()
    torch.testing.assert_close(view, plane.block_table[1:4])
    selected = plane.table([3, 0, 3])
    torch.testing.assert_close(selected, plane.block_table[[3, 0, 3]])
    assert plane.table([3, 0, 3]) is selected


@pytest.mark.parametrize('lengths,ends', [([1, 3], [1, 4]), ([3, 2], [4, 7]),
                                       ([2048, 2048], [31488, 31616])])
def test_prepared_compression_matches_completed_c2_pairs_and_source_coordinates(lengths, ends):
    from deepinfer.model_executor.models.w8a8_compressor910c import compress_batch
    config = json.loads((Path(__file__).parent/'data/ascend_w8a8_config.json').read_text())
    config['text_config']['num_hidden_layers'] = 1
    config['text_config']['kv_source_layer_ids'] = [2, 8, 20]
    cache = W8A8PagedCache(config, 'cpu', max_tokens=32769, requests=3, max_query_tokens=2048)
    batch = DSABatch.build(lengths, ends, 'cpu')
    slots = [2, 0]
    prepared = cache.prepare_compressed(batch, slots)
    x = torch.randn(sum(lengths), 4)
    wkv, gate, norm = torch.eye(4), torch.eye(4), torch.ones(4)
    for source, (expected, device_positions, plans) in prepared.items():
        ratio = config['text_config']['compress_ratios'][source]
        pending = {slot:(torch.zeros(1, 4), torch.zeros(1, 4))
                   for q, end, slot in zip(lengths, ends, slots) if (end-q)%2}
        _, positions, requests = compress_batch(x, batch, slots, ratio, wkv,
            gate if ratio == 2 else None, norm, 1e-5, pending)
        assert tuple(positions) == expected
        assert device_positions.tolist() == positions
        reference = cache.compressed[source].plan([p//ratio for p in positions], requests)
        assert all(torch.equal(plan.physical_rows, reference.physical_rows) for plan in plans)
    assert prepared[2] is prepared[8]
