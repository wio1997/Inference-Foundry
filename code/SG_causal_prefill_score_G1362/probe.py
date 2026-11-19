"""Run the causal score correctness gate on an explicitly leased Ascend device."""

import argparse
import ast
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec
import itertools
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from causal_score import causal_chunks, score_tiles


HEADS = 32
HEAD_DIM = 128
BUDGET = 128 << 20
PINS = {
    'srt/layers/attention/dsv4/v41_indexer/scoring.py':
        'fbba62901c066ffc85ff27ec44ccac2c2002a111f852b3840d85bf18958b035f',
    'srt/layers/attention/dsv4/v41_indexer/full_topk.py':
        'e7085e403db1ac0b6da9922fc6a897fee29e1e90101df64541c7b735676d7e18',
    'srt/layers/attention/dsv4/dsv41_sparse.py':
        '829c867262dfeaf1284f3fd47ff9268f4e3bc01ffce62255e92d9293d27f4197',
    'kernels/ops/quantization/fp4_fake_quant.py':
        '72770fc6635768f49d8c7dc53a053829a6a5e5acead41e555d22cbd385a4cdef',
}


@dataclass(frozen=True)
class Case:
    name: str
    host_lens: tuple[int, ...]
    weight_fp32: bool = False
    all_equal_scores: bool = False


def cases():
    yield Case('short-ties-and-empty-rows', (0, 1, 63, 64, 129), all_equal_scores=True)
    yield Case('odd-width-interleaved', (0, 7, 516, 515, 1, 509), weight_fp32=True)
    yield Case('all-masked-tile', (0,) * 512 + (4096,))
    for prefix in range(0, 32768, 8192):
        yield Case(f'32K-chunk-prefix-{prefix}', tuple((prefix + i + 1) // 2 for i in range(8192)))
    yield Case('odd-width-after-prefix', tuple((8195 + i + 1) // 2 for i in range(513)), weight_fp32=True)


def extract_native(source_root, torch, msgspec):
    sources = {}
    for relative, digest in PINS.items():
        raw = (source_root / relative).read_bytes()
        assert sha256(raw).hexdigest() == digest, relative
        sources[Path(relative).name] = ast.parse(raw)
    sparse = sources['dsv41_sparse.py']
    cls = next(item for item in sparse.body if isinstance(item, ast.ClassDef) and item.name == 'DeepseekV41Indexer')
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'scores')
    names = {'RequestScores', 'ChunkScores', '_score_chunks', 'write_prefill'}
    nodes = [item for item in sources['scoring.py'].body if getattr(item, 'name', None) in names]
    assert len(nodes) == len(names)
    tree = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], method, *nodes], type_ignores=[])
    context = {'torch': torch, 'msgspec': msgspec, '_TORCH_SCORE_BUDGET_BYTES': BUDGET}
    exec(compile(ast.fix_missing_locations(tree), '<pinned-native-score>', 'exec'), context)
    return context


def run_case(case, native, fake_quant_fp4, torch, device, seed):
    lens = case.host_lens
    rows, width = len(lens), max(lens)
    generator = torch.Generator().manual_seed(seed)
    q = (torch.randn(rows, HEADS, HEAD_DIM, generator=generator) * 0.125).bfloat16().to(device)
    k = (torch.randn(width, HEAD_DIM, generator=generator) * 0.125).bfloat16().to(device)
    weights = torch.rand(rows, HEADS, generator=generator).bfloat16().to(device)
    if case.weight_fp32:
        weights = weights.float()
    if case.all_equal_scores:
        weights.zero_()
    q, k = fake_quant_fp4(q), fake_quant_fp4(k)
    topk = min(width, 512)
    column_ids = torch.arange(width, device=device)
    # Exercise request-local selection and mapping through a nontrivial slot table.
    slot_ids = width * 3 - 2 * column_ids
    row_ids = torch.arange(rows - 1, -1, -1, device=device)
    req = native['RequestScores'](lc=width, k=topk, columns=column_ids, tok=row_ids,
                                  lens=torch.tensor(lens, device=device), slots=slot_ids)
    tiles = score_tiles(lens, heads=HEADS, key_count=width, budget_bytes=BUDGET)
    reference = SimpleNamespace()
    reference.scores = MethodType(native['scores'], reference)
    observed_widths = []

    def scores(q_tile, k_tile, weights_tile):
        observed_widths.append(k_tile.shape[0])
        return reference.scores(q_tile, k_tile, weights_tile)

    candidate = SimpleNamespace(scores=scores)
    original_chunks = native['_score_chunks'](indexer=reference, q=q, weights=weights, index_k=k, request=req)
    candidate_chunks = causal_chunks(candidate, q, weights, k, req, tiles, native['ChunkScores'],
                                     output_dtype=torch.float32)
    outputs = [SimpleNamespace(out_raw_indices=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
                               out_page_indices=torch.full((rows, topk), -1, dtype=torch.int32, device=device))
               for _ in range(2)]
    for original, trimmed in zip(original_chunks, candidate_chunks, strict=True):
        # The native final slice can end past the request. Compare the covered rows.
        assert original.rows.indices(rows) == trimmed.rows.indices(rows)
        assert torch.equal(original.tok, trimmed.tok) and torch.equal(original.lens, trimmed.lens)
        assert torch.equal(original.scores.view(torch.int32), trimmed.scores.view(torch.int32)), (case.name, original.rows)
        original_topk = original.scores.topk(topk, dim=-1, sorted=False).indices
        trimmed_topk = trimmed.scores.topk(topk, dim=-1, sorted=False).indices
        assert torch.equal(original_topk, trimmed_topk), (case.name, original.rows)
        native['write_prefill'](outputs[0], req, original, original_topk)
        native['write_prefill'](outputs[1], req, trimmed, trimmed_topk)
    assert all(torch.equal(getattr(outputs[0], field), getattr(outputs[1], field))
               for field in ('out_raw_indices', 'out_page_indices'))
    assert observed_widths == [tile.key_width for tile in tiles if tile.key_width > 0]
    return {'case': case.name, 'query_rows': rows, 'key_rows': width, 'weight_dtype': str(weights.dtype),
            'tile_count': len(tiles), 'observed_native_call_key_widths': observed_widths,
            'baseline_logical_score_cells': rows * HEADS * width,
            'candidate_logical_score_cells': sum((tile.end - tile.start) * HEADS * tile.key_width for tile in tiles),
            'FP32_scores_bitwise_equal': True, 'native_topk_indices_equal': True, 'native_writeback_equal': True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device', type=int, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    import msgspec
    import torch
    import torch_npu
    from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    root = Path(next(iter(find_spec('sglang').submodule_search_locations)))
    native = extract_native(root, torch, msgspec)
    with args.output.open('x', buffering=1) as stream:
        header = {'device': device, 'source_SHA256': PINS, 'probe_SHA256': sha256(Path(__file__).read_bytes()).hexdigest(),
                  'candidate_SHA256': sha256(Path(__file__).with_name('causal_score.py').read_bytes()).hexdigest(),
                  'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__}
        print(json.dumps(header), file=stream)
        for case, seed in zip(cases(), itertools.count(136200 + args.device * 100)):
            print(json.dumps(run_case(case, native, fake_quant_fp4, torch, device, seed)), file=stream)
        print(json.dumps({'stage': 'complete'}), file=stream)


if __name__ == '__main__':
    main()
