"""Compare stock A2 LightningIndexer against the pinned SGLang C2 Prefill path."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec
import json
from pathlib import Path
from time import perf_counter
from types import MethodType, SimpleNamespace

from c2_lightning_indexer_geometry_g1425 import c2_causal_groups
from probe import HEAD_DIM, HEADS, PINS, extract_native


@dataclass(frozen=True)
class Case:
    name: str
    prefix: int
    query: int
    zero_weights: bool = False
    float_weights: bool = False


SUITES = {
    'short': (
        Case('empty-visible-prefix', 0, 1),
        Case('short-equal-scores', 0, 33, zero_weights=True),
        Case('odd-prefix', 17, 18),
        Case('odd-query-with-fp32-weights', 8195, 513, float_weights=True),
        Case('top-k-ties', 1200, 19, zero_weights=True),
    ),
    '32k': tuple(Case(f'32K-chunk-{p}', p, 8192) for p in range(0, 32768, 8192)),
}


def fused_indices(torch, call, q, k, weights, prefix, topk):
    """Compute candidate request-local indices and preserve raw launch outputs."""
    width = k.shape[0]
    assert width == (prefix + q.shape[0]) // 2
    indices = torch.full((q.shape[0], topk), width, dtype=torch.int64, device=q.device)
    calls = []
    for group in c2_causal_groups(prefix, q.shape[0]):
        count = min(topk, group.key_prefix_rows)
        group_q = q[group.query_slice].contiguous().unsqueeze(0)
        group_k = k[:group.key_prefix_rows].contiguous().unsqueeze(0).unsqueeze(2)
        group_w = weights[group.query_slice].contiguous().unsqueeze(0)
        raw, _ = call(group_q, group_k, group_w, layout_query='BSND', layout_key='BSND',
                      sparse_count=count, sparse_mode=3, return_value=False)
        assert raw.shape == (1, group.query_rows, 1, count) and raw.dtype == torch.int32
        returned = raw[0, :, 0, :]
        calls.append((group, returned))
        # The native writer uses only idx < lens and indexes its slot table.
        # Convert library -1 invalids into a positive sentinel outside each lens.
        indices[group.query_slice, :count] = torch.where(returned < 0, width, returned).long()
    return indices, calls


def outputs(torch, rows, topk, device):
    return SimpleNamespace(
        out_raw_indices=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
        out_page_indices=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
    )



def selection_diagnostics(expected, actual):
    reference = expected.out_raw_indices.cpu()
    candidate = actual.out_raw_indices.cpu()
    changed = (reference != candidate).any(dim=-1)
    slots_changed = (expected.out_page_indices.cpu() != actual.out_page_indices.cpu()).any(dim=-1)
    # The pinned writer sorts valid local indices and masks the remaining tail.
    duplicates = ((candidate[:, 1:] >= 0) & (candidate[:, 1:] == candidate[:, :-1])).any(dim=-1)
    changed_positions = changed.nonzero().flatten()[:8].tolist()
    return {
        'canonical_selection_changed_rows': int(changed.sum()),
        'ordered_physical_slot_changed_rows': int(slots_changed.sum()),
        'rows_with_duplicate_candidate_valid_indices': int(duplicates.sum()),
        'examples': [{
            'row': row,
            'reference_valid': int((reference[row] >= 0).sum()),
            'candidate_valid': int((candidate[row] >= 0).sum()),
            'reference_first_indices': reference[row, :8].tolist(),
            'candidate_first_indices': candidate[row, :8].tolist(),
        } for row in changed_positions],
    }


def run_case(case, native, quantize, torch, call, device, repeat):
    rows = case.query
    width = (case.prefix + rows) // 2
    topk = min(width, 512)
    generator = torch.Generator().manual_seed(142600 + case.prefix + rows)
    q = quantize((torch.randn(rows, HEADS, HEAD_DIM, generator=generator) * .125).bfloat16().to(device))
    if width:
        k = quantize((torch.randn(width, HEAD_DIM, generator=generator) * .125).bfloat16().to(device))
    else:
        k = torch.empty((0, HEAD_DIM), dtype=torch.bfloat16, device=device)
    w = torch.rand(rows, HEADS, generator=generator).bfloat16().to(device)
    if case.zero_weights:
        w.zero_()
    if case.float_weights:
        w = w.float()
    columns = torch.arange(width, device=device)
    token_rows = torch.arange(rows - 1, -1, -1, device=device)
    lens = (torch.arange(rows, device=device) + case.prefix + 1) // 2
    request = native['RequestScores'](lc=width, k=topk, columns=columns,
                                     tok=token_rows, lens=lens, slots=width * 3 - 2 * columns)
    model = SimpleNamespace()
    model.scores = MethodType(native['scores'], model)

    def baseline():
        output = outputs(torch, rows, topk, device)
        for chunk in native['_score_chunks'](indexer=model, q=q, weights=w, index_k=k, request=request):
            idx = chunk.scores.topk(topk, dim=-1, sorted=False).indices
            native['write_prefill'](output, request, chunk, idx)
        return output

    def candidate():
        output = outputs(torch, rows, topk, device)
        idx, calls = fused_indices(torch, call, q[token_rows], k, w[token_rows], case.prefix, topk)
        chunk = SimpleNamespace(tok=token_rows, lens=lens)
        native['write_prefill'](output, request, chunk, idx)
        return output, calls

    if width:
        expected = baseline()
        actual, launches = candidate()
    else:
        expected = outputs(torch, rows, topk, device)
        empty, launches = fused_indices(torch, call, q, k, w, case.prefix, topk)
        assert empty.shape == (rows, 0) and not launches
        actual = outputs(torch, rows, topk, device)

    invalid = 0
    for group, indices in launches:
        visible = lens[group.query_slice, None]
        invalid += int(((indices < -1) | (indices >= visible)).sum().item())
    exact_indices = torch.equal(expected.out_raw_indices, actual.out_raw_indices)
    exact_slots = torch.equal(expected.out_page_indices, actual.out_page_indices)
    passed = invalid == 0 and exact_indices and exact_slots
    result = {
        'case': case.name, 'prefix_tokens': case.prefix,
        'query_rows': rows, 'compressed_key_rows': width,
        'head_weight_dtype': str(w.dtype),
        'actual_fused_calls': len(launches), 'invalid_library_indices': invalid,
        'exact_native_effective_topk_indices': exact_indices,
        'exact_native_ordered_physical_slots': exact_slots,
        'changed_raw_index_rows': int((expected.out_raw_indices != actual.out_raw_indices).any(dim=1).sum().item()),
        'passed_exact_component_gate': passed,
    }
    if not passed:
        result['failed_selection_diagnostics'] = selection_diagnostics(expected, actual)
    if passed and width:
        times = {'native_ms': [], 'candidate_ms': []}
        for index in range(repeat):
            pair = (('native_ms', baseline), ('candidate_ms', candidate))
            if index % 2:
                pair = pair[::-1]
            for label, function in pair:
                torch.npu.synchronize()
                when = perf_counter()
                function()
                torch.npu.synchronize()
                times[label].append((perf_counter() - when) * 1000)
        result['synchronized_component_pass_times'] = times
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, choices=range(8), required=True)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, choices=range(1, 8), default=3)
    args = parser.parse_args()

    import msgspec
    import torch
    import torch_npu
    from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    schema = str(torch._C._dispatch_find_schema_or_throw('npu::npu_lightning_indexer', '').schema())
    call = torch.ops.npu.npu_lightning_indexer.default
    root = Path(next(iter(find_spec('sglang').submodule_search_locations)))
    native = extract_native(root, torch, msgspec)
    with args.output.open('x', buffering=1) as stream:
        print(json.dumps({
            'stage': 'input_contract', 'suite': args.suite, 'device': args.device,
            'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__,
            'torch_operator_schema': schema, 'native_source_sha256': PINS,
            'probe_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'geometry_sha256': sha256(Path(__file__).with_name('c2_lightning_indexer_geometry_g1425.py').read_bytes()).hexdigest(),
        }), file=stream)
        torch.npu.set_device(args.device)
        device = f'npu:{args.device}'
        passed = True
        for case in SUITES[args.suite]:
            row = run_case(case, native, fake_quant_fp4, torch, call, device, args.repeat)
            print(json.dumps(row), file=stream)
            passed &= row['passed_exact_component_gate']
        print(json.dumps({'stage': 'complete', 'all_exact': passed}), file=stream)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
