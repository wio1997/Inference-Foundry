"""Validate CPU-side candidate wiring and registry without NPU devices."""
import glob
import json
from importlib.util import find_spec
from pathlib import Path

assert not glob.glob('/dev/davinci*') and not glob.glob('/dev/davinci_manager')
import msgspec
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
import torch_npu

from probe import extract_native
from probe_c2_lightning_indexer_g1426 import Case, run_case

schema = torch._C._dispatch_find_schema_or_throw('npu::npu_lightning_indexer', '').schema()
assert str(schema).startswith('npu::npu_lightning_indexer(Tensor query, Tensor key, Tensor weights')
assert tuple(schema.returns)[0].type.kind() == 'TensorType'
assert schema.arguments[-1].name == 'return_value'
source_root = Path(next(iter(find_spec('sglang').submodule_search_locations)))
native = extract_native(source_root, torch, msgspec)


def torch_only_test_double(q, k, w, *, layout_query, layout_key, sparse_count,
                          sparse_mode, return_value):
    assert (layout_query, layout_key, sparse_mode, return_value) == ('BSND', 'BSND', 3, False)
    assert q.device.type == k.device.type == w.device.type == 'cpu'
    assert q.is_contiguous() and k.is_contiguous() and w.is_contiguous()
    assert q.dtype == k.dtype == torch.bfloat16
    assert q.shape == (1, q.shape[1], 32, 128) and k.shape[2:] == (1, 128)
    assert w.shape == q.shape[:3]
    score = torch.einsum('bshd,bkd->bshk', q, k[:, :, 0, :])
    score = (score.relu() * w.unsqueeze(-1)).sum(dim=2).float()
    q_rows, k_rows = q.shape[1], k.shape[1]
    lens = (torch.arange(q_rows) + k_rows - q_rows + 1)
    valid = torch.arange(k_rows)[None, :] < lens[:, None]
    selected = score.masked_fill(~valid, -torch.inf).topk(sparse_count, dim=-1).indices
    selected = torch.where(selected < lens[None, :, None], selected, -1)
    return selected.int().unsqueeze(2), torch.empty(0)


def deliberately_invalid(*args, **kwargs):
    indices, values = torch_only_test_double(*args, **kwargs)
    indices[0, 0, 0, 0] = -2
    return indices, values


def deliberately_changed(*args, **kwargs):
    original, values = torch_only_test_double(*args, **kwargs)
    # Change one visible selected key without generating an out-of-range index.
    original[0, 0, 0, 0] = original[0, 0, 0, 1]
    return original, values


def deliberately_reordered(*args, **kwargs):
    indices, values = torch_only_test_double(*args, **kwargs)
    last = indices[0, -1, 0, :2]
    assert last[0] >= 0 and last[1] >= 0 and last[0] != last[1]
    last.copy_(last.flip(0))
    return indices, values


cases = [
    Case('empty-visible-prefix', 0, 1),
    Case('zero-visible-leading-row', 0, 9, zero_weights=True),
    Case('odd-prefix-physical-permutation', 17, 18),
    Case('fp32-head-weights-and-odd-end', 17, 19, float_weights=True),
]
outcomes = [run_case(case, native, lambda tensor: tensor, torch,
                     torch_only_test_double, 'cpu', 0) for case in cases]
assert all(row['passed_exact_component_gate'] for row in outcomes), outcomes
error_case = Case('malformed-stock-result', 17, 18)
invalid = run_case(error_case, native, lambda tensor: tensor, torch,
                   deliberately_invalid, 'cpu', 0)
changed = run_case(error_case, native, lambda tensor: tensor, torch,
                   deliberately_changed, 'cpu', 0)
assert not invalid['passed_exact_component_gate'] and invalid['invalid_library_indices'] > 0
assert not changed['passed_exact_component_gate'] and changed['invalid_library_indices'] == 0
assert not changed['exact_native_ordered_physical_slots']
assert changed['failed_selection_diagnostics']['canonical_selection_changed_rows'] == changed['actual_fused_calls']
assert changed['failed_selection_diagnostics']['rows_with_duplicate_candidate_valid_indices'] == changed['actual_fused_calls']
reordered = run_case(error_case, native, lambda tensor: tensor, torch,
                     deliberately_reordered, 'cpu', 0)
assert reordered['passed_exact_component_gate'] and reordered['invalid_library_indices'] == 0
assert reordered['exact_native_ordered_physical_slots'] and reordered['exact_native_effective_topk_indices']
assert not glob.glob('/dev/davinci*')
print('G1426_ACTUAL_A2_CPU_WIRING_FIXTURE_JSON ' + json.dumps({
    'device': 'cpu', 'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__,
    'actual_registered_torch_schema': str(schema),
    'native_writer_loaded_from_pinned_immutable_image_files': True,
    'physical_npu_visible': False, 'stock_operator_kernel_called': False,
    'all_positive_cpu_wiring_fixtures_passed': True,
    'invalid_index_fixture_rejected': True, 'different_native_slots_fixture_rejected': True,
    'positive_cases': outcomes,
    'pre_writeback_order_change_canonicalized_by_native_writer': True,
    'failed_duplicate_diagnostics': changed['failed_selection_diagnostics'],
    'reordered_selection': reordered,
}, sort_keys=True))
