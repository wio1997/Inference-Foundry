"""Exercise the pinned causal trim scorer and writeback on the fixed A2 CPU stack."""

import glob
from importlib.util import find_spec
import json
from pathlib import Path

import msgspec
import torch
import torch_npu

from host_geometry import contiguous_request_rows
import probe


def fixture_case(case, budget):
    probe.BUDGET = budget
    package = Path(next(iter(find_spec('sglang').submodule_search_locations)))
    native = probe.extract_native(package, torch, msgspec)
    # Supply generated BF16 Q/K directly to test the scorer independent of the upstream quantizer.
    report = probe.run_case(case, native, lambda tensor: tensor, torch, 'cpu', 145500)
    return {**report, 'fixture_score_budget_bytes': budget}


def main():
    assert not glob.glob('/dev/davinci*')
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    cases = (
        fixture_case(probe.Case('native_budget_short_equal_scores', (0, 1, 8, 17), all_equal_scores=True),
                     128 << 20),
        fixture_case(probe.Case('small_budget_all_masked_tile', (0, 0, 0, 0, 1, 3, 9, 32),
                                all_equal_scores=True), 8192),
        fixture_case(probe.Case('small_budget_nonmonotonic_fp32_weights', (4, 17, 3, 22, 1, 28, 2, 55),
                                weight_fp32=True), 8192),
        fixture_case(probe.Case('small_budget_C2_contiguous', tuple((15 + row + 1) // 2 for row in range(24))),
                     8192),
    )
    assert cases[1]['tile_count'] == 2 and cases[1]['observed_native_call_key_widths'] == [32]
    assert all(row['native_topk_indices_equal'] and row['native_writeback_equal'] for row in cases)
    geometries = contiguous_request_rows([7, 15], [21, 24], [28, 39], row_count=45)
    assert [item.row_start for item in geometries] == [0, 21]
    for ratio in (1, 2):
        for request in geometries:
            lens = [(request.prefix_length + i + 1) // ratio for i in range(request.row_count)]
            budget = 8192
            keys = max(lens)
            assert request.score_tiles(compress_ratio=ratio, heads=32, key_count=keys, budget_bytes=budget) == (
                probe.score_tiles(lens, heads=32, key_count=keys, budget_bytes=budget))
    assert contiguous_request_rows([7], [21], [29], row_count=21) is None
    assert contiguous_request_rows([7], [21], [28], row_count=20) is None
    print('G1455_A2_CAUSAL_TRIM_CPU_FIXTURE_JSON ' + json.dumps({
        'torch': str(torch.__version__), 'torch_npu': torch_npu.__version__,
        'native_source_sha256': probe.PINS, 'CPU_generated_BF16_Q_K': True,
        'physical_NPU_devices_visible': False, 'NPU_scoring_kernel_executed': False,
        'original_upstream_FP4_quantizer_executed': False, 'wall_time_conclusion_emitted': False,
        'cases': cases, 'host_tile_geometry_agrees_with_exact_visibility': True,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
