"""Find the generated Ascend copy that precedes the logical TopK column mask."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audit(group):
    ttir_path = group / 'kernel.ttir'
    adapter_path = group / 'kernel.ttadapter'
    meta = json.loads((group / 'kernel.json').read_text())
    assert meta['target']['backend'] == 'npu'
    assert meta['target']['arch'] == 'Ascend910B1'
    assert meta['kernel_name'] == '_combine_topk_swa_indices_kernel'

    def locations(text):
        return {label: int(line) for label, line in re.findall(
            r'(#loc\d+) = loc\("[^"]+?sparse_prefill_kernels\.py":(\d+):\d+\)', text)}

    def source_line(text, line):
        return [row.strip() for row in text.splitlines()
                if any(f'loc({label})' in row for label, value in locations(text).items() if value == line)]

    ttir = ttir_path.read_text()
    assert not any('tt.load' in row for line in (109, 128, 131) for row in source_line(ttir, line))
    assert any('tt.load' in row for row in source_line(ttir, 121))
    search_steps = sum('tt.load' in row for row in source_line(ttir, 87))
    assert search_steps in (1, 2, 3, 4)

    target = adapter_path.read_text()
    copy_rows = source_line(target, 121)
    copies = [row for row in copy_rows if row.startswith('memref.copy')]
    assert len(copies) == 1
    copied, written = re.search(r'memref.copy (%\w+), (%\w+) :', copies[0]).groups()
    assert 'memref<?x128xi32, strided<[1, 1],' in copies[0]
    subview = next(row for row in copy_rows if row.startswith(f'{copied} = memref.subview'))
    region = re.search(r'memref.subview (%\w+)\[', subview)[1]
    assert re.search(r'\[%\w+, 128\]', subview)
    layout_line = next(row.strip() for row in target.splitlines()
                       if row.strip().startswith(f'{region} = memref.reinterpret_cast'))
    assert re.search(r'sizes: \[2, 128\], strides: \[1, 1\]', layout_line)
    written_def = next(row for row in copy_rows if row.startswith(f'{written} = memref.subview'))
    allocation = re.search(r'memref.subview (%\w+)\[', written_def)[1]
    tensor_row = next(row for row in copy_rows if 'bufferization.to_tensor ' + allocation + ' ' in row)
    tensor_var = tensor_row.split()[0]
    masked = [row for row in copy_rows if 'arith.select ' in row and f', {tensor_var},' in row]
    assert len(masked) == 1
    assert target.index(copies[0]) < target.index(masked[0])
    return {'search_steps': search_steps, 'num_warps': meta['num_warps'],
            'explicit_swa_load_present': False, 'tile_rows': 2, 'physical_copy_columns': 128,
            'physical_source_row_stride_i32': 1, 'column_mask_applied_after_copy': True,
            'ttir_sha256': digest(ttir_path), 'ttadapter_sha256': digest(adapter_path)}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('cache_export', type=Path)
    cases = sorted((audit(d) for d in cli.parse_args().cache_export.iterdir()),
                   key=lambda row: row['search_steps'])
    assert [row['search_steps'] for row in cases] == [1, 2, 3, 4]
    print(json.dumps({'cases': cases, 'C0_logical_TopK_columns': 1,
                      'last_row_elements_outside_logical_storage': 127}, sort_keys=True))
