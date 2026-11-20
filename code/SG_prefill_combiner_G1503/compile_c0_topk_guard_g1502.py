"""Verify that a compile-time C0 guard removes the speculative TopK copy."""

from hashlib import sha256
import importlib.util
import inspect
import json
from pathlib import Path
import re

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compiler as compiler_module

EXPECTED_SOURCE = '6d6dbae0db0d9bfab2adc5abb8ca108338f899420ebd4c8a772ec464340d91fe'
INSTALLED = Path('/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/sparse_prefill_kernels.py')
OUT = Path('/work')
KERNEL = '_combine_topk_swa_indices_kernel'
NAME = 'sparse_prefill_kernels'
VARIANTS = (
    ('SEARCH_STEPS: tl.constexpr,\n):', 'SEARCH_STEPS: tl.constexpr,\n    READ_TOPK: tl.constexpr,\n):'),
    (
        '''    topk_vals = tl.load(
        topk_indices_ptr + topk_row + offset, mask=valid[:, None] & in_topk, other=-1
    )
''',
        '''    if READ_TOPK:
        topk_vals = tl.load(
            topk_indices_ptr + topk_row + offset,
            mask=valid[:, None] & in_topk,
            other=-1,
        )
    else:
        # Ascend can lower a masked vector load to a copy before applying
        # the column mask. The C0 caller supplies a one-column dummy tensor.
        topk_vals = tl.full((BLOCK_T, PADDED_WIDTH), -1, tl.int32)
''',
    ),
    (
        '        SEARCH_STEPS=max(num_reqs, 1).bit_length(),\n        num_warps=2,',
        '        SEARCH_STEPS=max(num_reqs, 1).bit_length(),\n        READ_TOPK=topk > 0,\n        num_warps=2,',
    ),
)


def sha(raw):
    return sha256(raw).hexdigest()


def define_candidate(path, raw):
    assert sha(raw) == EXPECTED_SOURCE
    text = raw.decode()
    for old, new in VARIANTS:
        assert text.count(old) == 1
        text = text.replace(old, new)
    with path.open('x') as f:
        f.write(text)
    return sha(text.encode())


def load(spec_name, path):
    spec = importlib.util.spec_from_file_location(spec_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, KERNEL)


def compile_case(fn, search, is_candidate, read_topk):
    constants = {
        'EXPLICIT_SWA': False,
        'COMPRESS_RATIO': 1 if not read_topk else 128,
        'WINDOW_SIZE': 128,
        'WIDTH': 128 if not read_topk else 256,
        'PADDED_WIDTH': 128 if not read_topk else 256,
        'BLOCK_T': 2,
        'SEARCH_STEPS': search,
        'swa_indices_ptr': None,
        'swa_indices_stride': 0,
        'swa_lengths_ptr': None,
    }
    if is_candidate:
        constants['READ_TOPK'] = read_topk
    signature = {}
    for name in fn.arg_names:
        if name in constants:
            continue
        signature[name] = '*i32' if name.endswith('_ptr') else 'i32'
    if not read_topk:
        constants['topk_indices_stride'] = 1
        signature.pop('topk_indices_stride')
    source = ASTSource(fn, signature, constants)
    target = GPUTarget('npu', 'Ascend910B1', 0)
    backend = compiler_module.make_backend(target)
    options = backend.parse_options({'num_warps': 2})
    context = compiler_module.ir.context()
    compiler_module.ir.load_dialects(context)
    compiler_module.buffer_ir.load_dialects(context)
    compiler_module.ascend_ir.load_dialects(context)
    backend.load_dialects(context)
    stages = {}
    backend.add_stages(stages, options)
    metadata = {'target': target, **options.__dict__}
    ttir = adapter = None
    try:
        module = source.make_ir(options, backend.get_codegen_implementation(),
                                backend.get_module_map(), context)
        first = list(stages).index('ttir')
        for name, lower in list(stages.items())[first:]:
            module = lower(module, metadata)
            if name == 'ttir':
                ttir = str(module)
            if name == 'ttadapter':
                adapter = str(module)
                break
    finally:
        context.disable_multithreading()
    assert ttir and adapter

    lines, fn_start = inspect.getsourcelines(source.fn.fn)
    topk_line = next(fn_start + i for i, row in enumerate(lines)
                     if 'topk_indices_ptr + topk_row + offset' in row)
    def mapped(text, line_op):
        locs = {name for name, location in re.findall(
            r'(#loc\d+) = loc\("[^"]+\.py":(\d+):\d+\)', text)
                if abs(int(location) - topk_line) <= 2}
        return [row.strip() for row in text.splitlines()
                if line_op in row and any(f'loc({loc})' in row for loc in locs)]

    ttir_loads = mapped(ttir, 'tt.load')
    adapter_copies = mapped(adapter, 'memref.copy')
    # Positive control C128 uses the nonzero branch and must keep a named load.
    assert bool(ttir_loads) == (not is_candidate or read_topk), (topk_line, ttir_loads)
    assert bool(adapter_copies) == (not is_candidate or read_topk), (topk_line, adapter_copies)
    tag = f'{"candidate" if is_candidate else "original"}_{search}_{"topk" if read_topk else "c0"}'
    for suffix, value in (('ttir', ttir), ('ttadapter', adapter)):
        (OUT / f'{tag}.{suffix}').write_text(value)
    return {
        'tag': tag, 'search_steps': search, 'row_tile': 2,
        'ttir_topk_loads': len(ttir_loads), 'lowered_topk_copies': len(adapter_copies),
        'ttir_sha256': sha(ttir.encode()), 'ttadapter_sha256': sha(adapter.encode()),
    }


if __name__ == '__main__':
    candidate_path = OUT / (NAME + '_guarded.py')
    candidate_sha = define_candidate(candidate_path, INSTALLED.read_bytes())
    baseline = load(NAME, INSTALLED)
    candidate = load(NAME + '_guarded', candidate_path)
    configurations = [compile_case(baseline, 1, False, False)]
    configurations.extend(compile_case(candidate, steps, True, False) for steps in range(1, 5))
    configurations.append(compile_case(candidate, 1, True, True))
    report = {'installed_source_sha256': EXPECTED_SOURCE, 'candidate_source_sha256': candidate_sha,
              'scope': 'Ascend910B1, A2, compiler only, no NPU or model request',
              'last_stage': 'ttadapter', 'binary_compiled': False, 'configurations': configurations}
    (OUT / 'compile_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))
