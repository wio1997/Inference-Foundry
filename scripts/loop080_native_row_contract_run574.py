#!/usr/bin/env python3
"""Source/build/package audit for the 910B sparse-attention query row order.

This is a conditional ABI certificate, not a live Runtime row-map witness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend')
KERNEL = Path('csrc/attention/sparse_attn_sharedkv/op_kernel')
BUILD = Path('csrc/build/binary/ascend910b/src/sparse_attn_sharedkv/op_kernel')
BIN = Path('csrc/build/binary/ascend910b/bin/sparse_attn_sharedkv')
PKG = Path('vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/sparse_attn_sharedkv')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exactly(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise ValueError(f'{label}: expected one {needle!r}, got {count}')


def kernel_gate(text: str, name: str) -> None:
    exactly(text, 'uint64_t tndBIdxOffsetForQ = tempLoopInfo.actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;', name)
    exactly(text, 'tensorACoreOffset = tndBIdxOffsetForQ + info.gS1Idx * constInfo.headDim;', name)
    exactly(text, 'info.attenOutOffset = tensorACoreOffset;', name)
    if 'constInfo.outputLayout == SAS_LAYOUT::TND' not in text:
        raise ValueError(f'{name}: TND output branch absent')


def vector_gate(text: str, name: str) -> None:
    if 'attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount]' not in text:
        raise ValueError(f'{name}: missing output copy using attention offset')


def negative_gate() -> None:
    sample = '''uint64_t tndBIdxOffsetForQ = tempLoopInfo.actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
tensorACoreOffset = tndBIdxOffsetForQ + info.gS1Idx * constInfo.headDim;
info.attenOutOffset = tensorACoreOffset;
constInfo.outputLayout == SAS_LAYOUT::TND'''
    kernel_gate(sample, 'positive')
    for bad in (
        sample.replace('info.attenOutOffset = tensorACoreOffset;', 'info.attenOutOffset = tensorACoreOffset + 1;'),
        sample.replace('actualSeqQPrefixSum', 'actualSeqKVPrefixSum'),
        sample.replace('SAS_LAYOUT::TND', 'SAS_LAYOUT::BSND'),
    ):
        try:
            kernel_gate(bad, 'negative')
        except ValueError:
            pass
        else:
            raise AssertionError('mutated output-row contract accepted')
    vector_gate('attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount]', 'positive')
    try:
        vector_gate('attentionOutGm[info.tensorBOffset + wsMStart * actualColumnCount]', 'negative')
    except ValueError:
        pass
    else:
        raise AssertionError('mutated vector output address accepted')


def audit(container: str | None) -> dict:
    negative_gate()
    pins = {}
    for kind in ('scfa', 'swa'):
        rel = Path(f'arch32/sparse_attn_sharedkv_{kind}_kernel.h')
        src, copied = ASC / KERNEL / rel, ASC / BUILD / rel
        if sha(src) != sha(copied):
            raise ValueError(f'{kind}: source/build copy differs')
        kernel_gate(src.read_text(), kind)
        pins[kind] = dict(source=str(src), source_sha256=sha(src),
                          build_copy=str(copied), build_copy_sha256=sha(copied))
        vec_rel = Path(f'arch32/sparse_attn_sharedkv_{kind}_block_vector.h')
        vec_src, vec_copied = ASC / KERNEL / vec_rel, ASC / BUILD / vec_rel
        if sha(vec_src) != sha(vec_copied):
            raise ValueError(f'{kind}: vector source/build copy differs')
        vector_gate(vec_src.read_text(), kind)
        pins[f'{kind}_vector'] = dict(source=str(vec_src), source_sha256=sha(vec_src),
                                     build_copy=str(vec_copied), build_copy_sha256=sha(vec_copied))
    entry = ASC / KERNEL / 'sparse_attn_sharedkv.cpp'
    entry_text = entry.read_text()
    for needle in ('SparseAttnSharedkvScfa', 'SparseAttnSharedkvSwa',
                   'static_cast<SAS_LAYOUT>(LAYOUT_T)'):
        if needle not in entry_text:
            raise ValueError(f'kernel entry missing {needle}')
    copied_entry = ASC / BUILD / 'sparse_attn_sharedkv.cpp'
    if sha(entry) != sha(copied_entry):
        raise ValueError('entry source/build copy differs')
    pins['entry'] = dict(source=str(entry), source_sha256=sha(entry),
                         build_copy_sha256=sha(copied_entry))
    template = ASC / KERNEL / 'sparse_attn_sharedkv_template_tiling_key.h'
    template_copy = ASC / BUILD / 'sparse_attn_sharedkv_template_tiling_key.h'
    template_package = (ASC / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/'
                        'custom_transformer_impl/ascendc/sparse_attn_sharedkv/sparse_attn_sharedkv_template_tiling_key.h')
    if not (sha(template) == sha(template_copy) == sha(template_package)):
        raise ValueError('template source/build/package mismatch')
    tpl = template.read_text()
    if tpl.count('ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0)') != 3 or 'ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 1)' in tpl:
        raise ValueError('FLASH_DECODE=1 selected in a supported template')
    pins['template'] = dict(source=str(template), sha256=sha(template),
                            build_copy_sha256=sha(template_copy), package_copy_sha256=sha(template_package),
                            supported_flash_decode_values=[0])
    tiling = ASC / 'csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp'
    if 'GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(tilingInfo->perfMode))' not in tiling.read_text():
        raise ValueError('host tiling no longer selects FLASH_DECODE=0')
    pins['tiling_source'] = dict(path=str(tiling), sha256=sha(tiling))
    cpp = ASC / 'csrc/torch_binding.cpp'
    binding = cpp.read_text()
    for needle in ('at::empty(q.sizes(), q.options().dtype(q.dtype()))',
                   'EXEC_NPU_CMD(aclnnSparseAttnSharedkv, q,',
                   'at::Tensor attn_out = std::get<0>(output);'):
        if needle not in binding:
            raise ValueError(f'C++ binding missing {needle}')
    pins['cpp_binding'] = dict(path=str(cpp), sha256=sha(cpp))
    device = ASC / 'vllm_ascend/device/device_op.py'
    if 'return torch.ops._C_ascend.npu_sparse_attn_sharedkv' not in device.read_text():
        raise ValueError('DeviceOperator sparse attention selector changed')
    pins['device_selector'] = dict(path=str(device), sha256=sha(device))
    dsa = ASC / 'vllm_ascend/attention/context_parallel/dsa_cp.py'
    dsa_text = dsa.read_text()
    for needle in ('layout_q="TND"', 'attn_op = DeviceOperator.get_dsa_sparse_attn_op()',
                   'local_attn_output.view(num_tokens, self.tp_size, self.n_local_heads, self.head_dim)',
                   'dist.all_to_all_single(recv, send, group=self.tp_group.device_group)'):
        if needle not in dsa_text:
            raise ValueError(f'DSA source missing {needle}')
    pins['dsa'] = dict(path=str(dsa), sha256=sha(dsa))
    binaries = []
    for manifest in sorted((ASC / PKG).glob('SparseAttnSharedkv_*.json')):
        record = json.loads(manifest.read_text())
        name = record['binFileName'] + record['binFileSuffix']
        installed = manifest.with_name(name)
        built = ASC / BIN / name
        built_manifest = ASC / BIN / manifest.name
        if record['sha256'] != sha(installed) or sha(built) != sha(installed) or sha(built_manifest) != sha(manifest):
            raise ValueError(f'build/package binary or manifest mismatch: {name}')
        if not record.get('kernelList') or not any(x.get('tilingKey') is not None for x in record['kernelList']):
            raise ValueError(f'no tiling keys: {name}')
        binaries.append(dict(name=name, package_object_sha256=sha(installed),
                             build_object_sha256=sha(built),
                             package_manifest_sha256=sha(manifest),
                             tiling_keys=sorted(x['tilingKey'] for x in record['kernelList'])))
    if len(binaries) != 2:
        raise ValueError(f'expected two BF16/F16 package variants, got {len(binaries)}')
    build_generators = []
    for generator in sorted((ASC / 'csrc/build/binary/ascend910b/gen').glob('SparseAttnSharedkv-sparse_attn_sharedkv-*.sh')):
        if '--soc_version=Ascend910B1' not in generator.read_text():
            raise ValueError(f'generated compiler target changed: {generator}')
        build_generators.append(dict(path=str(generator), sha256=sha(generator), soc_version='Ascend910B1'))
    if len(build_generators) != 2:
        raise ValueError('expected two compiled generator scripts')
    kernel_config = ASC / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/sparse_attn_sharedkv.json'
    container_binding = None
    if container:
        mounts = json.loads(subprocess.check_output(
            ['docker', 'inspect', container, '--format', '{{json .Mounts}}'], text=True))
        expected = dict(Source=str(ASC), Destination='/vllm-workspace/vllm-ascend')
        if not any(all(row.get(k) == v for k, v in expected.items()) for row in mounts):
            raise ValueError('container does not bind the audited source tree')
        module = subprocess.check_output(
            ['docker', 'exec', container, 'python3', '-c',
             'import importlib.util; print(importlib.util.find_spec("vllm_ascend").origin)'],
            text=True).strip()
        if module != '/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py':
            raise ValueError(f'container resolves a different vllm_ascend: {module}')
        container_binding = dict(container=container, source_bind=expected,
                                 imported_module=module,
                                 qualification='import path only; service op binary/tiling load unobserved')
    return dict(schema=2, status='conditional_native_query_output_row_order',
                applicability='Ascend910B source/build/package compiled for Ascend910B1; actual 910B3 dispatch unproved',
                invariant='For source-supported FLASH_DECODE=0 SCFA/SWA TND branches, output offset equals query offset for each token/head tile.',
                source_pins=pins, package_binaries=binaries,
                build_generators=build_generators,
                installed_kernel_config=dict(path=str(kernel_config), sha256=sha(kernel_config)),
                container_import_binding=container_binding,
                unresolved=['prove running 910B3 service loads this B1-targeted packaged object and selected tiling',
                            'bind actual SpecDecoding branch, TND input shape and query_start_loc',
                            'compose TP8 DSA A2A/head order, wo_b RS, MoE prepare and every layer',
                            'bind first-post-parking FULL Graph replay and W4A8 operand formats'],
                formal_current_tps=571.681, finite_resource_floor_s=None,
                finite_scheduling_floor_s=None, finite_product_tps_ceiling=None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--container')
    args = ap.parse_args()
    negative_gate()
    if not args.self_test:
        if args.output is None or args.output.exists():
            raise ValueError('new --output required')
        result = audit(args.container)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(dict(status=result['status'], packages=len(result['package_binaries']))))
    else:
        print('Run574 native row contract negative gate PASS')
