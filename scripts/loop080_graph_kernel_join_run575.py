#!/usr/bin/env python3
"""Join admitted Run502 FULL96 Target graph task names to Run574 package objects."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/data/wio/Inference_Foundry')
RUN502 = ROOT / 'evidence/20260927_loop079_identity/run502/b_candidate'
RUN574 = ROOT / 'evidence/20260928_loop080_bound/run574/native_row_contract.json'
RUN574_SHA = 'ff86c527cd191effcb05740a90e483725eae8916e26c8ace8d71937c73500fc2'
ASC = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend')
PKG = ASC / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/sparse_attn_sharedkv'
CANN_KEY_HEADER = '/usr/local/Ascend/cann-9.1.0/aarch64-linux/asc/include/tiling/template_argument.h'
CANN_KEY_HEADER_SHA = '0993117c58ee7bff1e8ef972c63504e9f1a38375ed3618b63baed99927d3ac79'
CONTAINER = 'vllm-ascend26-dsv4f-w4a8'
TASK = re.compile(r'^(SparseAttnSharedkv_[0-9a-f]{32})_(\d+)$')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_name(name: str, packages: dict[str, dict]) -> tuple[str, int]:
    match = TASK.fullmatch(name)
    if match is None:
        raise ValueError(f'not an exact package task name: {name}')
    base, key = match.group(1), int(match.group(2))
    package = packages.get(base)
    if package is None or key not in package['tiling_keys']:
        raise ValueError(f'task object/tiling key absent from package: {name}')
    return base, key


def decode_key(key: int) -> dict:
    # Installed CANN9.1 FastEncodeTilingKeyDirect appends fields in declaration
    # order; the sparse-attention declaration is bool1, q4, kv4, mode4.
    flash, q, kv, mode = key & 1, (key >> 1) & 15, (key >> 5) & 15, (key >> 9) & 15
    if key >> 13 or flash != 0 or q not in (0, 1) or kv not in (0, 1, 2) or mode not in (0, 1, 2):
        raise ValueError(f'unsupported sparse-attention tiling key: {key}')
    return dict(flash_decode=flash, query_layout=('BSND', 'TND')[q],
                kv_layout=('PA_ND', 'BSND', 'TND')[kv],
                template_mode=('SWA', 'CFA', 'SCFA')[mode])


def self_test() -> None:
    fake = {'SparseAttnSharedkv_' + 'a' * 32: {'tiling_keys': [2, 514]}}
    assert join_name('SparseAttnSharedkv_' + 'a' * 32 + '_514', fake)[1] == 514
    for bad in ('SparseAttnSharedkv_' + 'b' * 32 + '_514',
                'SparseAttnSharedkv_' + 'a' * 32 + '_515',
                'SparseAttnSharedkv_' + 'a' * 32 + '_-2',
                'SparseAttnSharedkv_' + 'a' * 32 + '_514_other'):
        try:
            join_name(bad, fake)
        except ValueError:
            pass
        else:
            raise AssertionError(f'bad Graph task admitted: {bad}')
    assert decode_key(2) == dict(flash_decode=0, query_layout='TND', kv_layout='PA_ND', template_mode='SWA')
    assert decode_key(514)['template_mode'] == 'CFA'
    assert decode_key(1026)['template_mode'] == 'SCFA'
    for bad in (3, 4, 66, 1538, 8194):
        try:
            decoded = decode_key(bad)
            if decoded['query_layout'] == 'TND' and decoded['kv_layout'] == 'PA_ND':
                raise AssertionError(f'bad key accepted as selected branch: {bad}')
        except ValueError:
            pass


def audit() -> dict:
    self_test()
    if sha(RUN574) != RUN574_SHA:
        raise ValueError('Run574 reviewed contract bytes changed')
    contract = json.loads(RUN574.read_text())
    if contract['schema'] != 2 or contract['status'] != 'conditional_native_query_output_row_order':
        raise ValueError('Run574 certificate changed')
    packages = {row['name'][:-2]: row for row in contract['package_binaries']}
    if len(packages) != 2:
        raise ValueError('expected two packaged object variants')
    template = contract['source_pins']['template']
    if sha(Path(template['source'])) != template['sha256']:
        raise ValueError('reviewed template source changed')
    template_text = Path(template['source']).read_text()
    for needle in ('ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1)',
                   'ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW',
                   'SAS_LAYOUT_BSND,', 'SAS_LAYOUT_TND)',
                   'ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW',
                   'SAS_LAYOUT_PA_ND, SAS_LAYOUT_BSND, SAS_LAYOUT_TND',
                   'ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW',
                   'SWA_TEMPLATE,', 'CFA_TEMPLATE, SCFA_TEMPLATE'):
        if needle not in template_text:
            raise ValueError(f'template declaration changed: {needle}')
    header = subprocess.check_output(['docker', 'exec', CONTAINER, 'cat', CANN_KEY_HEADER])
    if hashlib.sha256(header).hexdigest() != CANN_KEY_HEADER_SHA:
        raise ValueError('reviewed installed CANN tiling-key encoder bytes changed')
    header_text = header.decode()
    for needle in ('#define ASCENDC_TPL_1_BW 1',
                   '#define ASCENDC_TPL_4_BW 4',
                   '#define ASCENDC_TPL_BOOL_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW',
                   'tilingKey |= (encodeVal << totalBits);',
                   'totalBits += param.bitWidth;',
                   'uint64_t index = iter - param.vals.cbegin();',
                   'encodeVal = index;'):
        if needle not in header_text:
            raise ValueError(f'installed CANN key encoder changed: {needle}')
    variant_dtype = {}
    for name, row in packages.items():
        manifest = PKG / (name + '.json')
        if sha(manifest) != row['package_manifest_sha256']:
            raise ValueError(f'package manifest changed: {name}')
        support = json.loads(manifest.read_text())['supportInfo']
        q_dtype = next(x['dtype'] for x in support['inputs'] if x['name'] == 'q')
        out_dtype = next(x['dtype'] for x in support['outputs'] if x['name'] == 'attn_out')
        if q_dtype != out_dtype or q_dtype not in ('bfloat16', 'float16'):
            raise ValueError(f'package q/output dtype mismatch: {name}')
        variant_dtype[name] = q_dtype
    admission_path = RUN502 / 'graph_dump_admission.json'
    admission = json.loads(admission_path.read_text())
    if admission.get('ranks') != 8 or not admission.get('nonempty_json') or not admission.get('meta_join'):
        raise ValueError('Run502 admission is incomplete')
    rows = []
    for rank in range(8):
        graph_path = RUN502 / f'graph_dump/rank{rank}_cohort5_acl_graph.json'
        meta_path = RUN502 / f'graph_dump/rank{rank}_cohort5_acl_graph.meta.json'
        graph = json.loads(graph_path.read_text())
        meta = json.loads(meta_path.read_text())
        if not isinstance(graph, list) or meta['rank'] != rank or meta['cohort'] != 5:
            raise ValueError(f'rank/cohort/Graph format mismatch: {rank}')
        if meta['sha256'] != sha(graph_path) or meta['bytes'] != graph_path.stat().st_size:
            raise ValueError(f'Graph SHA/size mismatch: {rank}')
        if meta['node_count'] != len(graph) or meta['task_count'] != len(graph):
            raise ValueError(f'Graph task count mismatch: {rank}')
        if meta['run_id'] != admission['run_id'] or meta['capture_generation'] < 1:
            raise ValueError(f'Graph acquisition/generation mismatch: {rank}')
        if 'num_tokens=96, num_reqs=12' not in meta['batch_descriptor']:
            raise ValueError(f'not FULL96 Target: {rank}')
        if not meta.get('output_owner_sha256') or not meta.get('entry_id') or not meta.get('graph_id'):
            raise ValueError(f'Graph owner/entry absent: {rank}')
        selected = [event for event in graph if event.get('name', '').startswith('SparseAttnSharedkv_')]
        if len(selected) != 43:
            raise ValueError(f'expected 43 sparse attention tasks, got {len(selected)} on rank{rank}')
        names = Counter()
        task_ids = set()
        for event in selected:
            base, key = join_name(event['name'], packages)
            branch = decode_key(key)
            if variant_dtype[base] != 'bfloat16' or branch != dict(
                    flash_decode=0, query_layout='TND', kv_layout='PA_ND',
                    template_mode=branch['template_mode']):
                raise ValueError(f'Graph selects unexpected dtype/layout branch: rank{rank} {event["name"]}')
            args = event.get('args', {})
            if args.get('Task Type') != 'KERNEL_MIX_AIC' or args.get('Model Id') not in meta['native_model_ids']:
                raise ValueError(f'wrong native model/task class: rank{rank}')
            task_id = (args.get('Model Id'), args.get('Stream Id'), args.get('Task Id'))
            if task_id in task_ids:
                raise ValueError(f'duplicate native task identity: rank{rank}')
            task_ids.add(task_id)
            names[(base, key)] += 1
        rows.append(dict(rank=rank, run_id=meta['run_id'], cohort=5,
                         capture_generation=meta['capture_generation'],
                         graph_sha256=sha(graph_path), graph_meta_sha256=sha(meta_path),
                         graph_task_count=len(graph), sparse_attention_tasks=len(selected),
                         task_object_key_counts={f'{base}_{key}': n for (base, key), n in sorted(names.items())},
                         native_model_ids=meta['native_model_ids'],
                         batch_descriptor=meta['batch_descriptor'],
                         output_owner_sha256=meta['output_owner_sha256'],
                         dump_completion=meta['completion']))
    first = rows[0]['task_object_key_counts']
    if any(row['task_object_key_counts'] != first for row in rows):
        raise ValueError('sparse attention object/key distribution differs by rank')
    return dict(schema=1, status='selected_graph_kernel_name_to_installed_package_join',
                run502_admission_sha256=sha(admission_path),
                run574_contract_sha256=sha(RUN574),
                selected_task_count=sum(row['sparse_attention_tasks'] for row in rows),
                object_key_counts_per_rank=first, ranks=rows,
                decoded_selected_keys={str(key): decode_key(key) for (_, key) in sorted(
                    {join_name(name, packages) for name in first})},
                selected_package_q_output_dtype='bfloat16',
                installed_cann_key_encoder=dict(path=CANN_KEY_HEADER, sha256=hashlib.sha256(header).hexdigest()),
                packages={name: dict(object_sha256=row['package_object_sha256'],
                                     manifest_sha256=row['package_manifest_sha256'])
                          for name, row in packages.items()},
                interpretation='Run502 same-process selected FULL96 Target graph exporter names 43 native sparse attention tasks per rank that join Run574 packaged BF16 object and CANN9.1-decoded FLASH_DECODE=0/TND/PA_ND tiling keys. Name+manifest join is not a memory-level loaded-byte proof.',
                unresolved=['actual op-api and tiling library loaded SHA at Run502 capture',
                            'source-to-object reproducible build and B1-targeted binary behavior on 910B3',
                            'actual per-task dynamic query prefix/head metadata and layer ordinal mapping',
                            'Run502 graph dump after later replays is structural, not cycle64 dynamic parameters',
                            'first-post-parking same-trajectory row/expert/weight witness'],
                formal_current_tps=571.681, finite_resource_floor_s=None,
                finite_scheduling_floor_s=None, finite_product_tps_ceiling=None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    self_test()
    if args.self_test:
        print('Run575 Graph task-name negative gate PASS')
    else:
        if args.output is None or args.output.exists():
            raise ValueError('new --output required')
        result = audit()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(dict(status=result['status'], tasks=result['selected_task_count'],
                              names=result['object_key_counts_per_rank'])))
