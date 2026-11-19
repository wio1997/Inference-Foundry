#!/usr/bin/env python3
"""Validate/render two already-extracted P rank0 windows; no trace or network access."""
import argparse
import collections
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'evidence/p_timeline_036'
NAMES = {
    'sglang': {'device_notify_wait': 'NOTIFY_WAIT', 'device_wait': 'EVENT_WAIT',
        'grouped_matmul': 'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul',
        'kda': 'chunk_kda_fwd', 'lightning_indexer_score': '_glm5_next_lightning_indexer_score_kernel',
        'moe_combine': 'CamMoeCombineNormal', 'moe_combine_aicpu': 'CamMoeCombineNormalMc2AicpuKernel',
        'moe_dispatch': 'CamMoeDispatchNormal', 'moe_dispatch_aicpu': 'CamMoeDispatchNormalMc2AicpuKernel',
        'notify_dispatch_aicpu': 'NotifyDispatchMc2AicpuKernel', 'sdma': 'SDMA',
        'sparse_flash_attention': 'SparseFlashAttention', 'swiglu_activation': 'ClippedSwiglu',
        'swiglu_quant': '_swiglu_quant_kernel'},
    'vllm': {'fused_grouped_matmul_swiglu_quant': 'aclnnGroupedMatmulSwigluQuantWeightNZ_GroupedMatmulSwigluQuant_GroupedMatmulSwigluQuant',
        'grouped_matmul': 'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul',
        'kda': 'aclnnChunkKdaFwd_KdaChunkForward_ChunkKdaFwd',
        'lightning_indexer_score': '_glm5_next_lightning_indexer_score_kernel',
        'sparse_flash_attention': 'SparseFlashAttention'},
}
EXCLUDED_ANCHORS = frozenset(('device_wait', 'device_notify_wait', 'notify_dispatch_aicpu', 'moe_dispatch_aicpu', 'moe_combine_aicpu'))
COLUMNS = ['category', 'stream', 'start_ms', 'duration_ms', 'visible_start_ms', 'visible_duration_ms', 'clipped_left', 'clipped_right']
EXPECTED_KEYS = {'schema_version', 'timing_precision_ms', 'source_positive_duration_filter', 'zero_duration_scope', 'framework', 'rank', 'capture_type', 'trace_sha256', 'trace_bytes', 'rank_receipt_sha256', 'source_sample_sha256', 'frozen_components', 'window', 'columns', 'event_names', 'window_event_counts', 'events'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(data, fw):
    require(set(data) == EXPECTED_KEYS and data['framework'] == fw and data['rank'] == 0 and data['schema_version'] == 1, 'schema, framework, or rank mismatch')
    require(data['timing_precision_ms'] == .001 and data['source_positive_duration_filter'] is True
            and data['zero_duration_scope'] == 'rounded to zero is not zero source duration', 'duration precision boundary missing')
    require(data['event_names'] == NAMES[fw] and data['columns'] == COLUMNS, 'whitelist or columns mismatch')
    for name in ('trace_sha256', 'rank_receipt_sha256', 'source_sample_sha256'):
        require(isinstance(data[name], str) and re.fullmatch(r'[0-9a-f]{64}', data[name]), 'invalid SHA')
    require(isinstance(data['trace_bytes'], int) and not isinstance(data['trace_bytes'], bool) and data['trace_bytes'] > 0, 'invalid source size')
    expected_capture = 'G240_frozen_export' if fw == 'sglang' else 'G148_analysis_copy_derived_from_G138'
    require(data['capture_type'] == expected_capture, 'capture provenance mismatch')
    rel = ('evidence/matched_SG_G181/profile_G240/CPU_p/review_g252/rank_0.json' if fw == 'sglang'
           else 'evidence/G4_profile_G138/CPU_p/review/rank_0.json')
    raw = (ROOT / rel).read_bytes()
    receipt = json.loads(raw)
    require(digest(raw) == data['rank_receipt_sha256'] and receipt['rank'] == 0 and receipt['trace_sha256'] == data['trace_sha256'], 'rank receipt or trace SHA mismatch')
    components = data['frozen_components']
    if fw == 'sglang':
        require(components == [], 'unexpected frozen-component claims')
    else:
        require(len(components) == 2 and {x['component'] for x in components} == {'torch.op_range', 'torch.op_mark'}, 'bad component list')
        source = receipt['raw_layout_proof']['raw_files']
        for item in components:
            require(set(item) == {'component', 'bytes', 'sha256', 'mtime_match'} and item['mtime_match'] is True, 'bad component metadata')
            hits = [x for x in source if Path(x['relative_to_capture']).name == item['component']]
            require(len(hits) == 1 and hits[0]['sha256'] == item['sha256'] and hits[0]['bytes'] == item['bytes'], 'component SHA mismatch')
    w = data['window']
    require(set(w) == {'offset_from_first_selected_event_ms', 'width_ms', 'anchor_start_count', 'event_count', 'limit'}
            and w['width_ms'] == 250 and w['limit'] == 1000 and 0 < len(data['events']) <= w['limit']
            and w['event_count'] == len(data['events']) and type(w['anchor_start_count']) is int
            and 0 < w['anchor_start_count'] <= w['event_count']
            and not isinstance(w['offset_from_first_selected_event_ms'], bool)
            and isinstance(w['offset_from_first_selected_event_ms'], (int, float))
            and math.isfinite(w['offset_from_first_selected_event_ms']) and w['offset_from_first_selected_event_ms'] >= 0,
            'sample bound mismatch')
    counts = collections.Counter()
    anchors = 0
    zeros = 0
    for row in data['events']:
        require(isinstance(row, list) and len(row) == 8, 'bad event row')
        cat, stream, start, duration, visible_start, visible_duration, left, right = row
        require(cat in NAMES[fw] and isinstance(stream, str) and re.fullmatch(r'\d+', stream), 'event label outside whitelist')
        require(all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
                    for x in (start, duration, visible_start, visible_duration)), 'nonfinite event time')
        require(type(left) is bool and type(right) is bool and duration >= 0 and visible_duration >= 0, 'invalid clip flags or duration')
        require(-.001 <= visible_start <= 250.001 and visible_start + visible_duration <= 250.001, 'visible event outside sample')
        require(abs(visible_start - max(start, 0)) <= .0011 and abs(visible_start + visible_duration - min(start + duration, 250)) <= .0021, 'clipping arithmetic mismatch')
        require((not left or start <= .001) and (not right or start + duration >= 249.998), 'bad clip direction')
        counts[cat] += 1
        anchors += cat not in EXCLUDED_ANCHORS and not left
        zeros += duration == 0
    require(dict(counts) == data['window_event_counts'] and anchors == w['anchor_start_count'], 'sample category or anchor counts mismatch')
    return {'events': len(data['events']), 'anchors': anchors, 'rounded_zero_durations': zeros,
            'width_ms': w['width_ms'], 'categories': dict(sorted(counts.items()))}


def verify_reference(data, reference_dir):
    raw = (reference_dir / (data['framework'] + '_rank0.json')).read_bytes()
    require(digest(raw) == data['source_sample_sha256'], 'original sample SHA mismatch')
    original = json.loads(raw)
    for key in ('framework', 'rank', 'trace_sha256', 'trace_bytes', 'rank_receipt_sha256', 'window', 'window_event_counts', 'frozen_components'):
        require(original[key] == data[key], 'original sample metadata mismatch')
    rows = [[e[k] for k in COLUMNS] for e in original['events']]
    require(rows == data['events'], 'original relative sample event mismatch')
    require(all(e['name'] == NAMES[data['framework']][e['category']] for e in original['events']), 'original name mismatch')


def render(data):
    panels = []
    for d in data:
        lanes = sorted({(e[0], e[1]) for e in d['events']})
        height = 50 + len(lanes) * 24
        pieces = [f'<section><h2>{html.escape(d["framework"])} · rank0</h2>',
                  f'<p>独立归零 · 250 ms · {len(d["events"])} 条白名单事件</p>',
                  f'<svg viewBox="0 0 1100 {height}" role="img" aria-label="独立相对时间轴">']
        for tick in range(0, 251, 50):
            x = 300 + tick * 3
            pieces.append(f'<line x1="{x}" y1="25" x2="{x}" y2="{height-5}" stroke="#ccd"/><text x="{x}" y="18">{tick} ms</text>')
        for i, (cat, stream) in enumerate(lanes):
            y = 40 + i * 24
            pieces.append(f'<text x="2" y="{y+8}">{html.escape(cat)} / stream {html.escape(stream)}</text>')
            for e in d['events']:
                if e[0] != cat or e[1] != stream:
                    continue
                x, length = 300 + e[4] * 3, e[5] * 3
                tip = html.escape(f'{d["event_names"][cat]} · start {e[2]} ms · duration {e[3]} ms' + (' · rounded to zero; source duration > 0' if e[3] == 0 else ''))
                if length == 0:
                    pieces.append(f'<line x1="{x:.3f}" y1="{y}" x2="{x:.3f}" y2="{y+11}" stroke="#124caa"><title>{tip}</title></line>')
                else:
                    pieces.append(f'<rect x="{x:.3f}" y="{y}" width="{length:.3f}" height="11" fill="#2780c5"><title>{tip}</title></rect>')
        pieces.append('</svg></section>')
        panels.append(''.join(pieces))
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P rank0独立时间轴</title>'
            '<style>body{font:14px sans-serif}section{margin:1em 0;overflow-x:auto}svg{min-width:1100px}text{font-size:10px}</style>'
            '<h1>GLM P rank0 单窗口局部样本</h1><p>各框架独立归零，未匹配step/请求/shape；事件可能重叠，不能累加为墙钟或根据数量排名。'
            '0.001ms量化显示0不代表源事件耗时为0。诊断样本非正式性能。</p>' + ''.join(panels) + '</html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--html', action='store_true', help='render two independent axes on stdout')
    parser.add_argument('--reference-dir', type=Path, help='optional original local sample directory, checked in memory')
    args = parser.parse_args()
    try:
        data = [json.loads((DIR / (fw + '_rank0.json')).read_text()) for fw in ('sglang', 'vllm')]
        summary = {d['framework']: validate(d, d['framework']) for d in data}
        if args.reference_dir is not None:
            for d in data:
                verify_reference(d, args.reference_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        print('timeline validation failed: input, schema, SHA or event bounds', file=sys.stderr)
        return 1
    if args.html:
        print(render(data))
    else:
        print(json.dumps({'samples': summary, 'source_reference_verified': args.reference_dir is not None,
                          'scope': 'independent selected rank0 windows; no aligned steps or causal timing'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
