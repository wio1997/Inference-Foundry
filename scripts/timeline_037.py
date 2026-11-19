#!/usr/bin/env python3
"""Validate/render two fixed D rank0 samples; no remote, trace or model access."""
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
DIR = ROOT / 'evidence/d_timeline_037'
SOURCE_SHA = '468607f30a4d6a8723478ff0ed6be6bd20e70b3ea84e2f81aefd25c8256fb455'
HTML_SHA = '526d192f3dc7cd7c25c06f7bff9d86c658aad32fb0edd25c16ff6875109fe154'
NAMES = {
    'sglang': {'QuantBatchMatmulV3': 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3',
        'MoE LowLatency Dispatch': 'MoeLowLatencyDispatchV2',
        'GroupedMatmul WeightNz': 'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul',
        'MoE Combine AICPU': 'MoeLowLatencyCombineV2Mc2AicpuKernel',
        'MoE LowLatency Combine': 'MoeLowLatencyCombineV2',
        'MoE Dispatch AICPU': 'MoeLowLatencyDispatchV2Mc2AicpuKernel',
        'lightning indexer score': '_glm5_next_lightning_indexer_score_kernel',
        'SparseFlashAttention': 'SparseFlashAttention'},
    'vllm': {'QuantBatchMatmulV3': 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3',
        'GroupedMatmul+SwiGLU+Quant fused': 'aclnnGroupedMatmulSwigluQuantWeightNZ_GroupedMatmulSwigluQuant_GroupedMatmulSwigluQuant',
        'GroupedMatmul WeightNz': 'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul',
        'lightning indexer score': '_glm5_next_lightning_indexer_score_kernel',
        'SparseFlashAttention': 'SparseFlashAttention'},
}
COLUMNS = ['category', 'stream', 'start_ms', 'duration_ms']
EXPECTED_KEYS = {'schema_version', 'framework', 'rank', 'capture_type', 'trace_sha256', 'trace_bytes', 'rank_receipt_sha256', 'source_sample_sha256', 'source_html_sha256', 'source_positive_duration_filter', 'timing_precision_ms', 'zero_duration_scope', 'source_stability', 'source_checks', 'frozen_component_checks', 'window', 'columns', 'event_names', 'event_counts', 'events', 'shape_available'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def validate(d, fw):
    require(set(d) == EXPECTED_KEYS and d['schema_version'] == 1 and d['framework'] == fw and d['rank'] == 0, 'schema/framework/rank mismatch')
    require(d['columns'] == COLUMNS and d['event_names'] == NAMES[fw], 'event whitelist mismatch')
    require(d['source_positive_duration_filter'] is True and d['timing_precision_ms'] == .001 and d['shape_available'] is False
            and d['zero_duration_scope'] == 'rounded to zero is not zero source duration', 'precision/shape boundary mismatch')
    require(d['source_sample_sha256'] == SOURCE_SHA and d['source_html_sha256'] == HTML_SHA, 'historical sample identity mismatch')
    require(d['source_stability'] == 'two source-side passes matched SHA, byte size, mtime and metadata; no trace re-read in this publication', 'source stability scope mismatch')
    for key in ('trace_sha256', 'rank_receipt_sha256'):
        require(isinstance(d[key], str) and re.fullmatch(r'[0-9a-f]{64}', d[key]), 'bad source SHA')
    require(type(d['trace_bytes']) is int and d['trace_bytes'] > 0, 'bad source bytes')
    capture = 'G240_frozen_export' if fw == 'sglang' else 'G148_analysis_copy_derived_from_G138'
    require(d['capture_type'] == capture, 'capture provenance mismatch')
    path = ('evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/rank_0.json' if fw == 'sglang'
            else 'evidence/G4_profile_G138/CPU_d/review/rank_0.json')
    raw = (ROOT / path).read_bytes(); receipt = json.loads(raw)
    require(digest(raw) == d['rank_receipt_sha256'] and receipt['rank'] == 0 and receipt['trace_sha256'] == d['trace_sha256'], 'rank receipt/trace SHA mismatch')
    checks = d['frozen_component_checks']
    if fw == 'sglang':
        require(checks == [] and d['source_checks'] == {'trace_sha_matches_frozen_G240_index': True, 'bytes_mtime_match_index': True}, 'SG source scope mismatch')
    else:
        require(d['source_checks'] == {'derived_trace_matches_G148_rank_report': True, 'frozen_raw_components': 13, 'analysis_copy_sdk_conversion_replayed': False}, 'vLLM source scope mismatch')
        require(len(checks) == 13 and all(set(x) == {'sha256', 'bytes', 'matched_existing_index'} and x['matched_existing_index'] is True for x in checks), 'bad component checks')
        prior = json.loads((ROOT / 'evidence/trace_030/vllm_rank0.json').read_text())['raw_frozen_checks']
        require(collections.Counter((x['sha256'], x['bytes']) for x in checks) == collections.Counter((x['sha256'], x['bytes']) for x in prior), 'frozen components differ from 030 receipt')
    w = d['window']
    require(set(w) == {'offset_from_first_selected_us', 'bin_ms', 'axis_end_ms', 'candidate_event_count', 'rendered_event_count', 'uniformly_downsampled_to_1000', 'limit'}, 'window metadata mismatch')
    require(w['bin_ms'] == 100 and w['limit'] == 1000 and w['uniformly_downsampled_to_1000'] is False
            and 0 < len(d['events']) <= 1000 and len(d['events']) == w['candidate_event_count'] == w['rendered_event_count'], 'window/count bound mismatch')
    offset = w['offset_from_first_selected_us']; axis = w['axis_end_ms']
    require(all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (offset, axis)) and 100 <= axis <= 110, 'invalid relative axis')
    counts = collections.Counter(); zeros = 0
    for row in d['events']:
        require(isinstance(row, list) and len(row) == 4, 'invalid event row')
        cat, stream, start, duration = row
        require(cat in NAMES[fw] and isinstance(stream, str) and re.fullmatch(r'\d+', stream), 'event name/stream outside whitelist')
        require(all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (start, duration)), 'bad event time')
        require(start <= 100.001 and start + duration <= axis + .0011, 'event outside axis')
        counts[NAMES[fw][cat]] += 1
        zeros += duration == 0
    require(dict(counts) == d['event_counts'] and min(x[2] for x in d['events']) == 0, 'event counts/zero origin mismatch')
    require(math.isclose(max(100, max(x[2] + x[3] for x in d['events'])), axis, abs_tol=.0011), 'axis endpoint mismatch')
    return {'events': len(d['events']), 'bin_ms': w['bin_ms'], 'axis_end_ms': axis,
            'rounded_zero_durations': zeros, 'event_counts': dict(sorted(counts.items()))}


def verify_reference(data, path):
    raw = path.read_bytes(); require(digest(raw) == SOURCE_SHA, 'source summary SHA mismatch')
    source = json.loads(raw)
    require(source['extracted'] is True and len(source['timeline']) == 2, 'source sample scope mismatch')
    for d in data:
        labels = 'SGLang' if d['framework'] == 'sglang' else 'vLLM'
        matches = [t for t in source['timeline'] if t['engine'] == labels]
        require(len(matches) == 1, 'source engine mismatch')
        t = matches[0]; w = d['window']
        require(t['rank'] == 0 and t['trace_sha256'] == d['trace_sha256'] and t['trace_bytes'] == d['trace_bytes']
                and t['window_bin_ms'] == w['bin_ms'] and t['window_axis_end_ms'] == w['axis_end_ms']
                and t['window_start_offset_from_first_selected_us'] == w['offset_from_first_selected_us']
                and t['candidate_events_in_window'] == w['candidate_event_count'] and t['rendered_events'] == w['rendered_event_count']
                and t['uniformly_downsampled_to_1000'] is False and t['source_checks'] == d['source_checks']
                and t['event_counts_in_full_window'] == d['event_counts'], 'source metadata mismatch')
        require(all(e['shape'] is None and e['event_name'] == NAMES[d['framework']][e['category']] for e in t['events']), 'source event/shape mismatch')
        require([[e['category'], e['stream'], e['start_ms'], e['duration_ms']] for e in t['events']] == d['events'], 'source event values differ')
        require(t['frozen_component_checks'] == d['frozen_component_checks'], 'source frozen checks mismatch')


def render(data):
    panels = []
    for d in data:
        axis = d['window']['axis_end_ms']; lanes = sorted({(e[0], e[1]) for e in d['events']})
        height = 50 + len(lanes) * 24
        pieces = [f'<section><h2>{html.escape(d["framework"])} · rank0</h2>',
                  f'<p>独立归零 · 100 ms选窗 · 坐标轴 {axis:.3f} ms · {len(d["events"])} 事件</p>',
                  f'<svg viewBox="0 0 1100 {height}" role="img" aria-label="独立相对时间轴">']
        for i in range(11):
            x = 300 + i * 75
            pieces.append(f'<line x1="{x}" y1="25" x2="{x}" y2="{height-5}" stroke="#ccd"/><text x="{x}" y="18">{axis*i/10:.1f} ms</text>')
        for i, (cat, stream) in enumerate(lanes):
            y = 40 + i * 24
            pieces.append(f'<text x="2" y="{y+8}">{html.escape(cat)} / stream {html.escape(stream)}</text>')
            for e in d['events']:
                if e[0] != cat or e[1] != stream:
                    continue
                x, width = 300 + e[2] / axis * 750, e[3] / axis * 750
                tip = html.escape(f'{d["event_names"][cat]} · start {e[2]} ms · duration {e[3]} ms' + (' · rounded to zero; source duration > 0' if e[3] == 0 else ''))
                if width == 0:
                    pieces.append(f'<line x1="{x:.3f}" y1="{y}" x2="{x:.3f}" y2="{y+11}" stroke="#124caa"><title>{tip}</title></line>')
                else:
                    pieces.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="11" fill="#2780c5"><title>{tip}</title></rect>')
        pieces.append('</svg></section>'); panels.append(''.join(pieces))
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D rank0独立时间轴</title>'
            '<style>body{font:14px sans-serif}section{margin:1em 0;overflow-x:auto}svg{min-width:1100px}text{font-size:10px}</style>'
            '<h1>GLM D rank0 单窗口局部样本</h1><p>各框架独立归零，未匹配step/请求/shape；事件可能重叠，不能累加为墙钟或根据数量排名。'
            '源时长大于0，0.001ms量化显示0不代表真实为0；诊断样本非正式性能。</p>' + ''.join(panels) + '</html>')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--html', action='store_true')
    ap.add_argument('--reference', type=Path, help='optional original local sample JSON; checked in memory')
    args = ap.parse_args()
    try:
        data = [json.loads((DIR / (fw + '_rank0.json')).read_text()) for fw in ('sglang', 'vllm')]
        summary = {d['framework']: validate(d, d['framework']) for d in data}
        if args.reference is not None:
            verify_reference(data, args.reference)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        print('timeline validation failed: input, schema, SHA or event bounds', file=sys.stderr); return 1
    if args.html:
        print(render(data))
    else:
        print(json.dumps({'samples': summary, 'source_reference_verified': args.reference is not None,
                          'scope': 'independent 100ms selected-start bins; no aligned steps or causality'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
