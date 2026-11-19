#!/usr/bin/env python3
"""Conservatively sanitize D aggregate data while preserving exact rank/kernel counts.

Only known dtype tokens and numeric shapes survive. Unknown streams are a fixed
missing-value sentinel, never an original arbitrary identifier. This does not parse
or access raw device CSV, trace, network, or services.
"""
import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_D_SHA = '0003b1e4be8cadb2a097c1489318cff5b5780a84303a35f3a8dd956bd4230732'
DTYPES = frozenset(('BOOL', 'DOUBLE', 'DT_BF16', 'DT_UNDEFINED', 'FLOAT', 'FLOAT16', 'INT32', 'INT64', 'INT8', 'UINT64', 'unknown'))
SHAPE = re.compile(r'[0-9,;\[\]()\-\s]+\Z')
NAME = re.compile(r'[A-Za-z0-9_:.<>+(),\- ]{1,512}\Z')


def dtype(value):
    if not isinstance(value, str):
        return 'unknown'
    tokens = value.split(';')
    return value if tokens and all(x.strip() in DTYPES for x in tokens) else 'unknown'


def shape(value):
    return value if isinstance(value, str) and value and (value == 'unknown' or SHAPE.fullmatch(value)) else 'unknown'


def stream(value):
    if isinstance(value, str) and (value.isdigit() or value == 'unknown'):
        return value
    raise ValueError('non-numeric stream other than fixed missing sentinel')


def merge_hist(original, transform):
    result = collections.Counter()
    for label, n in original.items():
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError('invalid histogram count')
        result[transform(label)] += n
    if sum(result.values()) != sum(original.values()):
        raise ValueError('histogram mass changed')
    return dict(sorted(result.items()))


def kernel_invariants(kernel):
    n = kernel['duration_us']['count']
    if type(n) is not int or n < 0:
        raise ValueError('invalid kernel count')
    for field in ('input_shape_counts', 'output_shape_counts', 'input_dtype_counts', 'output_dtype_counts', 'stream_id_counts'):
        if sum(kernel[field].values()) != n:
            raise ValueError('kernel histogram count mismatch')
    if sum(x['count'] for x in kernel['shape_dtype_stream_counts']) != n:
        raise ValueError('kernel signature count mismatch')
    if set(kernel['stream_duration_stats']) != set(kernel['stream_id_counts']):
        raise ValueError('stream stats do not match stream counts')
    for key, stats in kernel['stream_duration_stats'].items():
        if stats['count'] != kernel['stream_id_counts'][key]:
            raise ValueError('stream duration count mismatch')
    return n


def sanitize_d(doc):
    if doc.get('schema') != 'glm53-d-diagnostic-all-exact-kernel-names-v1' or len(doc.get('ranks', [])) != 16:
        raise ValueError('unexpected D aggregate schema')
    stats = collections.Counter(); seen = set()
    for rank in doc['ranks']:
        ident = (rank['engine'], rank['rank'])
        if ident in seen or rank['engine'] not in ('sg', 'vllm') or rank['rank'] not in range(8):
            raise ValueError('duplicate or invalid rank')
        seen.add(ident); rows_before = 0
        for kernel in rank['exact_kernels']:
            if not isinstance(kernel['kernel_name'], str) or not NAME.fullmatch(kernel['kernel_name']):
                raise ValueError('kernel name is not a safe symbol')
            n = kernel_invariants(kernel); rows_before += n
            for f in ('input_dtype_counts', 'output_dtype_counts'):
                for label, count in kernel[f].items():
                    if dtype(label) != label:
                        stats['invalid_dtype_histogram_entries'] += 1
                        stats['invalid_dtype_histogram_weight'] += count
                kernel[f] = merge_hist(kernel[f], dtype)
            for f in ('input_shape_counts', 'output_shape_counts'):
                for label, count in kernel[f].items():
                    if shape(label) != label:
                        stats['sanitized_shape_histogram_entries'] += 1
                        stats['sanitized_shape_histogram_weight'] += count
                kernel[f] = merge_hist(kernel[f], shape)
            for key in kernel['stream_id_counts']:
                stream(key)
            signatures = collections.Counter()
            for item in kernel['shape_dtype_stream_counts']:
                if set(item) != {'input_shape', 'output_shape', 'input_dtype', 'output_dtype', 'stream_id', 'count'}:
                    raise ValueError('unexpected signature fields')
                count = item['count']
                if type(count) is not int or count < 0:
                    raise ValueError('invalid signature count')
                if dtype(item['input_dtype']) != item['input_dtype'] or dtype(item['output_dtype']) != item['output_dtype']:
                    stats['invalid_dtype_signature_entries'] += 1; stats['invalid_dtype_signature_weight'] += count
                if shape(item['input_shape']) != item['input_shape'] or shape(item['output_shape']) != item['output_shape']:
                    stats['sanitized_shape_signature_entries'] += 1; stats['sanitized_shape_signature_weight'] += count
                sig = (shape(item['input_shape']), shape(item['output_shape']), dtype(item['input_dtype']), dtype(item['output_dtype']), stream(item['stream_id']))
                signatures[sig] += count
            kernel['shape_dtype_stream_counts'] = [{'input_shape': s[0], 'output_shape': s[1], 'input_dtype': s[2], 'output_dtype': s[3], 'stream_id': s[4], 'count': count} for s, count in sorted(signatures.items())]
            if kernel_invariants(kernel) != n:
                raise ValueError('kernel count changed')
            stats['kernel_groups'] += 1
        if rows_before != rank['csv_row_count'] or rows_before != rank['rows_with_kernel_name'] or rank['rows_without_kernel_name'] != 0:
            raise ValueError('rank row count mismatch')
        if len(rank['exact_kernels']) != rank['exact_kernel_name_count']:
            raise ValueError('rank kernel group mismatch')
        stats['source_csv_rows'] += rows_before
    if seen != {(engine, rank) for engine in ('sg', 'vllm') for rank in range(8)}:
        raise ValueError('missing rank')
    if stats['source_csv_rows'] != 556800 or stats['kernel_groups'] != 15040:
        raise ValueError('unexpected total rows or groups')
    return doc, dict(sorted(stats.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='caller-supplied D aggregate')
    parser.add_argument('--output', type=Path, required=True, help='new output path inside the clone')
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_D_SHA:
            raise ValueError('source SHA mismatch')
        output = args.output
        if not output.parent.resolve(strict=True).is_relative_to(ROOT.resolve()) or output.exists() or output.is_symlink():
            raise ValueError('output must be new inside the clone')
        data, stats = sanitize_d(json.loads(raw))
        encoded = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        with output.open('xb') as f:
            f.write(encoded)
        print(json.dumps({'source_sha256': SOURCE_D_SHA, 'sanitized_sha256': hashlib.sha256(encoded).hexdigest(), 'bytes': len(encoded), 'statistics': stats}, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError):
        print('sanitization failed: source, schema, privacy or output check', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
