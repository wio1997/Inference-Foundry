#!/usr/bin/env python3
"""Read-only, bounded JSON-array Ascend trace analysis; stdout has no input text.

The caller supplies a trace path, SHA-256, engine and rank.  This script does not
access networks, launch models, or certify source trace provenance.  Only exact
allowlisted event names and aggregate numeric fields are emitted.
"""
import argparse
import codecs
import collections
import hashlib
import json
import math
import os
import re
import sys

INDEXER = '_glm5_next_lightning_indexer_score_kernel'
GMM = 'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul'
FUSED = 'aclnnGroupedMatmulSwigluQuantWeightNZ_GroupedMatmulSwigluQuant_GroupedMatmulSwigluQuant'
ALLOWLIST = frozenset((INDEXER, GMM, FUSED))
CHUNK = 65536
MAX_BYTES = 50_000_000
MAX_EVENT_CHARS = 2_000_000


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError('invalid quantile input')
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def union(intervals):
    result, right = 0.0, None
    for a, b in sorted(intervals):
        if right is None or a > right:
            result += b - a
            right = b
        elif b > right:
            result += b - right
            right = b
    return result


def analyze_stream(stream, expected_sha256, rank, engine, max_bytes=MAX_BYTES):
    """Stream a JSON array without retaining raw payload or emitting arbitrary names."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256) or rank < 0 or engine not in ('sg', 'vllm'):
        raise ValueError('invalid expected SHA, rank or engine')
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder('utf-8')()
    jd = json.JSONDecoder()
    buf, state, total, eof = '', 'start', 0, False
    metadata, labels, chosen, steps = {}, {}, [], []

    def consume(e):
        if not isinstance(e, dict):
            return
        pid = str(e.get('pid'))
        name = e.get('name')
        args = e.get('args') if isinstance(e.get('args'), dict) else {}
        if e.get('ph') == 'M' and name == 'process_name':
            metadata[pid] = args.get('name')
        elif e.get('ph') == 'M' and name == 'process_labels':
            labels[pid] = args.get('labels')
        if e.get('ph') != 'X':
            return
        try:
            ts, dur = float(e.get('ts')), float(e.get('dur'))
        except (TypeError, ValueError):
            return
        if not math.isfinite(ts) or not math.isfinite(dur) or dur <= 0:
            return
        if e.get('cat') == 'cpu_op' and isinstance(name, str) and name.startswith('step['):
            steps.append((ts, ts + dur))
        if name in ALLOWLIST:
            tid = str(e.get('tid'))
            if not re.fullmatch(r'[0-9]+', tid):
                raise ValueError('invalid selected stream identifier')
            chosen.append((pid, tid, name, ts, dur))

    while not eof:
        block = stream.read(CHUNK)
        eof = not block
        if block:
            total += len(block)
            if total >= max_bytes:
                raise ValueError('trace exceeds size limit')
            digest.update(block)
            buf += decoder.decode(block)
        else:
            buf += decoder.decode(b'', final=True)
        pos = 0
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos == len(buf):
                break
            ch = buf[pos]
            if state == 'start':
                if ch != '[':
                    raise ValueError('expected JSON array')
                state = 'first_or_end'
                pos += 1
            elif state in ('first_or_end', 'value'):
                if ch == ']' and state == 'first_or_end':
                    state = 'done'
                    pos += 1
                    continue
                try:
                    obj, end = jd.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError('invalid or truncated JSON event')
                    break
                consume(obj)
                pos = end
                state = 'comma_or_end'
            elif state == 'comma_or_end':
                if ch == ',':
                    state = 'value'
                elif ch == ']':
                    state = 'done'
                else:
                    raise ValueError('invalid JSON array delimiter')
                pos += 1
            else:
                raise ValueError('trailing data after JSON array')
        buf = buf[pos:]
        if len(buf) > MAX_EVENT_CHARS:
            raise ValueError('JSON event exceeds size limit')
    if state != 'done' or buf.strip():
        raise ValueError('truncated JSON array')
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError('trace SHA mismatch')
    hardware = {pid for pid, name in metadata.items() if name in ('Ascend Hardware', 'NPU Hardware')}
    if not hardware:
        raise ValueError('hardware metadata absent')
    for pid in hardware:
        label = labels.get(pid)
        match = re.search(r'\bNPU\s+(\d+)\b', label) if isinstance(label, str) else None
        if not match or int(match.group(1)) != rank:
            raise ValueError('hardware rank label mismatch')
    selected = [x for x in chosen if x[0] in hardware]
    if not selected:
        raise ValueError('no selected device events')
    groups = []
    for name in sorted(ALLOWLIST):
        rows = [x for x in selected if x[2] == name]
        if not rows:
            continue  # not observed; never emit a synthetic zero
        durations = [x[4] for x in rows]
        streams = {}
        for stream_id in sorted({x[1] for x in rows}, key=int):
            vals = [x[4] for x in rows if x[1] == stream_id]
            streams[stream_id] = {'count': len(vals), 'median_us': quantile(vals, .5),
                                  'p90_us': quantile(vals, .9), 'sum_us': sum(vals)}
        bins = collections.Counter(str(int(x // 20) * 20) + '-' + str(int(x // 20) * 20 + 20) for x in durations)
        groups.append({'name': name, 'count': len(rows), 'min_us': min(durations),
                       'median_us': quantile(durations, .5), 'p90_us': quantile(durations, .9),
                       'p99_us': quantile(durations, .99), 'max_us': max(durations),
                       'sum_us_overlapping_not_wallclock': sum(durations),
                       'event_union_us': union([(x[3], x[3] + x[4]) for x in rows]),
                       'span_us': max(x[3] + x[4] for x in rows) - min(x[3] for x in rows),
                       'stream_details': streams, 'duration_histogram_20us': dict(sorted(bins.items())),
                       'event_starts_in_cpu_step_window': sum(any(a <= x[3] < b for a, b in steps) for x in rows)})
    return {'engine': engine, 'rank': rank, 'trace_bytes': total, 'trace_sha256': actual,
            'cpu_step_window_count': len(steps), 'groups': groups,
            'limits': 'selected events only; durations may overlap; no workload/shape/step matching or causal attribution'}


def main():
    parser = argparse.ArgumentParser(description='Analyze one locally supplied trace, output aggregates only')
    parser.add_argument('--trace', required=True, help='caller-supplied local JSON-array trace')
    parser.add_argument('--sha256', required=True, help='expected SHA-256 for that trace')
    parser.add_argument('--engine', required=True, choices=('sg', 'vllm'))
    parser.add_argument('--rank', required=True, type=int)
    args = parser.parse_args()
    try:
        with open(args.trace, 'rb') as f:
            before = os.fstat(f.fileno())
            if not 0 < before.st_size < MAX_BYTES:
                raise ValueError('trace exceeds size limit')
            result = analyze_stream(f, args.sha256, args.rank, args.engine)
            after = os.fstat(f.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('trace changed while reading')
    except (OSError, ValueError, UnicodeError, TypeError):
        # Do not include exception details: caller-supplied paths and payload stay private.
        print('trace analysis failed: input, hash, format, rank, or stability check', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
