#!/usr/bin/env python3
"""Bound selected histogram slow-event sums, from two public summaries only.

Bucket counts are meaningful window observations only if histogram counters
were not reset within the endpoint window. Source decimals are as recorded in
the derived JSON; this cannot recover precision discarded during extraction.
No services, HTTP metrics, client records, or private identity data are read.
"""
import argparse
import hashlib
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/hist_bounds_061/summary.json'
SG = ROOT / 'evidence/stage_snapshots_056/summary_v7.json'
VLLM = ROOT / 'evidence/vllm_formal_059/summary_v2.json'
SG_SHA = 'ffb8f03c31b6d1608ec021a99d15cdf591a70880d70049b8a76f461f04774378'
VLLM_SHA = 'fa23c910c57a4221478a7dc30aedbacb3cb2304e51d5440501493668c4d1d8e6'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def decode_constant(_):
    raise ValueError('nonfinite JSON number')


def read_public(path, expected_sha):
    need(path.is_file() and not path.is_symlink(), 'invalid public source')
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'source changed')
    need(hashlib.sha256(data).hexdigest() == expected_sha, 'source SHA mismatch')
    return json.loads(data, parse_float=Decimal, parse_constant=decode_constant)


def dec(value):
    need(not isinstance(value, bool) and isinstance(value, (Decimal, int)), 'expected decimal')
    result = Decimal(value)
    need(result.is_finite(), 'nonfinite decimal')
    return result


def scope_one(items, **keys):
    rows = [item for item in items if all(item.get(k) == v for k, v in keys.items())]
    need(len(rows) == 1, 'scope is missing or ambiguous')
    return rows[0]


def hist_one(scope, **keys):
    return scope_one(scope['histograms'], **keys)


def _outward_percent(num, total, rounding):
    # Six decimal places in percent; directed rounding expands the interval.
    return format((num * Decimal(100) / total).quantize(Decimal('0.000001'), rounding=rounding), 'f')


def calculate(hist, threshold):
    """Find sharp interval bounds implied by disjoint buckets and total sum.

    Upper_s=None means +Inf; the first bucket has no lower endpoint. A finite
    bucket stands for (previous_upper, upper], including when intervening
    zero-count buckets establish the lower bound. Unknown boundaries stay
    unknown; individual observations are never assumed nonnegative.
    """
    threshold = dec(threshold)
    total = dec(hist['delta_sum_s'])
    count = hist['delta_count']
    need(type(count) is int and count > 0 and total > 0, 'invalid count or total')
    buckets = hist['buckets']
    need(isinstance(buckets, list) and len(buckets) >= 2, 'missing buckets')

    previous_upper = None
    previous_count = 0
    has_threshold = False
    fast_count = slow_count = 0
    fast_lower = fast_upper = Decimal(0)
    slow_lower = slow_upper = Decimal(0)
    fast_lower_known = fast_upper_known = True
    slow_lower_known = slow_upper_known = True

    for index, bucket in enumerate(buckets):
        need(isinstance(bucket, dict) and set(bucket) == {'upper_s', 'cumulative_count'}, 'bad bucket')
        raw_upper = bucket['upper_s']
        is_inf = raw_upper is None
        need(not is_inf or index == len(buckets) - 1, '+Inf must be last')
        need(is_inf or index < len(buckets) - 1, 'last bucket must be +Inf')
        upper = None if is_inf else dec(raw_upper)
        if upper is not None and previous_upper is not None:
            need(upper > previous_upper, 'unordered bucket boundaries')
        cumulative = bucket['cumulative_count']
        need(type(cumulative) is int and previous_count <= cumulative <= count, 'invalid cumulative count')
        in_bucket = cumulative - previous_count
        if upper == threshold:
            has_threshold = True
        if upper is not None and upper <= threshold:
            if in_bucket:
                fast_count += in_bucket
                fast_upper += in_bucket * upper
                if previous_upper is None:
                    fast_lower_known = False
                else:
                    fast_lower += in_bucket * previous_upper
        else:
            if in_bucket:
                need(previous_upper is not None and previous_upper >= threshold, 'threshold splits a nonempty bucket')
                slow_count += in_bucket
                if previous_upper is None:
                    slow_lower_known = False
                else:
                    slow_lower += in_bucket * previous_upper
                if upper is None:
                    slow_upper_known = False
                else:
                    slow_upper += in_bucket * upper
        previous_upper = upper
        previous_count = cumulative

    need(has_threshold, 'threshold is not a finite bucket boundary')
    need(previous_count == count and fast_count + slow_count == count, 'histogram count mismatch')
    # Finite summed lower boundaries are open for nonempty groups; upper
    # boundaries are closed. An empty group has the exact sum zero.
    fast_lower_strict = fast_count > 0 and fast_lower_known
    slow_lower_strict = slow_count > 0 and slow_lower_known
    if fast_lower_known and slow_lower_known:
        sample_lower = fast_lower + slow_lower
        need(total > sample_lower or (count == 0 and total == sample_lower), 'total below open lower bounds')
    if fast_upper_known and slow_upper_known:
        need(total <= fast_upper + slow_upper, 'total above upper bounds')

    # slow >= slow_lower (strict when any row has a finite open lower)
    # slow >= total - fast_upper (closed since bucket upper is inclusive)
    lower_candidates = []
    if slow_lower_known:
        lower_candidates.append((slow_lower, slow_lower_strict))
    if fast_upper_known:
        lower_candidates.append((total - fast_upper, False))
    # slow <= slow_upper (closed) and slow < total - fast_lower when fast
    # has nonzero rows and all their lower bounds are finite.
    upper_candidates = []
    if slow_upper_known:
        upper_candidates.append((slow_upper, False))
    if fast_lower_known:
        upper_candidates.append((total - fast_lower, fast_lower_strict))
    need(lower_candidates, 'no finite slow lower bound')
    lo = max(value for value, _ in lower_candidates)
    lo_open = any(open_ for value, open_ in lower_candidates if value == lo)
    if upper_candidates:
        hi = min(value for value, _ in upper_candidates)
        hi_open = any(open_ for value, open_ in upper_candidates if value == hi)
        need(lo < hi or (lo == hi and not lo_open and not hi_open), 'infeasible slow bounds')
    else:
        hi = None
        hi_open = False

    return {
        'threshold_s': str(threshold),
        'slow_condition': 'observation > threshold_s',
        'total_count': count,
        'slow_count': slow_count,
        'fast_count': fast_count,
        'total_sum_s_as_recorded': str(total),
        'fast_sum_lower_s': str(fast_lower) if fast_lower_known else None,
        'fast_sum_lower_exclusive': fast_lower_strict if fast_lower_known else False,
        'fast_sum_upper_s': str(fast_upper) if fast_upper_known else None,
        'slow_sum_lower_s': str(lo),
        'slow_sum_lower_exclusive': lo_open,
        'slow_sum_upper_s': str(hi) if hi is not None else None,
        'slow_sum_upper_exclusive': hi_open if hi is not None else False,
        'slow_share_percent_outward_6dp': {
            'lower_inclusive': _outward_percent(lo, total, ROUND_FLOOR),
            'upper_inclusive': _outward_percent(hi, total, ROUND_CEILING) if hi is not None else None,
        },
    }


def summarize():
    with localcontext() as ctx:
        ctx.prec = 100
        sg = read_public(SG, SG_SHA)
        vllm = read_public(VLLM, VLLM_SHA)
        results = []
        for size, stage_threshold, queue_threshold in (
                (32768, Decimal('25.10818416014034'), Decimal('30.0')),
                (65536, Decimal('106.74814798599316'), Decimal('60.0'))):
            sg_scope = scope_one(sg['scopes'], framework='sglang', capture='formal',
                                 input_tokens=size, phase='cold', role='decode')
            sg_hist = hist_one(sg_scope, stage='decode_bootstrap')
            result = calculate(sg_hist, stage_threshold)
            result.update({'framework': 'sglang', 'role': 'decode', 'input_tokens': size,
                           'phase': 'cold', 'metric': 'decode_bootstrap',
                           'source_summary_sha256': SG_SHA})
            results.append(result)
            v_scope = scope_one(vllm['results'], input_tokens=size, phase='cold', role='prefill')
            v_hist = hist_one(v_scope, metric='vllm:request_queue_time_seconds')
            result = calculate(v_hist, queue_threshold)
            result.update({'framework': 'vllm', 'role': 'prefill', 'input_tokens': size,
                           'phase': 'cold', 'metric': 'vllm:request_queue_time_seconds',
                           'source_summary_sha256': VLLM_SHA})
            results.append(result)
        need([x['slow_count'] for x in results] == [22, 21, 21, 24], 'unexpected slow counts')
        return {
            'schema': 'histogram-slow-event-sum-bounds-v1',
            'results': results,
            'conditions_and_limits': [
                'Histogram counter deltas represent the window event multiset only if series identity remains stable and no unobserved reset occurs.',
                'Bounds are exact only for numeric literals recorded in the derived JSON, not higher precision potentially lost in source extraction.',
                'Percent intervals are conservatively rounded outwards to six decimal places; endpoint inclusivity there intentionally overcovers exact bounds.',
                'First-bucket lower bound and +Inf upper bound remain unknown; no nonnegativity is assumed.',
                'Event cumulative sum is not wall-clock, utilization, TTFT contribution, nor evidence of causality.',
                'P and D, and two frameworks, have different semantics and unmatched events; do not add or subtract their intervals.',
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    document = summarize()
    text = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open('x', encoding='utf8') as stream:
            stream.write(text)
    if args.check:
        need(OUT.read_text(encoding='utf8') == text, 'output mismatch')
    print(json.dumps({'status': 'pass', 'scopes': len(document['results']),
                      'sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'wrote': args.write, 'checked': args.check}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError, json.JSONDecodeError):
        raise SystemExit('FAILED_CLOSED: invalid source, histogram, or output')
