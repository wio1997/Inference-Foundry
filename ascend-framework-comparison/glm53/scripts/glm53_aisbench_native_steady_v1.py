"""Use the unchanged AISBench3.1 steady calculator on native client clocks.

Contract-valid-text TTFT/TPOT and the complete run clock are reported separately.
The numeric conversion below follows the captured default_perf.py formulas;
native empty/usage SSE time points remain unchanged. Failures with cleared
native clocks make this secondary report unavailable, never shrink the primary
full-run denominator.
"""
import importlib.metadata
import math


def native_steady(outputs, concurrency):
    assert importlib.metadata.version('ais_bench_benchmark') == '3.1.0'
    report = {'implementation': 'AISBench3.1 StablePerfMetricCalculator',
              'clock': 'unchanged native AISBench time_points',
              'full_clock_is_separate': True,
              'formal_requests_supplied': len(outputs)}
    if not outputs or any(not item.get('success') for item in outputs):
        return report | {'status': 'N/A', 'reason':
                         'Native failed-request clocks may be cleared; no successful-survivor selection.'}
    from ais_bench.benchmark.calculators.stable_perf_metric_calculator import StablePerfMetricCalculator
    from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError
    import numpy as np

    details = {key: [] for key in ['id', 'start_time', 'end_time', 'success',
                                  'latency', 'ttft', 'tpot', 'itl', 'input_tokens',
                                  'output_tokens', 'generate_tokens_speed']}
    for item in outputs:
        points = item.get('time_points', [])
        if len(points) < 2 or any(not math.isfinite(float(p)) for p in points):
            return report | {'status': 'N/A', 'reason': 'Missing native streaming clocks.'}
        if any(float(a) > float(b) for a, b in zip(points, points[1:])):
            return report | {'status': 'N/A', 'reason': 'Non-monotonic native streaming clocks.'}
        latency = float(points[-1]) - float(points[0])
        count = item.get('output_tokens')
        if latency <= 0 or not isinstance(count, int) or count <= 0:
            return report | {'status': 'N/A', 'reason': 'Invalid native latency/token count.'}
        first = float(points[1]) - float(points[0])
        row = {'id': item['uuid'], 'start_time': float(points[0]),
               'end_time': float(points[-1]), 'success': True,
               'latency': latency, 'ttft': first,
               'tpot': (latency-first)/(count-1) if count > 1 else 0,
               'itl': np.diff(points[1:]) if len(points) > 2 else [],
               'input_tokens': item['input_tokens'], 'output_tokens': count,
               'generate_tokens_speed': count/latency}
        for key in details:
            details[key].append(row[key])
    calculator = StablePerfMetricCalculator()
    try:
        calculator._init_datas(details, concurrency)
        calculator.calculate()
    except AISBenchDataContentError as error:
        return report | {'status': 'N/A', 'reason': repr(error)}
    return report | {'status': 'CALCULATED',
                     'window_native_perf_counter_seconds': calculator.stage_section,
                     'sample_request_ids': [outputs[i]['uuid'] for i in calculator.stage_dict['stable']],
                     'sample_count': len(calculator.stage_dict['stable']),
                     'native_metrics': calculator.metrics,
                     'native_common_metrics': calculator.common_metrics,
                     'native_ITL_is_SSE_chunk_interval': True}
