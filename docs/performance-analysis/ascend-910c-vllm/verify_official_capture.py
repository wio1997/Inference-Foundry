#!/usr/bin/env python3
"""Recompute published client/metric summaries; does not rerun the private workload."""
import json
from pathlib import Path
from statistics import mean, median

data = json.loads((Path(__file__).parent / 'official-capture.json').read_text())
phases = data['phases']
assert phases['baseline']['input_sha256'] == phases['profile']['input_sha256']
durations = {}
for name, p in phases.items():
    req = p['requests']
    assert p['valid'] is True and len(req) == 8
    assert all(x['prompt_tokens'] == 32768 and x['completion_tokens'] == 1024 for x in req)
    expected = {'prefix_cache_queries_total':262144, 'prefix_cache_hits_total':235904,
                'prompt_tokens_total':262144, 'generation_tokens_total':8192,
                'request_success_total':8, 'num_preemptions_total':0}
    assert all(p['counter_deltas'][k] == v for k,v in expected.items())
    assert p['counter_deltas_by_engine']['request_success_total'] == {'0':4,'1':4}
    assert p['counter_deltas_by_engine']['generation_tokens_total'] == {'0':4096,'1':4096}
    start, end = min(x['start_ns'] for x in req), max(x['end_ns'] for x in req)
    assert (start, end) == (p['start_ns'], p['end_ns'])
    duration = (end - start) / 1e9
    durations[name] = duration
    ttft = mean((x['first_nonempty_ns']-x['start_ns'])/1e6 for x in req)
    approx_tpot = mean((x['end_ns']-x['first_nonempty_ns'])/1e6/1023 for x in req)
    samples = p['npu_samples']
    assert all(start <= s['monotonic_ns'] <= end for s in samples)
    assert all(s['error'] is None and len(s['chips']) == 16 for s in samples)
    core = [c['aicore_pct'] for s in samples for c in s['chips']]
    hbm = [c['hbm_used_display_mb'] for s in samples for c in s['chips']]
    interval = median((b['monotonic_ns']-a['monotonic_ns'])/1e9 for a,b in zip(samples,samples[1:]))
    print(f'{name}: {duration:.6f} s, {8192/duration:.3f} output token/s; '
          f'client TTFT mean {ttft:.3f} ms; approximate TPOT {approx_tpot:.3f} ms')
    print(f'  coarse NPU: {len(samples)} frames, median interval {interval:.3f} s; '
          f'AICore display min/median/max {min(core)}/{median(core)}/{max(core)}%; '
          f'HBM capacity display {min(hbm)}..{max(hbm)} MB')
print(f'Profiled wall increase {(durations["profile"]/durations["baseline"]-1)*100:.3f}%; '
      f'throughput decrease {(1-durations["baseline"]/durations["profile"])*100:.3f}%')
for x in data['frontend']:
    assert x['ph_counts'] == {'M':52,'X':1,'i':2}
    assert x['name_counts']['PyTorch Profiler (0)'] == 1
assert data['restore']['state'] == 'RESTORED'
print('Frontend envelopes are not CPU work. Arithmetic and recorded-state check only; no raw-source authenticity or hardware-bound proof.')
