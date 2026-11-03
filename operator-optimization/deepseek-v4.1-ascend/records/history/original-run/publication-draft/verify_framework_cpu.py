#!/usr/bin/env python3
"""Check anonymous EngineCore timing/sample arithmetic, not causal attribution."""
import json
import math
from pathlib import Path

d = json.loads((Path(__file__).parent / 'framework-cpu.json').read_text())
assert d['case_valid'] is True and d['final_health']['owner_ok'] is True
expected = {'prefix_cache_queries_total':262144, 'prefix_cache_hits_total':235904,
            'prompt_tokens_total':262144, 'generation_tokens_total':8192,
            'request_success_total':8, 'num_preemptions_total':0}
assert all(d['total_counters'][k] == v for k,v in expected.items())
w, p = d['request_window'], d['proc_window']
assert w['start_monotonic_ns'] <= p['first_ns'] < p['last_ns'] <= w['end_monotonic_ns']
assert math.isclose((w['end_monotonic_ns']-w['start_monotonic_ns'])/1e9, d['request_elapsed_s'])
assert math.isclose((p['last_ns']-p['first_ns'])/1e9, d['proc_interval_s'])
for engine, e in d['engines'].items():
    t = e['proc_leader_time']
    assert math.isclose(t['user_s']+t['system_s'], e['main_thread_cpu_s'])
    assert math.isclose(e['main_thread_cpu_s']/d['proc_interval_s'],e['main_thread_mean_cpu_cores'])
    assert math.isclose(e['main_stack_sample_count']*.02,e['main_stack_sample_weight_s'])
    pct = e['response_wait_sample_weight_s']/e['main_stack_sample_weight_s']*100
    assert math.isclose(pct,e['response_wait_inclusive_sample_pct'])
    assert e['empty_stacks'] == 0 and e['stderr_empty'] is True
    print(f'{engine}: main CPU {e["main_thread_cpu_s"]:.2f}s / {d["proc_interval_s"]:.6f}s = '
          f'{e["main_thread_mean_cpu_cores"]:.3f} cores; response-wait inclusive samples {pct:.2f}%')
print('Whole capture includes idle tail. Stack weights are not CPU seconds, device wait, removable latency or critical-path proof.')
