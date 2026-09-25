#!/usr/bin/env python3
"""Offline accounting of Run155 cohort prefill calls and fixed-slot exposure."""
import json
import statistics as st
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root / 'evidence/20260925_loop045_boundary/run155'
measure = json.loads((base / 'measured12.json').read_text())
measured_start = min(r['start'] for r in measure['requests'])
rows = []
for cohort in range(1, 6):
    ranks = []
    for rank in range(8):
        b = json.loads((base / f'boundary/rank{rank}_cohort{cohort}.json').read_text())
        r = json.loads((base / f'runtime/rank{rank}_cohort{cohort}.json').read_text())
        assert r['pass'] and r['generated_output_counts'] == [1024] * 12
        assert len(b['counts_cpu']) == r['cycles']
        calls = b['calls']
        if cohort == 5:
            # The boundary opens with the previous warmup cohort's trailing
            # execute. Select the measured request envelope explicitly.
            calls = [c for c in calls if c['t_ns']/1e9 >= measured_start]
            assert calls and len(calls) == 12
        else:
            assert calls
        parks = {slot:p['cycle'] for p in b['park_log'] for slot in p['slots']}
        assert set(parks) == set(range(12))
        n = r['cycles']
        parked = sum(n-p-1 for p in parks.values())
        counts = b['counts_cpu']
        assert all(len(v) == 12 for v in counts)
        active_hist = {str(k):sum(sum(x > 0 for x in v) == k for v in counts) for k in range(13)}
        ranks.append({
            'rank':rank,'cycles':n,'first_call_ns':calls[0]['t_ns'],
            'handoff_ns':b['handoff_ns'],
            'prefill_call_count':len(calls),
            'prefill_scheduled_tokens':sum(c['scheduled_tokens'] for c in calls),
            'first_to_handoff_s':(b['handoff_ns']-calls[0]['t_ns'])/1e9,
            'serve_s':(b['serve_end_ns']-b['serve_start_ns'])/1e9,
            'parked_slot_cycles':parked,
            'parked_fraction':parked/(n*12),
            'active_histogram':active_hist,
            'call_times_s':[round((c['t_ns']-calls[0]['t_ns'])/1e9,6) for c in calls],
            'call_tokens':[c['scheduled_tokens'] for c in calls],
        })
    assert len(set(x['prefill_scheduled_tokens'] for x in ranks)) == 1
    assert len(set(x['active_histogram']['12'] for x in ranks)) == 1
    rows.append({'cohort':cohort,'rank0':ranks[0],
        'first_to_handoff_s_median':st.median(x['first_to_handoff_s'] for x in ranks),
        'serve_s_median':st.median(x['serve_s'] for x in ranks),
        'parked_fraction_median':st.median(x['parked_fraction'] for x in ranks),
        'cycles':ranks[0]['cycles'],
        'prefill_calls':ranks[0]['prefill_call_count'],
        'prefill_tokens':ranks[0]['prefill_scheduled_tokens']})
report = {
 'run':'run156','source':'Run155 all 8 rank-cohort boundary and runtime records',
 'rows':rows,
 'conclusions':[
  'Run155 measured cohort5 prefill boundary contains a trailing warmup call; only calls at/after measured client start are counted.',
  'Fixed-shape parked slot fraction is an exposure metric. No runtime scaling or achievable speedup is inferred from it.',
  'Prefill call spacing measures start-to-start including device and host work; it cannot isolate either without trace instrumentation.'
 ]}
p = base / 'cohort_audit.json'
p.write_text(json.dumps(report,indent=2)+'\n')
for row in rows:
 x=row['rank0']
 print(row['cohort'],row['prefill_calls'],row['prefill_tokens'],round(row['first_to_handoff_s_median'],3),row['cycles'],round(row['serve_s_median'],3),round(row['parked_fraction_median'],4),x['call_tokens'],x['call_times_s'])
