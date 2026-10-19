"""Original unprofiled arrivals; delivery overlap is not engine batch membership."""
from pathlib import Path
import hashlib, json, statistics

r = Path(__file__).resolve().parent
p = next((r/'prefix-count-c4-v1').glob('*/metrics.json'))
x = json.loads(p.read_text()); rows = [z for z in x['rows'] if z['phase'] == 'formal']
assert len(rows) == 384

def quantile(values, fraction):
    a = sorted(values); i = (len(a)-1)*fraction
    return a[int(i)]+(a[min(int(i)+1, len(a)-1)]-a[int(i)])*(i%1)

def overlap(row):
    first, last = row['arrival_ns'][0], row['arrival_ns'][-1]
    peers = [z for z in rows if z is not row and z['block'] == row['block']]
    amount = sum(max(0, min(last, z['arrival_ns'][-1])-max(first, z['arrival_ns'][0])) for z in peers)
    return amount/(last-first)

cases = []
for case in sorted({z['case_id'] for z in rows}):
    per = {m: [z for z in rows if z['case_id'] == case and z['mode'] == m] for m in ['native', 'adaptive']}
    budget = len(per['native'][0]['token_ids'])
    milestones = sorted({1, 2, 8, 16, 32, 64, budget})
    curves = {m: {str(n): quantile([(z['arrival_ns'][n-1]-z['sent_ns'])/1e6 for z in rr], .95) for n in milestones} for m, rr in per.items()}
    gap = {m: quantile([max(b-a for a, b in zip(z['arrival_ns'], z['arrival_ns'][1:]))/1e6 for z in rr], .95) for m, rr in per.items()}
    overlaps = {m: statistics.mean(overlap(z) for z in rr) for m, rr in per.items()}
    cases.append(dict(case=case, output_tokens=budget, p95_delivery_ms_by_token=curves,
                      p95_delivery_ratios={str(n): curves['adaptive'][str(n)]/curves['native'][str(n)] for n in milestones},
                      p95_max_observed_interarrival_gap_ms=gap, mean_other_delivery_intervals_overlapping=overlaps))
result = dict(completed=True, formal_rows=384, source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), cases=cases,
              scope='All original unprofiled rows retained. Per-token delivery percentiles and overlap observed at the D-host client; repeated timestamps reflect batched token chunks. Delivery intervals are not device execution or actual scheduler admission/batch membership. Closed-loop arrivals depend on earlier completions; a fixed-arrival replay is a new control condition, not a repeat or reinterpretation of the rejected joint gate.')
(r/'prefix-count-arrival-analysis-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps([dict(case=z['case'], p95_delivery_ratios=z['p95_delivery_ratios'], p95_max_gap=z['p95_max_observed_interarrival_gap_ms'], delivery_overlap=z['mean_other_delivery_intervals_overlapping']) for z in cases], indent=2))
