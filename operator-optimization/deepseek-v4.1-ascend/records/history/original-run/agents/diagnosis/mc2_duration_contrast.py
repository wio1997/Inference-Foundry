#!/usr/bin/env python3
"""Paired per-rank duration sensitivity; never treats cross-rank timestamps as causal."""
import csv, hashlib, json, re, statistics
from pathlib import Path

root = Path(__file__).resolve().parents[2]
timeline = root / 'mc2-preceding-analysis/rank_layer0_timeline.csv'
calls = root / 'mc2-arrival-analysis-v2/rank_target_calls.csv'
out = Path(__file__).with_name('mc2-duration-contrast.json')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
with timeline.open() as f: samples = [r for r in csv.DictReader(f) if r['interior_runs_1_2_3'] == 'True']
with calls.open() as f: rows = list(csv.DictReader(f))
lookup = {(r['rank'], int(r['target_call_ordinal'])): r for r in rows}
assert len(lookup) == len(rows) and len(samples) == 48
grouped = {}
for s in samples:
    rank = s['rank']; forward = int(s['target_forward_ordinal'])
    match = re.search(r'^dp([01])_.*_tp([0-7])_', rank); assert match
    dp, tp = map(int, match.groups())
    r0 = lookup[(rank, forward * 40)]; r1 = lookup[(rank, forward * 40 + 1)]
    assert r0['source_layer'] == '0' and r1['source_layer'] == '1'
    assert r0['physical_input_rows'] == r1['physical_input_rows'] == '3' and r0['model_id'] == r1['model_id'] == '49'
    assert abs(float(r0['duration_us']) - float(s['mc2_duration_us'])) < .001
    grouped[(forward, tp, dp)] = {'layer0': float(r0['duration_us']), 'layer1': float(r1['duration_us'])}
assert len(grouped) == 48
per_forward = []
for forward in sorted({k[0] for k in grouped}):
    pairs = [(grouped[forward,tp,0], grouped[forward,tp,1]) for tp in range(8)]
    diff0 = [b['layer0']-a['layer0'] for a,b in pairs]
    diff1 = [b['layer1']-a['layer1'] for a,b in pairs]
    # Double difference removes a persistent group-specific duration difference only as a descriptive contrast.
    per_forward.append({'forward_ordinal':forward, 'tp_pairs':8,
      'layer0_dp0_median_us': statistics.median(a['layer0'] for a,b in pairs),
      'layer0_dp1_median_us': statistics.median(b['layer0'] for a,b in pairs),
      'layer1_dp0_median_us': statistics.median(a['layer1'] for a,b in pairs),
      'layer1_dp1_median_us': statistics.median(b['layer1'] for a,b in pairs),
      'dp1_minus_dp0_layer0_median_us':statistics.median(diff0),
      'dp1_minus_dp0_layer1_median_us':statistics.median(diff1),
      'difference_of_paired_differences_median_us':statistics.median(x-y for x,y in zip(diff0,diff1)),
      'layer0_dp1_gt_dp0_pairs':sum(x>0 for x in diff0)})
payload={'scope':'16 ranks, three source-validated captured interior forward ordinals; identical physical M=3, model ID 49; each duration uses its own device clock',
 'source_sha256':{str(p.relative_to(root)):sha(p) for p in [timeline,calls]}, 'pairing':'forward ordinal and TP index; DP group operation identity remains conditional',
 'sample_group_results':per_forward,
 'limits':['Same-rank duration subtraction cancels fixed timestamp offset, but not frequency drift or differing profiler behavior.', 'Comparing DP groups still requires correct ordinal pairing. Layer 0 and layer 1 differ semantically and have different routed expert work; neither is a controlled intervention.', 'An early-arrival wait hypothesis is consistent with a group duration disparity, but dispatch, compute, load imbalance, data movement, or profiler effects are not separated.', 'These are kernel samples, not end-to-end exposed time; no claim of recoverable time or speedup.'],
 'followup_evidence':['Per-rank host launch and device stream timeline with stable request and collective identifiers.', 'Per-phase MC2 trace/counters or documented internal flags to separate dispatch/computation/combine and synchronization.', 'Repeat under same workload and uncertainty accounting only if free NPU is available.']}
out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(per_forward,indent=2))
