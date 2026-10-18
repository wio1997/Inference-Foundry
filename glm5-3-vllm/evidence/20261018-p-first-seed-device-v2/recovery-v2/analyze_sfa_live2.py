"""Reduce immutable live2 results; preserve the original cancel parser error."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics

r = Path(__file__).resolve().parent

def read(name):
    p = next((r / name).glob('*/metrics.json'))
    return json.loads(p.read_text()), {'path': str(p.relative_to(r)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def quantile(values, q):
    x = sorted(values); k = (len(x) - 1) * q
    return x[int(k)] + (x[min(int(k) + 1, len(x) - 1)] - x[int(k)]) * (k % 1)

c1, c1_source = read('sfa-replay-order-live2-c1')
c4, c4_source = read('sfa-replay-order-live2-c4')
quality, quality_source = read('sfa-replay-order-live2-quality')
cancel, cancel_source = read('sfa-replay-order-live2-cancel')
assert all(x['completed'] for x in [c1, c4, quality, cancel])
formal = [x for x in c4['rows'] if x['phase'] == 'formal']
assert len(formal) == 384 and all(x['numeric_pass'] and x['functional_pass'] for x in formal)
assert len(c4['blocks']) == 12
pooled = {}
for mode in ['native', 'adaptive']:
    rows = [x for x in formal if x['mode'] == mode]
    pooled[mode] = {k: {'median': statistics.median(x[k] for x in rows),
                        'p95_linear': quantile([x[k] for x in rows], .95)}
                    for k in ['TTFT_ms', 'last_token_ms', 'mean_TPOT_ms']}
cycles = []
for index in range(3):
    block = c4['blocks'][index * 4:(index + 1) * 4]
    rates = {m: sum(x['output_tokens'] for x in block if x['mode'] == m) /
             sum(x['seconds'] for x in block if x['mode'] == m) for m in ['native', 'adaptive']}
    cycles.append({'index': index, 'rates': rates, 'candidate_over_native': rates['adaptive'] / rates['native']})
ratios = {k: math.exp(statistics.mean(math.log(c1['summary']['adaptive']['cases'][case][k] /
                                                   c1['summary']['native']['cases'][case][k])
                                      for case in c1['summary']['native']['cases']))
          for k in ['TTFT_ms', 'last_token_ms', 'mean_TPOT_ms']}
matched = {rid: [s for s in lines if re.search(
    r'\b(?:abort(?:ed|ing)?|cancel(?:led|ed|ling|ing)?|disconnect(?:ed)?)\b', s.replace(rid, ''), re.I)]
    for rid, lines in cancel['backend_abort_log_evidence'].items()}
corrected_cancel = {
    'source': cancel_source, 'completed': cancel['completed'],
    'client_close_requests': cancel['client_close_requests'],
    'full_reuse_and_peer_responses': cancel['full_reuse_and_peer_responses'],
    'original_parser_reported_backend_log_count': cancel['backend_abort_log_request_count'],
    'corrected_backend_log_count': sum(bool(v) for v in matched.values()),
    'corrected_backend_log_evidence': matched, 'abort_counter_deltas': cancel['abort_counter_deltas'],
    'interpretation': '10 client closes and 16 complete peers/reuses passed, all P/D running/waiting/KV returned idle. Original substring parser counted cancel in request IDs; corrected count is 0 and abort counter delta is 0. Do not claim explicit backend-abort proof.',
    'raw_results_preserved': True}
(r / 'sfa-replay-order-live2-cancel-correction-v1.json').write_text(json.dumps(corrected_cancel, indent=2) + '\n')
result = {
    'decision': 'INCONCLUSIVE performance; native default restored. Balanced C4 aggregate gain below1%, cycle signs alternate. No stable service-capacity/Pareto claim, no more benchmark on this version.',
    'sources': [c1_source, c4_source, quality_source, cancel_source],
    'C1': c1['summary'], 'C1_equal_case_ratios': ratios,
    'C4': c4['summary'], 'C4_blocks': c4['blocks'], 'C4_cycles': cycles, 'C4_pooled': pooled,
    'C4_candidate_over_native': c4['summary']['adaptive']['output_tokens_per_s'] / c4['summary']['native']['output_tokens_per_s'],
    'quality_sequence_comparisons': quality.get('sequence_comparisons'),
    'quality_summary': quality['summary'], 'cancel': corrected_cancel,
    'scope': 'Same pinned TP16/TP16 all-SDMA stock-proxy ordinary64/128, four closed-loop slots, rotated four cases, 8requests/slot/block, equal768 output tokens/slot, ABBAx3. Finite point, not maximum sustained SLO capacity. Static prior point retained separately.',
    'next': 'Keep native and model loaded; examine source-supported partial graph of pure MTP FFN, leaving attention/KV eager. Do not turn on unsupported whole-draft graph.'}
(r / 'sfa-replay-order-live2-verdict-v1.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['C1_equal_case_ratios', 'C4_cycles', 'C4_candidate_over_native', 'C4_pooled']}))
