import json
import re
import statistics
from pathlib import Path

root = Path('/data/wio/Inference_Foundry/evidence/20260920_baseline')
runs = []
for n in (1, 2):
    path = root / f'bench48_{n}.json'
    if path.exists():
        data = json.loads(path.read_text())
        runs.append({'run': n, **data['summary'],
                     'completion_tokens': sum(r['output_tokens'] for r in data['requests'] if not r['error']),
                     'input_tokens': sum((r['input_tokens'] or 0) for r in data['requests'] if not r['error'])})

metrics = {}
for stage in ('before', 'after'):
    path = root / f'metrics_{stage}.txt'
    if not path.exists():
        continue
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.rpartition(' ')
        if not sep:
            continue
        name = key.split('{')[0]
        if name in ('vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_draft_tokens_total',
                    'vllm:spec_decode_num_accepted_tokens_total', 'vllm:prefix_cache_queries_total',
                    'vllm:prefix_cache_hits_total', 'vllm:request_success_total'):
            try:
                values[name] = values.get(name, 0) + float(value)
            except ValueError:
                pass
    metrics[stage] = values

diff = {k: metrics.get('after', {}).get(k, 0) - metrics.get('before', {}).get(k, 0)
        for k in metrics.get('after', {})}
drafts = diff.get('vllm:spec_decode_num_drafts_total', 0)
draft_tokens = diff.get('vllm:spec_decode_num_draft_tokens_total', 0)
accepted = diff.get('vllm:spec_decode_num_accepted_tokens_total', 0)
queries = diff.get('vllm:prefix_cache_queries_total', 0)
hits = diff.get('vllm:prefix_cache_hits_total', 0)

util = []
hbm = []
path = root / 'npu_samples.txt'
if path.exists():
    pattern = re.compile(r'\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*65536')
    for line in path.read_text(errors='replace').splitlines():
        m = pattern.search(line)
        if m:
            util.append(int(m.group(1)))
            hbm.append(int(m.group(2)))

result = {'runs': runs, 'metrics_delta': diff,
          'spec_acceptance': accepted / draft_tokens if draft_tokens else None,
          'accepted_per_draft': accepted / drafts if drafts else None,
          'prefix_hit_rate': hits / queries if queries else None,
          'npu_ai_core_samples': len(util),
          'npu_ai_core_median': statistics.median(util) if util else None,
          'npu_ai_core_p90': sorted(util)[int(.9 * (len(util)-1))] if util else None,
          'npu_hbm_median_mb': statistics.median(hbm) if hbm else None}
(root / 'analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
