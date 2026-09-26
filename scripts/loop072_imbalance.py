"""Summarize per-rank live MoE row ownership from existing Run287 masks."""

import json
from collections import defaultdict
from pathlib import Path

root = Path('/data/wio/Inference_Foundry/evidence/20260926_loop064_cp/run287/ownership')
summary = defaultdict(lambda: {'cycles': 0, 'max_local_hist': defaultdict(int),
                               'zero_rank_hist': defaultdict(int), 'local_rows_total': [0] * 8})
for path in sorted(root.glob('rank0_cohort*.jsonl')):
    for line in path.read_text().splitlines():
        mask = json.loads(line)['active_mask']
        if len(mask) != 12:
            raise RuntimeError(f'unexpected active mask in {path}')
        local = [sum(mask[row // 8] for row in range(rank * 12, (rank + 1) * 12))
                 for rank in range(8)]
        if sum(local) != 8 * sum(mask):
            raise RuntimeError('local/global row total mismatch')
        bucket = summary[sum(mask)]
        bucket['cycles'] += 1
        bucket['max_local_hist'][max(local)] += 1
        bucket['zero_rank_hist'][local.count(0)] += 1
        bucket['local_rows_total'] = [a + b for a, b in zip(bucket['local_rows_total'], local)]
result = {str(k): {'cycles': v['cycles'],
                   'max_local_hist': dict(sorted(v['max_local_hist'].items())),
                   'zero_rank_hist': dict(sorted(v['zero_rank_hist'].items())),
                   'mean_local_rows_by_rank': [round(x / v['cycles'], 3)
                                               for x in v['local_rows_total']]}
          for k, v in sorted(summary.items())}
print(json.dumps(result, indent=2))
