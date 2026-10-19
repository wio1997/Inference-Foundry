#!/usr/bin/env python3
"""Summarize existing per-role metric deltas after completed official-PD runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket


def parse(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, value = line.rsplit(' ', 1)
        name = name.split('{', 1)[0]
        values[name] = values.get(name, 0) + float(value)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', action='append', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd/tests')
    report = {'scope': 'Native histogram means within each host, outside timing. '
              'Do not subtract these from client percentiles or compare cross-host clocks.',
              'sources': [], 'rows': []}
    wanted = ('time_to_first_token_seconds', 'request_queue_time_seconds',
              'request_inference_time_seconds', 'request_prefill_time_seconds',
              'request_prefill_kv_computed_tokens')
    for run_id in args.run_id:
        assert re.fullmatch(r'[a-z0-9-]+', run_id)
        root = base / run_id
        result = root / 'result.json'
        data = json.loads(result.read_text())
        assert data['completed'] and data['passed'], 'Only summarize completed formal runs'
        report['sources'].append({'run_id': run_id,
                                  'result_sha256': hashlib.sha256(result.read_bytes()).hexdigest()})
        for case in data['cases']:
            for role in ('P', 'D'):
                before = parse(root / (case['case'] + '-before-' + role + '.txt'))
                after = parse(root / (case['case'] + '-after-' + role + '.txt'))
                row = {'run_id': run_id, 'case': case['case'], 'role': role, 'metrics': {}}
                for kind in wanted:
                    key = 'vllm:' + kind
                    count = after.get(key + '_count', 0) - before.get(key + '_count', 0)
                    if count <= 0:
                        continue
                    total = after[key + '_sum'] - before[key + '_sum']
                    value = total / count
                    metric = {'count': count}
                    metric['mean_tokens' if kind.endswith('_tokens') else 'mean_ms'] = (
                        value if kind.endswith('_tokens') else value * 1000)
                    row['metrics'][key] = metric
                report['rows'].append(row)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
