"""校验公开分解数据；不运行模型或浏览器，不判定性能收益。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9), (a, b)


def main():
    d = json.loads((HERE / 'data.json').read_text())
    html = (HERE / 'index.html').read_text()
    embedded = re.search(r'<script id="flow-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(embedded) == d
    requests = d['requests']
    assert len(requests) == 1344 and len(d['components']) == 13
    assert len(d['profiles']) == 500 and len(d['diagnostic_steps']) == 262
    keys = [(r['framework'], r['n'], r['c'], r['repetition'], r['ordinal']) for r in requests]
    assert len(set(keys)) == len(keys)
    waves = set(k[:4] for k in keys)
    assert len(waves) == 168
    for fw, n, c, rep in waves:
        assert sorted(k[4] for k in keys if k[:4] == (fw, n, c, rep)) == list(range(c))
    for r in requests:
        assert r['valid'] and r['completion_tokens'] == 1024
        assert r['start_s'] <= r['first_text_s'] <= r['eof_s']
        close(r['ttft_s'], r['first_text_s'] - r['start_s'])
        close(r['after_first_s'], r['eof_s'] - r['first_text_s'])
        close(r['e2e_s'], r['eof_s'] - r['start_s'])
        close(r['ttft_s'] + r['after_first_s'], r['e2e_s'])
        close(r['endpoint_tpot_ms'], r['after_first_s'] / 1023 * 1000)
        assert ('sglang' if r['framework'] == 'vllm' else 'vllm', r['n'], r['c'], r['repetition'], r['ordinal']) in keys
    csvrows = list(csv.DictReader((HERE / 'formal-requests.csv').open()))
    assert len(csvrows) == len(requests)
    for row, r in zip(csvrows, requests):
        for k, v in r.items():
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                close(row[k], v)
            else:
                assert row[k] == str(v)
    assert len(d['summaries']) == 28
    for s in d['summaries']:
        for fw in ['vllm', 'sglang']:
            for metric, value in s[fw].items():
                rr = [r for r in requests if (r['n'], r['c'], r['framework']) == (s['n'], s['c'], fw)]
                close(value, statistics.median(statistics.median(r[metric] for r in rr if r['repetition'] == rep) for rep in [1, 2, 3]))
    assert len(d['decode']) == 8 and len(d['phases']) == 5
    for x in d['decode']:
        v = x['intervals_ms']  # Historical field suffixes are us; enclosing object values are ms.
        assert x['outer_dependency_closed'] is False
        close(v['collective_envelope_union_us'] - v['collective_compute_overlap_us'], v['collective_outside_compute_union_us'])
        close(v['compute_union_us'] + v['collective_outside_compute_union_us'] + x['uncovered_by_compute_collective_ms'], x['fence_ms'])
    sg = next(x for x in d['diagnostic_steps'] if x['framework'] == 'sglang' and str(x['step_id']) == '5291')
    assert sg['chunks']['482b498e3704479f9330df16fe9d8bc6'] == 1
    assert sg['chunks']['f38c6269fdc54aa995636b114ecc26b8'] == 512
    assert sg['correction_generation'] == 112
    for edge in d['chains']['sglang']['edges']:
        if edge.get('from_event', '').startswith('ASGI_'):
            assert '不是await返回' in edge['restriction']
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    assert manifest['sources'] == d['sources'] and len(d['sources']) == 16
    for source in d['sources']:
        assert re.fullmatch('[0-9a-f]{64}', source['sha256'])
    for component in d['components']:
        for source in component['sources']:
            if source['url']:
                assert source['commit'] in source['url'] and '/blob/' in source['url']
    javascript = re.findall(r'<script>(.*?)</script>', html, re.S)
    assert len(javascript) == 1
    with tempfile.TemporaryDirectory() as temp:
        script = Path(temp) / 'flow.js'
        script.write_text(javascript[0])
        subprocess.run(['node', '--check', str(script)], check=True, capture_output=True)
    forbidden = [r'/home/', r'/root/', r'/data/', r'vllm-ascend-perf-lab', r'BEGIN .*PRIVATE KEY', r'gh[pousr]_[A-Za-z0-9]{20}']
    # Scan deliverable content, not the literal patterns in this verifier.
    for file in [HERE / 'data.json', HERE / 'index.html', HERE.parent / 'REQUEST_FLOW_COMPARISON.md']:
        for pattern in forbidden:
            assert not re.search(pattern, file.read_text()), (file.name, pattern)
    receipt = dict(status='passed', formal_waves=len(waves), formal_requests=len(requests),
                   paired_request_positions=672, components=13, profile_windows=500,
                   diagnostic_steps=262, decode_windows=8, phase_subsets=5,
                   data_sha256=hashlib.sha256((HERE / 'data.json').read_bytes()).hexdigest(),
                   javascript_syntax='passed', browser_validation='not_run',
                   performance_acceptance='not_evaluated_by_this_checker')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
