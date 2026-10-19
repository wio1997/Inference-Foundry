#!/usr/bin/env python3
"""One native P profile window for the validated crossover inputs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
import zlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--cases-base64', required=True)
    p.add_argument('--expected-cases-sha256', required=True)
    a = p.parse_args()
    assert a.execute and socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live3-20261018/node32')
    cid = json.loads((root / 'launch.json').read_text())['container_id']
    assert cid == '4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157'
    ins = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    plan = json.loads((root / 'plan.json').read_text())
    assert ins['State']['Running'] and ins['Image'] == plan['image']['id']
    assert ins['Name'] == '/' + plan['container']['name']
    assert plan['env']['GLM53_P_FIRST_ADOPT_TRACE'] == '0'
    assert shutil.disk_usage(root).free > 12 * 1024**3
    data = zlib.decompress(base64.b64decode(a.cases_base64, validate=True))
    assert hashlib.sha256(data).hexdigest() == a.expected_cases_sha256
    cases = json.loads(data)['cases']
    assert len(cases) == 4 and {c['N'] for c in cases} == {256, 257}
    assert all(c['math_suffix_ids_unchanged'] for c in cases)
    out = root / 'prefill-boundary-profile-v1'
    out.mkdir(exist_ok=False)
    (out / 'cases.json').write_bytes(data)
    before = {x.name for x in (root / 'profile').glob('*_ascend_pt')}
    report = {'completed': False, 'rows': [], 'container_id': cid, 'profile_role': 'P',
              'cases_sha256': a.expected_cases_sha256,
              'scope': 'Native P local completion, same original model/K1/operators/precision/SDMA. Two arm calls then8 serial crossover calls in one bounded native profile session. Installed NPU wrapper ignores active_iterations and records until explicit stop (max_iterations0); no end-to-end timing accepted from this profile. All16 raw provenance, representative rank0 compute analysis only.'}

    def post(path, body, timeout=180, request_id=None):
        req = urllib.request.Request('http://172.16.12.32:29481' + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', **({'X-Request-Id': request_id} if request_id else {})})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def one(case, phase):
        rid = 'boundary-profile-' + uuid.uuid4().hex
        body = {'model': 'glm5', 'prompt': case['prompt_ids'], 'max_tokens': 1, 'min_tokens': 1,
                'temperature': 0, 'top_p': 1, 'seed': 1024, 'ignore_eos': True,
                'return_token_ids': True, 'cache_salt': rid, 'stream': False,
                'kv_transfer_params': {'do_remote_decode': False, 'do_remote_prefill': False}}
        row = {'case_id': case['case_id'], 'N': case['N'], 'phase': phase, 'request_id': rid,
               'sent_monotonic_ns': time.monotonic_ns(), 'sent_wall_ns': time.time_ns()}
        report['rows'].append(row)
        row['response'] = post('/v1/completions', body, request_id=rid)
        row['received_monotonic_ns'] = time.monotonic_ns()
        row['received_wall_ns'] = time.time_ns()
        response = row['response']
        assert response['choices'][0]['token_ids'] == [case['expected_token_id']]
        assert response['choices'][0]['text'].strip() == case['answer']
        assert response['usage']['prompt_tokens'] == case['N'] and response['usage']['completion_tokens'] == 1
        assert not response.get('kv_transfer_params')
        row['complete_answer_pass'] = True
        (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')

    attempted = False
    try:
        attempted = True
        report['start_profile'] = post('/start_profile', {}, timeout=360)
        for case in cases[:2]:
            one(case, 'arm')
        for case_id in ['sum-three-4', 'sum-three-1']:
            by_n = {c['N']: c for c in cases if c['case_id'] == case_id}
            for n in [256, 257, 257, 256]:
                one(by_n[n], 'active-request')
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if attempted:
            try:
                report['stop_profile'] = post('/stop_profile', {}, timeout=360)
                report['profile_stopped'] = True
            except Exception as exc:
                report['stop_failure'] = type(exc).__name__ + ': ' + str(exc)
    selected = sorted(x for x in (root / 'profile').glob('*_ascend_pt') if x.name not in before)
    report['new_profile_dirs'] = [str(x) for x in selected]
    report['ranks'] = sorted(int(re.search(r'_rank(\d+)_', x.name).group(1)) for x in selected)
    report['source_files'] = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size} for d in selected for p in sorted(d.rglob('*')) if p.is_file()]
    report['source_bytes'] = sum(p['bytes'] for p in report['source_files'])
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'] and report.get('profile_stopped'), report.get('failure') or report.get('stop_failure')
    assert report['ranks'] == list(range(16))
    assert report['source_bytes'] < 4 * 1024**3


if __name__ == '__main__':
    main()
