"""One paired D-admission fragment: prepare native P KV, then release four D requests."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, re, shutil, socket, subprocess, threading, time, urllib.request, uuid

assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefix-count-live1-20261019/node33')
cid = '8f434957c821dec55a77d352a68d16c14f2558d0b4c7e11287e0e37181111fdd'
plan = json.loads((root/'plan.json').read_text()); d = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
assert d['State']['Running'] and d['Name'] == '/'+plan['container']['name'] and d['Image'] == plan['image']['id']
left = plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds()
assert left > 900 and shutil.disk_usage(root).free > 16*1024**3
fixed = json.loads((root/'moe-prefix-gmm2-count-c4-fixed-arrival-v2/result.json').read_text())
assert fixed['completed'] and fixed['numeric_all_pass'] and fixed['fixed_arrival_control']['timing_valid']
probe = root/'moe-prefix-gmm2-count-probe'; marker = probe/'enable'; disabled = probe/'disabled'
assert disabled.exists() and not marker.exists() and (probe/'target-only').exists()
assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists()
cases = json.loads((root/'prefill-padding-cases-v1.json').read_text())['mixed']
assert sorted(len(c['prompt_token_ids']) for c in cases) == [256, 257, 1536, 1537]
out = root/'prefix-count-c4-supply-profile-v1'; out.mkdir(exist_ok=False)
report = dict(completed=False, rows=[], profiles=[], profile_attempts=[], CID=cid, remaining_model_s_at_start=left,
              scope='One native and one prefix-count paired synchronized-D-admission diagnostic, each four8-token native PD requests after one unprofiled four-request warm. P produces KV first, D requests then start at a barrier. Changed condition isolates concurrent D execution; not original stock-proxy C4 arrivals or a service performance estimate. Existing metadata observer enabled for diagnostic warm/profile groups, first-use inventory warmed before capture; RoPE observer stays off. No tensor snapshots or new operators.')

def post(node, path, body, rid=None, timeout=180):
    req = urllib.request.Request('http://172.16.12.%d:%d%s' % (node, 29481 if node == 32 else 29482, path), json.dumps(body).encode(), {'Content-Type': 'application/json', **({'X-Request-Id': rid} if rid else {})})
    with urllib.request.urlopen(req, timeout=timeout) as f: raw = f.read()
    return json.loads(raw) if raw else None

def idle():
    for _ in range(121):
        values = []
        for node, port in [(32, 29481), (33, 29482)]:
            with urllib.request.urlopen('http://172.16.12.%d:%d/metrics' % (node, port), timeout=5) as f: data = f.read().decode()
            lines = [z for z in data.splitlines() if z.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{', 'vllm:kv_cache_usage_perc{'))]
            assert len(lines) >= 3; values.extend(float(z.rsplit(' ', 1)[1]) for z in lines)
        if all(v == 0 for v in values): return
        time.sleep(.25)
    raise AssertionError(values)

def prepare(case, mode, phase):
    rid = 'c4supply-'+uuid.uuid4().hex
    body = dict(model='glm5', prompt=case['prompt_token_ids'], max_tokens=8, min_tokens=1, ignore_eos=True, temperature=0, top_p=1, seed=1024, return_token_ids=True, cache_salt=rid, stream=False)
    producer = dict(body, max_tokens=1, kv_transfer_params=dict(do_remote_decode=True, do_remote_prefill=False, remote_engine_id=None, remote_block_ids=None, remote_host=None, remote_port=None))
    response = post(32, '/v1/completions', producer, rid)
    params = response['kv_transfer_params']; assert params['do_remote_prefill'] and params['remote_num_prompt_tokens'] == len(body['prompt'])
    body['kv_transfer_params'] = params
    row = dict(mode=mode, phase=phase, case_id=case['case_id'], N=len(body['prompt']), expected=case['expected'], request_id=rid, P_response=response, D_request=body)
    report['rows'].append(row)
    return body, row

def consume(prepared, barrier):
    body, row = prepared; barrier.wait(timeout=20); row['D_sent_monotonic_ns'] = time.monotonic_ns(); row['D_sent_wall_ns'] = time.time_ns()
    result = post(33, '/v1/completions', body, row['request_id']); row['D_finished_monotonic_ns'] = time.monotonic_ns(); row['D_response'] = result
    tokens = result['choices'][0]['token_ids']; match = re.search(r'[-+]?\d+(?:\.\d+)?', result['choices'][0]['text'])
    row['numeric_pass'] = bool(match and float(match.group()) == row['expected']); row['budget_pass'] = len(tokens) == result['usage']['completion_tokens'] == 8
    assert row['numeric_pass'] and row['budget_pass']

def group(mode, phase, profiling=False):
    idle()
    if mode == 'native': marker.unlink(missing_ok=True)
    else: marker.touch()
    # Do not ask for zero P KV usage between producer completion and D consumption.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: prepared = list(pool.map(lambda c: prepare(c, mode, phase), cases))
    before = {p.name for p in (root/'profile').glob('*_ascend_pt')}
    offsets = {p.name: p.stat().st_size for p in probe.glob('rank-*.jsonl')}
    attempted = False; record = dict(mode=mode, phase=phase)
    try:
        disabled.unlink()
        if profiling:
            attempted = True; report['profile_attempts'].append(record); record['start_profile'] = post(33, '/start_profile', {})
        barrier = threading.Barrier(4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda p: consume(p, barrier), prepared))
        idle()
    finally:
        disabled.touch()
        if attempted:
            record['stop_profile'] = post(33, '/stop_profile', {}); record['profile_stopped'] = True
    if profiling:
        dirs = sorted(p for p in (root/'profile').glob('*_ascend_pt') if p.name not in before)
        record['dirs'] = [str(p) for p in dirs]; record['ranks'] = sorted(int(re.search(r'_rank(\d+)_', p.name).group(1)) for p in dirs)
        record['files'] = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size) for directory in dirs for p in sorted(directory.rglob('*')) if p.is_file()]
        record['source_bytes'] = sum(z['bytes'] for z in record['files']); record['metadata_offsets'] = offsets
        frames = []
        for p in sorted(probe.glob('rank-*.jsonl')):
            with p.open('rb') as f: f.seek(offsets.get(p.name, 0)); raw = f.read()
            frames.extend(json.loads(s) for s in raw.decode().splitlines())
        record['metadata_frames'] = len(frames); record['metadata_ranks'] = sorted({z['rank'] for z in frames})
        record['selected_N_by_rank'] = {str(rank): sorted({g['tokens'] for z in frames if z['rank'] == rank for g in z['graphs']}) for rank in range(16)}
        for frame in frames:
            if not frame['request_ids']: continue
            assert frame['requested'] == (mode == 'adaptive')
            assert all(g['selected_family'] == ('prefix' if mode == 'adaptive' else 'native') and g['selected'] == (mode == 'adaptive') for g in frame['graphs'])
            if frame['phase'] == 'sample_tokens': assert all(not z['selected'] and z['output_list_type'] == 1 for z in frame['dispatch'])
        (out/(mode+'-metadata.json')).write_text(json.dumps(frames)+'\n')
        report['profiles'].append(record)
        assert record['profile_stopped'] and record['ranks'] == list(range(16)) and record['metadata_ranks'] == list(range(16))
        assert record['source_bytes'] < 4*1024**3
    (out/'progress.json').write_text(json.dumps(report)+'\n')

try:
    for mode in ['native', 'adaptive']:
        group(mode, 'unprofiled_warm'); group(mode, 'profiled_concurrent', True)
    idle(); report['completed'] = True
except Exception as exc: report['failure'] = type(exc).__name__+': '+str(exc)
finally:
    marker.unlink(missing_ok=True); disabled.touch()
    # Consume producer responses which never reached D, without retrying an
    # already sent/unknown-outcome D request or touching unrelated work.
    for row in report['rows']:
        if 'D_sent_monotonic_ns' not in row:
            try:
                row['cleanup_D_response'] = post(33, '/v1/completions', row['D_request'], row['request_id'])
                row['unsent_own_KV_drained'] = True
            except Exception as exc: row['cleanup_failure'] = str(exc)
    for attempt in report['profile_attempts']:
        if not attempt.get('profile_stopped'):
            try: attempt['cleanup_stop_profile'] = post(33, '/stop_profile', {}); attempt['profile_stopped'] = True
            except Exception as exc: attempt['cleanup_stop_failure'] = str(exc)
report['candidate_restored_native'] = not marker.exists(); report['observer_restored_off'] = disabled.exists()
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'], report.get('failure')
