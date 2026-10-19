"""910C lifecycle client: cancel an observed chunk, finish survivors, reuse RID without flush."""
import concurrent.futures, hashlib, json, math, os, pathlib, sys, threading, time, traceback, urllib.error, urllib.request

j = json.load(sys.stdin)
plan = j['plan']; tag = plan['run']; root = pathlib.Path(__file__).resolve().parent
assert tag == 'L039-R092' and len(j['inputs']) == 3
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
handle = {'run': tag, 'pid': os.getpid(), 'proc_start_ticks': pathlib.Path('/proc/self/stat').read_text().split()[21], 'source_path': __file__, 'client_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(root / 'worker-handle.json').write_text(json.dumps(handle, indent=2) + '\n')
v = {'run': tag, 'execution_node': 46, 'status': 'STARTING', 'phase': 'INITIAL_FLUSH', 'requests': {}, 'checks': [], 'abort_all_used': False, 'intermediate_flushes': 0, 'performance': 'LIFECYCLE_ONLY_NOT_PERFORMANCE', 'worker_handle': handle}

def save():
    q = root / 'result.tmp'; q.write_text(json.dumps(v, indent=2) + '\n'); q.replace(root / 'result.json')

def http(base, path, body=None, timeout=10):
    raw_body = None if body is None else json.dumps(body, separators=(',', ':')).encode()
    req = urllib.request.Request(base + path, data=raw_body, headers={'Content-Type': 'application/json'}, method='GET' if body is None else 'POST')
    start = time.monotonic_ns()
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(4 * 1024**2); code = response.status
    except urllib.error.HTTPError as error:
        raw = error.read(4 * 1024**2); code = error.code
    except BaseException as error:
        return {'status': None, 'start_ns': start, 'end_ns': time.monotonic_ns(), 'error': type(error).__name__ + ': ' + str(error)}
    try: obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError): obj = None
    return {'status': code, 'start_ns': start, 'end_ns': time.monotonic_ns(), 'body_sha256': hashlib.sha256(raw).hexdigest(), 'body': obj, 'text': raw.decode(errors='replace') if obj is None else None}

def generate(role, index, attempt, room, barrier):
    item = j['inputs'][index]; rid = plan['rids'][index]
    body = {'rid': rid, 'input_ids': item['input_ids'], 'sampling_params': plan['sampling_params'], 'stream': False, 'return_logprob': True, 'logprob_start_len': -1, 'top_logprobs_num': 2, 'return_text_in_logprobs': False, 'return_prompt_token_ids': True, 'bootstrap_host': plan['bootstrap_host'], 'bootstrap_port': plan['bootstrap_port'], 'bootstrap_room': room}
    barrier.wait(timeout=15)
    result = http(plan['endpoints'][role], '/generate', body, plan['request_timeout_seconds'])
    result.update(role=role, index=index, attempt=attempt, rid=rid, room=room, request_sha256=hashlib.sha256(json.dumps(body, separators=(',', ':')).encode()).hexdigest())
    return attempt + '-' + role, result

def aborted(result, role):
    obj = result.get('body')
    finish = obj.get('meta_info', {}).get('finish_reason', {}) if isinstance(obj, dict) else {}
    if result['status'] == 200 and finish.get('type') == 'abort':
        return {'valid_abort': True, 'representation': 'HTTP200_FINISH_ABORT', 'finish': finish, 'output_tokens': len(obj.get('output_ids', []))}
    # Actual decode transfer-failure path converts explicit AbortReq to HTTP500.
    detail = json.dumps(obj, ensure_ascii=False) if obj is not None else result.get('text', '')
    valid = role == 'decode' and result['status'] in [500, 503] and 'Aborted by AbortReq.' in detail and plan['rids'][0] in detail
    return {'valid_abort': valid, 'representation': 'EXPLICIT_ABORT_TRANSFER_HTTP_ERROR' if valid else 'UNEXPECTED_RESPONSE', 'status': result['status'], 'finish': finish, 'detail': detail}

def inspect_pair(index, attempt):
    p = v['requests'][attempt + '-prefill']; d = v['requests'][attempt + '-decode']
    assert p['status'] == d['status'] == 200, (attempt, p['status'], d['status'])
    pb = p['body']; db = d['body']; pm = pb['meta_info']; dm = db['meta_info']; ids = db['output_ids']
    assert pm['id'] == dm['id'] == plan['rids'][index]
    assert len(pb['output_ids']) == 1 and len(ids) == dm['completion_tokens'] == 256
    assert pm['prompt_tokens'] == dm['prompt_tokens'] == 32768
    assert pb['prompt_token_ids'] == db['prompt_token_ids'] == j['inputs'][index]['input_ids']
    assert all(type(t) is int and 0 <= t < 154880 for t in ids) and pb['output_ids'][0] == ids[0]
    assert dm['finish_reason']['type'] == 'length' and dm['spec_verify_ct'] > 0
    hist = dm['spec_correct_drafts_histogram']
    assert len(hist) == 4 and sum(hist) == dm['spec_verify_ct']
    assert sum(i*n for i,n in enumerate(hist)) == dm['spec_num_correct_drafts']
    assert dm['spec_num_proposed_drafts'] == dm['spec_verify_ct'] * 3
    assert 0 <= dm['spec_num_correct_drafts'] <= dm['spec_num_proposed_drafts']
    selected = dm['output_token_logprobs']; top = dm['output_top_logprobs']; offset = len(ids) - len(selected)
    assert offset in [0, 1] and len(top) == len(selected) and [int(x[1]) for x in selected] == ids[offset:]
    rows = []
    for k, (chosen, alternatives) in enumerate(zip(selected, top)):
        position = offset + k
        if position == 0 and (chosen[0] is None or not alternatives): continue
        assert chosen[0] is not None and math.isfinite(float(chosen[0])) and len(alternatives) == 2
        assert all(math.isfinite(float(a[0])) and 0 <= int(a[1]) < 154880 for a in alternatives)
        rows.append({'position': position, 'chosen_id': ids[position], 'chosen_logprob': float(chosen[0]), 'top2': [[float(a[0]), int(a[1])] for a in alternatives], 'chosen_at_max': float(chosen[0]) == max(float(a[0]) for a in alternatives)})
    assert len(rows) >= 255 and all(x['chosen_at_max'] for x in rows)
    digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    return {'index': index, 'attempt': attempt, 'input_tokens': 32768, 'output_tokens': 256, 'P_D_first_token_equal': True, 'prompt_IDs_exact': True, 'output_ids_sha256': digest, 'reference_R090_sha256': plan['reference_output_hashes'][index], 'reference_output_matches': digest == plan['reference_output_hashes'][index], 'MTP_verify': dm['spec_verify_ct'], 'MTP_correct': dm['spec_num_correct_drafts'], 'MTP_proposed': dm['spec_num_proposed_drafts'], 'histogram': hist, 'target_score_rows': rows}

pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
futures = []
try:
    assert plan['max_new_tokens'] == 256 and plan['sampling_params']['ignore_eos'] is True
    assert len(set(plan['rids'])) == 3 and all(not a.startswith(b) for a in plan['rids'] for b in plan['rids'] if a != b)
    for item in j['inputs']:
        ids = item['input_ids']; assert len(ids) == 32768 and all(type(t) is int and 0 <= t < 154880 for t in ids)
        assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == item['wire_ids_sha256']
    v['initial_flush'] = {role: http(url, '/flush_cache?timeout=10', timeout=15) for role, url in plan['endpoints'].items()}
    assert all(x['status'] == 200 for x in v['initial_flush'].values())
    barrier = threading.Barrier(2)
    victim = [pool.submit(generate, role, 0, 'victim', plan['rooms'][0], barrier) for role in ['prefill', 'decode']]; futures += victim
    v['phase'] = 'WAIT_OBSERVED_PREFILL_CHUNK'; save(); gate = root / 'chunk-gate.json'; deadline = time.monotonic() + 130
    while not gate.exists():
        assert time.monotonic() < deadline, 'No verified native chunk event; do not pretend prefill cancellation'
        time.sleep(0.05)
    event = json.loads(gate.read_text()); assert event['run'] == tag and event['P_event']['node'] == 44 and event['P_event']['pending_tokens'] > 0
    assert event['P_event']['new_tokens'] == 4096
    v['chunk_gate'] = event; v['gate_observed_D_monotonic_ns'] = time.monotonic_ns(); v['victim_done_before_gate'] = [f.done() for f in victim]
    barrier = threading.Barrier(5)
    survivors = [pool.submit(generate, role, i, 'survivor-' + str(i), plan['rooms'][i], barrier) for i in [1, 2] for role in ['prefill', 'decode']]; futures += survivors
    barrier.wait(timeout=15); v['survivor_threads_released_D_monotonic_ns'] = time.monotonic_ns()
    v['phase'] = 'ABORT_VICTIM_ONLY'; save()
    abort_barrier = threading.Barrier(2)
    def abort_one(role):
        abort_barrier.wait(timeout=10)
        return role, http(plan['endpoints'][role], '/abort_request', {'rid': plan['rids'][0], 'abort_all': False})
    af = [pool.submit(abort_one, role) for role in ['prefill', 'decode']]
    v['abort_ack'] = dict(f.result(timeout=20) for f in af); save()
    for f in concurrent.futures.as_completed(victim + survivors):
        key, result = f.result(); v['requests'][key] = result; save()
    v['victim_abort'] = {role: aborted(v['requests']['victim-' + role], role) for role in ['prefill', 'decode']}
    # No explicit cache flush between original cancellation, survivors, and same-RID retry.
    v['phase'] = 'RETRY_SAME_RID_NEW_ROOM'; v['retry_start_D_monotonic_ns'] = time.monotonic_ns(); save()
    barrier = threading.Barrier(2)
    retry = [pool.submit(generate, role, 0, 'retry', plan['retry_room'], barrier) for role in ['prefill', 'decode']]; futures += retry
    for f in concurrent.futures.as_completed(retry):
        key, result = f.result(); v['requests'][key] = result; save()
    for index, attempt in [(1, 'survivor-1'), (2, 'survivor-2'), (0, 'retry')]:
        v['checks'].append(inspect_pair(index, attempt)); save()
    assert not any(v['victim_done_before_gate']), 'Victim already finished before observed cancellation gate'
    assert all(x['status'] == 200 for x in v['abort_ack'].values()), 'Abort dispatch failed'
    assert all(x['valid_abort'] for x in v['victim_abort'].values()), 'Victim did not return explicit abort'
    assert all(x['reference_output_matches'] for x in v['checks']), 'Completed output differs from frozen R090 reference; retain evidence'
    v['status'] = 'PASS_PREFILL_CANCEL_SURVIVORS_RETRY'
except BaseException as error:
    v['status'] = 'FAIL'; v['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    v['own_abort_on_failure'] = {role + '-' + rid: http(url, '/abort_request', {'rid': rid, 'abort_all': False}) for role, url in plan['endpoints'].items() for rid in plan['rids']}
finally:
    # Drain our bounded HTTP calls, preserving their real responses on failure.
    for f in futures:
        try:
            key, result = f.result(); v['requests'][key] = result
        except BaseException as error: v.setdefault('future_errors', []).append(str(error))
    pool.shutdown(wait=True)
    v['final_ready'] = {role: http(url, '/ready') for role, url in plan['endpoints'].items()}
    v['final_flush'] = {role: http(url, '/flush_cache?timeout=20', timeout=25) for role, url in plan['endpoints'].items()}
    if not all(x['status'] == 200 for x in [*v['final_ready'].values(), *v['final_flush'].values()]): v['status'] = 'FAIL'
    v['phase'] = 'TERMINAL'; v['scope'] = 'Observed chunk-triggered cancellation, survivor completion and sameRID/newroom reuse without intermediate flush. No physicalKV byte/page-generation oracle, universal leak proof, independent model quality or performance gain.'
    save(); print(json.dumps(v))
sys.exit(0 if v['status'] == 'PASS_PREFILL_CANCEL_SURVIVORS_RETRY' else 1)
