"""Frozen C8/P90/32K/1K diagnostic client; only run in the isolated copy."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

ROOT = Path(os.environ["WIO_OUTPUT"])
PHASE = os.environ["WIO_PHASE"]
INPUT = Path(os.environ['WIO_FROZEN_ROOT']) / ('warmup' if PHASE == 'warmup' else 'measure1')
URL = os.environ['WIO_API_URL']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(path, body=None, timeout=900, headers=None):
    data = None if body is None else json.dumps(body).encode()
    return opener.open(urllib.request.Request(URL+path, data=data,
                       headers={'Content-Type':'application/json', **(headers or {})}), timeout=timeout)

def metrics():
    with request('/metrics', timeout=15) as response:
        return response.read().decode()

def metric_by_engine(raw, name):
    values = {'0':[], '1':[]}
    for row in raw.splitlines():
        if not row.startswith('vllm:'+name+'{'): continue
        match = re.search(r'(?:\{|,)engine="([01])"(?:,|\})',row)
        if match: values[match.group(1)].append(float(row.rsplit(' ',1)[1]))
    return {engine:sum(xs) if xs else None for engine,xs in values.items()}

def run_request(row, index, stream, barrier, firsts, finished):
    body = {'model':'dsv41', 'prompt':row['prompt'], 'max_tokens':row['max_tokens'],
            'temperature':0, 'ignore_eos':True, 'stream':stream,
            'request_id':f'official-{PHASE}-{index}'}
    if stream: body['stream_options'] = {'include_usage':True}
    barrier.wait(timeout=120)
    result = {'start_ns':time.monotonic_ns(), 'chunks':[], 'text':''}
    try:
        headers = {'X-data-parallel-rank':str(row['dp_rank'])} if row.get('dp_rank') is not None else {}
        with request('/v1/completions', body, headers=headers) as response:
            if not stream:
                data = json.load(response)
                result.update(usage=data['usage'], text=data['choices'][0]['text'])
            else:
                for line in response:
                    line = line.decode()
                    if not line.startswith('data:'): continue
                    line = line[5:].strip()
                    if line == '[DONE]': continue
                    data = json.loads(line)
                    now = time.monotonic_ns()
                    s = ''.join(x.get('text','') for x in data.get('choices',[]))
                    result['chunks'].append({'at_ns':now,'nonempty':bool(s)})
                    if s and not firsts[index].is_set():
                        result['first_nonempty_ns'] = now
                        firsts[index].set()
                    result['text'] += s
                    if data.get('usage'): result['usage'] = data['usage']
    finally:
        result['end_ns'] = time.monotonic_ns()
        finished[index].set()
    return result

def parallel(rows, stream=False, window_control=None):
    firsts = [threading.Event() for _ in rows]
    finished = [threading.Event() for _ in rows]
    barrier = threading.Barrier(len(rows))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        futures = [pool.submit(run_request, row, i, stream, barrier, firsts, finished)
                   for i,row in enumerate(rows)]
        if window_control: window_control(firsts, finished)
        return [f.result() for f in futures]

def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    status = {'phase':PHASE,'case':'C8 P90 32K/1K','windows':[],
              'semantics':'first nonempty text is a client-visible boundary, not a device prefill-completion timestamp'}
    rows = [json.loads(x) for x in next(INPUT.rglob('*.requests.jsonl')).read_text().splitlines()]
    assert len(rows) == 8
    status['input_sha256'] = hashlib.sha256(json.dumps([x['question'] for x in rows], ensure_ascii=False).encode()).hexdigest()
    seeds = json.loads((INPUT/'warmup.json').read_text())
    seeded = parallel(seeds)
    assert all(x['usage']['prompt_tokens']==29696 and x['usage']['completion_tokens']==1 for x in seeded)
    (ROOT/'seeds.json').write_text(json.dumps(seeded, ensure_ascii=False))
    before = metrics()
    (ROOT/'metrics-before.txt').write_text(before)
    started = False
    try:
        if PHASE == 'profile':
            with request('/start_profile', {}, timeout=120) as r: assert r.status == 200
            started = True
            status['windows'].append({'name':'initial','start_ns':time.monotonic_ns()})
        def window(firsts, finished):
            nonlocal started
            if PHASE != 'profile': return
            # CPU frontend profiling is not governed by max_iterations.
            deadline = time.monotonic()+30
            while time.monotonic()<deadline and not all(e.is_set() for e in firsts):
                if any(done.is_set() and not first.is_set() for done,first in zip(finished,firsts)): break
                time.sleep(.05)
            if not all(e.is_set() for e in firsts):
                status['decode_gap'] = 'not all requests produced visible text before timeout/completion'
                try:
                    with request('/stop_profile', {}, timeout=600) as r: assert r.status == 200
                    started = False
                    status['windows'][0]['stop_ns'] = time.monotonic_ns()
                except Exception as exc:
                    status['early_stop_error'] = repr(exc)
                return
            status['all_first_visible_ns'] = time.monotonic_ns()
            try:
                with request('/stop_profile', {}, timeout=600) as r: assert r.status == 200
                started = False
                status['windows'][0]['stop_ns'] = time.monotonic_ns()
                if all(e.is_set() for e in finished):
                    status['decode_gap'] = 'requests finished before second window could start'
                    return
                with request('/start_profile', {}, timeout=120) as r: assert r.status == 200
                started = True
                status['windows'].append({'name':'later_decode_candidate','start_ns':time.monotonic_ns()})
                # Frontend has no iteration bound: cap the second window by time.
                end = time.monotonic()+2
                while time.monotonic()<end and not all(e.is_set() for e in finished): time.sleep(.05)
                with request('/stop_profile', {}, timeout=600) as r: assert r.status == 200
                started = False
                status['windows'][-1]['stop_ns'] = time.monotonic_ns()
            except Exception as exc:
                status['decode_gap'] = 'second window unavailable: '+repr(exc)
        requests = [{'prompt':x['question'],'max_tokens':1024,
                     **({'dp_rank':x['dp_rank']} if x.get('dp_rank') is not None else {})} for x in rows]
        responses = parallel(requests, True, window)
    finally:
        if started:
            try:
                with request('/stop_profile', {}, timeout=600) as r: status['final_stop_http'] = r.status
            except Exception as exc: status['final_stop_error'] = repr(exc)
        (ROOT/'window-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    after = metrics()
    (ROOT/'metrics-after.txt').write_text(after)
    (ROOT/'responses.json').write_text(json.dumps(responses, ensure_ascii=False))
    names = ['prefix_cache_queries_total','prefix_cache_hits_total','prompt_tokens_total',
             'generation_tokens_total','request_success_total','num_preemptions_total',
             'spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total']
    before_engines = {name:metric_by_engine(before,name) for name in names}
    after_engines = {name:metric_by_engine(after,name) for name in names}
    by_engine = {name:{engine:(after_engines[name][engine]-before_engines[name][engine]
                        if after_engines[name][engine] is not None and before_engines[name][engine] is not None
                        else None) for engine in ('0','1')} for name in names}
    delta = {name:(sum(by_engine[name].values()) if all(v is not None for v in by_engine[name].values())
                   else None) for name in names}
    status['counter_deltas_by_engine'] = by_engine
    status['counter_deltas'] = delta
    status['valid'] = (len(responses)==8 and all(x.get('usage',{}).get('prompt_tokens')==32768 and
        x.get('usage',{}).get('completion_tokens')==1024 for x in responses) and
        delta['prefix_cache_queries_total']==262144 and delta['prefix_cache_hits_total']==235904 and
        delta['prompt_tokens_total']==262144 and
        delta['generation_tokens_total']==8192 and delta['request_success_total']==8 and
        delta['num_preemptions_total']==0)
    (ROOT/'window-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    if not status['valid']: raise RuntimeError('Frozen case validation failed')

if __name__ == '__main__': main()
