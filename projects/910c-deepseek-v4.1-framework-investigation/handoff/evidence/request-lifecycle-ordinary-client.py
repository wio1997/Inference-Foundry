"""Exercise a stock service with trace-sized requests and retain per-engine counters."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Event
import difflib
import inspect
import json
import re
import sys
import textwrap
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:30075'
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(p, data):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(p)


def post(path, data):
    req = urllib.request.Request(URL+path, data=json.dumps(data).encode(),
                                 headers={'Content-Type':'application/json'})
    return HTTP.open(req, timeout=100)


def token_ids(value):
    with post('/tokenize', {'model':'dsv41', 'prompt':value, 'add_special_tokens':False}) as response:
        return json.load(response)['tokens']


def build_inputs():
    text = {'chat': inspect.getsource(difflib.SequenceMatcher),
            'code': inspect.getsource(textwrap.TextWrapper)}
    bodies = {k:token_ids(v) for k,v in text.items()}
    pairs = json.loads((ROOT/'RUN.json').read_bytes())['sizes']
    result = {}
    for name, spec in pairs.items():
        kind = spec['kind']
        n, out = spec['input'], spec['output']
        # The third request for each kind changes the question after an identical document prefix.
        tag = name.replace('return', 'single')
        head = token_ids('Reading session '+ROOT.name+' '+tag+'.\n'+{'chat':'Read this reference about sequence comparison:\n','code':'Inspect this Python text wrapping implementation:\n'}[kind])
        questions = {'chat': 'Explain how the comparison works using a short example.',
                     'code':'Complete the next line of Python and return only the code.'}
        suffix = token_ids('\nQuestion: '+('Explain a boundary condition briefly.' if 'return' in name else questions[kind])+'\nAnswer:')
        content = bodies[kind][:n-len(head)-len(suffix)]
        ids = head + content + suffix
        assert len(ids) == n and n > len(head) + len(suffix)
        result[name] = {'prompt':ids, 'max_tokens':out, 'sha256':sha256(json.dumps(ids).encode()).hexdigest()}
    return result


SERIES = re.compile(r'^vllm:([a-z_]+)\{([^}]+)\} (\S+)')
COUNTERS = {'request_success_total','prompt_tokens_total','generation_tokens_total','prefix_cache_hits_total',
            'prefix_cache_queries_total','num_preemptions_total','spec_decode_num_drafts_total',
            'spec_decode_num_accepted_tokens_total','spec_decode_num_draft_tokens_total',
            'iteration_tokens_total_bucket','iteration_tokens_total_sum','iteration_tokens_total_count',
            'num_requests_running','num_requests_waiting'}


def snapshot():
    with HTTP.open(URL+'/metrics', timeout=8) as response:
        lines = response.read().decode().splitlines()
    result = {}
    for line in lines:
        m = SERIES.match(line)
        if not m or m[1] not in COUNTERS:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m[2]))
        if labels.get('model_name') != 'dsv41' or 'engine' not in labels:
            continue
        name = m[1] + (('/'+labels['le']) if 'le' in labels else '')
        key = labels['engine'] + '/' + name
        result[key] = result.get(key, 0) + float(m[3])
    assert {k.split('/')[0] for k in result} == {'0','1'}
    return result


def generate(name, prompt, initial=None, wait_for=None):
    if wait_for is not None and not wait_for.wait(timeout=30):
        raise RuntimeError('No first content for mixed arrival')
    start = time.monotonic_ns()
    chunks = []
    usage = None
    text_hash = sha256()
    finished = False
    with post('/v1/completions', {'model':'dsv41','prompt':prompt['prompt'], 'max_tokens':prompt['max_tokens'],
              'temperature':0, 'ignore_eos':True, 'stream':True, 'stream_options':{'include_usage':True}}) as response:
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            body = raw[5:].strip()
            if body == b'[DONE]':
                finished = True
                break
            row = json.loads(body)
            if row.get('usage'):
                usage = row['usage']
            for c in row.get('choices',[]):
                if c.get('text'):
                    chunks.append(time.monotonic_ns())
                    text_hash.update(c['text'].encode())
                    if initial is not None: initial.set()
    end = time.monotonic_ns()
    assert finished and chunks and usage['prompt_tokens'] == len(prompt['prompt']) and usage['completion_tokens'] == prompt['max_tokens']
    return {'name':name,'request_sha256':prompt['sha256'],'output_sha256':text_hash.hexdigest(),
            'input_tokens':len(prompt['prompt']),'output_tokens':prompt['max_tokens'],'start_ns':start,
            'end_ns':end,'content_chunk_ns':chunks,'ttft_ms':(chunks[0]-start)/1e6,'elapsed_ms':(end-start)/1e6}


def run(label):
    folder = ROOT/label
    folder.mkdir()
    inputs = build_inputs()
    cases = [('chat-single',['chat-single']),('code-single',['code-single']),
             ('mixed',['chat-mixed','code-mixed']),('followups',['chat-return','code-return'])]
    for stage,names in cases:
        before = snapshot()
        assert all(before[f'{dp}/num_requests_{s}']==0 for dp in ('0','1') for s in ('running','waiting'))
        begin = time.monotonic_ns()
        if len(names) == 1:
            results = [generate(names[0],inputs[names[0]])]
        else:
            event = Event()
            with ThreadPoolExecutor(2) as pool:
                a = pool.submit(generate,names[0],inputs[names[0]],event)
                b = pool.submit(generate,names[1],inputs[names[1]],None,event)
                results = [a.result(),b.result()]
            assert results[0]['content_chunk_ns'][0] <= results[1]['start_ns'] < results[0]['end_ns']
        stop = time.monotonic_ns()
        for _ in range(40):
            after = snapshot()
            delta = {k:after[k]-before.get(k,0) for k in after}
            def count(name): return sum(delta.get(dp+'/'+name,0) for dp in ('0','1'))
            if count('request_success_total') == len(names) and count('generation_tokens_total') == sum(r['output_tokens'] for r in results):
                break
            time.sleep(.25)
        save(folder/(stage+'.json'), {'stage':stage,'begin_ns':begin,'stop_ns':stop,'requests':results,'delta':delta})
        assert count('request_success_total')==len(names) and count('num_preemptions_total')==0
        assert count('prompt_tokens_total')==sum(r['input_tokens'] for r in results)
        assert count('generation_tokens_total')==sum(r['output_tokens'] for r in results)
        assert count('iteration_tokens_total_count')>0


if __name__ == '__main__':
    run(sys.argv[1])
