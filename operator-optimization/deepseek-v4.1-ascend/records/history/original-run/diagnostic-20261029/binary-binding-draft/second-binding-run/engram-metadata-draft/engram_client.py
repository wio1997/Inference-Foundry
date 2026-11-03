"""Frozen 8-request diagnostic client; no profiler endpoints, no text logs.

Intended only for an independently reviewed owned diagnostic container. This
module is never executed by the draft's static or CPU-only tests.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import stat
import sys
import threading
import time
import urllib.request
import uuid

if sys.flags.optimize:
    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')

APP=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash')
INPUTS=Path('/bench-inputs/workload/frozen')
URL='http://127.0.0.1:8172'
RUN=os.environ.get('WIO_ENGRAM_RUN_ID','')
if len(RUN)!=32 or any(c not in '0123456789abcdef' for c in RUN):
    raise RuntimeError('missing diagnostic run identity')
OUT=APP/'engram-client-results'
GATE=APP/'metadata/PROFILE_ACTIVE'
METRICS=('prefix_cache_queries_total','prefix_cache_hits_total','prompt_tokens_total',
         'generation_tokens_total','request_success_total','num_preemptions_total',
         'spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total')
FROZEN_FILES={
 'warmup':{'seed':'de3aa10aefe1287fca42ad831cc8bdc27dc0118b6dd2be68f7329f8a39890fc0',
           'requests':'c5fb896a2d9d5fe13c0b27eb1cf92932edb31bf3cd2007de1f8a60b5900c20e8'},
 'measure1':{'seed':'6dafe45dd0225129ac9e3e79193619a284df4ea410bd48437739c0eb03c029b5',
             'requests':'7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404',
             'references':'c00806e9438e56bc7485dfba73a36ab7eea3bf08009fe167129d4d03cabb69f4'}}


def save(obj):
    tmp=OUT/'progress.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,OUT/'progress.json')


def request(path,body=None,headers=None,timeout=900):
    req=urllib.request.Request(URL+path,data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type':'application/json',**(headers or {})})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout)


def metrics():
    with request('/metrics',timeout=15) as response:raw=response.read().decode()
    out={name:0.0 for name in METRICS};seen={name:0 for name in METRICS}
    for row in raw.splitlines():
        for name in METRICS:
            if row.startswith('vllm:'+name+'{') or row.startswith('vllm:'+name+' '):
                out[name]+=float(row.rsplit(' ',1)[1]);seen[name]+=1
    if any(seen[n]==0 for n in METRICS):raise RuntimeError('required metric missing')
    return out


def complete(row,phase,index,stream):
    rid='wio-engram-'+RUN+'-'+phase+'-'+str(index)
    body={'model':'dsv41','prompt':row['prompt'],'max_tokens':row['max_tokens'],
          'temperature':0,'ignore_eos':True,'stream':stream,'request_id':rid}
    if stream:body['stream_options']={'include_usage':True}
    headers={'X-data-parallel-rank':str(row['dp_rank'])} if row.get('dp_rank') is not None else {}
    prediction='';usage=None
    with request('/v1/completions',body,headers) as response:
        if not stream:
            data=json.load(response);usage=data.get('usage');prediction=data['choices'][0]['text']
        else:
            for line in response:
                text=line.decode()
                if not text.startswith('data:'):continue
                text=text[5:].strip()
                if text=='[DONE]':continue
                data=json.loads(text)
                for choice in data.get('choices',[]):prediction+=choice.get('text','')
                if data.get('usage'):usage=data['usage']
    if not isinstance(usage,dict):raise RuntimeError('response usage missing')
    return {'usage':usage,'prediction':prediction}


def parallel(rows,phase,stream=False):
    if not 1<=len(rows)<=8:raise RuntimeError('invalid frozen parallel request count')
    barrier=threading.Barrier(len(rows))
    def worker(pair):
        index,row=pair;barrier.wait(timeout=30);return complete(row,phase,index,stream)
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:return list(pool.map(worker,enumerate(rows)))


def gate_open(window):
    if GATE.exists() or GATE.is_symlink():raise RuntimeError('preexisting gate')
    tmp=GATE.parent/('.gate-'+uuid.uuid4().hex)
    data=(json.dumps({'run_id':RUN,'window_id':window,'event':'open',
                      'monotonic_ns':time.monotonic_ns()})+'\n').encode()
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        if os.write(fd,data)!=len(data):raise RuntimeError('gate short write')
        os.fsync(fd)
    finally:os.close(fd)
    try:os.link(tmp,GATE,follow_symlinks=False)
    finally:tmp.unlink(missing_ok=True)


def gate_close(window):
    try:fd=os.open(GATE,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    except FileNotFoundError:return
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise RuntimeError('invalid gate type')
        data=json.loads(os.read(fd,4096))
        if data.get('run_id')!=RUN or data.get('window_id')!=window:raise RuntimeError('gate identity mismatch')
        current=os.lstat(GATE)
        if (current.st_dev,current.st_ino)!=(s.st_dev,s.st_ino):raise RuntimeError('gate replaced')
        GATE.unlink()
    finally:os.close(fd)


def phase(name,window=None):
    folder=INPUTS/name
    files=list(folder.rglob('*.requests.jsonl'))
    if len(files)!=1:raise RuntimeError('ambiguous frozen requests')
    items={'seed':folder/'warmup.json','requests':files[0]}
    if window is not None:items['references']=folder/'frozen-vllm-details.jsonl'
    for key,path in items.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_FILES[name][key]:raise RuntimeError('frozen file mismatch')
    seeds=json.loads(items['seed'].read_text())
    seeded=parallel(seeds,name+'-seed')
    if any(x['usage'].get('prompt_tokens')!=29696 or x['usage'].get('completion_tokens')!=1 for x in seeded):
        raise RuntimeError('seed usage mismatch')
    rows=[json.loads(line) for line in files[0].read_text().splitlines()]
    if len(rows)!=8:raise RuntimeError('wrong frozen request count')
    digest=hashlib.sha256(json.dumps([x['question'] for x in rows],ensure_ascii=False).encode()).hexdigest()
    before=metrics()
    try:
        if window is not None:gate_open(window)
        output=parallel([{'prompt':x['question'],'max_tokens':1024} for x in rows],name,stream=window is not None)
    finally:
        if window is not None:gate_close(window)
    after=metrics()
    if any(x['usage'].get('prompt_tokens')!=32768 or x['usage'].get('completion_tokens')!=1024 for x in output):
        raise RuntimeError('measurement usage mismatch')
    delta={n:after[n]-before[n] for n in METRICS}
    if any(delta[k]!=v for k,v in {'prompt_tokens_total':262144,'generation_tokens_total':8192,
        'request_success_total':8,'prefix_cache_queries_total':262144,'prefix_cache_hits_total':235904,
        'num_preemptions_total':0}.items()):raise RuntimeError('frozen counter mismatch')
    record={'prompt_sha256':digest,'counter_deltas':delta,'request_count':len(output)}
    if window is not None:
        refs={str(x['data_id']):x['prediction'] for x in map(json.loads,(folder/'frozen-vllm-details.jsonl').read_text().splitlines())}
        record['exact_frozen_prediction_matches']=sum(x['prediction'].encode()==refs[str(i)].encode() for i,x in enumerate(output))
    return record


def main():
    OUT.mkdir(exist_ok=False)
    state={'state':'STARTING','run_id':RUN,'window_id':None,'not_a_benchmark_score':True}
    save(state)
    try:
        with request('/health',timeout=10) as response:
            if response.status!=200:raise RuntimeError('diagnostic not healthy')
        state['warmup']=phase('warmup');state['state']='WARMUP_COMPLETE';save(state)
        window=uuid.uuid4().hex;state['window_id']=window;state['state']='MEASURING';save(state)
        state['measure1']=phase('measure1',window);state['state']='COMPLETE';save(state)
    except BaseException as exc:
        state.update(state='FAILED',error_type=type(exc).__name__);save(state);raise


if __name__=='__main__':main()
