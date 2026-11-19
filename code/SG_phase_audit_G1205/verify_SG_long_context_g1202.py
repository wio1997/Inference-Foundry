"""Replay the four original exact-length vLLM retrieval cases through the verified SG pair."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import urllib.error
import urllib.request

MODEL='ds41flash'
LENGTHS={32768,65536,131072,262144}
MANIFEST_SHA='2b2a82681901b3e19e0ed8b799b1e12085a9f7bd4f2caf90d30d62750e484b5d'
PREFIX_KEY='sglang:prefill_effective_tokens_total'
SOURCES=('device_hit','host_hit','storage_hit')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def fetch(url, payload=None, timeout=10):
    req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as f:
        assert f.status==200
        return f.read()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def counter(text, mode):
    expected={'engine_type':'prefill','model_name':MODEL,'moe_ep_rank':'0',
              'pp_rank':'0','tp_rank':'0','mode':mode}
    found=[]
    for line in text.decode().splitlines():
        if not line.startswith(PREFIX_KEY+'{'):
            continue
        left,right=line.rsplit(' ',1)
        labels=dict(re.findall(r'([\w_]+)="([^"]*)"',left))
        if all(labels.get(k)==v for k,v in expected.items()):
            found.append(float(right))
    assert len(found)==1,(mode,found)
    return found[0]

def main(args):
    assets=args.assets.resolve(strict=True)
    gate=json.loads(args.gate.read_text())
    report=json.loads(args.runtime_proof.read_text())
    assert gate['engine']=='sglang' and gate['profile_epoch']==1178
    assert gate['model_id']==MODEL and all(gate[k] for k in ('ready','pd_transfer_verified','configuration_verified','resources_verified'))
    assert gate['max_model_len'] >= 262144+128 and gate['workload_guard_actual_verified']
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(gate['validated_at'])).total_seconds()
    assert -60<=age<600,age
    assert report['math1']=='91' and report['completed']==3
    for key in ('transfer_count_delta_by_P_rank','D_full_graph_counter_delta_by_rank'):
        assert set(report[key])=={str(i) for i in range(8)} and all(x>0 for x in report[key].values())
    raw_manifest=(assets/'manifest.json').read_bytes()
    assert sha(raw_manifest)==MANIFEST_SHA
    manifest=json.loads(raw_manifest)
    assert {v['input_tokens'] for v in manifest['records']}==LENGTHS and manifest['model']==MODEL
    assert manifest['benchmark_prefixes_checked']==448
    for row in manifest['records']:
        assert sha((assets/row['request_file']).read_bytes())==row['request_sha256']
    output=args.output.resolve()
    output.mkdir(exist_ok=False)
    def store(name,value):
        p=output/name
        tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        tmp.replace(p)
    def alarm(*_):
        raise TimeoutError('Bounded long-context job reached its deadline')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(3600)
    prefix=gate['proxy_url'].rstrip('/')
    roles=gate['roles']
    assert set(roles)=={'prefill','decode'}
    store('manifest.json',{'at':now(),'request_manifest_sha256':MANIFEST_SHA,'gates':gate,
        'one_attempt_per_case':True,'selected_lengths':sorted(LENGTHS),
        'scope':'Selected strict long-context retrieval; semantic and measured reuse evidence for the tested prompts.'})
    def snap(tag):
        result={}
        for role,v in roles.items():
            base=v['base_url'].rstrip('/')
            fetch(base+'/ready')
            raw=fetch(base+'/metrics')
            (output/f'{tag}_{role}.prom').write_bytes(raw)
            result[role]=raw
        return result
    complete=[]
    try:
        assert json.loads(fetch(prefix+'/v1/models'))['data'][0]['id']==MODEL
        for row in manifest['records']:
            length=row['input_tokens']
            req=(assets/row['request_file']).read_bytes()
            expected=row['expected']
            prior=None
            for kind in ('fresh','replay'):
                name=f'{kind}_{length}'
                (output/f'{name}_request.json').write_bytes(req)
                before=snap(name+'_before')
                store('state.json',{'phase':'request_running','case':name,'at':now(),'completed':len(complete)})
                try:
                    raw=fetch(prefix+'/v1/completions',req,timeout=900)
                except urllib.error.HTTPError as error:
                    (output/f'{name}_HTTP_error.txt').write_bytes(error.read())
                    raise
                (output/f'{name}_response.json').write_bytes(raw)
                response=json.loads(raw)
                assert response['model']==MODEL and len(response['choices'])==1
                choice=response['choices'][0]
                after=snap(name+'_after')
                v=choice['text'].strip()
                usage=response['usage']
                local={mode:counter(after['prefill'],mode)-counter(before['prefill'],mode) for mode in SOURCES}
                cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                evidence={'case':name,'input_tokens':length,'strict_retrieval_passed':v==expected,
                    'valid_usage':usage.get('prompt_tokens')==length and 0<usage.get('completion_tokens',0)<=128,
                    'finish':choice['finish_reason'],'expected':expected,'actual':v,'usage':usage,
                    'prefill_same_rank_local_cache_deltas':local,'response_cached_tokens_from_PD':cached,
                    'decode_local_hits_separately_measured':False,'response_SHA256':sha(raw),
                    'request_SHA256':sha(req)}
                store(f'{name}_audit.json',evidence)
                assert all(x>=0 for x in local.values()) and (cached is None or 0<=cached<=length)
                assert evidence['strict_retrieval_passed'] and evidence['valid_usage'] and evidence['finish']=='stop',evidence
                if kind=='fresh':
                    assert sum(local.values())==0 and cached in (None,0),evidence
                else:
                    assert prior==v
                prior=v
                complete.append(evidence)
        assert len(complete)==8
        store('result.json',{'phase':'complete_requires_parent_review','selected_cases_passed':True,'cases':complete,
            'cache_scope':'P local source counters plus response cache usage. Decode local hits are not separately inferred.'})
        store('state.json',{'phase':'complete_requires_parent_review','completed':len(complete),'at':now()})
    except BaseException as exc:
        store('state.json',{'phase':'failed_requires_review','completed':len(complete),'at':now(),'error':repr(exc)})
        raise

if __name__=='__main__':
    assert socket.gethostname()=='<ROLE_HOSTNAME>'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--runtime-proof',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
