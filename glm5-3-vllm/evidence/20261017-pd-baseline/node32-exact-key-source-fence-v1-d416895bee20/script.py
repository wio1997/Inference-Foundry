#!/usr/bin/env python3
"""Exact traced31-key minimal B contrast on existing models, with bounded native LRU eviction."""
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request
import uuid

D_BLOCKS=1288


def main():
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
    base=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    previous=base/'tests/post-pool-state-control-v1'
    test=json.loads((previous/'result.json').read_text());assert test['completed'] and len(test['waves'])==4
    selected=test['waves'][0];assert not selected['pool_matches_P_native']
    original=json.loads((previous/'wave-0-producer.json').read_text())['request']
    native=json.loads((previous/'D-native-reference.json').read_text())['response']
    p_old=json.loads((previous/'wave-0-P-native-reference.json').read_text())['response']
    fixture=json.loads((base/'tests/baseline-c1-v1/fixture-4096-128.json').read_text())
    assert native['choices'][0]['token_ids']==p_old['choices'][0]['token_ids']==fixture['reference_ids']
    out=base/'tests/exact-key-source-fence-v1';out.mkdir(exist_ok=False)
    keys=[]
    for path in (base/'pd-node32-v15/kv-control-audit').glob('buffers-*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['phase']=='native_prefix' and r['tp_rank']==0 and r['request_id'].startswith(selected['P_native_id']+'-'):keys.append(r['key'])
    prefix='GLM-5.3-w8a8c8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'
    assert len(keys)==len(set(keys))==31 and all(k.startswith(prefix) and re.fullmatch('[0-9a-f]{64}',k[len(prefix):]) for k in keys)
    layouts=[json.loads(f.read_text()) for f in (base/'pd-node32-v15/kv-control-audit').glob('layout-*.json')]
    assert len(layouts)==16
    P_blocks=max(r['num_blocks'] for r in layouts);assert P_blocks==1108
    P_evict=math.ceil(P_blocks/31);D_evict=math.ceil(D_BLOCKS/31)
    deadline=time.monotonic()+1200
    report={'completed':False,'waves':[],'selected_historical_salt':original['cache_salt'],'key_set_sha256':hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            'keys':keys,'P_blocks':P_blocks,'D_blocks':D_BLOCKS,'P_eviction_requests_per_arm':P_evict,'D_eviction_requests_per_arm':D_evict,
            'D_native_control_id':test['D_native_control_id'],'D_native_reference_origin':'Same D14 native and existing controls from post-pool-state-control-v1, original D golden preserved.',
            'scope':'One B on exact same31 traced keys after the recorded failing A; each arm first evicts local P/D caches using ordinary unique-salt native requests, then metadata-only remove_batch on these31 keys with exist1/remove0/absent0/close0 gates. Initial P1 cold and immediate D128 external3968 required; after D, P128/Dlocal8 four-view controls. Native eviction requests are not correctness/performance evidence. No model reload/new endpoint/RPC/kernel/precision or transport change. Independent pool reads are separately counted diagnostics.'}
    def rpc(node,payload):
        port=29481 if node==32 else 29482
        headers={'Content-Type':'application/json'}
        if 'request_id' in payload:headers['X-Request-Id']=payload['request_id']
        q=urllib.request.Request(f'http://172.16.12.{node}:{port}/v1/completions',data=json.dumps(payload).encode(),headers=headers)
        with urllib.request.urlopen(q,timeout=min(90,max(1,deadline-time.monotonic()))) as response:return json.load(response)
    def save(name,payload,value):
        (out/(name+'.json')).write_text(json.dumps({'request':payload,'response':value},indent=2)+'\n')
    def metrics(node,name):
        with urllib.request.urlopen(f'http://172.16.12.{node}:{29481 if node==32 else 29482}/metrics',timeout=10) as response:raw=response.read().decode()
        (out/(name+'.txt')).write_text(raw);values={}
        for line in raw.splitlines():
            if not line or line.startswith('#'):continue
            k,v=line.rsplit(' ',1);k=k.split('{',1)[0];values[k]=values.get(k,0)+float(v)
        return values
    def delta(a,b,key):return b.get(key,0)-a.get(key,0)
    def remove_exact(arm):
        root=out/f'remove-{arm}';root.mkdir(exist_ok=False)
        (root/'keys.json').write_text(json.dumps(keys)+'\n')
        code='''import json
from pathlib import Path
from memcache_hybrid import DistributedObjectStore
keys=json.loads(Path('/out/keys.json').read_text())
assert len(keys)==len(set(keys))==31
store=DistributedObjectStore();report={}
try:
 report['init']=store.init(0,init_bm=False);assert report['init']==0
 report['before']=list(store.batch_is_exist(keys));assert report['before']==[1]*31
 report['remove']=list(store.remove_batch(keys));assert report['remove']==[0]*31
 report['after']=list(store.batch_is_exist(keys));assert report['after']==[0]*31
finally:
 report['close']=store.close()
 Path('/out/result.json').write_text(json.dumps(report,indent=2)+'\\n')
assert report['close']==0
print(json.dumps(report),flush=True)
'''
        (root/'probe.py').write_text(code)
        image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
        fix=Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3')
        assert hashlib.sha256((fix/'aliases.json').read_bytes()).hexdigest()=='2a10023d36c2480332a24dadd2e24142a68d07a04a9cf358ced1f16adf167d03'
        env=json.loads(subprocess.check_output(['docker','image','inspect',image,'--format','{{json .Config.Env}}']))
        ld=next((x.split('=',1)[1] for x in env if x.startswith('LD_LIBRARY_PATH=')), '')
        py=next((x.split('=',1)[1] for x in env if x.startswith('PYTHONPATH=')), '')
        config='/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-config-node32-fixed-v3'
        argv=['docker','run','--rm','--pull','never','--network','host','--read-only','--tmpfs','/tmp:rw,size=128m','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','2','--memory','2g','--pids-limit','512','-v',str(root)+':/out','-v',str(fix/'site-packages')+':/opt/memcache-runtime:ro','-v',config+':/memcache-config:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONPATH=/opt/memcache-runtime'+(':'+py if py else ''),'-e','LD_LIBRARY_PATH=/opt/memcache-runtime/memcache_hybrid/lib:/opt/memcache-runtime/memfabric_hybrid/lib:'+ld,'-e','MMC_LOCAL_CONFIG_PATH=/memcache-config/mmc-local.conf','-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/usr/bin/timeout',image,'--signal=TERM','--kill-after=5','40','python3','/out/probe.py']
        p=subprocess.run(argv,capture_output=True,text=True,timeout=50)
        (root/'stdout.txt').write_text(p.stdout);(root/'stderr.txt').write_text(p.stderr)
        (root/'launch.json').write_text(json.dumps({'returncode':p.returncode,'argv':argv},indent=2)+'\n')
        assert p.returncode==0,'Exact-key metadata operation failed; inspect immutable remove subdirectory, do not blindly retry'
        return json.loads((root/'result.json').read_text())
    def source_controls(rid):
        rows=[]
        for path in (base/'pd-node32-v15/kv-control-audit').glob('buffers-*.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row['phase']=='native_prefix' and row['request_id'].startswith(rid+'-'):rows.append(row)
        return rows
    try:
        for arm,fenced in enumerate([True]):
            assert deadline-time.monotonic()>200
            for node,count in [(32,P_evict),(33,D_evict)]:
                before=metrics(node,f'arm-{arm}-node{node}-before-eviction')
                for key in ('vllm:num_requests_running','vllm:num_requests_waiting'):assert before.get(key,0)==0
                for i in range(count):
                    payload=dict(original,max_tokens=1,cache_salt=f'evict-{node}-'+uuid.uuid4().hex,request_id='glm53-evict-'+uuid.uuid4().hex)
                    value=rpc(node,payload);save(f'arm-{arm}-node{node}-evict-{i:02d}',payload,value)
                    assert value['usage']['prompt_tokens']==4096
                after=metrics(node,f'arm-{arm}-node{node}-after-eviction')
                assert delta(before,after,'vllm:prefix_cache_hits_total')==0
                assert delta(before,after,'vllm:ascend_store_load_get_keys_total')==0
                print(f'arm{arm}: node{node} {count} native eviction requests complete',flush=True)
            removal=remove_exact(arm)
            p_before=metrics(32,f'arm-{arm}-P-before-cold')
            d_before=metrics(33,f'arm-{arm}-D-before-cold')
            producer_payload=dict(original,request_id=('glm53-source-ready-' if fenced else 'glm53-source-control-')+uuid.uuid4().hex)
            producer=rpc(32,producer_payload)
            consumer_payload=dict(producer_payload,max_tokens=128);consumer_payload.pop('request_id',None)
            pool=rpc(33,consumer_payload)
            save(f'wave-{arm}-producer',producer_payload,producer);save(f'wave-{arm}-pool',consumer_payload,pool)
            p_after=metrics(32,f'arm-{arm}-P-after-cold');d_after=metrics(33,f'arm-{arm}-D-after-cold')
            assert delta(p_before,p_after,'vllm:prefix_cache_hits_total')==0
            assert delta(d_before,d_after,'vllm:prefix_cache_hits_total')==0
            assert delta(d_before,d_after,'vllm:ascend_store_load_get_keys_total')==496
            p_payload=dict(consumer_payload,request_id='glm53-postkv-source-'+uuid.uuid4().hex)
            p_native=rpc(32,p_payload);save(f'wave-{arm}-P-native-reference',p_payload,p_native)
            p_local=metrics(32,f'arm-{arm}-P-after-native')
            assert delta(p_after,p_local,'vllm:prefix_cache_hits_total')==3968
            source=source_controls(p_native['id']);assert len(source)==496 and {r['tp_rank'] for r in source}==set(range(16))
            d_payload=dict(consumer_payload,max_tokens=8,logprobs=20,request_id='glm53-postkv-target-'+uuid.uuid4().hex)
            d_local=rpc(33,d_payload);save(f'wave-{arm}-D-pool-local',d_payload,d_local)
            d_final=metrics(33,f'arm-{arm}-D-after-local')
            assert delta(d_after,d_final,'vllm:prefix_cache_hits_total')==3968
            assert delta(d_after,d_final,'vllm:ascend_store_load_get_keys_total')==0
            row={'wave':arm,'source_fence_selected':fenced,'producer_id':producer['id'],'consumer_id':pool['id'],'P_native_id':p_native['id'],'D_local_id':d_local['id'],
                 'pool_matches_P_native':pool['choices'][0]['token_ids']==p_native['choices'][0]['token_ids'],
                 'pool_matches_original_D_reference':pool['choices'][0]['token_ids']==fixture['reference_ids'],
                 'pool_matches_current_D_native':pool['choices'][0]['token_ids']==native['choices'][0]['token_ids'],
                 'P_native_matches_original_D_reference':p_native['choices'][0]['token_ids']==fixture['reference_ids'],
                 'source_control_records':len(source),'P_initial_local_hits':0,'D_initial_local_hits':0,'D_model_loader_get_keys':496,
                 'metadata_keys_removed':len(removal['remove'])}
            report['waves'].append(row);(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END',flush=True)
    assert report['completed'],report.get('failure')


if __name__=='__main__':main()
