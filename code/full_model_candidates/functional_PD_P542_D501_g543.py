"""Bounded native PD router + real-model functional checks; no service restart."""
from pathlib import Path
import datetime
import hashlib
import fcntl
import signal
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT=Path('/PRIVATE_TASK_ROOT')
RUN=ROOT/'sg_pd_g542'
OUT=RUN/'functional_PD_P542_D501_g543'
os.chdir(ROOT)
assert socket.gethostname()=='<ROLE_HOSTNAME>'
P='http://<ROLE_HOST>:19093'
D='http://<ROLE_HOST>:19094'
BASE='http://<ROLE_HOST>:18090'
started=time.monotonic()
workload_lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a');fcntl.flock(workload_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
OUT.mkdir(exist_ok=False)


def save(name,data):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)

def deadline_exceeded(sig,frame):
    raise TimeoutError('Functional-only process reached 3600s lifecycle bound')
signal.signal(signal.SIGALRM,deadline_exceeded)
signal.alarm(3600)

def state(phase,**extra):
    data=dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,**extra)
    save('state.json',data);print(json.dumps(data),flush=True)

def http(url,body=None,timeout=20):
    req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as f:
        assert f.status==200
        return f.read()

def own_P():
    s=json.loads((RUN/'state.json').read_text())
    assert s['generation']==542 and s['phase']=='model_starting'
    os.kill(s['pid'],0)
    info=json.loads(subprocess.check_output(['docker','inspect',s['container']],text=True,cwd=ROOT))[0]
    assert info['Id']==s['container_id'] and info['State']['Running']
    assert info['Config']['Labels']['codex.task']==ROOT.name and info['Config']['Labels']['codex.run']==RUN.name
    return s,info

try:
    state('waiting_specific_P542_D501_pair',max_ready_wait_s=1200)
    deadline=time.monotonic()+1200
    while True:
        own_P()
        try:
            info={role:json.loads(http(base+'/server_info',timeout=3)) for role,base in [('p',P),('d',D)]}
            for role,base in [('p',P),('d',D)]:
                http(base+'/health',timeout=30)
            break
        except (urllib.error.URLError,TimeoutError,socket.timeout):
            assert time.monotonic()<deadline,'Specific P542_D501 readiness observation bound; no restart'
            time.sleep(5)
    for role,data in info.items():
        save('startup_'+role+'_server_info.json',data)
        args=data.get('server_args',data)
        assert args['tp_size']==8 and args['ep_size']==8 and args['pp_size']==1
        assert args['context_length']==263296
        assert data['max_total_num_tokens']>=263296 and args['max_running_requests']==16
        if role=='d':assert args['cuda_graph_config']['decode']['backend']=='full' and not args['disable_cuda_graph']
        assert args['disaggregation_mode']==('prefill' if role=='p' else 'decode')
        assert args['disaggregation_transfer_backend']=='ascend'
        assert args['deepep_dispatcher_output_dtype']=='auto' and args['quantization']=='modelslim'
        assert abs(args['mem_fraction_static']-(0.88 if role=='p' else 0.80))<1e-9
        assert args['enable_metrics'] is True and args['enable_cache_report'] is True
        assert args['disable_radix_cache'] is False
        if role=='d':
            assert args['disaggregation_decode_enable_radix_cache'] is True
            assert args['cuda_graph_config']['decode']['max_bs']==16
        assert args['speculative_num_steps']==1
        assert args['disable_overlap_schedule'] is False,'Requested overlap not effective'
        assert args['speculative_algorithm']=='DSPARK' and args['speculative_draft_model_path']=='/models/ds41' and args['speculative_num_draft_tokens']==6,'Native DSPARK5 own-checkpoint geometry changed'
    model_ids={}
    for role,base in [('p',P),('d',D)]:
        models=json.loads(http(base+'/v1/models'))
        save('startup_'+role+'_models.json',models)
        assert models.get('data'),'Actual served model ID missing'
        model_ids[role]=models['data'][0]['id']
    assert model_ids['p']==model_ids['d'],'PD served model IDs differ'
    served_model=model_ids['p']
    s,container=own_P()
    assert not (RUN/'router_handle.json').exists()
    with socket.socket() as sock:sock.bind(('0.0.0.0',18090))
    argv=['flock','--nonblock','/root/.codex/coord/<ROLE_HOSTNAME>-port18090.lock','docker','exec','-w','/task/sg_pd_g542',s['container'],'python3','-m','sglang_router.launch_router','--host','0.0.0.0','--port','18090','--pd-disaggregation','--prefill',P,'18998','--decode',D,'--max-concurrent-requests','16','--queue-size','100','--request-timeout-secs','3600','--disable-retries']
    with (RUN/'router.log').open('x') as log:
        router=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (RUN/'router_handle.json').write_text(json.dumps(dict(pid=router.pid,argv=argv,container_id=container['Id'],at=datetime.datetime.now(datetime.timezone.utc).isoformat(),lifecycle='Owned P542_D501 P container, port flock lasts for client process'),indent=2))
    deadline=time.monotonic()+120
    while True:
        assert router.poll() is None,'Native router exited; inspect log, do not restart'
        try:http(BASE+'/health',timeout=3);break
        except (urllib.error.URLError,TimeoutError,socket.timeout):
            assert time.monotonic()<deadline,'Router health observation bound'
            time.sleep(2)
    deadline=time.monotonic()+180
    while True:
        workers=json.loads(http(BASE+'/workers',timeout=10))
        active={(w['worker_type'],w['url']) for w in workers['workers'] if w['is_healthy'] and w['model_id']==served_model}
        if {('prefill',P),('decode',D)}<=active:
            save('router_workers_before.json',workers)
            break
        assert time.monotonic()<deadline,('Actual native PD workers not registered',workers)
        time.sleep(2)
    state('native_PD_router_ready_not_functional_accepted')
    audits=[]

    def case(tag,path,body,expected_prompt=None,expected_out=None,answer=None):
        assert time.monotonic()-started<3500
        if 'model' in body:body['model']=served_model
        own_P()
        ready_deadline=time.monotonic()+90
        stable=0
        while stable<2:
            w=json.loads(http(BASE+'/workers',timeout=5))
            active={v['worker_type'] for v in w['workers'] if v['is_healthy'] and v['model_id']==served_model and v['load']==0}
            stable=stable+1 if active=={'prefill','decode'} else 0
            assert time.monotonic()<ready_deadline,'Native router eligibility unavailable; no submission'
            if stable<2:time.sleep(5)
        save(tag+'_request.json',body);state('request_running',case=tag)
        before=time.monotonic()
        try:raw=http(BASE+path,body,timeout=300)
        except urllib.error.HTTPError as e:
            (OUT/(tag+'_error.txt')).write_bytes(e.read());raise
        (OUT/(tag+'_response.json')).write_bytes(raw);data=json.loads(raw)
        usage=data.get('usage',data.get('meta_info',{}))
        assert usage.get('completion_tokens',0)>0,(tag,'No actual completion token',usage)
        if expected_prompt is not None:assert usage['prompt_tokens']==expected_prompt,(tag,usage)
        if expected_out is not None:assert usage['completion_tokens']==expected_out,(tag,usage)
        if answer is not None:
            content=data['choices'][0]['message'].get('content') or ''
            final=content.split('</think>')[-1].strip().rstrip('。.!！ ')
            assert final==answer,(tag,final)
        if 'meta_info' in data:assert data['meta_info'].get('num_retractions',0)==0
        audit=dict(case=tag,passed=True,usage=usage,elapsed_s=time.monotonic()-before,response_sha256=hashlib.sha256(raw).hexdigest(),diagnostic_only=True)
        save(tag+'_audit.json',audit);audits.append(audit);state('case_passed',case=tag)

    # Native fixed-image chat encoding resolves deepseek_v41 to dsv41 DeepSeekRecipe.
    # Missing generic HF chat_template alone does not prevent this native encoder.
    # Semantic guard first, then compressed odd tails and chunk boundaries.
    case('math','/v1/chat/completions',dict(model='ds41flash',messages=[dict(role='user',content='计算 2+2。最终答案只写一个数字。')],temperature=0,max_tokens=512,stream=False,thinking={'type':'disabled'}),answer='4')
    for length in [8191,8193,127,129,131]:
        ids=[1000+(i%100) for i in range(length)]
        case('tail_'+str(length),'/generate',dict(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=32,min_new_tokens=32,ignore_eos=True),stream=False),expected_prompt=length,expected_out=32)
    case('openai_short_32','/v1/completions',dict(model='ds41flash',prompt='Explain the role of matrix multiplication in neural networks.',max_tokens=32,min_tokens=32,ignore_eos=True,temperature=0,stream=False),expected_out=32)
    ids=[1000+(i%100) for i in range(32768)]
    case('native_32768_32','/generate',dict(input_ids=ids,sampling_params=dict(temperature=0,max_new_tokens=32,min_new_tokens=32,ignore_eos=True),stream=False),expected_prompt=32768,expected_out=32)
    # Exact same diagnostic bytes, different new request slots. This never
    # uses the reserved AISBench formal dataset.
    for length in [129,8193,32768]:
        original_tag='native_32768_32' if length==32768 else 'tail_'+str(length)
        body=json.loads((OUT/(original_tag+'_request.json')).read_text())
        tag='cache_replay_'+str(length)
        case(tag,'/generate',body,expected_prompt=length,expected_out=32)
        old=json.loads((OUT/(original_tag+'_response.json')).read_text())
        new=json.loads((OUT/(tag+'_response.json')).read_text())
        assert old['text']==new['text'],(tag,'Greedy diagnostic output changed on identical-input cache replay')
        assert new['meta_info']['cached_tokens']>=128,(tag,'Requested prefix cache did not report a nonzero full-page hit',new['meta_info'])
        save(tag+'_coherence_audit.json',dict(passed=True,cached_tokens=new['meta_info']['cached_tokens'],same_greedy_text=True,scope='Selected native response/cache diagnostic. Not all-component bytewise cross-host proof or performance gain.'))
    for role,base in [('p',P),('d',D)]:
        save('after_'+role+'_server_info.json',json.loads(http(base+'/server_info')))
    save('result.json',dict(passed=True,cases=audits,scope='Real native router full model PD protocol and selected semantic/length plus selected cache replay gates only; not C16 performance, all-rank profile, accuracy suite, or direct bytewise proof of every registered state component'))
    hits=list((RUN/'gate_lifetime_hits').glob('*.json')); records=[json.loads(p.read_text()) for p in hits]; assert {x['device_index'] for x in records}==set(range(8)), 'Actual patched Engram path not observed on all8 ranks'; save('engram_lifetime_path_hits.json',records)
    state('functional_cases_passed_requires_PD_and_effective_graph_evidence_review')
except BaseException as exc:
    state('failed_requires_review',error=repr(exc));raise
