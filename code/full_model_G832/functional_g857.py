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
RUN=ROOT/'sg_pd_g832'
OUT=RUN/'functional_PD_P832_D832_g857'
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
    assert s['generation']==832 and s['phase']=='model_starting'
    os.kill(s['pid'],0)
    info=json.loads(subprocess.check_output(['docker','inspect',s['container']],text=True,cwd=ROOT))[0]
    assert info['Id']==s['container_id'] and info['State']['Running']
    assert info['Config']['Labels']['codex.task']==ROOT.name and info['Config']['Labels']['codex.run']==RUN.name
    return s,info

try:
    state('waiting_specific_P832_D832_pair',max_ready_wait_s=1200)
    deadline=time.monotonic()+1200
    while True:
        own_P()
        try:
            info={role:json.loads(http(base+'/server_info',timeout=3)) for role,base in [('p',P),('d',D)]}
            for role,base in [('p',P),('d',D)]:
                http(base+'/ready',timeout=30)
            break
        except (urllib.error.URLError,TimeoutError,socket.timeout):
            assert time.monotonic()<deadline,'Specific P832_D832 readiness observation bound; no restart'
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
    argv=['flock','--nonblock','/root/.codex/coord/<ROLE_HOSTNAME>-port18090.lock','docker','exec','-w','/task/sg_pd_g832',s['container'],'python3','-m','sglang_router.launch_router','--host','0.0.0.0','--port','18090','--pd-disaggregation','--prefill',P,'18998','--decode',D,'--max-concurrent-requests','16','--queue-size','100','--request-timeout-secs','3600','--disable-retries','--health-check-endpoint','/ready']
    with (RUN/'router.log').open('x') as log:
        router=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (RUN/'router_handle.json').write_text(json.dumps(dict(pid=router.pid,argv=argv,container_id=container['Id'],at=datetime.datetime.now(datetime.timezone.utc).isoformat(),lifecycle='Owned P832_D832 P container, port flock lasts for client process'),indent=2))
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
    for tag,prompt,answer in [('math','计算 2+2。最终答案只写一个数字。','4'),('capital','What is the capital of France? Reply with only the city name.','Paris')]:
        body={'model':served_model,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':0,'max_tokens':512,'stream':False,'chat_template_kwargs':{'thinking':False}}
        save(tag+'_request.json',body);state('selected_natural_request_running',case=tag)
        raw=http(BASE+'/v1/chat/completions',body,timeout=180);(OUT/(tag+'_response.json')).write_bytes(raw);response=json.loads(raw)
        text=response['choices'][0]['message'].get('content') or '';final=text.split('</think>')[-1].strip().rstrip('。.!！ ');assert final==answer,(tag,final)
        assert response['usage']['completion_tokens']>0 and response['usage'].get('reasoning_tokens',0)==0,response
        audit={'case':tag,'passed':True,'usage':response['usage'],'response_sha256':hashlib.sha256(raw).hexdigest()};save(tag+'_audit.json',audit);audits.append(audit)
    text='SGLang832 native protocol check. Explain how a matrix multiplication combines rows and columns.'
    body={'model':served_model,'prompt':text,'temperature':0,'seed':0,'max_tokens':1024,'min_tokens':1024,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
    save('usage_request.json',body);state('standard_SSE_1024_request_running')
    raw=http(BASE+'/v1/completions',body,timeout=1200);(OUT/'usage_response.sse').write_bytes(raw)
    events=[line[5:].strip() for line in raw.decode().splitlines() if line.startswith('data:')];assert events[-1]=='[DONE]'
    objects=[json.loads(line) for line in events if line!='[DONE]'];usage=[x['usage'] for x in objects if x.get('usage')][-1];assert usage['prompt_tokens']>0 and usage['completion_tokens']==1024,usage
    save('usage_audit.json',{'passed':True,'usage':usage,'SSE_DONE':True,'sha256':hashlib.sha256(raw).hexdigest(),'scope':'ShortvalidUTF8textprotocolprobe; actualAISBenchhandlerbindingandfull32Kcontractstillrequired'})
    for role,base in [('p',P),('d',D)]:save('after_'+role+'_server_info.json',json.loads(http(base+'/server_info')))
    save('result.json',{'passed':True,'cases':audits,'usage':usage,'scope':'Two matched natural-text answers and1024-outputSSE only; notfullGSMquality/longcontext/C16performance/profile'})
    state('selected_functional_and_usage_passed_requires_parent_review')
except BaseException as exc:
    state('failed_requires_review',error=repr(exc));raise
