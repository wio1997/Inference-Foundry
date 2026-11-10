"""Bounded profiling clone of the owned 171 service, with unconditional restoration."""
import concurrent.futures,copy,datetime,fcntl,hashlib,http.client as http_client,json,os,re,shlex,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
ROOT=Path(sys.argv[1]);ROOT.mkdir(parents=True,exist_ok=True)
BASE='wio-vllm-v41-repro171-20261102';NAME='wio-idle-scopes171-'+ROOT.name[-16:]
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
PORT=30076;APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
HOSTROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime')
assert ROOT.is_relative_to(HOSTROOT)
CROOT=APP+'/'+str(ROOT.relative_to(HOSTROOT))
state={'status':'PREFLIGHT','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_host':str(ROOT),'output_container':CROOT,'original':BASE,'diagnostic':NAME,'phases':[]}
def write(name,obj):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def status(s):state['status']=s;write('receipt.json',state)
def command(args,timeout=90,check=True):return subprocess.run(args,capture_output=True,text=True,timeout=timeout,check=check)
def inspect(name):return json.loads(command(['docker','inspect',name]).stdout)[0]
def http(path,body=None,port=PORT,timeout=120):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}));data=None if body is None else json.dumps(body).encode()
 return op.open(urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Content-Type':'application/json'}),timeout=timeout)
def health(port):
 try:
  with http('/health',port=port,timeout=5) as r:return r.status==200
 except:return False
def idle():
 with http('/metrics',port=30075,timeout=8) as r:s=r.read().decode()
 rows=[float(l.rsplit(' ',1)[1]) for l in s.splitlines() if re.match(r'vllm:num_requests_(running|waiting)\{',l)]
 return len(rows)>=4 and all(v==0 for v in rows)
def wait_health(port,seconds):
 until=time.monotonic()+seconds
 while time.monotonic()<until:
  if health(port):return
  time.sleep(5)
 raise RuntimeError(f'health timeout port {port}')
lock=open(HOSTROOT/'framework-profile-controller.lock','a');status('WAITING_FOR_PREVIOUS_RESTORE');fcntl.flock(lock,fcntl.LOCK_EX)
old=inspect(BASE);assert old['State']['Running'] and old['Image']==IMAGE
assert idle(),'other requests active'
serve=(HOSTROOT/'scripts/serve.sh').read_text();assert serve.count('exec /usr/local/python3.12.13/bin/vllm serve ')==1
profile={'profiler':'torch','torch_profiler_dir':CROOT+'/traces','torch_profiler_with_stack':True,'torch_profiler_with_memory':False,'torch_profiler_record_shapes':False,'ignore_frontend':True,'torch_profiler_dump_cuda_time_total':False,'max_iterations':12}
serve_diagnostic=serve.replace('exec /usr/local/python3.12.13/bin/vllm serve ', 'export VLLM_SERVER_DEV_MODE=1\nexport VLLM_CUSTOM_SCOPES_FOR_PROFILING=1\nexec /usr/local/python3.12.13/bin/vllm serve ')
script=serve_diagnostic.rstrip()+' --profiler-config '+shlex.quote(json.dumps(profile))+'\n'
(ROOT/'serve-profile.sh').write_text(script)
state['profiler_config']=profile;state['base_serve_sha256']=hashlib.sha256(serve.encode()).hexdigest();state['load']='frozen C8 / 32768 input / 1024 output / 90% cached prefix';state['capture']='CPU frontend full-request; CPU+NPU workers first 12 iterations including prefill and early decode'
inputs=HOSTROOT/'benchmark-20261028/workload/frozen/measure1'
original_rows=[json.loads(l) for l in next(inputs.rglob('*.requests.jsonl')).read_text().splitlines()]
original_seeds=json.loads((inputs/'warmup.json').read_text())
assert len(original_rows)==8 and len(original_seeds)==2
rows=[dict(original_rows[i],dp_rank=dp,pair_id=i) for dp in [0,1] for i in range(4)]
seeds=[dict(seed,dp_rank=dp) for dp in [0,1] for seed in original_seeds]
state['request_assignment']='first 4 frozen prompts mirrored across DP0/DP1; X-data-parallel-rank; same texts and same two seed types on each DP'
state['capture']='one steady CPU+NPU window; max 12 worker iterations; VLLM_CUSTOM_SCOPES_FOR_PROFILING=1; metrics verify DP4+4; ignore frontend'
state['scope_flag']='VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 (diagnostic instance only)'
state['input_sha256']=hashlib.sha256(next(inputs.rglob('*.requests.jsonl')).read_bytes()).hexdigest()
cfg=copy.deepcopy(old['Config']);hc=copy.deepcopy(old['HostConfig']);cfg['Image']=IMAGE;cfg['Hostname']=''
cfg['Entrypoint']=['bash'];cfg['Cmd']=['-c',f'exec bash {shlex.quote(CROOT+"/serve-profile.sh")} >{shlex.quote(CROOT+"/service.log")} 2>&1']
hc['RestartPolicy']={'Name':'no','MaximumRetryCount':0};hc['AutoRemove']=False
hc['PortBindings']={'8172/tcp':[{'HostIp':'127.0.0.1','HostPort':str(PORT)}]}
for m in hc.get('Mounts',[]):
 if m['Target']=='/tmp':
  tmp=ROOT/'tmp';tmp.mkdir();tmp.chmod(0o1777);m['Source']=str(tmp)
assert hc['NetworkMode']=='bridge'
class UnixConnection(http_client.HTTPConnection):
 def connect(self):
  import socket
  self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.connect('/var/run/docker.sock')
def docker_create():
 c=UnixConnection('localhost');body=json.dumps(dict(cfg,HostConfig=hc)).encode();c.request('POST','/v1.41/containers/create?name='+NAME,body,{'Content-Type':'application/json'});r=c.getresponse();s=r.read();assert r.status==201,s.decode()[:400];return json.loads(s)['Id']
METRIC_NAMES=['num_requests_running','num_requests_waiting','request_success_total','prompt_tokens_total','generation_tokens_total','prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total','spec_decode_num_drafts_total','spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total']
import threading

def metrics():
    with http('/metrics',timeout=20) as response:
        raw=response.read().decode()
    result={name:{'0':0.0,'1':0.0} for name in METRIC_NAMES}
    seen=set()
    for line in raw.splitlines():
        if not line.startswith('vllm:') or line.startswith('#'):
            continue
        match=re.match(r'vllm:(\w+)\{([^}]+)\}\s+([\d.eE+-]+)',line)
        if not match or match[1] not in result:
            continue
        engine=re.search(r'engine="([01])"',match[2])
        if engine:
            result[match[1]][engine[1]]+=float(match[3])
            seen.add(match[1])
    return {key:value for key,value in result.items() if key in seen}

def delta(before,after):
    return {key:{dp:after[key][dp]-before[key][dp] for dp in ['0','1']} for key in before if key in after}

def reset_cache():
    for attempt in range(10):
        with http('/reset_prefix_cache',{},timeout=120) as response:
            result=json.load(response)
        if result.get('success'):
            return
        time.sleep(.5)
    raise RuntimeError('prefix cache reset did not succeed')

def batch(data,label,stream=True,first_events=None):
    start=time.monotonic_ns()
    barrier=threading.Barrier(len(data))
    monitor_stop=threading.Event()
    samples=[]
    def monitor():
        while not monitor_stop.is_set():
            try:
                current=metrics()
                samples.append({'mono_ns':time.monotonic_ns(),'running':current.get('num_requests_running'),'waiting':current.get('num_requests_waiting'),'generated':current.get('generation_tokens_total')})
            except Exception as exc:
                samples.append({'mono_ns':time.monotonic_ns(),'error':str(exc)})
            monitor_stop.wait(.1)
    poller=threading.Thread(target=monitor,daemon=True)
    def one(index,row):
        prompt=row.get('question',row.get('prompt'))
        body={'model':'dsv41','prompt':prompt,'max_tokens':row.get('max_tokens',1024),'temperature':0,'ignore_eos':True,'stream':stream,'request_id':f'balanced-{ROOT.name}-{label}-{index}'}
        if stream:
            body['stream_options']={'include_usage':True}
        headers={'Content-Type':'application/json','X-data-parallel-rank':str(row['dp_rank'])}
        req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',data=json.dumps(body).encode(),headers=headers)
        barrier.wait()
        result={'index':index,'dp_rank':row['dp_rank'],'pair_id':row.get('pair_id'),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'start_mono_ns':time.monotonic_ns(),'start_wall_ns':time.time_ns()}
        pieces=[]
        chunks=[]
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=900) as response:
            if not stream:
                decoded=json.load(response)
                result['usage']=decoded['usage']
                pieces=[choice.get('text','') for choice in decoded.get('choices',[])]
            else:
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    line=line[5:].strip()
                    if line==b'[DONE]':
                        break
                    packet=json.loads(line)
                    text=''.join(choice.get('text','') for choice in packet.get('choices',[]))
                    if text:
                        now=time.monotonic_ns()
                        if 'first_token_mono_ns' not in result:
                            result['first_token_mono_ns']=now
                            if first_events:
                                first_events[index].set()
                        pieces.append(text)
                        chunks.append(now)
                    if packet.get('usage'):
                        result['usage']=packet['usage']
        text=''.join(pieces)
        result.update(end_mono_ns=time.monotonic_ns(),end_wall_ns=time.time_ns(),output_sha256=hashlib.sha256(text.encode()).hexdigest(),output_chars=len(text),chunks_mono_ns=chunks)
        return result
    before=metrics()
    poller.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(data)) as pool:
            results=list(pool.map(lambda item:one(*item),enumerate(data)))
    finally:
        monitor_stop.set()
        poller.join(timeout=25)
    after=metrics()
    diff=delta(before,after)
    obj={'label':label,'start_mono_ns':start,'end_mono_ns':time.monotonic_ns(),'before':before,'after':after,'counter_deltas':diff,'requests':results,'samples':samples}
    if len(data)==8 and stream:
        checks={
            'eight_requests':len(results)==8,
            'lengths_32k_1k':all(r.get('usage',{}).get('prompt_tokens')==32768 and r.get('usage',{}).get('completion_tokens')==1024 for r in results),
            'four_success_each_dp':all(diff.get('request_success_total',{}).get(dp)==4 for dp in ['0','1']),
            'input_tokens_equal':all(diff.get('prompt_tokens_total',{}).get(dp)==131072 for dp in ['0','1']),
            'output_tokens_equal':all(diff.get('generation_tokens_total',{}).get(dp)==4096 for dp in ['0','1']),
            'cache_hits_equal':diff.get('prefix_cache_hits_total',{}).get('0')==diff.get('prefix_cache_hits_total',{}).get('1'),
            'observed_four_running_each':any(s.get('running',{}).get('0')==4 and s.get('running',{}).get('1')==4 for s in samples if s.get('running')),
            'matching_inputs':all(results[i]['prompt_sha256']==results[i+4]['prompt_sha256'] for i in range(4)),
            'matching_outputs':all(results[i]['output_sha256']==results[i+4]['output_sha256'] for i in range(4)),
            'no_preemption':all(v==0 for v in diff.get('num_preemptions_total',{}).values())
        }
        obj['balance_checks']=checks
        obj['batch_seconds']=(obj['end_mono_ns']-start)/1e9
        obj['output_tokens_per_second']=8192/obj['batch_seconds']
        obj['validated_balance']=all(v for k,v in checks.items() if k!='matching_outputs')
    write(label+'.json',obj)
    if len(data)==8 and stream and not obj['validated_balance']:
        raise RuntimeError('balance validation failed: '+json.dumps(obj['balance_checks']))
    return obj

def prepare(label):
    reset_cache()
    batch(seeds,label+'-seed',False)

def record_phase(label,result):
    state['phases'].append({'phase':label,'batch_seconds':result['batch_seconds'],'output_tokens_per_second':result['output_tokens_per_second'],'balance_checks':result['balance_checks'],'counter_deltas':result['counter_deltas']})
    write('receipt.json',state)

created=None
stopped=False
profiling=False
try:
    status('CREATING_PROFILING_CLONE')
    created=docker_create()
    state['diagnostic_id']=created
    assert idle(),'request arrived before stop'
    status('STOPPING_OWNED_SERVICE')
    command(['docker','stop','-t','120',BASE],timeout=150)
    stopped=True
    status('LOADING_PROFILING_SERVICE')
    command(['docker','start',created])
    wait_health(PORT,2400)
    with http('/openapi.json',timeout=30) as response:
        routes=json.load(response)['paths']
    assert all(path in routes for path in ['/start_profile','/stop_profile','/reset_prefix_cache'])
    status('WARMUP_BALANCED')
    prepare('warmup')
    batch(rows,'warmup-full')
    for round_index in range(1):
        label=f'balanced-reference-{round_index}'
        status(label.upper())
        prepare(label)
        record_phase(label,batch(rows,label))
    status('PREPARING_BALANCED_STEADY')
    prepare('balanced-steady')
    first_events=[threading.Event() for _ in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as client_pool:
        future=client_pool.submit(batch,rows,'balanced-steady',True,first_events)
        deadline=time.monotonic()+120
        while not all(event.is_set() for event in first_events):
            if future.done():
                future.result()
                raise RuntimeError('requests ended before all first visible outputs')
            if time.monotonic()>deadline:
                raise RuntimeError('first output timeout')
            time.sleep(.05)
        time.sleep(2)
        assert not future.done(),'request ended before steady sampling'
        status('COLLECTING_BALANCED_STEADY')
        state['steady_start_metrics']=metrics()
        with http('/start_profile',{},timeout=180) as response:
            assert response.status==200
        profiling=True
        started=time.time_ns()
        time.sleep(2)
        with http('/stop_profile',{},timeout=1200) as response:
            assert response.status==200
        profiling=False
        state.setdefault('capture_windows',[]).append({'name':'steady','start_wall_ns':started,'stop_wall_ns':time.time_ns(),'scope':'after all first outputs plus 2 seconds; max 12 worker iterations'})
        result=future.result(timeout=900)
    record_phase('balanced-steady',result)
    status('CAPTURE_COMPLETE')
except BaseException as exc:
    state['error']=str(exc)
    state['traceback']=traceback.format_exc()
    status('FAILED_RECOVERING')
finally:
    if profiling:
        try:
            with http('/stop_profile',{},timeout=1200) as response:
                state['final_profile_stop']=response.status
        except Exception as exc:
            state['final_profile_stop_error']=str(exc)
    if created:
        command(['docker','stop','-t','90',created],timeout=120,check=False)
    if stopped:
        status('RESTORING_ORIGINAL')
        command(['docker','start',BASE],timeout=120)
        wait_health(30075,2400)
    state['original_running']=inspect(BASE)['State']['Running']
    state['original_health_200']=health(30075)
    state['original_serve_unchanged']=hashlib.sha256((HOSTROOT/'scripts/serve.sh').read_bytes()).hexdigest()==state['base_serve_sha256']
    status('FAILED_RESTORED' if 'error' in state else 'CAPTURED_RESTORED')
    print(json.dumps({key:state[key] for key in ['status','output_host','original_health_200']}),flush=True)
