import pathlib,json,hashlib,socket,re,subprocess,time,urllib.request
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030');R=P/'analysis/whole-scope-continuation-20261030';R.mkdir(parents=True,exist_ok=True)
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b';ids={'wio-vllm-v41-repro166-20261030':'8757dcc7a2f8db672bbf6ddfc1ed29e6cff74d5dd89199770b7b6f3b27fd48e3','wio-vllm-v41-repro166-stage-20261030':'bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'}
cs={}
for n,cid in ids.items():
 c=json.loads(subprocess.check_output(['docker','inspect',n]))[0];assert c['Id']==cid and c['Image']==IMAGE and c['State']['Running'];cs[n]={'id':cid,'image':c['Image'],'state':c['State'],'command':c['Config']['Cmd']}
base=list(ids)[0];stage=list(ids)[1];smi=subprocess.check_output(['npu-smi','info'],text=True);pids=set();chips=set();health=[]
for l in smi.splitlines():
 a=[x.strip() for x in l.split('|')]
 if len(a)>4 and re.fullmatch(r'\d+\s+\d+',a[1]) and a[2].isdigit():pids.add(int(a[2]));chips.add(tuple(map(int,a[1].split())))
 if len(a)>3 and 'Ascend910' in a[1]:health.append(a[2])
own={int(x) for x in subprocess.check_output(['docker','top',base,'-eo','pid'],text=True).splitlines() if x.strip().isdigit()};assert pids and pids<=own,('Other NPU tasks',sorted(pids-own));assert len(health)==16 and all(x=='OK' for x in health),health
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));http={}
for endpoint in ['health','v1/models','metrics']:
 with op.open('http://127.0.0.1:30075/'+endpoint,timeout=10) as response:text=response.read().decode();http[endpoint]={'status':response.status,'body':text if endpoint!='metrics' else None}
 if endpoint=='metrics':http[endpoint]['requests']=[l for l in text.splitlines() if not l.startswith('#') and any(k in l.split('{')[0] for k in ['num_requests_running','num_requests_waiting'])]
assert http['metrics']['requests'] and all(float(x.rsplit(None,1)[-1])==0 for x in http['metrics']['requests']),'Service has in-flight requests'
pkg='''import importlib.metadata as m,json,pathlib,hashlib,importlib.util
p=pathlib.Path(m.distribution('torch_npu').locate_file('torch_npu/npu/graphs.py'))
print(json.dumps({'torch_npu_version':m.version('torch_npu'),'graphs_path':str(p),'graphs_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'torchair_spec':str(importlib.util.find_spec('torchair'))}))'''
q=subprocess.run(['docker','exec',stage,'python3','-c',pkg],capture_output=True,text=True,timeout=25);assert q.returncode==0,q.stderr
installed=json.loads(q.stdout)
serve=hashlib.sha256((P/'runtime/scripts/serve.sh').read_bytes()).hexdigest();assert serve=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
out={'status':'PASS_READ_ONLY_RESUME_PREFLIGHT','asset_ip':'10.55.0.166','actual_hostname':socket.gethostname(),'project':str(P),'analysis_root':str(R),'containers':cs,'all_npu_processes_owned':True,'logical_chips_covered':len(chips),'health':health,'npu_processes':sorted(pids),'http':http,'installed':installed,'serve_sha256':serve,'driver':pathlib.Path('/usr/local/Ascend/driver/version.info').read_text(),'user_resumption':'开始基于这个继续任务','new_measurements_must_use_166_environment_id':True,'historical_169_timings_not_relabelled':True,'checked_unix':time.time(),'source_archive_root':'/data/tiankuan/wio/migration-166-20261030/source169-project','overall_complete':False}
(R/'resume-preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
