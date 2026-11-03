import json,pathlib,subprocess,re,socket,time,urllib.request,hashlib,datetime,uuid
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
R=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');B='wio-vllm-v41-repro166-20261030';S='wio-vllm-v41-repro166-stage-20261030';IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b';IDS={B:'8757dcc7a2f8db672bbf6ddfc1ed29e6cff74d5dd89199770b7b6f3b27fd48e3',S:'bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'}
runid='dump-api-probe-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6];hostroot=R/runid;hostroot.mkdir();outpath='/work/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/'+runid+'/device'
EXPECTED='f8ba6f797542c3a2ee874f4e3e764c33f7926cb536ec7c9af1160110e6c1aa5d'
def call(args,t=25):return subprocess.run(args,capture_output=True,text=True,timeout=t)
def guard():
 containers={}
 for n in [B,S]:
  c=json.loads(subprocess.check_output(['docker','inspect',n]))[0];assert c['Id']==IDS[n] and c['Image']==IMAGE and c['State']['Running'];containers[n]={'id':c['Id'],'state':c['State'],'image':c['Image']}
  if n==B:
   env=dict(x.split('=',1) for x in c['Config']['Env']);assert env['CANN_VERSION']=='9.2.0'
 smi=subprocess.check_output(['npu-smi','info'],text=True);pids=set();health=[]
 for line in smi.splitlines():
  a=[x.strip() for x in line.split('|')]
  if len(a)>4 and re.fullmatch(r'\d+\s+\d+',a[1]) and a[2].isdigit():pids.add(int(a[2]))
  if len(a)>3 and 'Ascend910' in a[1]:health.append(a[2])
 own={int(x) for x in subprocess.check_output(['docker','top',B,'-eo','pid'],text=True).splitlines() if x.strip().isdigit()};assert pids and pids<=own,('Other NPU owner',sorted(pids-own));assert health==['OK']*16
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with op.open('http://127.0.0.1:30075/health',timeout=5) as response:assert response.status==200
 with op.open('http://127.0.0.1:30075/metrics',timeout=5) as response:metrics=response.read().decode()
 requests=[l for l in metrics.splitlines() if not l.startswith('#') and any(k in l.split('{')[0] for k in ['num_requests_running','num_requests_waiting'])];assert requests and all(float(x.rsplit(None,1)[-1])==0 for x in requests)
 return {'containers':containers,'npu_pids':sorted(pids),'health':'OK','service_health':200,'request_metrics':requests,'cann_version':env['CANN_VERSION']}
TEST=r'''
import os,pathlib,json,hashlib,time,threading,traceback,shutil,importlib.metadata
root=pathlib.Path(OUTPUT);root.mkdir(parents=True,exist_ok=False)
for n in ['cache','logs','work','torch-cache']:(root/n).mkdir()
os.environ.update(ASCEND_RT_VISIBLE_DEVICES='0',ASCEND_PROCESS_LOG_PATH=str(root/'logs'),ASCEND_CACHE_PATH=str(root/'cache'),ASCEND_WORK_PATH=str(root/'work'),ASCEND_GLOBAL_LOG_LEVEL='3',ASCEND_SLOG_PRINT_TO_STDOUT='0',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_CACHE_DIR=str(root/'torch-cache'))
p=pathlib.Path(importlib.metadata.distribution('torch_npu').locate_file('torch_npu/npu/graphs.py'));assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED
state={'scope':'FINITE_SYNTHETIC_OBSERVER_CAPABILITY_NOT_MODEL_COUNTS_OR_PERFORMANCE','values':[3,8,1],'elements':24,'device':0,'observations':[],'target_service_changed':False}
timer=threading.Timer(50,lambda:os._exit(124));timer.daemon=True;timer.start()
try:
 import torch,torch_npu
 torch.npu.set_device(0)
 free,total=torch.npu.mem_get_info();state['memory_before']={'free':free,'total':total};assert free>2*1024**3,'Insufficient free memory for finite observer probe'
 single=torch.ops.npu.save_npugraph_tensor.default;multiple=torch.ops.npu.save_npugraph_tensor.tensor_list
 src=torch.full((24,),-1,dtype=torch.int32,device='npu:0');out=torch.empty_like(src);epoch=torch.zeros((1,),dtype=torch.int64,device='npu:0')
 out.copy_(src);single(out,save_path=str(root/'warmup.pt'));torch.npu.synchronize()
 graph=torch.npu.NPUGraph()
 with torch.npu.graph(graph):
  out.copy_(src);epoch.add_(1)
  single(out,save_path=str(root/'single.pt'))
  multiple([out,epoch],save_path=str(root/'multiple.pt'))
  out.zero_()
 epoch.zero_();torch.npu.synchronize()
 for ordinal,value in enumerate([3,8,1],1):
  src.fill_(value);graph.replay();torch.npu.synchronize();deadline=time.monotonic()+3;good=None;error=None
  while time.monotonic()<deadline:
   try:
    singles=list(root.glob('single*.pt'));multiples=list(root.glob('multiple*.pt'));assert len(singles)<=8 and len(multiples)<=8
    for a in singles:
     x=torch.load(a,map_location='cpu',weights_only=True)
     if not(isinstance(x,torch.Tensor) and x.dtype==torch.int32 and tuple(x.shape)==(24,) and bool((x==value).all())):continue
     for b in multiples:
      y=torch.load(b,map_location='cpu',weights_only=True)
      if isinstance(y,list) and len(y)==2 and y[0].dtype==torch.int32 and tuple(y[0].shape)==(24,) and bool((y[0]==value).all()) and y[1].dtype==torch.int64 and y[1].tolist()==[ordinal]:
       aa=root/('archived_single_'+str(ordinal)+'.pt');bb=root/('archived_multiple_'+str(ordinal)+'.pt');shutil.copy2(a,aa);shutil.copy2(b,bb);good={'ordinal':ordinal,'expected_value':value,'single_values':x.tolist(),'multiple_values':y[0].tolist(),'epoch':y[1].tolist(),'single_source':a.name,'multiple_source':b.name,'archive_sha256':{aa.name:hashlib.sha256(aa.read_bytes()).hexdigest(),bb.name:hashlib.sha256(bb.read_bytes()).hexdigest()},'snapshot_preserved_before_out_zero':True};break
     if good:break
   except Exception as e:error=repr(e)
   if good:break
   time.sleep(.05)
  assert good,{'ordinal':ordinal,'expected':value,'last_error':error}
  state['observations'].append(good);assert sum(p.stat().st_size for p in root.glob('*.pt'))<1024*1024
 state['status']='PASS_SYNTHETIC_GRAPH_REPLAY_OBSERVER';state['raw_filenames']=[p.name for p in root.glob('single*.pt')]+[p.name for p in root.glob('multiple*.pt')]
except BaseException:state.update(status='FAIL_SYNTHETIC_GRAPH_REPLAY_OBSERVER',error=traceback.format_exc())
finally:
 (root/'result.json').write_text(json.dumps(state,indent=2)+'\n');timer.cancel()
print(json.dumps(state));raise SystemExit(0 if state['status']=='PASS_SYNTHETIC_GRAPH_REPLAY_OBSERVER' else 1)
'''
state={'scope':'SMALL_OWN_STAGE_OBSERVER_PROBE_WITH_OWN_IDLE_BASELINE_PRESERVED','run_id':runid,'host_output':str(hostroot),'before':guard(),'started_unix':time.time()};(hostroot/'plan.json').write_text(json.dumps(state,indent=2))
code='OUTPUT='+repr(outpath)+'\nEXPECTED='+repr(EXPECTED)+'\n'+TEST
q=call(['docker','exec',S,'timeout','--kill-after=5s','60s','python3','-c',code],t=80);state['test']={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
for _ in range(10):
 try:state['after']=guard();break
 except Exception:
  if _==9:raise
  time.sleep(.5)
state['finished_unix']=time.time();state['service_preserved']=True;state['status']='PASS_PROBE' if q.returncode==0 else 'FAIL_PROBE';(hostroot/'host-result.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state));raise SystemExit(q.returncode)
