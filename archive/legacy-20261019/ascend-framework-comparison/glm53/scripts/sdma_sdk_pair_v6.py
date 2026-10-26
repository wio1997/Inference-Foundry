#!/usr/bin/env python3
"""Small SDMA correctness probe, explicitly NOT model PD validation or a benchmark."""
import faulthandler,hashlib,json,os,socket,sys,threading,time,traceback
faulthandler.enable(all_threads=True)
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import urlopen,Request

ROLE=sys.argv[1];SOURCE=sys.argv[2];LOCAL=sys.argv[3]
CONTROL=38221;STORE=38220;SESSION_PORT=38222 if ROLE=='Prefill' else 38223
SIZE=4*1024*1024
state={};done=threading.Event();failed=threading.Event();cleaned=threading.Event()
def event(kind,**fields):
 print('GLM53_SDMA_EVENT='+json.dumps({'project_id':'glm53','role':ROLE,'event':kind,'monotonic_ns':time.monotonic_ns(),**fields}),flush=True)
def control(path,body=None):
 data=None if body is None else json.dumps(body).encode()
 req=Request('http://%s:%s/%s'%(SOURCE,CONTROL,path),data=data,headers={'Content-Type':'application/json'})
 with urlopen(req,timeout=5) as response:return json.load(response)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  data=json.dumps(state).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path=='/cleanup':
   state['peer_cleanup']=data;cleaned.set()
  else:state['peer_result']=data
  if not data.get('pass'):failed.set()
  self.send_response(200);self.end_headers();self.wfile.write(b'{}');done.set()
def main():
 event('before_torch')
 import torch,torch_npu
 event('torch_imported')
 torch.npu.set_device(0)
 event('device_initialized')
 import acl
 soc=acl.get_soc_name()
 assert isinstance(soc,str) and soc.startswith('Ascend910_93'), 'A3 required; do not invoke 950 provisioning'
 assert os.environ.get('ASCEND_HOME_PATH'), 'Keep native CANN environment'
 event('a3_provisioning_gate',soc=soc,ascend_home=os.environ['ASCEND_HOME_PATH'])
 mf=load_memfabric()
 event('memfabric_imported')
 mf.set_log_level(0);mf.set_conf_store_tls(False,'')
 engine=mf.TransferEngine()
 op=mf.TransferEngine.TransDataOpType.SDMA
 assert int(op.value)==1
 torch.npu.set_device(0)
 event('device_selected',local_ip=LOCAL,device=0,protocol_name=op.name,protocol_value=op.value,
       initialize_signature=engine.initialize.__doc__,model_requests=0)
 server=None
 if ROLE=='Prefill':
  server=ThreadingHTTPServer((SOURCE,CONTROL),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
  state['stage']='initializing'
 else:
  deadline=time.monotonic()+60
  while True:
   try:
    remote=control('state')
    if remote.get('stage')=='ready':break
   except Exception:pass
   if time.monotonic()>deadline:raise TimeoutError('source readiness timeout')
   time.sleep(.25)
 session='%s:%s'%(LOCAL,SESSION_PORT)
 rc=engine.initialize('tcp://%s:%s'%(SOURCE,STORE),session,ROLE,0,op,store_server_role='Prefill')
 event('initialize_completed',rc=rc,session=session,rpc_port=engine.get_rpc_port(),sdk_error=mf.get_last_err_msg())
 if rc:raise RuntimeError('SDMA initialize failed: %s'%rc)
 # Distinct nonzero deterministic bytes catch no-op and stale transfers.
 expected=((torch.arange(SIZE,dtype=torch.int32)*17+29)%251).to(torch.uint8)
 raw=torch.empty(SIZE+2*1024*1024,dtype=torch.uint8,device='npu')
 offset=(-raw.data_ptr())%(2*1024*1024);buffer=raw[offset:offset+SIZE]
 buffer.copy_(expected if ROLE=='Prefill' else torch.zeros_like(expected));torch.npu.synchronize()
 rc=engine.batch_register_memory([buffer.data_ptr()],[SIZE]);event('register_completed',rc=rc,bytes=SIZE,ptr=buffer.data_ptr())
 if rc:raise RuntimeError('SDMA register failed: %s'%rc)
 digest=hashlib.sha256(expected.numpy().tobytes()).hexdigest()
 if ROLE=='Prefill':
  state.update(stage='ready',session=session,ptr=buffer.data_ptr(),bytes=SIZE,sha256=digest,protocol='SDMA')
  event('source_ready',**state)
  if not done.wait(100):raise TimeoutError('destination completion timeout')
  event('peer_completion',result=state.get('peer_result'))
  if failed.is_set():raise RuntimeError('peer reported failure')
  if not cleaned.wait(60):raise TimeoutError('destination SDK cleanup timeout')
 else:
  # Control only carries address/size/hash; all data bytes below use the SDK SDMA READ.
  assert remote['bytes']==SIZE and remote['protocol']=='SDMA'
  start=time.monotonic_ns()
  rc=engine.batch_transfer_sync_read(remote['session'],[buffer.data_ptr()],[remote['ptr']],[SIZE])
  torch.npu.synchronize()
  event('sdma_read_completed',rc=rc,physical_bytes=SIZE,source_session=remote['session'],destination_session=session,elapsed_ns=time.monotonic_ns()-start,sdk_error=mf.get_last_err_msg())
  actual=buffer.cpu();wrong=int((actual!=expected).sum().item());got=hashlib.sha256(actual.numpy().tobytes()).hexdigest()
  result={'pass':rc==0 and wrong==0 and got==remote['sha256'],'wrong_bytes':wrong,'sha256':got,'bytes':SIZE,'protocol':'SDMA','rc':rc}
  event('data_verified',**result);control('done',result)
  if not result['pass']:raise RuntimeError('SDMA payload mismatch')
 maps=sorted(set(x.split()[-1] for x in open('/proc/self/maps') if any(s in x for s in ['libmf_','libhcom','_pymf_transfer','libascend_hal'])))
 event('loaded_libraries',paths=maps)
 # Release SDK mappings while torch backing storage and runtime are still alive.
 event('before_uninitialize')
 engine.unInitialize();event('uninitialize_completed')
 del engine
 if ROLE=='Decode':control('cleanup',{'pass':True})
 else:server.shutdown()
 event('probe_pass',qualification='SDK_ONLY_NOT_MODEL_PD')
if __name__=='__main__':
 try:main()
 except Exception:
  event('probe_failed',traceback=traceback.format_exc())
  if ROLE=='Decode':
   try:control('done',{'pass':False,'error':traceback.format_exc()})
   except Exception:pass
  sys.exit(1)
