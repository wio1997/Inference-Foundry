"""Task-only persistent CPU config store, no Torch or accelerator device nodes."""
import datetime,importlib.metadata,json,os,pathlib,sys,threading,traceback
root=pathlib.Path('/work');state={'status':'STARTING','store_url':'tcp://172.16.12.46:39440','model_loaded':False,'NPU_devices':False,'performance':'NOT_MEASURED'}
def write():
 p=root/'state.tmp';p.write_text(json.dumps(state));os.replace(p,root/'state.json')
try:
 assert not any(pathlib.Path('/dev').glob('davinci*'));write()
 import memfabric_hybrid
 state['MemFabric']=importlib.metadata.version('memfabric-hybrid');state['common_init_return']=memfabric_hybrid.initialize();assert state['common_init_return']==0
 state['create_config_store_return']=memfabric_hybrid.create_config_store(state['store_url']);assert state['create_config_store_return']==0
 state.update({'status':'READY','torch_imported':'torch' in sys.modules,'torch_npu_imported':'torch_npu' in sys.modules,'ready_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});write();print(json.dumps(state),flush=True)
 threading.Event().wait()
except BaseException as e:state['status']='FAILED';state['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};write();print(json.dumps(state),flush=True);sys.exit(1)
