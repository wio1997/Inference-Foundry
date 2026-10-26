"""Initialize the actual SGLang Ascend backend and register/deregister owned NPU memory."""
import hashlib,importlib.metadata,json,os,pathlib,sys,traceback
v={'status':'STARTING','model_loaded':False,'PD_MTP_serving_executed':False,'device_compute_executed':False,'transport_between_nodes_executed':False,'performance':'NOT_MEASURED','registered':False,'deregistered':False}
try:
 os.environ['ASCEND_MF_TRANSFER_PROTOCOL']='sdma'
 # Use the backend's supported default; never inherit a different task's store URL.
 os.environ.pop('ASCEND_MF_STORE_URL',None)
 import torch,torch_npu
 v['stage']='torch_context_before_memfabric';torch.npu.set_device(0);v['torch_context_initialized_before_memfabric']=True
 v['torch_native_libraries_before_memfabric']=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(n in line for n in ['libascendcl','libascend_hal','libruntime.so','libacl_rt'])})
 import memfabric_hybrid
 v['import_order']=['torch','torch_npu','memfabric_hybrid'];v['memfabric_module_file']=memfabric_hybrid.__file__
 try:v['memfabric_version']=importlib.metadata.version('memfabric-hybrid')
 except importlib.metadata.PackageNotFoundError:v['memfabric_version']='UNKNOWN'
 v['supported_protocol_names']=[n for n in dir(memfabric_hybrid.TransferEngine.TransDataOpType) if n.isupper()]
 v['stage']='after_memfabric_import';v['torch_native_libraries_after_memfabric']=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(n in line for n in ['libascendcl','libascend_hal','libruntime.so','libacl_rt'])});v['device_name']=torch.npu.get_device_name(0)
 from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
 from sglang.srt.disaggregation.utils import DisaggregationMode
 file=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt/disaggregation/ascend/transfer_engine.py');v['source_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
 engine=AscendTransferEngine(hostname='172.16.12.46',npu_id=0,disaggregation_mode=DisaggregationMode.DECODE)
 v['initialized']=True;v['protocol']=engine._get_transfer_protocol();v['session_id']=engine.get_session_id();assert v['protocol']=='sdma' and not v['session_id'].endswith(':0')
 buffer=torch.zeros(4*1024**2,dtype=torch.uint8,device='npu:0');torch.npu.synchronize();v['device_compute_executed']=True
 engine.register(buffer.data_ptr(),buffer.numel());v['registered']=True;v['buffer_bytes']=buffer.numel()
 engine.deregister(buffer.data_ptr());v['deregistered']=True
 del engine,buffer
 torch.npu.synchronize();v['status']='PASS'
except BaseException as e:v['status']='UNAVAILABLE';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print('L039_RESULT='+json.dumps(v));sys.exit(0 if v['status']=='PASS' else 1)
