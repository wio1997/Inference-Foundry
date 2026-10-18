#!/usr/bin/env python3
"""Inspect installed native Python types in a device-free container; never construct them."""
import argparse,ast,datetime,hashlib,importlib.metadata,inspect,json,pathlib,sys,types
p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--stdout',action='store_true');p.add_argument('--expected-hashes',required=True);args=p.parse_args()
expected=json.loads(args.expected_hashes);snapshot=pathlib.Path(args.source_root)
nodes=sorted({str(x) for pattern in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for x in pathlib.Path('/dev').glob(pattern)});assert not nodes
dist=importlib.metadata.distribution('torch-npu');root=pathlib.Path(dist.locate_file('')).resolve();hashes={}
for f,h in expected.items():
 b=(root/f).read_bytes();hashes[f]=hashlib.sha256(b).hexdigest();assert hashes[f]==h and hashlib.sha256((snapshot/f).read_bytes()).hexdigest()==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'correctness':'PASS','probe_status':'PENDING','source_hashes':hashes,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'distribution_version':dist.version,'accelerator_device_nodes':nodes,'scope':'Native class descriptors only inside new CPU-only runc container. No Stream/Event instances, no device query/tensor, no transfer adapter import, no model or NPU kernel work. Class API evidence does not prove live service object identity or branch execution.'}
try:
 import torch
 import torch_npu
except Exception as exc:
 result.update({'probe_status':'UNAVAILABLE','import_error_type':type(exc).__name__,'import_error':str(exc)[:3000],'torch_npu_import_completed':False})
else:
 Stream=torch_npu.npu.Stream;Event=torch_npu.npu.Event;sentinel=object()
 def has(cls,name):return inspect.getattr_static(cls,name,sentinel) is not sentinel
 result.update({'probe_status':'INTROSPECTED','torch':torch.__version__,'torch_npu_import_completed':True,'stream_mro':[c.__module__+'.'+c.__name__ for c in Stream.__mro__],'event_mro':[c.__module__+'.'+c.__name__ for c in Event.__mro__],'stream_has_wait':has(Stream,'wait'),'event_has_wait':has(Event,'wait'),'stream_has_wait_event':has(Stream,'wait_event'),'stream_has_wait_stream':has(Stream,'wait_stream'),'stream_has_getattr':has(Stream,'__getattr__'),'cuda_Stream_is_npu_Stream_before_adapter':torch.cuda.Stream is Stream,'npu_initialized_module_flag':torch_npu.npu.__dict__.get('_initialized')})
 assert result['event_has_wait'] and result['stream_has_wait_event'] and result['stream_has_wait_stream']
 assert torch_npu.npu.__dict__.get('_initialized') is False
 source=inspect.getsource(Stream.wait_event);result['actual_wait_event_source_ast_sha256']=hashlib.sha256(ast.dump(ast.parse(source.lstrip()),include_attributes=False).encode()).hexdigest()
 trace=[]
 class EventSpy:
  def wait(self,receiver):trace.append(['event.wait','main'])
 class StreamSpy:
  def record_event(self):trace.append(['alt.record_event']);return EventSpy()
 receiver=types.SimpleNamespace();receiver.wait_event=types.MethodType(Stream.wait_event,receiver)
 Stream.wait_stream(receiver,StreamSpy());assert trace==[['alt.record_event'],['event.wait','main']];result['wait_stream_python_delegate_trace']=trace
 try:Stream.wait_event(receiver,StreamSpy())
 except AttributeError as exc:result['stream_spy_as_event_error']={'type':type(exc).__name__,'message':str(exc)}
 else:raise AssertionError('Stream-like object without wait unexpectedly accepted')
 assert torch_npu.npu.__dict__.get('_initialized') is False
print(json.dumps(result))
