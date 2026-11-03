"""CPU mocks: forwarding, privacy, gated recording, concurrency and caps."""
from pathlib import Path
import importlib,multiprocessing as mp,os,sys,tempfile,types,json
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parent));import engram_metadata as m
class Storage:pass
class Device:type='npu';index=8
class Tensor:
 def __init__(self, shape, stride, store):self.shape=shape;self._stride=stride;self.dtype='torch.int32';self.device=Device();self.store=store;self.secret='PRIVATE_TENSOR_CONTENT_SHOULD_NEVER_APPEAR'
 def stride(self):return self._stride
 def storage_offset(self):return 24
 def is_contiguous(self):return False
 def untyped_storage(self):return self.store
class Model:
 def embed_gathered(self,gathered,num_tokens):
  if num_tokens==0:raise ValueError('original error')
  return gathered
sys.modules['torch']=types.SimpleNamespace(Tensor=Tensor)
with tempfile.TemporaryDirectory() as td:
 os.environ.pop('WIO_ENGRAM_META_ENABLE',None);assert m.install(Model) is False
 os.environ['WIO_ENGRAM_META_ENABLE']='1';os.environ['WIO_DIAG_DIR']=td;os.environ['WIO_ENGRAM_META_PATH']=td+'/engram-metadata.jsonl'
 assert m.install(Model) is True and m.install(Model) is False
 s=Storage();t=Tensor((8184,24),(48,1),s);obj=Model()
 assert obj.embed_gathered(t,8184) is t and not Path(td+'/engram-metadata.jsonl').exists()
 Path(td+'/PROFILE_ACTIVE').touch()
 assert obj.embed_gathered(t,8184) is t and obj.embed_gathered(t,8184) is t
 try:obj.embed_gathered(t,0)
 except ValueError as e:assert str(e)=='original error'
 else:raise AssertionError('original exception lost')
 # Metadata exceptions cannot change the original return.
 with mock.patch.object(m._STATE,'metadata',side_effect=RuntimeError('diagnostic')):assert obj.embed_gathered(t,8184) is t
 with mock.patch.object(m._STATE,'write_locked',side_effect=RuntimeError('diagnostic')):assert obj.embed_gathered(t,8184) is t
 m._STATE.finish();data=Path(td+'/engram-metadata.jsonl').read_bytes();rows=[json.loads(x) for x in data.splitlines()]
 begins=[r for r in rows if r['event']=='embed_begin'];ends=[r for r in rows if r['event']=='embed_end'];assert len(begins)==4 and len(ends)==4
 assert begins[0]['tensor']['device_type']=='npu' and begins[0]['tensor']['device_index']==8
 assert begins[0]['tensor']['shape']==[8184,24] and begins[0]['tensor']['stride']==[48,1] and begins[0]['tensor']['storage_offset']==24
 assert begins[0]['tensor']['tensor_instance']==begins[1]['tensor']['tensor_instance'] and begins[0]['tensor']['storage_wrapper_instance']==begins[1]['tensor']['storage_wrapper_instance']
 assert ends[2]['outcome']=='raised' and begins[3]['tensor'] is None
 assert b'PRIVATE_TENSOR_CONTENT' not in data and b'_cdata' not in data and b'data_ptr' not in data
 print(json.dumps({'mode':'forward_identity_gate_privacy','status':'PASS','records':len(rows),'bytes':len(data)}))
 # Exercise fixed identity capacity with live tensor/storage objects.
 observer=m._Observer(td+'/engram-metadata.jsonl',td);keep=[]
 for i in range(300):
  store=Storage();tensor=Tensor((i+1,24),(48,1),store);keep.append((tensor,store));observer.metadata(tensor)
 assert len(observer.ids)<=m.MAX_IDS and observer.stats['identity_evicted']>0
 print(json.dumps({'mode':'bounded_live_identities','status':'PASS','slots':len(observer.ids),'evictions':observer.stats['identity_evicted']}))
# Multiprocess shared cap and bounded JSON serialization; children reset after fork.
def child(td):
 m.MAX_BYTES=16384
 st=Storage();t=Tensor((4888,24),(48,1),st);obj=Model()
 for _ in range(250):obj.embed_gathered(t,4888)
 if m._STATE is not None:m._STATE.finish()
with tempfile.TemporaryDirectory() as td:
 os.environ['WIO_DIAG_DIR']=td;os.environ['WIO_ENGRAM_META_PATH']=td+'/engram-metadata.jsonl';Path(td+'/PROFILE_ACTIVE').touch()
 m._after_fork();ctx=mp.get_context('fork');ps=[ctx.Process(target=child,args=(td,)) for _ in range(4)]
 for p in ps:p.start()
 for p in ps:p.join(20);assert p.exitcode==0
 raw=Path(td+'/engram-metadata.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
 assert len(raw)<=16384 and sum(r['event']=='trace_cap' for r in rows)==1 and rows[-1]['event']=='trace_cap'
 print(json.dumps({'mode':'multiprocess_shared_cap','status':'PASS','bytes':len(raw),'records':len(rows)}))
