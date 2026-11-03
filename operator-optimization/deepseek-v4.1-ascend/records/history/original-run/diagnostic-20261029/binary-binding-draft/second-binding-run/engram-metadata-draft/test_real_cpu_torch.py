"""CPU-only synthetic tensor test; never loads torch_npu or NPU devices."""
import json,os,sys,tempfile
from pathlib import Path
import torch
import engram_metadata as observer
assert not any(name=='torch_npu' or name.startswith('torch_npu.') for name in sys.modules)
assert torch.device('cpu').type=='cpu'
class FakeEngram:
 def embed_gathered(self,gathered,num_tokens):return gathered.contiguous(),num_tokens
base=torch.arange(13*2*24,device='cpu',dtype=torch.int32).reshape(13,2,24)
views=[base[:,0,:],base[:,1,:]]
assert all(x.device.type=='cpu' and tuple(x.shape)==(13,24) and tuple(x.stride())==(48,1) and not x.is_contiguous() for x in views)
expected=[(x.contiguous(),13) for x in views]
with tempfile.TemporaryDirectory() as d:
 os.environ.pop('WIO_ENGRAM_META_ENABLE',None);os.environ['WIO_DIAG_DIR']=d;os.environ['WIO_ENGRAM_META_PATH']=d+'/engram-metadata.jsonl'
 before=FakeEngram.__dict__['embed_gathered'];assert observer.install(FakeEngram) is False and FakeEngram.__dict__['embed_gathered'] is before
 os.environ['WIO_ENGRAM_META_ENABLE']='1';assert observer.install(FakeEngram) is True
 Path(d,'PROFILE_ACTIVE').touch();actual=[FakeEngram().embed_gathered(x,13) for x in views]
 assert all(a[1]==e[1] and torch.equal(a[0],e[0]) for a,e in zip(actual,expected))
 observer._STATE.finish();rows=[json.loads(x) for x in Path(d,'engram-metadata.jsonl').read_bytes().splitlines()];begins=[x for x in rows if x['event']=='embed_begin']
 assert len(begins)==2
 for i,b in enumerate(begins):
  t=b['tensor'];assert t['shape']==[13,24] and t['stride']==[48,1] and t['storage_offset']==24*i and t['is_contiguous'] is False
  assert t['device_type']=='cpu' and t['device_index'] is None and t['dtype']=='torch.int32' and isinstance(t['tensor_instance'],int)
  assert t['storage_wrapper_instance'] is None or isinstance(t['storage_wrapper_instance'],int)
 assert 'torch_npu' not in sys.modules
 print(json.dumps({'status':'PASS','torch_version':str(torch.__version__),'device':'cpu','shape':[13,24],'stride':[48,1],'storage_offsets':[0,24],'contiguous':False,'default_disabled':'PASS','original_method_return_equivalence':'PASS','npu_imported':False,'storage_wrapper_identity_is_not_alias_or_lifetime_proof':True,'graph_replay':'not tested'}))
