# SPDX-License-Identifier: Apache-2.0
"""CPU mocks only: ownership/fail-closed/protocol/counters; never SDK/NPU."""
import argparse,ast,importlib.util,json,os,pathlib,sys,types
from unittest import mock
ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();source=pathlib.Path(__file__).with_name('vllm_flash_sdma_performance_engine.py');spec=importlib.util.spec_from_file_location('candidate_sdma',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FakeEngine:
 def __init__(self):self.reads=[];self.regions=[];self.ret=0
 def get_rpc_port(self):return 24400
 def initialize(self,*args,**kwargs):self.binding=(args,kwargs);return 0
 def register_memory(self,p,n):self.regions.append((p,n));return 0
 def batch_transfer_sync_read(self,*v):self.reads.append(v);return self.ret
class Func:
 def __call__(self,*args):return 0
fake=types.ModuleType('memfabric_hybrid');fake.TransferEngine=FakeEngine;fake.set_conf_store_tls=lambda *a:None;fake.set_log_level=lambda *a:None;enum=types.ModuleType('_pymf_transfer');enum.TransDataOpType=types.SimpleNamespace(SDMA='SDMA_ONLY')
cases=[];events=[]
def rejects(name,fn):
 try:fn()
 except (ValueError,RuntimeError):cases.append(name)
 else:raise AssertionError(name)
with mock.patch.dict(sys.modules,{'memfabric_hybrid':fake,'_pymf_transfer':enum}),mock.patch.dict(os.environ,{'ASCEND_MF_STORE_URL':'tcp://172.16.12.14:19048','FLASH_SDMA_VERIFY_BYTES':'0'}),mock.patch('ctypes.CDLL',return_value=types.SimpleNamespace(aclrtSetDevice=Func())),mock.patch.object(m,'event',side_effect=lambda k,**v:events.append({'kind':k,**v})),mock.patch.object(m.atexit,'register'):
 # Global DP identity differs from logical current device after visibility remap.
 r={'tp_size':1,'dp_size':16,'tp_rank':0,'dp_rank':7,'global_device_ordinal':7};g=m.GlobalFlashPerformanceTE();g.configure('Decode',0,r);e=g.get_transfer_engine('172.16.12.15');assert e.device_id==0 and e.rank_info['global_device_ordinal']==7 and e.protocol=='SDMA_ONLY';assert e._engine.binding[1]['data_op_type']=='SDMA_ONLY';cases.append('global_DP7_logical_device0_and_explicit_SDMA_binding')
 g.configure('Decode',0,r);assert g.get_transfer_engine('172.16.12.15') is e;rejects('owner_cannot_change',lambda:g.configure('Decode',1,r))
 g.register_buffer([1000,2000],[512,256]);g.register_buffer([1000,2000],[512,256]);assert e._engine.regions==[(1000,512),(2000,256)];cases.append('native_registration_once')
 assert e.batch_transfer_sync_read('172.16.12.14:24401',[1000],[9000],[32])==0;assert e._engine.reads==[('172.16.12.14:24401',[1000],[9000],[32])];assert e.hasher is e.hash_server is None;cases.append('audit0_exact_SDK_vectors_no_hash_import')
 e._engine.ret=-3;rejects('SDK_failure_raises_no_fallback',lambda:e.batch_transfer_sync_read('172.16.12.14:24401',[1000],[9000],[32]));assert len(e._engine.reads)==2
 rejects('unowned_remote_session',lambda:e.batch_transfer_sync_read('172.16.12.46:24401',[1000],[9000],[32]));rejects('negative_transfer_size',lambda:e.batch_transfer_sync_read('172.16.12.14:24401',[1000],[9000],[-1]));rejects('vector_length_mismatch',lambda:e.batch_transfer_sync_read('172.16.12.14:24401',[1000],[],[32]))
 assert e._stats=={'calls':2,'successful_calls':1,'failed_calls':1,'ranges':2,'bytes':64,'producer_fences':0,'successful_bytes':32};cases.append('attempted_and_successful_byte_counters_distinct')
 rejects('consumer_cannot_publish_producer_fence',e.synchronize_producer_stream)
 for rank in range(2):
  for tp in range(8):
   z=m.GlobalFlashPerformanceTE();z.configure('Prefill',tp,{'tp_size':8,'dp_size':2,'tp_rank':tp,'dp_rank':rank,'global_device_ordinal':rank*8+tp})
 cases.append('all16_official_P_DP2_TP8_rank_bindings')
 rejects('old_diagnostic_DP1_TP16_not_silently_qualified',lambda:m.GlobalFlashPerformanceTE().configure('Prefill',0,{'tp_size':16,'dp_size':1,'tp_rank':0,'dp_rank':0,'global_device_ordinal':0}))
 with mock.patch.dict(os.environ,{'FLASH_SDMA_VERIFY_BYTES':'unexpected'}):rejects('audit_flag_explicit_0_or_1',lambda:m.FlashPerformanceSdmaEngine('172.16.12.15','Decode',0,r))
 with mock.patch.dict(os.environ,{'ASCEND_MF_STORE_URL':'tcp://172.16.12.14:18500'}):rejects('control_store_exact_project_port',lambda:m.FlashPerformanceSdmaEngine('172.16.12.15','Decode',0,r))
 e.emit_summary();assert events[-1]['successful_bytes']==32
out={'project_id':'glm53-flash','passed':True,'cases':cases,'case_count':len(cases),'actual_SDK_NPU_network_model_calls':0,'scope_limit':'CPU mocks only; no actual DP/graph/MTP/SDMA/performance qualification'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
