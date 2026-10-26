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
 # Bound audit RPCs even when a native80Kpacket exceeds64MiB/4096ranges.
 for lengths in [[8*1024*1024-1],[8*1024*1024],[8*1024*1024+1],[72*1024*1024+3],[1]*1025,[13]*4097]:
  ls=[1000+i*1000000000 for i in range(len(lengths))];rs=[900000+i*1000000000 for i in range(len(lengths))];chunks=list(m.audit_chunks(ls,rs,lengths));cursor=offset=0
  for left,right,sizes in chunks:
   assert 0<len(sizes)<=1024 and sum(sizes)<=32*1024*1024
   assert len(json.dumps({'addresses':right,'sizes':sizes}).encode())<=128*1024
   for l,r,n in zip(left,right,sizes):
    assert l==ls[cursor]+offset and r==rs[cursor]+offset and 0<n<=8*1024*1024 and offset+n<=lengths[cursor]
    offset+=n
    if offset==lengths[cursor]:cursor+=1;offset=0
  assert cursor==len(lengths) and offset==0
 cases.append('six_large_and_many_range_audit_partitions_cover_every_byte_exactly')
 import io,hashlib,urllib.request
 def digest(sizes):return {'range_sha256':[hashlib.sha256(str(n).encode()).hexdigest() for n in sizes],'sha256':hashlib.sha256(json.dumps(sizes).encode()).hexdigest()}
 rpc_bodies=[]
 def fake_urlopen(request,timeout):
  q=json.loads(request.data);rpc_bodies.append(q);return io.BytesIO(json.dumps(digest(q['sizes'])).encode())
 e.byte_audit=True;e.hasher=types.SimpleNamespace(hash_ranges=lambda ls,ns:digest(ns));e._engine.ret=0
 ls=[1000,1000000000];rs=[900000,2000000000];ns=[72*1024*1024+3,17];old_reads=len(e._engine.reads);old_events=len(events)
 peer={'172.16.12.14:24401':{'global_device_ordinal':0,'url':'http://172.16.12.14:18500/hash'}}
 with mock.patch.object(pathlib.Path,'read_text',return_value=json.dumps(peer)),mock.patch.object(urllib.request,'urlopen',side_effect=fake_urlopen):
  assert e.batch_transfer_sync_read('172.16.12.14:24401',ls,rs,ns)==0
 assert len(e._engine.reads)==old_reads+1 and e._engine.reads[-1]==('172.16.12.14:24401',ls,rs,ns)
 audited=[v for v in events[old_events:] if v['kind']=='qualification_payload_checksum'];assert len(audited)==3 and all(v['passed'] for v in audited) and sum(v['bytes'] for v in audited)==sum(ns) and len(rpc_bodies)==6
 cases.append('audit1_72MiB_six_hash_RPCs_but_exactly_one_original_SDMA_read')
 e.emit_summary();assert events[-1]['successful_bytes']==e._stats['successful_bytes']
out={'project_id':'glm53-flash','passed':True,'cases':cases,'case_count':len(cases),'actual_SDK_NPU_network_model_calls':0,'scope_limit':'CPU mocks only; no actual DP/graph/MTP/SDMA/performance qualification'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
