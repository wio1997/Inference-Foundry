#!/usr/bin/env python3
"""CPU-only positive/negative tests for Run589 identity capture and patch guards."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj

capture=load('capture589',ROOT/'scripts/loop081_slice_capture_run589.py')
patch=load('patch589',ROOT/'scripts/loop081_slice_patch_run589.py')

class Storage:
    def __init__(self,p): self.p=p
    def data_ptr(self): return self.p
    def nbytes(self): return 4096
class Tensor:
    dtype='torch.bfloat16'
    device='npu:0'
    def __init__(self,p,shape=(96,4096)):
        self.p=p;self.shape=shape
    def stride(self): return (self.shape[-1],1)
    def data_ptr(self): return self.p
    def untyped_storage(self): return Storage(self.p)
    def storage_offset(self): return 0
    def element_size(self): return 2
fake_torch=types.ModuleType('torch')
fake_torch.Tensor=Tensor
fake_torch.npu=types.SimpleNamespace(is_current_stream_capturing=lambda: True)
sys.modules['torch']=fake_torch
tp=types.SimpleNamespace(rank_in_group=0,world_size=8,ranks=list(range(8)))
distributed=types.ModuleType('vllm.distributed')
distributed.get_tp_group=lambda:tp
sys.modules['vllm']=types.ModuleType('vllm')
sys.modules['vllm.distributed']=distributed
utils=types.ModuleType('vllm_ascend.utils')
utils.oproj_tp_enable=lambda:False
utils.enable_sp=lambda:True
utils.enable_dsa_cp=lambda:True
sys.modules['vllm_ascend']=types.ModuleType('vllm_ascend')
sys.modules['vllm_ascend.utils']=utils
ctx=types.ModuleType('vllm_ascend.ascend_forward_context')
ctx._EXTRA_CTX=types.SimpleNamespace(flash_comm_v1_enabled=True,mmrs_fusion=False,pad_size=0)
sys.modules['vllm_ascend.ascend_forward_context']=ctx

class Entry:
    batch_descriptor=types.SimpleNamespace(num_tokens=96,num_reqs=12,uniform=True,
                                           has_lora=False,num_active_loras=0)
    class FakeGraph:
        def debug_dump(self,path):
            Path(path).write_text('[]')
    aclgraph=FakeGraph()
owner=types.SimpleNamespace(runnable=object())

def raises(fn,typ=Exception):
    try:fn()
    except typ:return
    raise AssertionError('expected rejection')

with tempfile.TemporaryDirectory() as tmp:
    os.environ[capture.ENV]=tmp
    os.environ['RUN_TS']='LOOP081-RUN589-B'
    tags=[]
    capture.sidecar._loaded=types.SimpleNamespace(enqueue_tag=tags.append)
    entry=Entry()
    raises(lambda:capture.current(),RuntimeError)
    capture.graph_begin(entry,owner)
    raises(lambda:capture.graph_begin(entry,owner),RuntimeError)
    wrong='model.layers.1.self_attn'
    name='model.layers.0.self_attn'
    impl=types.SimpleNamespace(wo_b=types.SimpleNamespace(custom_op=types.SimpleNamespace(input_is_parallel=True),
                                                       quant_method=object()))
    capture.attention_before(wrong,impl,Tensor(100),Tensor(200),False)
    assert capture.GRAPH[capture.key(entry)]['layer0']=={}
    capture.attention_before(name,impl,Tensor(100),Tensor(200),False)
    raises(lambda:capture.attention_before(name,impl,Tensor(100),Tensor(200),False),RuntimeError)
    capture.before_sequence_rs(name+'.wo_b',Tensor(100),Tensor(150),True,False)
    capture.sequence_rs(name+'.wo_b',Tensor(100),Tensor(150),Tensor(180),True,False)
    assert capture.projected(name,Tensor(180)).data_ptr()==180
    capture.attention_after(name,Tensor(200))
    assert len(tags)==2 and 'role=partial' in tags[0] and 'role=copy' in tags[1]
    capture.graph_end()
    raises(lambda:capture.graph_end(),RuntimeError)
    runtime=types.SimpleNamespace(state=types.SimpleNamespace(cycle_index=64))
    req_ids=tuple(f'req{i}' for i in range(12))
    capture.bind_cohort(5,req_ids,'LOOP081-RUN589-B',runtime)
    capture.target_begin(runtime)
    capture.graph_replay(entry,owner)
    capture.graph_replay(entry,owner)
    capture.target_end()
    files=list(Path(tmp).glob('rank*_captures.jsonl'))
    assert len(files)==1
    rows=[json.loads(x) for x in files[0].read_text().splitlines()]
    assert len(rows)==1 and rows[0]['rank']==0
    assert rows[0]['replay_submitted'] and not rows[0]['replay_completed']
    assert rows[0]['runtime_phase']=='target' and rows[0]['runtime_cycle']==64
    assert rows[0]['cohort']==5 and rows[0]['request_ids']==list(req_ids)
    edge=rows[0]['layer0']
    assert edge['attention_before']['output_destination']['data_ptr']==edge['attention_after']['output']['data_ptr']
    assert edge['sequence_rs']['partial']['data_ptr']==150
    assert edge['sequence_rs']['reduced']['data_ptr']==edge['projected']['data_ptr']
    assert edge['partial_pre_rs']['data_ptr']==edge['sequence_rs']['partial']['data_ptr']
    capture.cohort_end(5,req_ids,runtime,2)
    assert (Path(tmp)/'rank0_cohort5_acl_graph.json').exists()
    assert (Path(tmp)/'rank0_cohort5_graph_meta.json').exists()
    missing=Entry();missing.batch_descriptor='other'
    raises(lambda:capture.graph_replay(missing,owner),RuntimeError)
    capture.graph_begin(missing,owner);capture.graph_end();capture.graph_replay(missing,owner)
    assert len(files[0].read_text().splitlines())==1

for key,path in patch.SOURCES.items():
    src=path.read_text()
    assert patch.MARK not in src
    patched=patch.patch(key,src)
    compile(patched,str(path),'exec')
    raises(lambda k=key,s=patched:patch.patch(k,s),ValueError)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    copies={}
    for key,path in patch.SOURCES.items():
        copies[key]=root/(key+'.py')
        copies[key].write_bytes(path.read_bytes())
    helper=root/'helper.py';helper.write_bytes(patch.HELPER.read_bytes())
    patch.SOURCES=copies;patch.HELPER=helper
    state=root/'state';record=root/'record.json'
    old_argv=sys.argv
    try:
        sys.argv=['patch589','install','--state-dir',str(state),'--record',str(record),'--offline-confirmed']
        patch.main()
        assert all(patch.sha(copies[k].read_bytes())!=patch.ORIGINAL[k] for k in copies)
        helper.write_bytes(helper.read_bytes()+b'\n# helper drift in offline test\n')
        sys.argv=['patch589','restore','--state-dir',str(state),'--record',str(record),'--offline-confirmed']
        patch.main()
        assert all(patch.sha(copies[k].read_bytes())==patch.ORIGINAL[k] for k in copies)
        assert json.loads(record.read_text())['helper_drift_on_restore'] is True
    finally:
        sys.argv=old_argv

print(json.dumps({'status':'pass','note':'CPU fake modules and temp-copy install/restore only; live branch remains unobserved'}))
