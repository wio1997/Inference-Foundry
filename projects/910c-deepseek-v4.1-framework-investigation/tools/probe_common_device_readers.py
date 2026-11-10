#!/usr/bin/env python3
"""Execute original base-builder control flow with a shape-only poison input.

This is a stdlib symbolic reader/alias check, NOT Torch/NPU bit validation.
"""
import ast,itertools,json
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
from enum import Enum
ROOT=Path('audit/execution-search')
class ValueRead(Exception):pass
class S:
    def __init__(self,name,shape=(4,16),roots=None):self.name=name;self.shape=shape;self.roots=set(roots or [name]);self.device=NS(type='npu');self.dtype='int32'
    def __getitem__(self,key):return S(self.name+'[view]',self.shape,self.roots)
    def __setitem__(self,key,value):pass
    def __getattr__(self,name):
        def f(*args,**kwargs):return self if name.endswith('_') else S(self.name+'.'+name,self.shape,self.roots)
        return f
    def op(self,other=None):return S(self.name+'[op]',self.shape,self.roots| (other.roots if isinstance(other,S) else set()))
    __sub__=__add__=__mul__=__ge__=__gt__=__lt__=__eq__=__and__=__floordiv__=__mod__=op
    __rsub__=__radd__=__rmul__=__rand__=op
    __hash__=object.__hash__
class Poison(S):
    def __getitem__(self,key):raise ValueRead('fresh common.seq_lens VALUE reached')
class Stage(Enum):ATTENTION=1;INDEXER=2;COMPRESSOR=3
def main():
    deps=json.loads((ROOT/'common-device-entry.json').read_text());source=next(r for r in deps['sources'] if r['runtime_path'].endswith('/attention/dsa_v41.py'));p=Path(source['local_path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==source['runtime_sha256'];tree=ast.parse(raw)
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AscendDSAV41MetadataBuilder');methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ['build','_build_batch_metadata','_publish_task']];assert len(methods)==3
    calls=[]
    def native(name):
        def run(*args,**kwargs):calls.append(dict(name=name,inputs=[sorted(v.roots) for v in [*args,*kwargs.values()] if isinstance(v,S)]));return S('packet:'+name)
        return run
    def torop(*args,**kwargs):return kwargs.get('out') if kwargs.get('out') is not None else S('torchop',roots=set().union(*(v.roots for v in args if isinstance(v,S))))
    torch=NS(Tensor=S,int32='int32',ops=NS(_C_ascend=NS(npu_sparse_flash_mla_metadata=native('SMLA'),npu_quant_lightning_indexer_v2_metadata=native('QLI'))),where=torop,div=torop,remainder=torop,arange=torop,zeros_like=torop,gather=torop)
    ns=dict(torch=torch,Any=object,DeviceMetadataStage=Stage,DeviceMetadataTask=lambda *a:NS(stage=a[0],run=a[1],id=a[2]),get_kv_cache_compression_ratio=lambda s:s.ratio,get_storage_block_size=lambda s:16,compressed_slot_mapping=lambda *a:S('compressed_slots'),get_cos_and_sin_dsa=lambda *a,**k:(S('rope_cos'),S('rope_sin')),build_dspark_swa_indices=lambda *a,**k:(S('draft_indices',(24,64)),S('draft_lengths')),AscendDSAV41Metadata=lambda **k:NS(**k))
    config=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_config_value')
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),config,ast.ClassDef(name='Builder',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
    resources=[('swa',1),('long_kv',1),('long_kv',2),('index_k',1),('index_k',2),('compressor_state',2)];geometries=[(0,0,0),(4,0,0),(4,2,0),(4,2,12),(4,4,24)];records=[];native_kinds=set()
    for (kind,ratio),(nr,actualr,nt),(causal,device,full,deferred,skip) in itertools.product(resources,geometries,itertools.product([False,True],repeat=5)):
        b=ns['Builder'].__new__(ns['Builder']);b.kv_cache_spec=NS(ratio=ratio,block_size=64);b._cache_kind=kind;b._device_metadata_enabled=deferred;b._device_metadata_in_graph=False;b._uses_a5_packed_cache=False;b._supports_device_ops=device;b._build_query_metadata=True;b._build_compressor_metadata=True;b._a5_causal_swa_indices=None;b._dspark_swa_indices=None;b._dspark_swa_lengths=None;b._max_tokens=24
        for name in ['_seq_lens','_slot_mapping','_slot_mapping_2d','_flat_slot_mapping','_cache_seq_lens','_cmp_residual','_smla_metadata','_qli_metadata','_c2_ring_metadata','_c2_complete_mask','_c2_source_positions','_c2_source_cos','_c2_source_sin']:setattr(b,name,S(name))
        b._c2_full_source_rope=(S('full_cos'),S('full_sin'));b.vllm_config=NS(model_config=NS(hf_text_config=dict(sliding_window=128,num_attention_heads=64,head_dim=512,index_topk=2048,index_n_heads=64,index_head_dim=128)),parallel_config=NS(tensor_parallel_size=8),speculative_config=NS(num_speculative_tokens=5))
        canonical=S('canonical_seq_lens',(nr,));qsl=S('canonical_qsl',(nr+1,));batch=dict(seq_lens=canonical,positions=S('canonical_positions',(nt,)),query_start_loc=qsl,query_start_loc_cpu=S('canonical_cpu_qsl',(nr+1,)),seq_lens_cpu=S('canonical_cpu_lengths',(nr,)),max_cache_seq_len=32774,num_decodes=nr,num_decode_tokens=nt,num_prefills=0,num_prefill_tokens=0)
        common=NS(seq_lens=Poison('fresh_unread_seq_lens',(nr,)),num_reqs=nr,num_input_tokens=nt,num_actual_tokens=nt,slot_mapping=S('own_slots',(nt,)),query_start_loc=S('per_builder_qsl',(nr+1,)),block_table_tensor=S('own_table'),causal=causal,max_query_len=6 if nt else 0,max_seq_len=32774)
        shared={};batch_shared={'batch':batch};calls.clear()
        out=b.build(0,common,num_actual_reqs=actualr,full_graph_mode=full,skip_ring_state_update=skip,common_v41_metadata=shared,common_v41_batch_metadata=batch_shared)
        for task in b._device_metadata_tasks:task.run()
        assert out.seq_lens is canonical and out.query_start_loc is qsl
        assert all('fresh_unread_seq_lens' not in roots for c in calls for roots in c['inputs'])
        native_kinds.update(c['name'] for c in calls)
        records.append(dict(cache_kind=kind,ratio=ratio,num_reqs=nr,num_actual_reqs=actualr,actual_tokens=nt,causal=causal,device_ops=device,full_graph=full,deferred=deferred,skip_ring=skip,native_call_kinds=[c['name'] for c in calls],task_count=len(b._device_metadata_tasks),canonical_aliases_preserved=True))
    try:b._build_batch_metadata(common,nr,actualr,nt)
    except ValueRead:miss_reads=True
    else:raise AssertionError('miss must retain the original projection value')
    result=dict(status='PASS_ORIGINAL_BASE_CONTROL_FLOW_SHAPE_ONLY_INPUT_ON_BATCH_HIT',source_path=str(p),source_sha256=source['runtime_sha256'],methods=[dict(name=m.name,first=m.lineno,last=m.end_lineno) for m in methods],cases=len(records),records=records,native_call_kinds=sorted(native_kinds),cache_miss_negative_control_reads_value=miss_reads,qualification='Original full base build/publish functions, symbolic Tensor API and packet sinks. All960 hit cases force deferred callbacks and protect original canonical aliases. This proves bounded control-flow read/escape behavior only, not numerical/bitwise/Torch/NPU/serving execution; A5 metadata-in-graph/PCP are outside this transformation. CPU-device=False/state-local=True combinations are explicit broader controls, not observed frozen caller population.',new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'common-device-reader-poison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(cases=len(records),native_call_kinds=sorted(native_kinds),miss_negative=True,new_primitive_admissions=0)))
if __name__=='__main__':main()
