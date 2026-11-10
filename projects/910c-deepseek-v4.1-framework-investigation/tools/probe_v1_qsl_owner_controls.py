#!/usr/bin/env python3
"""Original A3 base builder: distinguish canonical coordinates and input readers.

Tensor/packet sinks are symbolic. This is not native arithmetic or bit evidence.
"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS,MethodType
from collections import defaultdict
import ast,json,itertools
from probe_common_device_readers import S,Poison,ValueRead,Stage
ROOT=Path('audit/execution-search')
def main():
    deps=json.loads((ROOT/'v1-input-dependencies.json').read_text());src=next(s for s in deps['sources'] if s['runtime_path'].endswith('/attention/dsa_v41.py'))
    p=Path(src['local_path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==src['runtime_sha256'];T=ast.parse(raw)
    c=next(n for n in T.body if isinstance(n,ast.ClassDef) and n.name=='AscendDSAV41MetadataBuilder')
    methods=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ['build','_publish_task','_build_batch_metadata']]
    helper=next(n for n in T.body if isinstance(n,ast.FunctionDef) and n.name=='_config_value')
    calls=[];reads=[];phase='build';current=''
    class QSL(S):
        def __getitem__(self,key):
            reads.append(dict(input_owner=self.name,reader=current,phase=phase));return super().__getitem__(key)
    def encoded(v):
        if isinstance(v,S):return dict(roots=sorted(v.roots))
        if isinstance(v,(list,tuple)):return [encoded(x) for x in v]
        return v
    def native(name):
        def run(*a,**k):calls.append(dict(native=name,reader=current,args=encoded(a),kwargs={x:encoded(v) for x,v in k.items()}));return S('packet:'+name)
        return run
    def op(*a,**k):return k['out'] if k.get('out') is not None else S('torchop',roots=set().union(*(v.roots for v in a if isinstance(v,S))))
    torch=NS(Tensor=S,int32='int32',ops=NS(_C_ascend=NS(npu_sparse_flash_mla_metadata=native('SMLA'),npu_quant_lightning_indexer_v2_metadata=native('QLI'))),where=op,div=op,remainder=op,arange=op,zeros_like=op,gather=op)
    ns=dict(torch=torch,Any=object,DeviceMetadataStage=Stage,DeviceMetadataTask=lambda *a:NS(stage=a[0],run=a[1],group_id=a[2]),get_kv_cache_compression_ratio=lambda s:s.ratio,get_storage_block_size=lambda s:16,compressed_slot_mapping=lambda *a:S('compressed_slots'),get_cos_and_sin_dsa=lambda *a,**k:(S('rope_cos'),S('rope_sin')),AscendDSAV41Metadata=lambda **k:NS(**k))
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),helper,ast.ClassDef(name='Builder',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
    resources=[('swaA','swa',1,0),('long2A','long_kv',2,1),('index2A','index_k',2,1),('stateA','compressor_state',2,2),('long1A','long_kv',1,3),('index1A','index_k',1,3),('long2B','long_kv',2,4),('index2B','index_k',2,4),('swaB','swa',1,5),('stateB','compressor_state',2,6),('long1B','long_kv',1,7),('index1B','index_k',1,7)]
    orders=[resources,list(reversed(resources)),resources[3:]+resources[:3],resources[6:]+resources[:6]]
    def builder(name,kind,ratio):
        b=ns['Builder']();b.kv_cache_spec=NS(ratio=ratio,block_size=128);b._cache_kind=kind;b._device_metadata_enabled=True;b._device_metadata_in_graph=False;b._uses_a5_packed_cache=False;b._supports_device_ops=True;b._build_query_metadata=True;b._build_compressor_metadata=False;b._a5_causal_swa_indices=None;b._dspark_swa_indices=None;b._max_tokens=24
        for field in ['_seq_lens','_slot_mapping','_slot_mapping_2d','_flat_slot_mapping','_cache_seq_lens','_cmp_residual','_smla_metadata','_qli_metadata','_c2_ring_metadata','_c2_complete_mask','_c2_source_positions','_c2_source_cos','_c2_source_sin']:setattr(b,field,S(name+field))
        b.vllm_config=NS(model_config=NS(hf_text_config=dict(sliding_window=128,num_attention_heads=64,head_dim=512,index_topk=512,index_n_heads=32,index_head_dim=128)),parallel_config=NS(tensor_parallel_size=8),speculative_config=NS(num_speculative_tokens=5));return b
    records=[]
    for oi,order in enumerate(orders):
     for nt,full in itertools.product([0,24],[False,True]):
      paired=[]
      for canonicalize in [False,True]:
        calls.clear();reads.clear();shared_by_gid=defaultdict(dict);owners={};tasks=[];batch_qsl=QSL('canonical_first_qsl',(5,));seq=S('canonical_seq',(4,))
        batch={'batch':dict(query_start_loc=batch_qsl,query_start_loc_cpu=S('canonical_cpu_qsl',(5,)),seq_lens=seq,seq_lens_cpu=S('canonical_cpu_seq',(4,)),positions=S('positions',(nt,)),max_cache_seq_len=32768,num_decodes=4,num_decode_tokens=nt,num_prefills=0,num_prefill_tokens=0)}
        for idx,(name,kind,ratio,gid) in enumerate(order):
            current=name;phase='eager_build';b=builder(name,kind,ratio);incoming=batch_qsl if idx==0 or canonicalize else QSL(name+'_incoming_qsl',(5,))
            cm=NS(seq_lens=Poison('unused_fresh_lengths',(4,)),num_reqs=4,num_input_tokens=nt,num_actual_tokens=nt,slot_mapping=S('physical_slots_'+str(gid),(nt,)),query_start_loc=incoming,block_table_tensor=S('physical_table_'+str(gid)),causal=True,max_query_len=6 if nt else 0,max_seq_len=32768)
            original=b._publish_task
            def publish(shared,key,buffer,stage,run,_original=original,_name=name):
                if key not in shared:owners[key]=_name
                return _original(shared,key,buffer,stage,run)
            b._publish_task=publish
            out=b.build(0,cm,num_actual_reqs=4,full_graph_mode=full,build_compressor_metadata=False,common_v41_metadata=shared_by_gid[gid],common_v41_batch_metadata=batch)
            assert out.query_start_loc is batch_qsl and out.seq_lens is seq
            tasks.extend((task,name) for task in b._device_metadata_tasks)
        for task,name in sorted(tasks,key=lambda x:x[0].stage.value):current=name;phase='deferred_'+task.stage.name;task.run()
        assert set(owners)=={'smla:c0','smla:c1','smla:c2','qli:c1','qli:c2'}
        normalized=[]
        for call in calls:
            x=json.loads(json.dumps(call));x['kwargs']['cu_seqlens_q']='<same-integer-QSL-value-not-native-validated>';normalized.append(x)
        eager_owners=sorted({x['input_owner'] for x in reads if x['phase']=='eager_build'})
        native_qsl=[dict(native=c['native'],reader=c['reader'],QSL_roots=c['kwargs']['cu_seqlens_q']['roots']) for c in calls]
        paired.append(dict(normalized=normalized,owners=dict(owners),task_count=len(tasks)))
        records.append(dict(order=oi,actual_tokens=nt,full=full,canonicalize_incoming=canonicalize,publication_owners=dict(owners),eager_QSL_owners=eager_owners,native_QSL=native_qsl,task_count=len(tasks),canonical_output_alias_preserved=True))
      assert paired[0]==paired[1]
    # A cached batch does not suppress fresh-key or physical-group QSL reads.
    negatives=[]
    for kind,ratio,expected in [('swa',1,'deferred_SMLA'),('index_k',1,'deferred_QLI'),('long_kv',2,'eager_C2_slot_valid_end')]:
        b=builder('negative',kind,ratio);q=Poison('unwritten_per_builder_qsl',(5,));cm=NS(seq_lens=Poison('unused',(4,)),num_reqs=4,num_input_tokens=24,num_actual_tokens=24,slot_mapping=S('slots',(24,)),query_start_loc=q,block_table_tensor=S('table'),causal=True,max_query_len=6,max_seq_len=32768)
        batch={'batch':dict(query_start_loc=S('canonical_qsl',(5,)),query_start_loc_cpu=S('cpu_qsl'),seq_lens=S('canonical_seq'),seq_lens_cpu=S('cpu_seq'),positions=S('positions',(24,)),max_cache_seq_len=32768,num_decodes=4,num_decode_tokens=24,num_prefills=0,num_prefill_tokens=0)}
        try:
            b.build(0,cm,num_actual_reqs=4,common_v41_metadata={},common_v41_batch_metadata=batch)
            for task in b._device_metadata_tasks:task.run()
        except ValueRead:negatives.append(dict(cache_kind=kind,ratio=ratio,expected_reader=expected,unwritten_incoming_QSL_rejected=True))
        else:raise AssertionError(expected)
    result=dict(status='PASS_BOUNDED_ORIGINAL_BASE_QSL_READER_AND_OWNER_CONTROLS',source=dict(runtime_path=src['runtime_path'],sha256=src['runtime_sha256']),controls=len(records),records=records,negative_controls=negatives,
        scope='A3 target-local CP base build: causal, query metadata on, local compressor off, seeded canonical batch; 12 synthetic builders/8 physical groups in four orders. Not an observed26-builder identity/order dump.',
        conclusions=['Output coordinate alias may be canonical while native callbacks still capture noncanonical incoming QSL.','Publication owner is first per smla:c0/c1/c2 and qli:c1/c2 key; C2 eager physical mapping is first per physical group/ratio/block key and can differ from packet owner.','Rebinding incoming common QSL to canonical preserves symbolic non-QSL native args, selected packet/task owners/count and returned aliases in32controls; it intentionally changes source pointer and extends its last-read.','Zero-query SMLA packet is zeroed but tasks retained; two QLI calls still execute. Global compressor ring uses global common and remains outside these local controls.'],
        qualification='Symbolic source execution only: QSL value equality, native bytes, actual addresses/events/retirement, frozen population and request cost NOT proved. Native admission0.',new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'v1-qsl-owner-controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(controls=len(records),negative_controls=len(negatives),publication_keys=5,new_native_admissions=0)))
if __name__=='__main__':main()
