#!/usr/bin/env python3
"""Derive frozen scalar cache groups with original dataclasses and planner bodies."""
from __future__ import annotations
import ast,copy,json,math,sys
from collections import Counter
from dataclasses import asdict,dataclass,fields,replace
from enum import Enum,IntEnum
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from types import ModuleType,SimpleNamespace as NS
sys.dont_write_bytecode=True
ROOT=Path('audit/execution-search')
def main():
    paths=[ROOT/'input-resource-dependencies.json',ROOT/'input-group-dispatch-dependencies.json',ROOT/'whole-forward-frozen-config.json'];dep,group,frozen=[json.loads(p.read_text()) for p in paths]
    sources={s['runtime_path']:s for s in dep['sources']+group['sources']};excerpts=[]
    for s in sources.values():assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
    def path(suffix):
        found=[s for k,s in sources.items() if k.endswith(suffix)];assert len(found)==1,suffix
        return Path(found[0]['local_path'])
    def node(p,name,owner=None):
        body=ast.parse(p.read_text()).body
        if owner:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==owner).body
        n=next(n for n in body if getattr(n,'name',None)==name)
        excerpts.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),name=name,owner=owner,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]));return copy.deepcopy(n)
    mod=ModuleType('_frozen_cache_group_source_controls');sys.modules[mod.__name__]=mod;g=mod.__dict__
    class Dtype(Enum):
        bfloat16=('bf16',2);float16=('fp16',2);float32=('fp32',4);int8=('i8',1);uint8=('u8',1)
        @property
        def itemsize(self):return self.value[1]
    hw=NS(supports=lambda capability:False)
    g.update(dataclass=dataclass,fields=fields,replace=replace,Enum=Enum,IntEnum=IntEnum,Counter=Counter,cached_property=cached_property,copy=copy,math=math,torch=NS(**{x.name:x for x in Dtype}),get_dtype_size=lambda x:x.itemsize,cdiv=lambda a,b:(a+b-1)//b,round_up=lambda a,b:(a+b-1)//b*b,get_current_hardware_profile=lambda:hw,HardwareCapability=NS(DSV41_PACKED_CACHE='A5',DSV4_COMPRESSED_CACHE='A5compressed'))
    def execute(nodes):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),'<original-cache-source>','exec'),g)
    p=path('/v1/kv_cache_interface.py')
    execute([node(p,n) for n in ['KVQuantMode','SparseCacheRole','KVCacheSpec','AttentionSpec','FullAttentionSpec','_apply_alignment_padding','MLAAttentionSpec','SlidingWindowSpec','CircularBufferSpec','SlidingWindowMLASpec','UniformTypeKVCacheSpecs','KVCacheGroupRole','KVCacheGroupSpec']])
    # Full-attention uniform-base registration is an explicit scalar stub here;
    # SWA/state use the original subtype-specific collection predicates.
    g['KVCacheSpecRegistry']=NS(get_uniform_type_base_spec=lambda spec:g['FullAttentionSpec'] if isinstance(spec,g['FullAttentionSpec']) else type(spec))
    p=path('/core/kv_cache_interface.py');execute([node(p,n) for n in ['get_kv_cache_compression_ratio','get_storage_block_size','AscendMLAAttentionSpec','AscendSlidingWindowMLASpec']])
    p=path('/models/deepseek_v41/cache_config.py');execute([node(p,n) for n in ['uses_a5_packed_cache','make_swa_cache_spec','make_mla_cache_spec','make_index_cache_spec','_layer_number','_draft_layer_number','get_layer_tuples','group_cache_specs','make_cache_groups']])
    p=path('/models/layer/attention/layer.py');execute([node(p,'get_dsv4_block_sizes')])
    config=frozen['model_config']['content']['text_config'];block=int(frozen['saved_serve']['selected_options']['--block-size']);physical_swa_block=g['get_dsv4_block_sizes']()[block][0][1]
    prefix='language_model.model';specs={}
    for i in range(config['num_hidden_layers']):specs[f'{prefix}.layers.{i}.self_attn.swa_cache']=g['make_swa_cache_spec'](block_size=physical_swa_block,window_size=config['sliding_window'],head_size=config['head_dim'],dtype=Dtype.bfloat16,cache_dtype='auto')
    for i in config['kv_source_layer_ids']:
        p=f'{prefix}.layers.{i}.self_attn';ratio=config['compress_ratios'][i]
        specs[p+'.long_kv_cache']=g['make_mla_cache_spec'](block_size=block,head_size=config['head_dim'],compress_ratio=ratio)
        specs[p+'.indexer.k_cache']=g['make_index_cache_spec'](block_size=block,head_size=config['index_head_dim'],compress_ratio=ratio)
        if ratio==2:specs[p+'.compressor.state_cache']=g['CircularBufferSpec'](block_size=32,num_kv_heads=1,head_size=2*config['head_dim'],dtype=Dtype.float32,head_size_v=0)
    draft_names=[f'mtp.{i}.self_attn.swa_cache' for i in range(3)]
    for name in draft_names:specs[name]=g['make_swa_cache_spec'](block_size=physical_swa_block,window_size=config['sliding_window'],head_size=config['head_dim'],dtype=Dtype.bfloat16,cache_dtype='auto')
    before={k:asdict(v) for k,v in specs.items()};groups=g['make_cache_groups'](g['group_cache_specs'](specs));assert before=={k:asdict(v) for k,v in specs.items()},'planner must not mutate source specs'
    full=[];draft_gid=None;builder_count=0
    for gid,grp in enumerate(groups):
        unique={grp.kv_cache_spec.kv_cache_specs[n] for n in grp.layer_names};builder_count+=len(unique)
        if any(n in draft_names for n in grp.layer_names):assert draft_gid is None;draft_gid=gid
        full.append(dict(group_id=gid,layers=grp.layer_names,layer_count=len(grp.layer_names),distinct_selected_backend_spec_pairs=len(unique),specs=[dict(name=n,type=type(grp.kv_cache_spec.kv_cache_specs[n]).__name__,fields=asdict(grp.kv_cache_spec.kv_cache_specs[n]),unpadded=grp.kv_cache_spec.kv_cache_specs[n].unpadded_page_size_bytes,page=grp.kv_cache_spec.kv_cache_specs[n].page_size_bytes) for n in grp.layer_names]))
    assert len(specs)==54 and len(groups)==13 and builder_count==26 and draft_gid==12
    # Execute original DSpark group initialization on these source-derived specs.
    class Backend:
        @staticmethod
        def full_cls_name():return 'selected_V41_cache_backend'
    class Group:
        def __init__(self,backend,names,spec,gid):self.layer_names=names;self.kv_cache_spec=spec;self.kv_cache_group_id=gid
        def create_metadata_builders(self,*a,**kw):pass
    all_layers={name:NS(get_attn_backend=lambda:Backend) for name in draft_names}
    g.update(get_layers_from_vllm_config=lambda *a:all_layers,AttentionLayerBase=object,AttentionGroup=Group)
    n=node(path('/spec_decode/dspark_proposer.py'),'initialize_attn_backend','AscendDSparkProposer');execute([n]);initialize=g['initialize_attn_backend']
    # Tensor creation is replaced by labeled allocation descriptors; no data,
    # device or host framework Tensor is allocated by these controls.
    g['torch'].int32='int32';g['torch'].zeros=lambda *a,**kw:NS(shape=a,dtype=kw.get('dtype'))
    d=NS(vllm_config=NS(),model=NS(get_draft_kv_cache_layer_names=lambda:draft_names),runner=None,dcp_size=1,device='stub',max_query_tokens=1024,max_num_tokens=4096)
    initialize(d,NS(kv_cache_groups=groups))
    assert len(d.draft_attn_groups)==1 and d.kv_cache_gid==12 and d._layer_group_idx==[12,12,12]
    # A changed padded page is a counterfactual, demonstrating why shared KV
    # group membership alone does not prove one attention-group/kernel call.
    changed=copy.deepcopy(groups);u=changed[12].kv_cache_spec;one=draft_names[-1];updated=dict(u.kv_cache_specs);updated[one]=replace(updated[one],page_size_padded=updated[one].page_size_bytes+128);changed[12]=g['KVCacheGroupSpec'](layer_names=list(updated),kv_cache_spec=g['UniformTypeKVCacheSpecs'].from_specs(updated))
    d2=NS(**{k:v for k,v in d.__dict__.items() if k in ['vllm_config','model','runner','dcp_size','device','max_query_tokens','max_num_tokens']});initialize(d2,NS(kv_cache_groups=changed));assert len(d2.draft_attn_groups)==2
    encoder_class_names={x['type'] for r in full for x in r['specs']};assert encoder_class_names=={'AscendMLAAttentionSpec','AscendSlidingWindowMLASpec','CircularBufferSpec'}
    out=dict(status='SOURCE_DERIVED_FROZEN_RESOURCE_GROUP_PLAN_NOT_INITIALIZED_OBJECT_DUMP',resource_specs=54,framework_KV_groups=13,runner_attention_backend_spec_groups=26,draft_framework_group_id=12,draft_attention_groups=1,draft_layer_group_ids=d._layer_group_idx,draft_group_layers=d.draft_attn_groups[0].layer_names,draft_kernel_block_size=d.kernel_block_size,physical_swa_block=physical_swa_block,groups=full,counterfactual_one_padded_page_changes_attention_group_count=len(d2.draft_attn_groups),source_specs_not_mutated=True,
        qualification='Original spec dataclasses, page properties, cache planner, DSV4 block table and DSpark initialize body. Explicit 910C capability(False), scalar dtype-size, selected uniform-base registry/backend and allocation-descriptor stubs; resource names/roles/dtype/config constructor arguments are source-derived, not actual live initialization or primitive bits. Full source registration/caller selection reviewed separately. No profile used to choose13/26/1.',
        implications=['Frozen source plan puts all3 draft SWA layers in one KV group and one identical backend/spec group, so per-group common expansion currently executes once; deleting supposed repeated common fields has no repeated execution to remove. Larger combined input/physical mapping kernels remain separate.','Runner26 metadata builders comprise4full+1state+20targetSWA+1draft, not26target-only resources.','Changing one padded page yields2 draft groups under the original initializer, so one shared KV group or raw caller count alone does not establish identity.'],
        inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],sources=list(sources.values()),every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    encode=lambda d:json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.name if isinstance(x,Enum) else str(x))+'\n'
    (ROOT/'input-resource-source-plan.json').write_text(encode(out));(ROOT/'input-resource-group-source-excerpts.json').write_text(encode(dict(excerpts=excerpts,qualification=out['qualification'])));assert 'torch' not in sys.modules
    print(json.dumps({k:out[k] for k in ['resource_specs','framework_KV_groups','runner_attention_backend_spec_groups','draft_framework_group_id','draft_attention_groups','draft_layer_group_ids','draft_kernel_block_size','counterfactual_one_padded_page_changes_attention_group_count','new_NPU_work']}))
if __name__=='__main__':main()
