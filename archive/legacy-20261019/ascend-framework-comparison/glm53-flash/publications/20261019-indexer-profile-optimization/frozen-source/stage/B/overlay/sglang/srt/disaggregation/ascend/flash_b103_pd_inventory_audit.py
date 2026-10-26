# SPDX-License-Identifier: Apache-2.0
"""Actual native Flash KVArgs inventory and source bindings, no tensor writes."""
import hashlib,json,pathlib,sys,inspect

def record(manager):
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    a=manager.kv_args;e=manager.engine
    if type(manager).__module__!='sglang.srt.disaggregation.ascend.conn':
        raise ValueError('Flash diagnostic mounted outside actual ASCEND manager')
    types=[v.value for v in a.state_types]
    if types!=['mamba','dsa','dsa_tail'] or len(a.kv_data_ptrs)!=11 or [len(v) for v in a.state_data_ptrs]!=[68,11,22]:
        raise ValueError('Actual Flash native PD state inventory is incomplete')
    full=list(range(3,45,4));linear=[i for i in range(45) if i not in full]
    if list(a.kv_layer_ids)!=full or list(a.state_layer_ids[0])!=linear*2:
        raise ValueError('Actual Flash global layer ownership differs')
    if a.kv_item_lens!=[131072]*11 or a.state_item_lens!=[[9216]*34+[262144]*34,[8192]*11,[2048]*22]:
        raise ValueError('Unexpected actual TP16 Flash state stride/dtype geometry')
    groups=[('main',a.kv_data_ptrs,a.kv_data_lens,a.kv_item_lens)]
    groups += [('KDA',a.state_data_ptrs[0],a.state_data_lens[0],a.state_item_lens[0]),
               ('index',a.state_data_ptrs[1],a.state_data_lens[1],a.state_item_lens[1]),
               ('tails',a.state_data_ptrs[2],a.state_data_lens[2],a.state_item_lens[2]),
               ('aux',a.aux_data_ptrs,a.aux_data_lens,a.aux_item_lens)]
    rows=[]
    for group,ptrs,lens,items in groups:
        if not len(ptrs)==len(lens)==len(items):raise ValueError('Unpaired native descriptor')
        for i,(p,n,item) in enumerate(zip(ptrs,lens,items)):
            if not p>0 or not n>0 or not item>0 or n%item:raise ValueError('Zero/invalid model state range')
            rows.append({'group':group,'entry':i,'ptr':int(p),'bytes':int(n),'item_bytes':int(item)})
    intervals=sorted((v['ptr'],v['ptr']+v['bytes']) for v in rows)
    if not all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Overlapping native model/aux regions')
    manifest=json.loads(pathlib.Path('/run-evidence/combined-manifest.json').read_text())
    bindings={}
    for row in manifest['files']:
        p=pathlib.Path('/sgl-workspace/sglang/python')/row['target']
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha!=row['sha256']:raise ValueError('Runtime source SHA mismatch: '+row['target'])
        name=row['target'].removesuffix('.py').replace('/','.')
        module=sys.modules.get(name)
        bindings[row['target']]={'sha256':sha,'imported':module is not None,'actual_module_path':getattr(module,'__file__',None)}
    from sglang.srt.managers.flash_pd_lifetime_audit import inventory as _lifetime_inventory
    _lifetime_inventory(e.npu_id, rows)
    append('native_inventory',device=e.npu_id,role=e.role,session_id=e.session_id,
           manager_class=type(manager).__module__+'.'+type(manager).__name__,
           manager_source_sha256=hashlib.sha256(pathlib.Path(inspect.getfile(type(manager))).read_bytes()).hexdigest(),
           state_types=types,rows=rows,typed_model_descriptors=112,
           kv_layer_ids=list(a.kv_layer_ids),state_layer_ids=a.state_layer_ids,
           bindings=bindings,SSM_FP32_conv_BF16_geometry=True,
           zero_byte_RoPE_absent=True,zero_sparse_aux_constant_not_model_state=True)


def record_startup_pool(full_pool,mamba_pool):
    """One startup metadata event; no tensor values, copies or synchronization."""
    import torch,os
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    def meta(t):
        return {'ptr':int(t.data_ptr()),'dtype':str(t.dtype),'device':str(t.device),
                'shape':list(t.shape),'strides':list(t.stride()),
                'bytes':int(t.numel()*t.element_size()),'element_size':int(t.element_size()),
                'exact_logical_storage':t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()}
    groups={name:[meta(t) for t in value] for name,value in (
        ('main',full_pool.k_buffer),('index',full_pool.index_k_buffer),
        ('tail_key',full_pool._flash_tail_key),('tail_score',full_pool._flash_tail_score))}
    cache=mamba_pool.mamba_cache
    append('native_startup_tensor_layout',device=torch.npu.current_device(),role=os.environ['FLASH_PD_ROLE'],
           groups=groups,mamba_conv=[meta(t) for t in cache.conv],mamba_SSM=meta(cache.temporal),
           actual_token_pool_size=int(full_pool.size),actual_req_tail_rows=int(full_pool.req_pool_size),
           no_tensor_value_read_copy_or_sync=True,before_API_ready=True,
           source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
