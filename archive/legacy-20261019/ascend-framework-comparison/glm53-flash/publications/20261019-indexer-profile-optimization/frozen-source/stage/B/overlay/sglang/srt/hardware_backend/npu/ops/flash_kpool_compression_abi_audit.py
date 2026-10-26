# SPDX-License-Identifier: Apache-2.0
"""Actual model compression ABI and producer/compress/store completion fences.

Source-only candidate. No math, tensor writes, synthetic transfer or fallback.
Completion occurs inside the compression bridge while its local contiguous
copies remain strongly referenced. This is not performance qualification.
"""
import hashlib,itertools,os,pathlib
_CALLS=itertools.count(1)


def validate(v):
    n=v['tokens'];seq=v['sequences'];lengths=v['query_lengths'];slots=v['request_slots'];shape=v['tail_shape'];cache=v['cache_shape']
    if n<=0 or not len(seq)==len(lengths)==len(slots)==len(v['query_ends'])==len(v['tail_table']) or not seq:raise ValueError('Compression request count mismatch')
    rows=v['tail_key_registered']['bytes']//2048
    if rows<1 or v['tail_key_registered']['bytes']%2048 or v['tail_score_registered']['bytes']!=rows*2048:raise ValueError('Invalid actual registered tail capacity')
    if shape!=[rows,4,128] or cache!=[65,32,1,128] or v['tail_strides']!=[512,128,1] or v['cache_strides']!=[4096,128,128,1] or sum(lengths)!=n:raise ValueError('Compression native bounded layout mismatch')
    if len(set(slots))!=len(slots) or any(type(slot)!=int or not 0<=slot<shape[0] for slot in slots):raise ValueError('Compression tail owner slot outside allocation')
    total=0;ends=[];positions=[];tail_slots=[]
    for length,sequence,slot in zip(lengths,seq,slots):
        if type(length)!=int or length<=0 or type(sequence)!=int or sequence<length:raise ValueError('Compression query/sequence bounds mismatch')
        total+=length;ends.append(total);pos=list(range(sequence-length,sequence));positions+=pos;tail_slots += [slot*4+p%4 for p in pos]
    if v['query_ends']!=ends or v['device_seq_lens']!=seq or v['positions']!=positions:raise ValueError('Compression actual device causal metadata mismatch')
    if v['tail_table']!=[[slot] for slot in slots] or v['tail_slots']!=tail_slots:raise ValueError('Compression actual tail addressing differs from request owner')
    if len(v['index_slots'])!=n or any(type(slot)!=int or not 0<=slot<cache[0]*cache[1] for slot in v['index_slots']):raise ValueError('Compression native compressed-cache slot out of bounds')
    for group in ('cache','tail_key','tail_score'):
        actual=v[group+'_region'];native=v[group+'_registered']
        if v.get('owned_boundary',False):
            original=v['owned_boundary_original_regions'][group]
            if original!=native or actual['ptr']<=0 or actual['ptr']==native['ptr'] or actual['bytes']!=native['bytes']:
                raise ValueError('Owned compression lost original registered owner or distinct scratch '+group)
            if v['owned_storage_bytes'][group]!=actual['bytes']:
                raise ValueError('Owned compression scratch storage capacity differs '+group)
        elif actual['ptr']!=native['ptr'] or actual['bytes']!=native['bytes']:
            raise ValueError('Compression tensor not actual native registered '+group)
    if v['tail_key_region']['bytes']!=rows*2048 or v['tail_score_region']['bytes']!=rows*2048 or v['cache_region']['bytes']!=532480:raise ValueError('Compression native allocation geometry mismatch')
    return {'tail_owner_slots':slots,'tail_indices_minmax':[min(tail_slots),max(tail_slots)],'index_indices_minmax':[min(v['index_slots']),max(v['index_slots'])],'completing_pool_rows':[i for i,p in enumerate(positions) if (p+1)%4==0],'all_native_region_bounds_valid':True}


def emit(state,phase,**fields):
    if state is None:return
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    return append('compression_abi_boundary',device=state['device'],role=os.environ['FLASH_PD_ROLE'],layer_id=state['layer_id'],requests=state['requests'],call_sequence=state['call_sequence'],phase=phase,source_sha256=state['source_sha256'],context=state['context'],**fields)


def fence(state,phase,**tensors):
    if state is None:return
    import torch
    from sglang.srt.hardware_backend.npu.ops.flash_kpool_score_abi_audit import tensor_meta
    if torch.npu.current_device()!=state['device']:raise ValueError('Compression fence lost actual owned scheduler device')
    emit(state,phase+'_begin',tensors={k:tensor_meta(v) for k,v in tensors.items()})
    torch.npu.synchronize()
    emit(state,phase+'_complete')


_LAYOUT_SEEN=set()

def record_layout_once(cache,tail_key,tail_score,context,pool):
    if os.environ.get('FLASH_PD_LAYOUT_ONCE')!='1':return
    from sglang.srt.managers.flash_pd_request_context import enabled
    if not enabled():return
    from sglang.srt.managers.flash_pd_lifetime_audit import _INVENTORIES
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    if context is None or len(_INVENTORIES)!=1 or pool!=4:raise ValueError('Missing actual long layout identity')
    device=next(iter(_INVENTORIES));layer=int(context['layer_id']);key=(device,layer)
    if key in _LAYOUT_SEEN:return
    if layer not in range(3,45,4):raise ValueError('Wrong actual long DSA layer')
    dense=(layer-3)//4;regions={}
    for name,tensor,group,entry in [('cache',cache,'index',dense),('tail_key',tail_key,'tails',dense),('tail_score',tail_score,'tails',dense+11)]:
        native=[v for v in _INVENTORIES[device] if v['group']==group and v['entry']==entry]
        if len(native)!=1:raise ValueError('Missing unique registered long region')
        item=native[0];nbytes=int(tensor.numel()*tensor.element_size());pointer=int(tensor.data_ptr())
        if tensor.device.type!='npu' or tensor.device.index!=device or pointer!=item['ptr'] or nbytes!=item['bytes']:raise ValueError('Long kernel tensor differs from native SDK registration')
        exact=tensor.storage_offset()==0 and tensor.untyped_storage().nbytes()==nbytes
        if os.environ['FLASH_PD_ROLE']=='Decode' and not exact:raise ValueError('Long D persistent storage is not exact independent logical allocation')
        if not tensor.is_contiguous():raise ValueError('Long actual kernel tensor is not contiguous')
        if str(tensor.dtype)!=('torch.bfloat16' if name=='cache' else 'torch.float32'):raise ValueError('Long actual kernel tensor dtype differs from fixed profile')
        regions[name]={'ptr':pointer,'bytes':nbytes,'native_ptr':item['ptr'],'native_bytes':item['bytes'],'shape':list(tensor.shape),'strides':list(tensor.stride()),'exact_logical_storage':exact,'dtype':str(tensor.dtype)}
    append('long_kernel_registered_layout_once',device=device,role=os.environ['FLASH_PD_ROLE'],layer_id=layer,regions=regions,no_tensor_value_read_or_sync=True,source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    _LAYOUT_SEEN.add(key)


def begin(tail_key,tail_score,cache,k,gate,ape,positions,ends,seq_lens,tail_slots,tail_table,index_slots,pool,context):
    record_layout_once(cache,tail_key,tail_score,context,pool)
    if os.environ.get('FLASH_PD_COMPRESSION_ABI_AUDIT')!='1':return None
    import torch
    from sglang.srt.managers.flash_pd_lifetime_audit import _INVENTORIES
    from sglang.srt.managers.flash_pd_request_context import forward_tags
    if context is None or len(_INVENTORIES)!=1 or pool!=4:raise ValueError('Compression ABI requires actual Flash owned forward/inventory')
    device=next(iter(_INVENTORIES));layer=int(context['layer_id']);tags=forward_tags()
    if tags is None or set(context['request_slots'])!=set(tags) or layer not in range(3,45,4):raise ValueError('Compression ABI actual scheduler layer/request mismatch')
    originals=context.get('_owned_boundary_original_tensors')
    context={key:value for key,value in context.items() if key!='_owned_boundary_original_tensors'}
    state={'call_sequence':next(_CALLS),'device':device,'layer_id':layer,'requests':[tags[slot] for slot in context['request_slots']],'context':context,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    tensors={'tail_key':tail_key,'tail_score':tail_score,'cache':cache,'k':k,'gate':gate,'ape':ape,'positions':positions,'query_ends':ends,'seq_lens':seq_lens,'tail_slots':tail_slots,'tail_table':tail_table,'index_slots':index_slots}
    for name,x in tensors.items():
        if x.device!=k.device or x.device.type!='npu' or x.device.index!=device:raise ValueError('Compression ABI device mismatch: '+name)
    if k.ndim!=2 or list(k.shape)!=list(gate.shape) or k.shape[1]!=128 or list(ape.shape)!=[4,128]:raise ValueError('Compression projection/APE shape mismatch')
    for x in (tail_key,tail_score,k,gate,ape):
        if x.dtype!=torch.float32 or not x.is_contiguous():raise ValueError('Compression requires actual contiguous F32 projections/tails/APE')
    if cache.dtype!=torch.bfloat16 or tail_table.ndim!=2 or tail_table.dtype not in (torch.int32,torch.int64):raise ValueError('Compression cache/table dtype mismatch')
    for x in (positions,ends,seq_lens,tail_slots,index_slots):
        if x.ndim!=1 or not x.is_contiguous() or x.dtype not in (torch.int32,torch.int64):raise ValueError('Compression metadata vector ABI mismatch')
    # Before the actual compression kernel: this fence distinguishes a previous
    # producer error from compression or tail-store execution.
    fence(state,'input_producers',**tensors)
    dense=(layer-3)//4;native={}
    for name,group,entry in [('cache','index',dense),('tail_key','tails',dense),('tail_score','tails',dense+11)]:
        rows=[r for r in _INVENTORIES[device] if r['group']==group and r['entry']==entry]
        if len(rows)!=1:raise ValueError('Missing unique actual registered compression region')
        native[name]=rows[0]
    def region(x):return {'ptr':int(x.data_ptr()),'bytes':int(x.numel()*x.element_size())}
    v={'tokens':int(k.shape[0]),'sequences':context['sequences'],'query_lengths':context['query_lengths'],'request_slots':context['request_slots'],'device_seq_lens':seq_lens.detach().cpu().tolist(),'query_ends':ends.detach().cpu().tolist(),'positions':positions.detach().cpu().tolist(),'tail_table':tail_table.detach().cpu().tolist(),'tail_slots':tail_slots.detach().cpu().tolist(),'index_slots':index_slots.detach().cpu().tolist(),'tail_shape':list(tail_key.shape),'tail_strides':list(tail_key.stride()),'cache_shape':list(cache.shape),'cache_strides':list(cache.stride()),'cache_region':region(cache),'tail_key_region':region(tail_key),'tail_score_region':region(tail_score),'cache_registered':{'ptr':native['cache']['ptr'],'bytes':native['cache']['bytes']},'tail_key_registered':{'ptr':native['tail_key']['ptr'],'bytes':native['tail_key']['bytes']},'tail_score_registered':{'ptr':native['tail_score']['ptr'],'bytes':native['tail_score']['bytes']}}
    if originals is not None:
        if set(originals)!=set(tensors):raise ValueError('Owned compression missing original boundary tensors')
        for name,x in tensors.items():
            original=originals[name]
            if original.device!=x.device or original.dtype!=x.dtype or original.shape!=x.shape or not original.is_contiguous():
                raise ValueError('Owned compression differs from actual source ABI '+name)
            if original.data_ptr()==x.data_ptr() or x.storage_offset()!=0 or x.untyped_storage().nbytes()!=x.numel()*x.element_size():
                raise ValueError('Owned compression is not an independent exact-size storage '+name)
        v.update(owned_boundary=True,
            owned_boundary_original_regions={name:region(originals[name]) for name in ('cache','tail_key','tail_score')},
            owned_storage_bytes={name:int(tensors[name].untyped_storage().nbytes()) for name in ('cache','tail_key','tail_score')})
    checks=validate(v);emit(state,'metadata_validated',actual=v,checks=checks)
    if os.environ['FLASH_PD_ROLE']=='Decode' and layer==3 and min(v['positions'])==24 and len(state['requests'])==2 and {t.get('sequence') for t in state['requests']}=={0,1}:
        # Preserve the one real model boundary needed for a minimal replay;
        # old records contained only hashes, which cannot recover these values.
        # No later full-model reload is needed for compression/store debugging
        # when this snapshot is successfully captured. Never publish raw tensors.
        import os as _os
        root=pathlib.Path('/run-evidence/compression-boundary-tensors');root.mkdir(exist_ok=True)
        path=root/(str(_os.getpid())+'-device'+str(device)+'-layer3-seq0.pt')
        cpu={name:x.detach().cpu().clone() for name,x in tensors.items()}
        payload={'project_id':'glm53-flash','role':'Decode','device':device,'layer_id':layer,'requests':state['requests'],'actual_metadata':v,'source_sha256':state['source_sha256'],'inputs':cpu}
        fd=_os.open(path,_os.O_CREAT|_os.O_EXCL|_os.O_WRONLY,0o600)
        with _os.fdopen(fd,'wb') as f:
            torch.save(payload,f);f.flush();_os.fsync(f.fileno())
        if not 0<path.stat().st_size<=2*1024*1024:raise ValueError('Actual compression snapshot exceeds frozen bound')
        emit(state,'actual_input_snapshot_saved',file=str(path.relative_to('/run-evidence')),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_actual_tensors_private_not_for_Git=True)
    return state
