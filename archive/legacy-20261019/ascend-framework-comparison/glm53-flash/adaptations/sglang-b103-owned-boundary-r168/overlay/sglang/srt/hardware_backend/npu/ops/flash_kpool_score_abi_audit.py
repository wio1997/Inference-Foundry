# SPDX-License-Identifier: Apache-2.0
"""Bounded eager score ABI/ownership and completion localization.

No model tensor writes, arithmetic changes, synthetic inputs or fallback.
The explicit fences keep score/expand temporaries alive through actual device
completion. This diagnostic is not graph or performance qualification.
"""
import hashlib,os,pathlib,itertools
_CALLS=itertools.count(1)


def enabled():return os.environ.get('FLASH_PD_SCORE_ABI_AUDIT')=='1'


def validate(v):
    """Validate a captured actual ABI using CPU metadata only."""
    n=v['tokens'];h=v['heads'];d=v['head_dim'];pool=v['pool'];seq=v['sequences'];lengths=v['query_lengths'];ends=v['query_ends'];lens=v['pool_lengths'];positions=v['positions'];table=v['table'];shape=v['cache_shape'];strides=v['cache_strides'];maximum=v['max_pool_seq_len']
    if n<=0 or h!=32 or d!=128 or pool!=4:raise ValueError('Unexpected actual Flash score dimensions')
    if len(shape)!=4 or shape[0]<=0 or shape[1]!=32 or shape[2:]!=[1,128] or len(strides)!=4 or min(strides)<=0:raise ValueError('Invalid actual native index-cache layout')
    if not len(seq)==len(lengths)==len(ends)==len(lens)==len(table) or not seq:raise ValueError('Score request-vector count mismatch')
    if len(v['request_slots'])!=len(seq) or len(set(v['request_slots']))!=len(seq) or min(v['request_slots'])<0:raise ValueError('Score request slots mismatch')
    if len(positions)!=n or any(type(x)!=int or x<=0 for x in lengths) or sum(lengths)!=n:raise ValueError('Score query lengths mismatch')
    total=0;expected=[]
    for length,sequence in zip(lengths,seq):
        if type(sequence)!=int or sequence<length:raise ValueError('Invalid actual sequence length')
        total+=length;expected.append(total)
    if ends!=expected or lens!=[x//pool for x in seq] or maximum!=max(seq)//pool:raise ValueError('Score actual boundaries/pool lengths mismatch')
    offset=0
    for length,sequence in zip(lengths,seq):
        if positions[offset:offset+length]!=list(range(sequence-length,sequence)):raise ValueError('Score causal positions mismatch')
        offset+=length
    columns=(maximum+shape[1]-1)//shape[1]
    if any(len(row)<columns for row in table):raise ValueError('Score kernel page-table read would exceed columns')
    for row,length in zip(table,lens):
        if any(type(page)!=int or not 0<=page<shape[0] for page in row[:(length+shape[1]-1)//shape[1]]):raise ValueError('Active score page outside native cache')
    span=sum((size-1)*stride for size,stride in zip(shape,strides))
    if span!=v['cache_span_elements'] or (span+1)*v['cache_element_size']>v['registered_cache_bytes']:raise ValueError('Score cache span outside registered native allocation')
    if v.get('owned_boundary',False):
        if v['owned_boundary_original_cache_ptr']!=v['registered_cache_ptr'] or v['owned_boundary_original_cache_bytes']!=v['registered_cache_bytes']:
            raise ValueError('Owned score lost actual original registered owner')
        if v['cache_ptr']<=0 or v['cache_ptr']==v['registered_cache_ptr'] or v['owned_cache_storage_bytes']!=v['registered_cache_bytes']:
            raise ValueError('Owned score scratch alias or capacity mismatch')
    elif v['cache_ptr']!=v['registered_cache_ptr'] or v['cache_ptr']<=0:
        raise ValueError('Score cache is not the actual native inventory row')
    return {'required_table_columns':columns,'active_pages':[row[:(length+shape[1]-1)//shape[1]] for row,length in zip(table,lens)],'cache_span_elements':span,'index32':span<=2147483647}


def tensor_meta(x):
    return {'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'device':str(x.device),'ptr':int(x.data_ptr()),'storage_offset':int(x.storage_offset()),'storage_bytes':int(x.untyped_storage().nbytes())}


def _emit(state,phase,**fields):
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    return append('score_abi_boundary',device=state['device'],role=os.environ['FLASH_PD_ROLE'],layer_id=state['layer_id'],requests=state['requests'],phase=phase,call_sequence=state['call_sequence'],source_sha256=state['source_sha256'],**fields)


def fence(state,phase,**tensors):
    if state is None:return
    import torch
    if torch.npu.current_device()!=state['device']:raise ValueError('Score fence lost actual owned scheduler device')
    _emit(state,phase+'_begin',tensors={k:tensor_meta(v) for k,v in tensors.items() if v is not None})
    torch.npu.synchronize()
    _emit(state,phase+'_complete')


def begin(query,cache,weights,ends,lens,table,positions,maximum,pool,context):
    if not enabled():return None
    import torch
    from sglang.srt.managers.flash_pd_lifetime_audit import _INVENTORIES
    from sglang.srt.managers.flash_pd_request_context import forward_tags
    if context is None or len(_INVENTORIES)!=1:raise ValueError('Score ABI requires actual forward and owned SDK inventory')
    device=next(iter(_INVENTORIES));layer=int(context['layer_id']);tags=forward_tags()
    if tags is None or set(context['request_slots'])!=set(tags):raise ValueError('Score ABI context differs from actual scheduler batch')
    state={'call_sequence':next(_CALLS),'device':device,'layer_id':layer,'requests':[tags[slot] for slot in context['request_slots']],'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    tensors={'query':query,'cache':cache,'weights':weights,'query_ends':ends,'pool_lengths':lens,'table':table,'positions':positions}
    for name,x in tensors.items():
        if x.device!=query.device or x.device.type!='npu' or x.device.index!=device:raise ValueError('Score ABI device mismatch: '+name)
        if any(stride<0 for stride in x.stride()):raise ValueError('Negative score tensor stride')
    if list(query.shape)[1:]!=[32,128] or list(weights.shape)!=list(query.shape[:2]) or query.dtype!=weights.dtype or query.dtype!=cache.dtype or query.dtype!=torch.bfloat16:raise ValueError('Score query/weights/cache BF16 ABI mismatch')
    if table.ndim!=2 or table.dtype not in (torch.int32,torch.int64):raise ValueError('Score page-table ABI mismatch')
    for x in (ends,lens,positions):
        if x.ndim!=1 or x.stride(0)!=1 or x.dtype not in (torch.int32,torch.int64):raise ValueError('Score metadata vector ABI mismatch')
    rows=[r for r in _INVENTORIES[device] if r['group']=='index' and r['entry']==(layer-3)//4]
    if layer not in range(3,45,4) or len(rows)!=1:raise ValueError('Score layer does not bind one actual native cache')
    # This first fence can attribute an earlier producer failure before any score
    # launch. All original tensors remain strongly referenced by the caller.
    fence(state,'entry_producers',**tensors)
    row=rows[0];record={'tokens':int(query.shape[0]),'heads':int(query.shape[1]),'head_dim':int(query.shape[2]),'pool':int(pool),'sequences':context['sequences'],'query_lengths':context['query_lengths'],'request_slots':context['request_slots'],'query_ends':ends.detach().cpu().tolist(),'pool_lengths':lens.detach().cpu().tolist(),'positions':positions.detach().cpu().tolist(),'table':table.detach().cpu().tolist(),'cache_shape':list(cache.shape),'cache_strides':list(cache.stride()),'cache_span_elements':sum((size-1)*stride for size,stride in zip(cache.shape,cache.stride())),'cache_element_size':cache.element_size(),'cache_ptr':int(cache.data_ptr()),'registered_cache_ptr':row['ptr'],'registered_cache_bytes':row['bytes'],'max_pool_seq_len':int(maximum)}
    original=context.get('_owned_boundary_original_cache')
    if original is not None:
        if original.device!=cache.device or original.dtype!=cache.dtype or original.shape!=cache.shape or original.stride()!=cache.stride():
            raise ValueError('Owned score cache differs from original native ABI')
        if cache.storage_offset()!=0:raise ValueError('Owned score cache is not zero-offset')
        record.update(owned_boundary=True,owned_boundary_original_cache_ptr=int(original.data_ptr()),
            owned_boundary_original_cache_bytes=int(original.numel()*original.element_size()),
            owned_cache_storage_bytes=int(cache.untyped_storage().nbytes()))
    result=validate(record);_emit(state,'metadata_validated',actual=record,checks=result)
    return state
