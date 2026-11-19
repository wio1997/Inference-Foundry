"""PP1 GLM target+independent NextN C8 tail state, native PD transport."""

def combined_tail_registration(target_root, target, draft, total_layers):
    import torch
    from glm53_sglang_adapter.pool import NPUKPoolTokenToKVPool
    from sglang.srt.runtime_context import get_parallel
    if get_parallel().pp_size != 1:
        raise NotImplementedError('Independent GLM draft tail aggregation currently requires PP1')
    if not isinstance(target,NPUKPoolTokenToKVPool) or not isinstance(draft,NPUKPoolTokenToKVPool):
        raise TypeError('Actual target and independent draft NPUKPool classes required')
    if target is draft or total_layers is None:
        raise ValueError('Independent draft and explicit target layer band required')
    if tuple(draft.indexer_layer_ids)!=(0,) or draft.layer_num!=1:
        raise ValueError('Scoped native NextN must own exactly dense indexer layer zero')
    mapping=target_root.full_attention_layer_id_mapping
    inverse={dense:global_id for global_id,dense in mapping.items()}
    if len(inverse)!=len(mapping) or any(i not in inverse for i in target.indexer_layer_ids):
        raise ValueError('Target global/dense layer mapping is incomplete')
    target_ids=[inverse[i] for i in target.indexer_layer_ids]
    if any(i<0 or i>=total_layers for i in target_ids):
        raise ValueError('Target indexer global layer ID lies outside target band')
    infos=[];rows=None;pointers=set()
    for pool in (target,draft):
        if (not pool.glm53_index_in_owner_kv or pool.index_kpool!=4
            or not pool.kpool_use_compress or pool.tail_extra_slots!=4):
            raise ValueError('Native owner-index and C8 KPool4+W4 geometry required')
        tensors=list(pool._compress_tail_k)+list(pool._compress_tail_score)
        if len(tensors)!=2*pool.num_indexer_layers or not tensors:
            raise ValueError('Tail key/score halves must cover native indexer layers')
        for tensor in tensors:
            if tensor.dtype!=torch.float32 or tensor.ndim!=3 or tensor.shape[1:]!=(8,128):
                raise ValueError('Registered tail must retain FP32 [request rows,8,128] layout')
            if rows is None:rows=tensor.shape[0]
            if tensor.shape[0]!=rows or tensor.device!=target._compress_tail_k[0].device:
                raise ValueError('Target/draft tail request rows and device must match')
            if tensor.data_ptr() in pointers:
                raise ValueError('Independent target/draft tail pointers must never alias')
            pointers.add(tensor.data_ptr())
        info=pool.get_compress_tail_buf_infos()
        expected=([x.data_ptr() for x in tensors],[x.nbytes for x in tensors],[x[0].nbytes for x in tensors])
        if info!=expected:
            raise ValueError('Native registration no longer matches authoritative tail arrays')
        infos.append(info)
    # One DSA_TAIL component: target K + draft K, target G + draft G.
    result=[]
    for component in range(3):
        t,d=infos[0][component],infos[1][component]
        th,dh=len(t)//2,len(d)//2
        result.append(t[:th]+d[:dh]+t[th:]+d[dh:])
    ids=target_ids+[int(total_layers)]
    return (*result,ids+ids)


def record_draft_prefill(pool, batch, forward_batch):
    """CPU shape/slot witness after real native draft prefill was enqueued.

    Scheduler's native copy_done fence completes after the whole worker returns;
    this witness is not a replacement GPU completion event.
    """
    if not getattr(pool,'glm53_index_in_owner_kv',False):return
    if forward_batch.forward_mode.is_idle():return
    prefix=forward_batch.extend_prefix_lens_cpu
    lengths=forward_batch.extend_seq_lens_cpu
    if prefix is None or lengths is None or len(prefix)!=len(batch.reqs) or len(lengths)!=len(batch.reqs):
        raise RuntimeError('Native draft prefill host position geometry must be available')
    original=getattr(forward_batch,'_original_batch_size',None)
    if original is not None and original!=len(batch.reqs):
        raise RuntimeError('Draft prefill native original request count disagrees')
    for req,start,count in zip(batch.reqs,prefix,lengths):
        end=int(start)+int(count)
        if min(int(start),int(count))<0 or end!=int(req.extend_end):
            raise RuntimeError('Draft prefill position endpoint differs from target committed length')
        req._glm53_pd_draft_prefill=(id(pool),int(req.kv.req_pool_idx),end)


def validate_draft_tail_send(draft_pool, req, committed_length):
    if draft_pool is None or not getattr(draft_pool,'glm53_index_in_owner_kv',False):return
    expected=(id(draft_pool),int(req.kv.req_pool_idx),int(committed_length))
    if getattr(req,'_glm53_pd_draft_prefill',None)!=expected:
        raise RuntimeError('PD refuses draft tails without same pool/request-slot/committed-L prefill witness')
