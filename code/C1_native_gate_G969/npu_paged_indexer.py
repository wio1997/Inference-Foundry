"""NPU-only decode candidate: reuse native FP4 paged math without dense slot maps.

Prepared for strict component/graph validation, not deployed to a model.
Keep FP4 operands, BF16 rounding, score capacity and torch top-k unchanged.
"""
from pathlib import Path
import hashlib,json,os
import torch
import triton
import triton.language as tl
from freqs_compat import gather as gather_freqs

@triton.jit
def _map_sorted_selected_slots(Indices,Lengths,Requests,ReqTable,Pages,
        BATCH:tl.constexpr,K:tl.constexpr,WIDTH:tl.constexpr,OUT_WIDTH:tl.constexpr,
        IDX_STRIDE:tl.constexpr,TABLE_STRIDE:tl.constexpr,PAGE_STRIDE:tl.constexpr,
        RATIO:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    idx=tl.load(Indices+row*IDX_STRIDE+col,(row<BATCH)&(col<K),WIDTH)
    length=tl.load(Lengths+row,row<BATCH,0)
    req=tl.load(Requests+row,row<BATCH,0).to(tl.int64)
    valid=(row<BATCH)&(col<K)&(idx>=0)&(idx<length)
    slot=tl.load(ReqTable+req*TABLE_STRIDE+idx.to(tl.int64)*RATIO,valid,0)
    tl.store(Pages+row*PAGE_STRIDE+col,tl.where(valid,slot.to(tl.int64)//RATIO,-1),col<OUT_WIDTH)

def score_paged(q,weights,req,req_table,lens,table,page_size,max_len,ratio):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import _fp4_index_logits_paged_kernel
    assert q.dtype==torch.bfloat16 and q.shape[-1]==128
    assert ratio in (1,2) and req_table.stride(1)==1
    assert max_len<=req_table.shape[1]//ratio and table.dtype==torch.uint8
    batch,heads,_=q.shape;q=q.contiguous();weights=weights.to(torch.bfloat16).contiguous()
    req=req.contiguous();lens=lens.contiguous()
    # The stock paged kernel writes only visible prefixes. Initialize every
    # other score to the same -inf consumed by the stock full-width torch top-k.
    scores=torch.full((batch,max_len),-torch.inf,dtype=torch.float32,device=q.device)
    if not batch or not max_len:return scores
    # Bounded A2 scheduling budget; changes launch geometry, not mathematical work.
    workers=max(1,min(triton.cdiv(max_len,64),triton.cdiv(96,batch)))
    _fp4_index_logits_paged_kernel[(batch,workers)](
        q,weights,req,req_table,lens,table,None,scores,max_len,page_size,
        table.stride(0),req_table.stride(0),0,scores.stride(0),q.stride(0),q.stride(1),weights.stride(0),
        H=heads,HALF_D=64,RATIO=ratio,HAS_MASK=False,BLOCK_L=64,num_warps=4)
    return scores

def select_slots(scores,lens,req,req_table,out_pages,ratio,topk):
    k=min(topk,scores.shape[1])
    # Use precisely the stock torch sort, avoiding the native finisher's
    # unsupported NPU bitonic lowering. Only the selected K positions are mapped.
    indices=scores.topk(k,dim=-1,sorted=False).indices.sort(dim=-1).values.to(torch.int32)
    _map_sorted_selected_slots[(out_pages.shape[0],)](
        indices,lens,req,req_table,out_pages,lens.numel(),k,scores.shape[1],out_pages.shape[1],
        indices.stride(0),req_table.stride(0),out_pages.stride(0),ratio,
        triton.next_power_of_2(max(k,out_pages.shape[1])),num_warps=4)
    return indices

def install(root):
    import sglang.srt.layers.attention.dsv4.v41_indexer.full_topk as full
    import sglang.srt.layers.attention.dsv4.v41_indexer.scoring as scoring
    import sglang.kernels.ops.attention.dsv4.fp4_indexer as native
    proof=json.loads((root/'paged_indexer_source_pins_g901.json').read_text())
    for module,key in [(full,'full_topk'),(scoring,'scoring'),(native,'FP4_kernel')]:
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==proof[key]
    cls=full.FullTopKIndexer
    assert not getattr(cls,'_npu_paged_candidate_g911',False)
    original=cls._torch_decode
    seen=set()
    def decode(self,inputs):
        if inputs.x.device.type!='npu':return original(self,inputs)
        assert not self.use_deep_gemm_decode and not self.use_deep_select_decode
        ratio=inputs.compress_ratio;assert ratio in (1,2)
        req,pos=inputs.req_rows,inputs.positions;batch=req.shape[0]
        assert pos.shape[0]==batch and inputs.indexer.n_local_heads==32
        inputs.reset_outputs()
        max_len=min(inputs.paged_metadata.max_compressed_seq_len,self.req_to_token.shape[1]//ratio)
        if not batch or not max_len:return
        lens=(pos+1)//ratio
        q=inputs.indexer.queries(inputs.q_lora,gather_freqs(inputs.freqs_cis,pos))
        weights=inputs.indexer.head_weights(inputs.x)
        table=self.token_to_kv_pool.get_index_k_with_scale_buffer(inputs.layer_id)
        scores=score_paged(q,weights,req,self.req_to_token,lens,table,table.shape[1]//68,max_len,ratio)
        select_slots(scores,lens,req,self.req_to_token,inputs.out_page_indices,ratio,inputs.indexer.index_topk)
        key=(inputs.layer_id,ratio,batch,max_len)
        if key not in seen:
            seen.add(key)
            receipt={'pid':os.getpid(),'layer_id':inputs.layer_id,'ratio':ratio,'query_rows':batch,'score_capacity':max_len,'query_shape':list(q.shape),'score_kernel':'native_fp4_index_logits_paged','selection':'same_torch_topk_and_sort_then_selectedK_mapping','candidate_source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Pythonpathhit/captureconstruction only; actualreplayrequiresdeviceprofile, notqualityorgainproof'}
            with (root/f'paged_indexer_hits.pid{os.getpid()}.jsonl').open('a') as log:log.write(json.dumps(receipt)+'\n')
    cls._torch_decode=decode;cls._npu_paged_candidate_g911=True
    (root/f'paged_indexer_install_g911.pid{os.getpid()}.json').write_text(json.dumps({'source_pins':proof,'scope':'NPUdecodecandidate only; nativepagedFP4math/nativeTorchTopK/stocktorchsort+NPU selectedK gather. NoPprefill/quant/weights/modelcontextchange. Requiresstrictscores/slots+changedinputgraphgate beforemodeldeployment.'},indent=2)+'\n')
