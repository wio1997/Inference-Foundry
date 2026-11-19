"""Graph-oriented NoPE consumer candidate. NPU graph correctness is UNVERIFIED."""
import torch
import torch_npu

class ZeroRopeSparseFA:
    def __init__(self,key_template,*,max_query_tokens,query_heads=8,rope_dim=64,sparse_width=2112,lanes=1):
        assert key_template.ndim==4 and key_template.shape[-2:]==(1,512)
        assert key_template.dtype==torch.bfloat16 and key_template.device.type=='npu'
        assert rope_dim==64 and sparse_width==2112 and max_query_tokens>0 and 1<=lanes<=2
        self.query_zeros=key_template.new_zeros((max_query_tokens,query_heads,64))
        self.key_zeros=key_template.new_zeros((*key_template.shape[:-1],64))
        self.index_scratch=[torch.full((max_query_tokens,1,2112),-1,dtype=torch.int32,device=key_template.device) for _ in range(lanes)]
        self.q_lens=[torch.zeros(max_query_tokens,dtype=torch.int32,device=key_template.device) for _ in range(lanes)]
        self.kv_lens=[torch.zeros(max_query_tokens,dtype=torch.int32,device=key_template.device) for _ in range(lanes)]
        self.request_ids=[torch.zeros(max_query_tokens,dtype=torch.int64,device=key_template.device) for _ in range(lanes)]
        self.rope_dim=64;self.sparse_width=2112;self.lanes=lanes;self.lane=0
    def select_lane(self,lane):
        # Shape/lane selection occurs in Python during eager/capture, never at graph replay.
        assert isinstance(lane,int) and 0<=lane<self.lanes;self.lane=lane
    def stage_request_ids(self,ids):
        assert ids.ndim==1 and ids.device==self.query_zeros.device
        assert ids.shape[0]<=self.request_ids[self.lane].shape[0]
        view=self.request_ids[self.lane][:ids.shape[0]];view.copy_(ids);return view
    def assert_graph_binding(self,ids):
        assert ids is not None and ids.device==self.query_zeros.device
        assert ids.data_ptr()==self.request_ids[self.lane].data_ptr(), 'graph reqids must use this lane persistent storage'
    def __call__(self,*,query,key,value,sparse_indices,scale_value,query_rope=None,key_rope=None,**kwargs):
        assert query_rope is None or query_rope.shape[-1]==0
        assert key_rope is None or key_rope.shape[-1]==0
        assert query.shape[1:]==self.query_zeros.shape[1:-1]+(512,)
        assert query.dtype==key.dtype==value.dtype==self.query_zeros.dtype
        assert key.shape==value.shape and key.shape[1:-1]==self.key_zeros.shape[1:-1]
        assert key.shape[0]<=self.key_zeros.shape[0] and query.shape[0]<=self.query_zeros.shape[0]
        assert kwargs.get('layout_query')=='TND' and kwargs.get('layout_kv')=='PA_BSND'
        assert kwargs.get('attention_mode')==2 and kwargs.get('sparse_block_size')==1
        if sparse_indices.ndim==2:sparse_indices=sparse_indices.unsqueeze(1)
        assert sparse_indices.dtype==torch.int32 and sparse_indices.shape[1]==1
        assert sparse_indices.shape[0]==query.shape[0] and sparse_indices.shape[2]<=2112
        qlen=kwargs.pop('actual_seq_lengths_query');kvlen=kwargs.pop('actual_seq_lengths_kv')
        assert qlen.ndim==kvlen.ndim==1 and qlen.shape==kvlen.shape
        assert qlen.dtype==kvlen.dtype==torch.int32
        assert qlen.device==kvlen.device==query.device==self.query_zeros.device
        assert qlen.shape[0]<=query.shape[0]
        table=kwargs['block_table'];assert table.dtype==torch.int32 and table.device==query.device and table.shape[0]>=qlen.shape[0]
        n=qlen.shape[0];qdst=self.q_lens[self.lane][:n];kdst=self.kv_lens[self.lane][:n]
        qdst.copy_(qlen);kdst.copy_(kvlen)
        indices=self.index_scratch[self.lane][:query.shape[0]]
        indices.fill_(-1);indices[:,:,:sparse_indices.shape[2]].copy_(sparse_indices)
        return torch_npu.npu_sparse_flash_attention(query=query,key=key,value=value,sparse_indices=indices,query_rope=self.query_zeros[:query.shape[0]],key_rope=self.key_zeros[:key.shape[0]],scale_value=scale_value,actual_seq_lengths_query=qdst,actual_seq_lengths_kv=kdst,**kwargs)
    def allocation_metadata(self):
        return {'rope_dim':64,'sparse_width':2112,'lanes':self.lanes,'query_zero_bytes':self.query_zeros.numel()*self.query_zeros.element_size(),'key_zero_bytes':self.key_zeros.numel()*self.key_zeros.element_size(),'lane_private_bytes':[sum(t.numel()*t.element_size() for t in xs) for xs in zip(self.index_scratch,self.q_lens,self.kv_lens,self.request_ids)],'scope':'candidate persistent addresses; no NPU graph/overlap/MTP acceptance'}
