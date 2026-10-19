"""P-only compatibility for native Memfabric HiCache IO with packed MTP host rows.

The upstream native switch enables allocation and AscendC IO together. Its
single-pool copy helper does not select the packed draft host rows. Reuse the
legacy path's exact pool views, then call the original native copy helper.
"""
import hashlib,json,os
from pathlib import Path

REVISION='glm53-C32-native-hicache-MTP-v1'
SOURCE_SHA='4009474db3858a74ca89e314bb4ea3a7840f0ce884d7c422fcb4fe6131ef5041'

def _validate(self,slices):
    assert self.page_size==128 and len(slices)==2
    assert slices[0][0] is self.device_pool and slices[1][0] is self.mtp_draft_device_pools[0]
    assert len(self.mtp_draft_device_pools)==1
    assert [s[0].k_buffer.shape[0] for s in slices]==[78,1]
    assert self.k_buffer.shape[1]==79
    for index,(pool,host_k,host_v,host_index,host_scale) in enumerate(slices):
        assert str(pool.k_buffer.dtype)==str(host_k.dtype)=='torch.bfloat16'
        assert str(pool.v_buffer.dtype)==str(host_v.dtype)=='torch.bfloat16'
        assert pool.k_buffer.shape[-1]==host_k.shape[-1]==512
        assert pool.v_buffer.shape[-1]==host_v.shape[-1]==64
        assert pool.k_buffer.shape[0]==host_k.shape[1]
        assert pool.v_buffer.shape[0]==host_v.shape[1]
        assert host_scale is None and getattr(pool,'index_k_scale_buffer',None) is None
        if index==0:
            assert host_index is not None and pool.index_k_buffer is not None
            assert str(host_index.dtype)==str(pool.index_k_buffer.dtype)=='torch.bfloat16'
            assert host_index.shape[1]==pool.index_k_buffer.shape[0]==78
            assert host_index.shape[-1]==pool.index_k_buffer.shape[-1]==128
        else:
            # Match the existing legacy HiCache policy: draft indexer is not
            # mirrored in this host pool. PD still transfers its live state.
            assert host_index is None

def install(module):
    assert os.environ.get('GLM53_C32_NATIVE_HICACHE_MTP')=='1'
    assert os.environ.get('SGLANG_HICACHE_HOST_MEM_BACKEND')=='memfabric'
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA
    cls=module.MLATokenToKVPoolHost
    assert not getattr(cls,'_glm53_C32_native_MTP_installed',False)
    original=cls._transfer_ascendc_sparse_copy
    def transfer(self,device_pool,host_indices,device_indices,direction,
                 layer_start=0,layer_num=-1,index_k_layer_start=0,index_k_layer_num=-1):
        slices=list(self._npu_transfer_pool_slices());_validate(self,slices)
        if layer_num<0:
            assert device_pool is self.device_pool and layer_start==0
            assert index_k_layer_start==0 and index_k_layer_num<0
            selected=slices
        else:
            assert layer_num==1
            selected=[s for s in slices if s[0] is device_pool];assert len(selected)==1
        if not getattr(self,'_glm53_C32_native_MTP_first_use',False):
            self._glm53_C32_native_MTP_first_use=True
            print('GLM53_C32_NATIVE_HICACHE_MTP_FIRST_USE='+json.dumps({
                'project_id':'glm53','revision':REVISION,'pid':os.getpid(),
                'target_layers':78,'draft_layers':1,'target_indexer_rows':78,
                'draft_indexer_host_mirror':False,'upstream_MLA_SHA256':SOURCE_SHA,
                'original_native_copy_helper_preserved':True}),flush=True)
        for pool,k,v,ik,scale in selected:
            # Preserve the original tensors, pitches, dtype, indices and stream
            # ownership. The view object only changes the host layer origin.
            view=cls.__new__(cls)
            view.page_size=self.page_size
            view.k_buffer=k;view.v_buffer=v
            view.index_k_buffer=ik;view.index_k_scale_buffer=scale
            original(view,pool,host_indices,device_indices,direction,
                     layer_start=layer_start,layer_num=layer_num,
                     index_k_layer_start=index_k_layer_start,
                     index_k_layer_num=index_k_layer_num if ik is not None else 0)
    cls._glm53_C32_original_native_transfer=original
    cls._transfer_ascendc_sparse_copy=transfer
    cls._glm53_C32_native_MTP_installed=True
    return {'revision':REVISION,'source_sha256':SOURCE_SHA,
            'variable':'Native mapped-host/AscendC HiCache IO, preserving packed target+draft views'}
