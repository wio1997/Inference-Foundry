"""Pure metadata audit of packed target/draft host origins; no framework imports."""
import ast,hashlib,importlib.util,json,os,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class DType:
    itemsize=2
    def __str__(self):return 'torch.bfloat16'
    def __eq__(self,other):return isinstance(other,DType)
class Tensor:
    def __init__(self,shape,ptr,strides=None):
        self.shape=tuple(shape);self.ptr=ptr;self.dtype=DType()
        if strides is None:
            strides=[];v=1
            for dim in reversed(shape):strides.insert(0,v);v*=dim
        self.strides=tuple(strides)
    def numel(self):
        v=1
        for dim in self.shape:v*=dim
        return v
    def stride(self,n):return self.strides[n]
    def data_ptr(self):return self.ptr
    def __getitem__(self,indices):
        dims=list(self.shape);ptr=self.ptr
        for n,index in enumerate(indices):
            assert isinstance(index,slice);a,b,c=index.indices(dims[n]);assert c==1
            dims[n]=b-a;ptr+=a*self.strides[n]*2
        return Tensor(dims,ptr,self.strides)
def main():
    path=ROOT/'scripts/glm53_C32_hicache_native_mtp_v1.py'
    spec=importlib.util.spec_from_file_location('compat',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=ROOT/'runs/20261018T-L11-P-trace-hotspots-readonly-v1/source/sgl-workspace__sglang__python__sglang__srt__mem_cache__pool_host__mla.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==m.SOURCE_SHA
    tree=ast.parse(source.read_text());cls=next(z for z in tree.body if isinstance(z,ast.ClassDef) and z.name=='MLATokenToKVPoolHost')
    wanted=['_ascendc_comp_meta','_build_ascendc_component_meta','_npu_transfer_pool_slices','_npu_draft_pool_compatible']
    namespace={};selected=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[z for z in cls.body if isinstance(z,ast.FunctionDef) and z.name in wanted],type_ignores=[])
    exec(compile(ast.fix_missing_locations(selected),'native-source-subset','exec'),namespace)
    class Pool:pass
    for name in wanted:setattr(Pool,name,namespace[name])
    calls=[]
    def native_copy(self,pool,host_indices,device_indices,direction,layer_start=0,layer_num=-1,index_k_layer_start=0,index_k_layer_num=-1):
        comps=self._build_ascendc_component_meta(pool,layer_start,layer_num,index_k_layer_start,index_k_layer_num)
        calls.append({'pool':pool,'host_k_base':self.k_buffer.data_ptr(),'components':comps,'direction':direction})
    Pool._transfer_ascendc_sparse_copy=native_copy
    module=types.SimpleNamespace(__file__=str(source),MLATokenToKVPoolHost=Pool)
    old={k:os.environ.get(k) for k in ['GLM53_C32_NATIVE_HICACHE_MTP','SGLANG_HICACHE_HOST_MEM_BACKEND']}
    try:
        os.environ['GLM53_C32_NATIVE_HICACHE_MTP']='1';os.environ['SGLANG_HICACHE_HOST_MEM_BACKEND']='memfabric';m.install(module)
        host=Pool();host.page_size=128
        host.k_buffer=Tensor((5,79,128,1,512),1000000);host.v_buffer=Tensor((5,79,128,1,64),100000000)
        host.index_k_buffer=Tensor((5,78,128,1,128),200000000);host.index_k_scale_buffer=None
        def device(layers,ptr,index):
            return types.SimpleNamespace(k_buffer=Tensor((layers,4,128,1,512),ptr),v_buffer=Tensor((layers,4,128,1,64),ptr+1000000000),index_k_buffer=Tensor((layers,4,128,1,128),ptr+2000000000) if index else None,index_k_scale_buffer=None)
        target=device(78,300000000,True);draft=device(1,400000000,False)
        host.device_pool=target;host.mtp_draft_device_pools=(draft,)
        host._transfer_ascendc_sparse_copy(target,None,None,'D2H')
        assert len(calls)==2 and [len(z['components']) for z in calls]==[3,2]
        expected_draft_k=host.k_buffer.data_ptr()+78*128*512*2
        expected_draft_v=host.v_buffer.data_ptr()+78*128*64*2
        assert calls[0]['host_k_base']==host.k_buffer.data_ptr() and calls[1]['host_k_base']==expected_draft_k
        assert calls[1]['components'][0][1]==expected_draft_k and calls[1]['components'][1][1]==expected_draft_v
        assert calls[1]['components'][0][4]==79*128*512*2
        assert calls[0]['components'][0][-2:]==(0,78) and calls[1]['components'][0][-2:]==(0,1)
        calls.clear();host._transfer_ascendc_sparse_copy(draft,None,None,'H2D',layer_num=1,index_k_layer_num=1)
        assert len(calls)==1 and calls[0]['host_k_base']==expected_draft_k and len(calls[0]['components'])==2
        calls.clear();host._transfer_ascendc_sparse_copy(target,None,None,'H2D',layer_start=77,layer_num=1,index_k_layer_start=77,index_k_layer_num=1)
        assert len(calls)==1 and calls[0]['components'][0][-2:]==(77,78) and calls[0]['components'][2][-2:]==(77,78)
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    result={'project_id':'glm53','pass':True,'kind':'PURE_METADATA_NOT_NPU_CORRECTNESS','compatibility_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_MLA_source_sha256':m.SOURCE_SHA,'checks':['exact78target+1draft layout','both pools in fullD2H','draft host layer78 origin and original79layer pagepitch','target lastlayer/indexer lastrow bounds','perlayer draft excludes indexer matching legacy'],'native_NPU_fixture_required':True}
    out=ROOT/'results/C32-L12-native-HiCache-MTP-CPU-review-v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
