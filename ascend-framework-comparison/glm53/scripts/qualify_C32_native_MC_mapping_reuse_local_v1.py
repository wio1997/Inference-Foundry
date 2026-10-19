"""CPU boundary tests for the guarded registration integration; no SDK/NPU calls."""
import ast,hashlib,importlib.util,json,math,os,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    os.environ.update(GLM53_NODE='34',GLM53_C32_NATIVE_MC_MAPPING_REUSE='1',GLM53_C32_NATIVE_HICACHE_MTP='1',SGLANG_HICACHE_HOST_MEM_BACKEND='memfabric')
    p=ROOT/'scripts/glm53_C32_native_MC_mapping_reuse_v1.py'
    spec=importlib.util.spec_from_file_location('reuse',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Tensor:
        dtype='torch.bfloat16';device=types.SimpleNamespace(type='cpu')
        def __init__(self,shape,ptr):self.shape=shape;self.ptr=ptr
        def data_ptr(self):return self.ptr
        def numel(self):return math.prod(self.shape)
        def element_size(self):return 2
        def is_contiguous(self):return True
    calls=[]
    class Store:
        def register_buffer(self,tensor):calls.append(tensor);return 'original-return'
    source=ROOT/'runs/20261017T2018-sglang-C64-native-cache-source-v1/source/srt__mem_cache__storage__npu_memcache__npu_memcache_store.py'
    module=types.SimpleNamespace(__file__=str(source),NpuMemcacheStore=Store)
    original=Store.register_buffer;m.install(module);store=Store();store.store=object()
    shapes=[(634,79,128,1,512),(634,79,128,1,64),(634,78,128,1,128)]
    tensors=[];ptr=m.START
    for shape in shapes:t=Tensor(shape,ptr);tensors.append(t);ptr+=t.numel()*2
    host=types.SimpleNamespace(layout='page_first_kv_split',page_size=128,k_buffer=tensors[0],v_buffer=tensors[1],index_k_buffer=tensors[2],index_k_scale_buffer=None,kv_buffer=tensors[0]);store.mem_pool_host=host
    mock=types.ModuleType('memfabric_hybrid');mock.offload=types.SimpleNamespace(get_dva=lambda ptr:ptr);sys.modules['memfabric_hybrid']=mock
    for t in tensors:assert store.register_buffer(t) is None
    assert not calls
    plain=Tensor((1024,),4096);assert store.register_buffer(plain)=='original-return' and calls==[plain]
    rejected=[]
    def reject(label,fn):
        try:fn()
        except (AssertionError,AttributeError):rejected.append(label)
        else:raise AssertionError('Invalid input accepted: '+label)
    reject('foreign-native-pointer',lambda:store.register_buffer(Tensor(shapes[0],m.START+8192)))
    reject('partial-overlap',lambda:store.register_buffer(Tensor((16,),m.START-2)))
    old=host.layout;host.layout='page_first';reject('wrong-layout',lambda:store.register_buffer(tensors[0]));host.layout=old
    old=tensors[0].shape;tensors[0].shape=(634,78,128,1,512);reject('wrong-packed-layer-count',lambda:store.register_buffer(tensors[0]));tensors[0].shape=old
    mock.offload.get_dva=lambda ptr:0;reject('unmapped',lambda:store.register_buffer(tensors[0]))
    mock.offload.get_dva=lambda ptr:ptr if ptr==m.START else 0;reject('truncated-mapping',lambda:store.register_buffer(tensors[0]))
    mock.offload.get_dva=lambda ptr:ptr+4096;reject('different-DVA',lambda:store.register_buffer(tensors[0]))
    reject('double-install',lambda:m.install(module));reject('source-mismatch',lambda:m.install(types.SimpleNamespace(__file__=str(p),NpuMemcacheStore=type('Bad',(),{'register_buffer':original}))))
    del sys.modules['memfabric_hybrid']
    old=json.loads((ROOT/'config/sglang-C32-target-graph-fullmesh-stage-v3.json').read_text())['bundles'][0];proof={}
    for kind in ['candidate','rollback']:
        path=ROOT/('config/sglang-C32-native-HiCache-L12C-'+kind+'-stage-v1.json');b=json.loads(path.read_text())['bundles'][0]
        for key,item in b['files'].items():
            p=ROOT/item['local'];assert sha(p)==item['sha256']
            if p.suffix=='.py':ast.parse(p.read_text())
        for n in [34,35,36,41]:
            key='config/node-'+str(n)+'.json';a=json.loads((ROOT/old['files'][key]['local']).read_text());z=json.loads((ROOT/b['files'][key]['local']).read_text());assert a['launch_args']==z['launch_args']
            key='supervisor-config/node-'+str(n)+'.json';a=json.loads((ROOT/old['files'][key]['local']).read_text());z=json.loads((ROOT/b['files'][key]['local']).read_text());a.pop('run_id');assert z.pop('run_id')==b['runtime_run_id']
            if kind=='candidate' and n in [34,35]:a['model_env'].update(GLM53_C32_NATIVE_HICACHE_MTP='1',GLM53_C32_NATIVE_MC_MAPPING_REUSE='1',SGLANG_HICACHE_HOST_MEM_BACKEND='memfabric',GLM53_NODE=str(n))
            assert a==z
        proof[kind]={'manifest_sha256':sha(path),'files':len(b['files']),'launch_args_all4_unchanged':True,'D_source_config_env_unchanged':True,'pool_SDK_config_unchanged':True}
    assert not any(k.startswith(('torch','sglang','memfabric_hybrid','memcache_hybrid')) for k in sys.modules)
    out={'project_id':'glm53','pass':True,'kind':'CPU_GUARD_BOUNDARIES_NOT_NATIVE_MODEL_QUALIFICATION','bundles':proof,'native_components_reused':3,'ordinary_registration_preserved':True,'rejected_cases':rejected,'source_guard':m.SOURCE_SHA,'helper_sha256':sha(ROOT/'scripts/glm53_C32_native_MC_mapping_reuse_v1.py'),'NPU_calls':0,'SDK_init_calls':0,'model_requests':0}
    p=ROOT/'results/C32-L12C-local-mapping-reuse-CPU-review-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
