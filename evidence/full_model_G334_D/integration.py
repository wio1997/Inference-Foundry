"""Scoped full-engine integration candidate; acceptance requires live gates."""
from pathlib import Path
import hashlib,json,os

def install(root):
    import sglang
    package=Path(sglang.__file__).resolve().parent.parent
    for filename in ['native_pool_sha256.json','native_integration_sha256.json']:
        for name,expected in json.loads((root/filename).read_text()).items():
            assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected, name
    from npu_backend import install_native_metadata_scope,install_bf16_request_window,DeepseekV41NPUBackend
    install_native_metadata_scope(root)
    from paged_indexer_scope_g314 import install as install_paged_indexer
    install_paged_indexer(root)
    install_bf16_request_window()
    from kv_geometry_g254 import install as install_geometry
    install_geometry(root)
    from lowratio_pool import LowRatioNPUTokenToKVPool
    import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pools
    original_pool=pools.DSV4NPUTokenToKVPool
    assert issubclass(LowRatioNPUTokenToKVPool,original_pool)
    pools.DSV4NPUTokenToKVPool=LowRatioNPUTokenToKVPool
    from lowratio_allocator import install as install_allocator
    install_allocator(root)
    import sglang.srt.layers.attention.attention_registry as registry
    def scoped_factory(prior):
        def create(runner):
            if (str(runner.device).startswith('npu') and
                getattr(runner.model_config.hf_config,'model_type',None)=='deepseek_v41'):
                from rope_install import install as install_rope
                install_rope(root)
                return DeepseekV41NPUBackend(runner)
            return prior(runner)
        return create
    for key in ['dsv4','ascend']:
        registry.ATTENTION_BACKENDS[key]=scoped_factory(registry.ATTENTION_BACKENDS[key])
    receipt={'pid':os.getpid(),'candidate_backend':'DeepseekV41NPUBackend',
             'pool':'LowRatioNPUTokenToKVPool','pricing':'G254',
             'native_compressor_indexer_candidates_and_DSPARK_inherited':True,
             'full_engine_PD_or_capacity_accepted':False,
             'prior_component_evidence':'G268 target metadata/attention 8ranks/48graphs/144replays; request_window=False; native source/indexer not yet executed'}
    (root/f'integration.pid{os.getpid()}.json').write_text(json.dumps(receipt,indent=2))
