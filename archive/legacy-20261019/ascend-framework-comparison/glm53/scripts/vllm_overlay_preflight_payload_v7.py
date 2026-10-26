"""Exact-image configuration validation; never creates an engine or reads weights."""
import hashlib
import importlib
import json
from pathlib import Path

root = Path('/opt/glm53/source')
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()=='1fc1d1e722b6dbb2d302980963c0285458e52acfa38abc95359828857a24d1ed'
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256(Path(entry['container_path']).read_bytes()).hexdigest() == entry['patched_sha256']
for entry in manifest['helpers']:
    assert hashlib.sha256((root / entry['local']).read_bytes()).hexdigest() == entry['sha256']
import memfabric_hybrid as mf
import memcache_hybrid as mc
assert mf.__file__.startswith('/opt/glm53/sdk13/') and mc.__file__.startswith('/opt/glm53/sdk13/')
import os,importlib.metadata
assert importlib.metadata.version('memcache_hybrid')=='1.3.0'
for key,value in {'PYTHONHASHSEED':'0','ACL_OP_INIT_MODE':'1','ASCEND_ENABLE_USE_FABRIC_MEM':'1'}.items():assert os.environ[key]==value
meta=Path('/opt/glm53/config/meta.conf').read_text();local=Path('/opt/glm53/config/local.conf').read_text()
assert 'ock.mmc.meta.lease_ttl_ms = 60000' in meta and 'ock.mmc.evict_threshold_high = 99' in meta and 'ock.mmc.evict_threshold_low = 98' in meta
assert 'ock.mmc.local_service.dram.size = 0GB' in local and 'ock.mmc.local_service.max.dram.size = 192GB' in local
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

summaries = []
for node in (30, 31, 43, 45):
    intended = json.loads(Path(f'/opt/glm53/config/node-{node}.json').read_text())
    parser = make_arg_parser(FlexibleArgumentParser())
    parsed = parser.parse_args(intended['launch_args'])
    assert parsed.enable_prompt_tokens_details is True
    args = EngineArgs.from_cli_args(parsed)
    config = args.create_engine_config()
    assert config.observability_config.cudagraph_metrics is True
    profiler=config.profiler_config
    assert profiler.profiler=='torch' and profiler.ignore_frontend is True
    assert profiler.delay_iterations==2 and profiler.max_iterations==8
    assert profiler.torch_profiler_with_stack is False and profiler.torch_profiler_with_memory is False and profiler.torch_profiler_record_shapes is False and profiler.capture_torch_profiler is False
    assert intended['required_meta_service_lease_ttl_ms']==60000
    parallel = config.parallel_config
    cls = KVConnectorFactory.get_connector_class(config.kv_transfer_config)
    children = list(cls._get_connector_classes_and_configs(config))
    assert cls.__name__ == 'AscendMultiConnector'
    assert [c.__name__ for c, _ in children] == ['AscendStoreConnector', 'MooncakePullConnector']
    for child, _ in children:
        importlib.import_module(child.__module__)
    assert parallel.tensor_parallel_size * parallel.data_parallel_size == 32
    assert parallel.tensor_parallel_size * parallel.data_parallel_size_local == 16
    assert parallel.enable_expert_parallel is True
    from vllm.config.compilation import CUDAGraphMode
    if node in (43,45):
        assert config.compilation_config.cudagraph_mode==CUDAGraphMode.FULL_DECODE_ONLY
        assert config.compilation_config.cudagraph_capture_sizes==[6,12,18,24]
        assert config.speculative_config.enforce_eager is True
        assert not config.model_config.enforce_eager
    else:assert config.model_config.enforce_eager
    draft=config.speculative_config.draft_model_config.hf_config
    from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import is_multi_kv_cache_group_mtp
    print('GLM53_GRAPH_PREFLIGHT='+json.dumps({'node':node,'target_mode':str(config.compilation_config.cudagraph_mode),'capture_sizes':config.compilation_config.cudagraph_capture_sizes,'draft_type':draft.model_type,'draft_architectures':draft.architectures,'multi_kv_mtp':is_multi_kv_cache_group_mtp(config),'draft_enforce_eager':config.speculative_config.enforce_eager}),flush=True)
    summaries.append({'node': node, 'parallel_config': str(parallel),
                      'model_config': str(config.model_config),
                      'speculative_config': str(config.speculative_config),
                      'cache_config': str(config.cache_config),
                      'compilation_config': str(config.compilation_config),
                      'connector_classes': [c.__module__ + '.' + c.__name__ for c, _ in children]})

import logging
from glm53_sdma_transport import audit
assert logging.getLogger('vllm.glm53_sdma_transport').isEnabledFor(logging.INFO)
audit('preflight_logging_fixture', model_requests=0)
from glm53_memcache_native import native_get_into_layers
assert callable(native_get_into_layers)
assert not (root/'python/glm53_memcache_batch.py').exists(), 'No experimental slicing helper in new source'
print('GLM53_PREFLIGHT_JSON=' + json.dumps({'configs': summaries, 'profile_activation':'INACTIVE_until_separate_afterbaseline_window',
      'model_requests': 0, 'weight_payload_read': False, 'sdk_paths': [mf.__file__, mc.__file__]}))
