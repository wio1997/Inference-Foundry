"""Exact-image configuration validation; never creates an engine or reads weights."""
import hashlib
import importlib
import json
from pathlib import Path

root = Path('/opt/glm53/source')
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256(Path(entry['container_path']).read_bytes()).hexdigest() == entry['patched_sha256']
for entry in manifest['helpers']:
    assert hashlib.sha256((root / entry['local']).read_bytes()).hexdigest() == entry['sha256']
import memfabric_hybrid as mf
import memcache_hybrid as mc
assert mf.__file__.startswith('/opt/glm53/sdk13/') and mc.__file__.startswith('/opt/glm53/sdk13/')
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

summaries = []
for node in (30, 31, 43, 45):
    intended = json.loads(Path(f'/opt/glm53/config/node-{node}.json').read_text())
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(intended['launch_args'])
    args = EngineArgs.from_cli_args(parsed)
    config = args.create_engine_config()
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

# Capture only the exact serving and native DP dispatch contracts needed next.
search = Path('/vllm-workspace/vllm')
snippets = []
for sub in ('vllm/entrypoints', 'examples/disaggregated'):
    for p in (search / sub).rglob('*.py'):
        if p.stat().st_size > 250000:
            continue
        text = p.read_text(); lines = text.splitlines(); indices = set()
        for i, line in enumerate(lines):
            if any(s in line for s in ('X-data-parallel', 'x-data-parallel', 'X-Data-Parallel',
                                       'data_parallel_rank=', 'last_token_id', 'do_remote_decode')):
                indices.update(range(max(0, i-6), min(len(lines), i+12)))
        if indices:
            snippets.append({'path': str(p), 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'lines': [[i+1, lines[i]] for i in sorted(indices)]})
import logging
from glm53_sdma_transport import audit
assert logging.getLogger('vllm.glm53_sdma_transport').isEnabledFor(logging.INFO)
audit('preflight_logging_fixture', model_requests=0)
from glm53_memcache_batch import batched_get_into_layers
assert callable(batched_get_into_layers)
print('GLM53_PREFLIGHT_JSON=' + json.dumps({'configs': summaries, 'snippets': snippets,
      'model_requests': 0, 'weight_payload_read': False, 'sdk_paths': [mf.__file__, mc.__file__]}))
