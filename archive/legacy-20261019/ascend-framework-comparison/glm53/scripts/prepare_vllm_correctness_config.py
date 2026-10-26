"""Local code preparation only; does not deploy or invoke the runtime."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'config/vllm-correctness-v1'


def main():
    OUT.mkdir(exist_ok=True)
    for node, role, tp, dp, local_dp, start, master in [
        (30, 'P', 8, 4, 2, 0, 30), (31, 'P', 8, 4, 2, 2, 30),
        (43, 'D', 2, 16, 8, 0, 43), (45, 'D', 2, 16, 8, 8, 43),
    ]:
        kv_role = 'kv_producer' if role == 'P' else 'kv_consumer'
        transfer = {
            'kv_connector': 'AscendMultiConnector', 'kv_role': kv_role,
            'kv_connector_extra_config': {'connectors': [
                {'kv_connector': 'AscendStoreConnector', 'kv_role': kv_role,
                 'kv_connector_extra_config': {'backend': 'memcache', 'use_layerwise': False,
                    'consumer_is_to_load': False, 'consumer_is_to_put': False,
                    'memcache_dp_init_barrier': True, 'lookup_rpc_port': 28100 if role == 'P' else 28200}},
                {'kv_connector': 'MooncakePullConnector', 'kv_role': kv_role,
                 'kv_port': 26000 if role == 'P' else 27000,
                 'kv_connector_extra_config': {'glm53_sdma': {
                    'store_url': 'tcp://172.16.12.30:29580',
                    'session_port_offset': 4000, 'sdk_log_level': 0}}},
            ]},
        }
        args = ['--model', '/data/wio/models/GLM-5.3-w8a8',
                '--served-model-name', 'glm53-w8a8', '--trust-remote-code',
                '--dtype', 'bfloat16', '--quantization', 'ascend',
                '--tensor-parallel-size', str(tp), '--data-parallel-size', str(dp),
                '--data-parallel-size-local', str(local_dp), '--data-parallel-start-rank', str(start),
                '--data-parallel-address', f'172.16.12.{master}',
                '--data-parallel-rpc-port', '29410' if role == 'P' else '29420',
                '--data-parallel-backend', 'mp', '--distributed-executor-backend', 'mp',
                '--enable-expert-parallel', '--max-model-len', '80640',
                '--gpu-memory-utilization', '0.92', '--block-size', '128',
                '--max-num-seqs', '2' if role == 'P' else '4',
                '--max-num-batched-tokens', '2048' if role == 'P' else '512',
                '--enable-chunked-prefill', '--enable-prefix-caching', '--enforce-eager',
                '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 5}),
                '--kv-transfer-config', json.dumps(transfer)]
        result = {'project_id': 'glm53', 'node': node, 'role': role,
                  'phase': 'FUNCTIONAL_EAGER_NOT_PERFORMANCE_BASELINE',
                  'image_id': 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83',
                  'api_port': 30300 if role == 'P' else 30400,
                  'launch_args': args,
                  'parallel_intent': {'role_die': 32, 'node_die': 16, 'attention_tp': tp,
                                      'attention_dp': dp, 'expert_parallel': 32, 'pp': 1, 'cp': 1},
                  'kv_transfer': transfer}
        (OUT / f'node-{node}.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'local.conf').write_text(
        'ock.mmc.meta_service_url = tcp://172.16.12.30:29680\n'
        'ock.mmc.local_service.config_store_url = tcp://172.16.12.30:29681\n'
        'ock.mmc.log_level = info\n'
        'ock.mmc.local_service.world_size = 128\n'
        'ock.mmc.local_service.protocol = device_sdma\n'
        'ock.mmc.local_service.dram.size = 16GB\n')
    (OUT / 'meta.conf').write_text(
        'ock.mmc.meta_service_url = tcp://172.16.12.30:29680\n'
        'ock.mmc.meta_service.config_store_url = tcp://172.16.12.30:29681\n'
        'ock.mmc.meta_service.metrics_url = http://172.16.12.30:29682\n'
        'ock.mmc.log_level = info\n')


if __name__ == '__main__':
    main()
