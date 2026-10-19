"""Validate compact six-case configuration/certificate comparability locally."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    16: 'dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9',
    32: '9bdba915c372eb2c23da018a18c2c9653ded9c5012d875db8b496effb5983b67',
    64: 'c4a6222088e7a8d7b9fafd1666e58601f3a0cf5055350871b1cb6a60aa7145d3',
}
CLIENT = '935a4a1c3188b3c7d2041f91e946423838dece7bcc5669daee6ae81c6e6c56a3'


def main():
    sources, cases, runtime, client_map = {}, [], {}, None
    for concurrency in (16, 32, 64):
        for framework in ('vllm', 'sglang'):
            config = (f'config/formal-runs-v2/vllm-C{concurrency}.json' if framework == 'vllm'
                      else f'config/optimization-overlays/routing-metadata-v1/formal-C{concurrency}-v2.json')
            certificate = (f'results/vllm-formal-C{concurrency}-v2-measurement-v1.json' if framework == 'vllm'
                           else f'results/sglang-L04-C{concurrency}-v1-measurement-v1.json')
            settings = json.loads((ROOT / config).read_text())
            measured = json.loads((ROOT / certificate).read_text())
            assert settings['framework'] == measured['framework'] == framework
            assert settings['run_id'] == measured['run_id']
            assert settings['concurrency'] == measured['concurrency'] == concurrency
            assert measured['measurement_valid'] is True and measured['client_audit_pass'] is True
            assert measured['native_postrun_audit_required'] is False
            assert settings['input_manifest_sha256'] == MANIFESTS[concurrency]
            assert settings['harness_revision'] == 'v2-uniform-fresh-upstream-HTTP-no-retries'
            assert settings['client_files_sha256']['glm53_formal_ais_client_v1.py'] == CLIENT
            if client_map is None:
                client_map = settings['client_files_sha256']
            assert client_map == settings['client_files_sha256']
            assert measured['formal_requests'] == 4 * concurrency
            assert measured['formal_successful_output_tokens'] == 600 * measured['formal_requests']
            assert measured['actual_client_peak_inflight'] == concurrency
            assert math.isclose(measured['output_tokens_per_second'], measured['formal_successful_output_tokens'] / measured['full_clock_seconds'], rel_tol=1e-12)
            cids = {node: item['container_id'] for node, item in settings['runtime_containers'].items()}
            assert measured['runtime_CIDs'] == cids
            identity = {'shared_run_id': settings['shared_run_id'], 'CIDs': cids,
                        'image': settings['model_image'], 'native_client_config_sha256': settings['native_client_config_sha256']}
            if framework not in runtime:
                runtime[framework] = identity
            assert runtime[framework] == identity
            native = measured['native_postrun_evidence']
            assert hashlib.sha256((ROOT / native).read_bytes()).hexdigest() == measured['native_postrun_evidence_sha256']
            for relative in (config, certificate, native):
                sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            cases.append({'framework': framework, 'concurrency': concurrency, 'certificate': certificate,
                          'full_clock_TPS': measured['output_tokens_per_second'],
                          'aggregate_SLA_pass': measured['aggregate_SLA_pass']})
    for name, digest in client_map.items():
        relative = json.loads((ROOT / 'config/formal-runs-v2/vllm-C16.json').read_text())['client_files'][name]
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
        sources[relative] = digest
    result = {
        'project_id': 'glm53', 'pass': True, 'cases': cases,
        'same_canonical_inputs_per_C': True, 'same_frozen_client_files_in_all_six_cases': True,
        'same_runtime_CIDs_image_native_client_config_within_each_framework': runtime,
        'all_primary_full_clock_rates_recomputed': True,
        'scope': 'Compact frozen configuration and accepted-certificate consistency. Per-case original client/native audits establish raw data validity; this is not a new raw-data audit.',
        'limits': ['Native deployments use distinct machine groups and native graph/cache/staging/SDK packaging.',
                   'C16 before/after is a separate single matched observation; C32/C64 have no unoptimized SGLang baseline.',
                   'Repeated full-run noise and pure hardware-independent framework attribution are not established.'],
        'source_evidence_sha256': sources,
        'new_model_SDK_init_SSH_calls': 0,
    }
    destination = ROOT / 'results/L04-matrix-comparability-v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'path': str(destination), 'cases': len(cases), 'pass': True}))


if __name__ == '__main__':
    main()
