"""Bind each framework's accepted AISBench config to one frozen 32K cohort."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

MANIFEST_SHA = '6a6a7dd8409d7b19ebbcfbee6bfec31673020ab1a6ecb5e001fae9c0a6633cef'
FORMAL_SHA = '1fba52e59b03579d49f028a5e83efead5eda5be85d92cb1ae496ecc68b058d14'
PHASES = {**{f'warmup{i}': 16 for i in range(1, 6)}, 'formal': 32}


def digest(raw):
    return sha256(raw).hexdigest()


def bind(dataset_dir, template_dirs, output):
    manifest_raw = (dataset_dir / 'manifest.json').read_bytes()
    assert digest(manifest_raw) == MANIFEST_SHA
    manifest = json.loads(manifest_raw)
    profiles = {p['phase']: p for p in manifest['profiles']}
    assert list(profiles) == list(PHASES)
    assert manifest['client_concurrency'] == 16
    assert manifest['required_request_output_tokens'] == 1024
    assert profiles['formal']['sha256'] == FORMAL_SHA
    assert manifest['submitted_model_requests'] == 0
    assert set(template_dirs) == {'sglang', 'vllm'}

    configs = {}
    bodies = {}
    for engine, template_dir in template_dirs.items():
        configs[engine] = {}
        for phase, count in PHASES.items():
            profile = profiles[phase]
            dataset_name = f'{phase}_32768.jsonl'
            assert profile['path'] == dataset_name and profile['requests'] == count
            raw = (dataset_dir / dataset_name).read_bytes()
            assert digest(raw) == profile['sha256']
            assert len(raw.splitlines()) == count
            filename = f'{engine}_{phase}_32768.py'
            source = (template_dir / filename).read_text()
            old_folder = 'client_ds41_g1122' if engine == 'sglang' else 'client_ds41_g503'
            old = f'/task/{old_folder}/datasets/{dataset_name}'
            new = f'/task/shared_retry32_32768/{dataset_name}'
            assert source.count(old) == 1
            config = source.replace(old, new)
            ast.parse(config, filename=filename)
            assert config.replace(new, old) == source
            bodies[filename] = config
            configs[engine][phase] = {
                'config': filename,
                'config_sha256': digest(config.encode()),
                'source_config_sha256': digest(source.encode()),
                'phase_dataset_sha256': profile['sha256'],
                'rows': count,
            }
    output.mkdir(mode=0o700, exist_ok=True)
    for filename, config in bodies.items():
        path = output / filename
        if path.exists():
            assert path.read_text() == config
        else:
            with path.open('x') as f:
                f.write(config)
    result = {
        'manifest_sha256': MANIFEST_SHA,
        'formal_and_replay_dataset_sha256': FORMAL_SHA,
        'config_mount': '/task/shared_retry32_32768',
        'config_changes_only_dataset_path': True,
        'frameworks': configs,
        'model_requests_sent_by_binder': 0,
    }
    raw = json.dumps(result, indent=2, sort_keys=True) + '\n'
    path = output / 'bindings.json'
    if path.exists():
        assert path.read_text() == raw
    else:
        with path.open('x') as f:
            f.write(raw)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-dir', 'sglang-templates', 'vllm-templates', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bind(args.dataset_dir, {
        'sglang': args.sglang_templates,
        'vllm': args.vllm_templates,
    }, args.output), sort_keys=True))
