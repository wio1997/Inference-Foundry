"""Load the pinned AISBench config and dataset without creating an API client."""

from hashlib import sha256
import inspect
import json
from pathlib import Path

from mmengine.config import Config


def digest(raw):
    return sha256(raw).hexdigest()


folder = Path('/configs')
raw = (folder / 'bindings.json').read_bytes()
contracts = json.loads(raw)
checks = {}
for framework, phase_data in contracts['frameworks'].items():
    checks[framework] = {}
    for phase, binding in phase_data.items():
        p = folder / binding['config']
        assert digest(p.read_bytes()) == binding['config_sha256']
        cfg = Config.fromfile(p)
        assert len(cfg.models) == len(cfg.datasets) == 1
        model, ds = cfg.models[0], cfg.datasets[0]
        assert model.type.__name__ == 'VLLMCustomAPI'
        assert model.batch_size == 16 and model.max_out_len == 1024 and model.stream
        assert model.generation_kwargs == {'temperature': 0.0, 'ignore_eos': True, 'min_tokens': 1024}
        assert ds.type.__name__ == 'CustomDataset'
        path = Path(ds.path)
        assert path == Path(contracts['config_mount']) / f'{phase}_32768.jsonl'
        assert digest(path.read_bytes()) == binding['phase_dataset_sha256']
        load = ds.type.load
        dataset = load(str(path))
        assert len(dataset) == binding['rows'], (inspect.signature(load), type(dataset))
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert all(x['input_tokens'] == 32768 for x in rows)
        assert all(dataset[i]['question'] == x['question'] and dataset[i]['max_out_len'] == 1024 for i, x in enumerate(rows))
        checks[framework][phase] = {
            'dataset_rows_loaded': len(dataset),
            'dataset_sha256': binding['phase_dataset_sha256'],
            'config_sha256': binding['config_sha256'],
        }
assert checks['sglang']['formal']['dataset_sha256'] == checks['vllm']['formal']['dataset_sha256'] == contracts['formal_and_replay_dataset_sha256']
print(json.dumps({
    'bindings_sha256': digest(raw),
    'actual_AISBench_dataset_loads': checks,
    'config_loader': Config.__module__,
    'installed_config_loader_sha256': digest(Path(inspect.getfile(Config)).read_bytes()),
    'actual_no_client_init_model_requests': 0,
}, sort_keys=True))
