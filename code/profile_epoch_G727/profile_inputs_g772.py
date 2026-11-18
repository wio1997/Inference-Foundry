"""Freeze fresh synthetic profile inputs; CPU tokenizer only, no serving calls.

The separate prefix lets the normal five independent warmups run unchanged.
Both frameworks must consume these exact new input bytes for diagnosis.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, manifest_path, configs):
    from tokenizers import Tokenizer
    assert root.is_absolute() and not root.exists()
    tokenizer_path = Path('/models/ds41/tokenizer.json')
    assert sha(tokenizer_path) == 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    manifest = json.loads(manifest_path.read_text())
    reserved = set()
    source_rows = None
    for item in manifest['profiles']:
        path = Path(item['path'])
        assert sha(path) == item['sha256']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            prefix = tuple(tokenizer.encode(row['question'][:5000], add_special_tokens=False).ids[:128])
            assert len(prefix) == 128
            reserved.add(prefix)
        if item['phase'] == 'warmup1' and item['input_tokens'] == 32768:
            source_rows = rows
            source_sha = item['sha256']
    assert len(reserved) == 448 and len(source_rows) == 16
    root.mkdir(parents=True)
    nonce = 'DS41-PROFILE-G772-' + uuid.uuid4().hex
    frozen = []
    records = []
    for index, old in enumerate(source_rows):
        text = nonce + '-' + str(index) + ' ' + old['question']
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:32768]
        assert len(ids) == 32768
        text = tokenizer.decode(ids, skip_special_tokens=False)
        assert tokenizer.encode(text, add_special_tokens=False).ids == ids
        prefix = tuple(ids[:128])
        assert prefix not in reserved
        reserved.add(prefix)
        frozen.append({'question': text, 'answer': '', 'max_out_len': 1024})
        records.append({'data_id': index, 'question_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'input_tokens': 32768, 'first128_token_ids_sha256': hashlib.sha256(json.dumps(list(prefix)).encode()).hexdigest(),
                        'token_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    dataset = root / 'profile_32768.jsonl'
    dataset.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in frozen))
    config_records = {}
    for engine in ('vllm', 'sglang'):
        original = configs / f'{engine}_warmup1_32768.py'
        text = original.read_text().replace('warmup1_32768_1024', 'profile16_32768_1024')
        old = '/task/client_ds41_g503/datasets/warmup1_32768.jsonl'
        assert old in text
        text = text.replace(old, str(dataset))
        target = root / f'{engine}_profile32768.py'
        target.write_text(text)
        config_records[engine] = {'path': str(target), 'sha256': sha(target), 'source_config_sha256': sha(original)}
    result = {'nonce': nonce, 'dataset_path': str(dataset), 'dataset_sha256': sha(dataset),
              'source_warmup1_dataset_sha256': source_sha, 'dataset_manifest_sha256': sha(manifest_path),
              'tokenizer_sha256': sha(tokenizer_path), 'CPU_tokenizers_version': importlib.metadata.version('tokenizers'),
              'records': records, 'configurations': config_records, 'original_reserved_prefixes_checked': 448,
              'new_prefixes_unique': 16, 'generation_requests_submitted': False,
              'scope': 'Fresh profile-only16/C16/L32768/O1024. Exact frozen bytes for bothframeworks; all5 existing independentwarmups remainunused bythispreparation. Not a scoredbenchmark or actualprofile.',
              'supersedes_unsubmitted_plan': 'profile_input_manifest_g736.json pointed tooldwarmup1; retainit asoriginalplan only. Newinputsavoidwarmingtheactualprofilecohort.'}
    (root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'model_requests_submitted': False, 'dataset_sha256': result['dataset_sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.manifest, args.configs)
