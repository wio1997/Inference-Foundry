"""Freeze fresh 32K inputs for the next shared SGLang and vLLM campaign."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re


TOKENIZER_SHA = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
CATALOG_SHA = '2491233b84c38ed476d885d8379d464af728b95ea810a4428e2ad65ca076513a'
PROFILE_SHA = '308d46f5ae05d7d30b8af9b4711b24582fe4b8a0c6fdc76f32473720f7acf4f4'
PHASES = {**{f'warmup{i}': 16 for i in range(1, 6)}, 'formal': 32}
LENGTH = 32768
OUTPUT = 1024
PREFIX_TOKENS = 128


def digest(raw):
    return sha256(raw).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f'Frozen input digest changed: {path.name}')
    return raw


def dataset_rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line]


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def first_prefix(tokenizer, text):
    prefix = tuple(encode(tokenizer, text[:5000])[:PREFIX_TOKENS])
    assert len(prefix) == PREFIX_TOKENS
    return prefix


def prepare(tokenizer_path, catalog, previous_profile, output, nonce):
    from tokenizers import Tokenizer

    assert re.fullmatch(r'[0-9a-f]{32}', nonce)
    assert output.is_absolute() and output.parent.is_dir() and not output.exists()
    checked(tokenizer_path, TOKENIZER_SHA)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    old = json.loads(checked(catalog, CATALOG_SHA))
    profile = json.loads(checked(previous_profile, PROFILE_SHA))
    assert len(old['profiles']) == 24 and old['tokenizer_sha256'] == profile['tokenizer_sha256'] == TOKENIZER_SHA
    assert profile['dataset_manifest_sha256'] == CATALOG_SHA

    reserved = set()
    retained = {}
    for row in old['profiles']:
        dataset_root = (catalog.parent / 'datasets').resolve(strict=True)
        path = (dataset_root / Path(row['path']).name).resolve(strict=True)
        assert path.is_relative_to(dataset_root)
        rows = dataset_rows(checked(path, row['sha256']))
        assert len(rows) == row['requests']
        for original in rows:
            prefix = first_prefix(tokenizer, original['question'])
            assert prefix not in reserved
            reserved.add(prefix)
        if row['input_tokens'] == LENGTH:
            phase = row['phase']
            assert phase not in retained and len(rows) == PHASES[phase]
            retained[phase] = rows, row['sha256']
    assert len(reserved) == 448 and set(retained) == set(PHASES)
    frozen_profile = (previous_profile.parent / Path(profile['dataset_path']).name).resolve(strict=True)
    assert frozen_profile.is_relative_to(previous_profile.parent.resolve())
    old_profile_rows = dataset_rows(checked(frozen_profile, profile['dataset_sha256']))
    assert len(old_profile_rows) == 16
    for index, row in enumerate(old_profile_rows):
        prefix = first_prefix(tokenizer, row['question'])
        assert prefix not in reserved
        expected = profile['records'][index]['first128_token_ids_sha256']
        assert digest(json.dumps(list(prefix)).encode()) == expected
        reserved.add(prefix)
    assert len(reserved) == 464

    staging = output.with_name(f'.{output.name}.{os.getpid()}.staging')
    staging.mkdir(mode=0o700)
    descriptions = []
    for phase, count in PHASES.items():
        body_rows, source_sha = retained[phase]
        name = f'{phase}_{LENGTH}.jsonl'
        records = []
        with (staging / name).open('x', encoding='utf-8') as stream:
            for index, old_row in enumerate(body_rows):
                assert old_row['max_out_len'] == OUTPUT
                text = f'DS41-RETRY32-{nonce}-{phase}-{index} ' + old_row['question']
                ids = encode(tokenizer, text)[:LENGTH]
                exact = tokenizer.decode(ids, skip_special_tokens=False)
                assert len(ids) == LENGTH and encode(tokenizer, exact) == ids
                prefix = tuple(ids[:PREFIX_TOKENS])
                assert prefix not in reserved and first_prefix(tokenizer, exact) == prefix
                reserved.add(prefix)
                row = {'question': exact, 'answer': '', 'max_out_len': OUTPUT, 'input_tokens': LENGTH}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                records.append({
                    'data_id': index, 'prompt_sha256': digest(exact.encode()),
                    'input_tokens': LENGTH,
                    'first128_token_ids_sha256': digest(json.dumps(list(prefix)).encode()),
                    'token_ids_sha256': digest(json.dumps(ids).encode()),
                })
        descriptions.append({'phase': phase, 'path': name, 'requests': count, 'input_tokens': LENGTH,
                             'sha256': digest((staging / name).read_bytes()),
                             'source_prompt_body_dataset_sha256': source_sha, 'records': records})
    assert len(reserved) == 576
    manifest = {
        'nonce': nonce, 'tokenizer_sha256': TOKENIZER_SHA,
        'tokenizers_package_version': importlib.metadata.version('tokenizers'),
        'original_catalog_sha256': CATALOG_SHA, 'previous_profile_manifest_sha256': PROFILE_SHA,
        'prior_prefixes_checked': 464, 'new_unique_prefixes': 112,
        'required_request_output_tokens': OUTPUT, 'client_concurrency': 16,
        'frameworks': ['sglang', 'vllm'], 'profiles': descriptions,
        'same_input_replay_rule': 'Use the exact new formal dataset bytes for the first replay.',
        'submitted_model_requests': 0,
    }
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    assert not output.exists()
    staging.rename(output)
    return {'new_unique_prefixes': 112, 'manifest_sha256': digest((output / 'manifest.json').read_bytes()),
            'formal_dataset_sha256': descriptions[-1]['sha256'], 'submitted_model_requests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tokenizer', 'catalog', 'previous-profile', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nonce', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.tokenizer, args.catalog, args.previous_profile, args.output, args.nonce)))
