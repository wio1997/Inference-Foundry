#!/usr/bin/env python3
"""Compare retained four-rank reference and DeepInfer results, without quality thresholds."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    raw = Path(path).read_bytes()
    result = json.loads(raw)
    ranks = result['ranks']
    if len(ranks) != 4 or sorted(r['rank'] for r in ranks) != list(range(4)) or len({r['die'] for r in ranks}) != 4:
        raise ValueError(f'{path}: expected four distinct ranks and dies')
    if result['summary']['semantic_status'] != 'pass' or any(r['status'] != 'passed' for r in ranks):
        raise ValueError(f'{path}: unsuccessful execution cannot be compared as a completed response')
    for key in ('prompt_ids', 'generated_ids', 'config_sha256', 'tokenizer_eos_id', 'generation_cap'):
        if any(r[key] != ranks[0][key] for r in ranks[1:]):
            raise ValueError(f'{path}: ranks disagree on {key}')
    for key in ('stop_reason', 'finish_reason', 'stop_token_id', 'cap_reached', 'stop_ids'):
        if any(r.get(key) != ranks[0].get(key) for r in ranks[1:]):
            raise ValueError(f'{path}: ranks disagree on {key}')
    return ranks[0], hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('deepinfer')
    args = parser.parse_args()
    ref, ref_sha = load(args.reference)
    impl, impl_sha = load(args.deepinfer)
    if (ref['index_sha256'] != impl['weight_index_sha256'] or
            ref['config_sha256'] != impl['config_sha256'] or
            ref['prompt_ids'] != impl['prompt_ids'] or
            ref['tokenizer_eos_id'] != impl['tokenizer_eos_id'] or
            ref['generation_cap'] != impl['generation_cap'] or
            ref['stop_ids'] != impl['stop_ids']):
        raise ValueError('reference and DeepInfer checkpoint, prompt or generation contract differs')
    rid, did = ref['generated_ids'], impl['generated_ids']
    first = next((i for i, (a, b) in enumerate(zip(rid, did)) if a != b), None)
    if first is None and len(rid) != len(did):
        first = min(len(rid), len(did))
    stop = ref.get('stop_reason')
    out = {
        'reference_sha256': ref_sha, 'deepinfer_sha256': impl_sha,
        'checkpoint_index_sha256': ref['index_sha256'], 'config_sha256': ref['config_sha256'],
        'prompt_tokens': len(ref['prompt_ids']), 'generation_cap': ref['generation_cap'],
        'reference_token_count': len(rid), 'deepinfer_token_count': len(did),
        'exact_token_match': rid == did, 'first_difference_zero_based': first,
        'reference_stop_id': stop, 'deepinfer_stop_id': impl.get('stop_token_id'),
        'stop_ids_match': stop == impl.get('stop_token_id'),
        'tokenizer_eos_id': ref['tokenizer_eos_id'],
        'reference_cap_reached': ref['cap_reached'], 'deepinfer_cap_reached': impl['cap_reached'],
        'reference_finish_reason': ref['finish_reason'],
        'reference_text': ref['generated_text'], 'deepinfer_final_text': impl['final_text'],
        'scope': 'single request; no whole-model accuracy or numerical-threshold conclusion',
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
