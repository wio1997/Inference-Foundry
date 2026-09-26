#!/usr/bin/env python3
"""SHA-guarded reversible hook for private layer2 full-producer value gate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
SOURCES = {
    'dsa_cp': {
        'path': Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py'),
        'base_sha': '27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e',
        'replacements': [(
            '''        hidden_states_cache = hidden_states[: common_attn_metadata.num_actual_tokens]
        kv = self.wkv(hidden_states_cache)
''',
            '''        hidden_states_cache = hidden_states[: common_attn_metadata.num_actual_tokens]
        if __import__("os").getenv("EXTREME_FULL_PRODUCER_DIR"):
            from scripts.loop068_full_producer_fixture import run_private_full_producer
            run_private_full_producer(
                self, x_full=hidden_states_cache, x_local=hidden_states_local,
                q=q, qr=qr_local, q_per_token_scale=qr_pertoken_scale_local,
                kv_cache=kv_cache, attn_metadata=attn_metadata, layer_name=layer_name,
            )
        kv = self.wkv(hidden_states_cache)
''')]},
    'handoff': {
        'path': ROOT / 'bootstrap/vllm_target_handoff.py',
        'base_sha': '2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5',
        'replacements': [(
            '''            context = get_forward_context()
            if self.graph_update is not None and self.graph_update_before:
''',
            '''            context = get_forward_context()
            if __import__("os").getenv("EXTREME_FULL_PRODUCER_DIR"):
                context.additional_kwargs["extreme_full_producer_runtime"] = True
            if self.graph_update is not None and self.graph_update_before:
''')]},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('preview', 'install', 'restore'))
    ap.add_argument('--record')
    args = ap.parse_args()
    if args.action in ('preview', 'install'):
        originals, changed = {}, {}
        for name, spec in SOURCES.items():
            raw = spec['path'].read_bytes()
            if sha(raw) != spec['base_sha']:
                raise SystemExit(f'{name} source SHA changed')
            code = raw.decode()
            for old, new in spec['replacements']:
                if code.count(old) != 1:
                    raise SystemExit(f'{name} hook anchor count != 1')
                code = code.replace(old, new, 1)
            patched = code.encode()
            compile(patched, str(spec['path']), 'exec')
            originals[name], changed[name] = raw, patched
        info = {name: {'source': str(spec['path']), 'base_sha': spec['base_sha'],
                       'patched_sha': sha(changed[name])}
                for name, spec in SOURCES.items()}
        print(json.dumps(info))
        if args.action == 'preview':
            return
        if not args.record:
            raise SystemExit('--record required')
        record = Path(args.record)
        record.parent.mkdir(parents=True, exist_ok=True)
        for name in SOURCES:
            backup = record.with_name(record.stem + f'.{name}.original.py')
            backup.write_bytes(originals[name])
            info[name]['backup'] = str(backup)
        record.write_text(json.dumps(info, indent=2) + '\n')
        for name, spec in SOURCES.items():
            spec['path'].write_bytes(changed[name])
    else:
        if not args.record:
            raise SystemExit('--record required')
        info = json.loads(Path(args.record).read_text())
        for name, spec in SOURCES.items():
            backup = Path(info[name]['backup']).read_bytes()
            if sha(backup) != spec['base_sha'] or sha(spec['path'].read_bytes()) != info[name]['patched_sha']:
                raise SystemExit(f'{name} restore SHA guard failed')
            spec['path'].write_bytes(backup)
            print(f'restored {name} {sha(backup)}')


if __name__ == '__main__':
    main()
