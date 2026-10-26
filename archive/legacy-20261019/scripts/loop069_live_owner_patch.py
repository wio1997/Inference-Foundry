#!/usr/bin/env python3
"""SHA-guarded reversible layer2 owner16 live Target integration diagnostic."""
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
        from scripts.loop069_live_owner import enabled, OwnerContext
        owner_layer = enabled(
            self, layer_name=layer_name, hidden_states_cache=hidden_states_cache,
            hidden_states_local=hidden_states_local, has_prefill=has_prefill,
        )
        owner_ctx = OwnerContext(
            self, layer_name=layer_name, full_x=hidden_states_cache,
            attn_metadata=attn_metadata,
        ) if owner_layer else None
        if owner_layer:
            hidden_states_cache = owner_ctx.x
        kv = self.wkv(hidden_states_cache)
'''), (
            '''            cos[: kv.shape[0]],
            sin[: kv.shape[0]],
''',
            '''            (owner_ctx.cos if owner_layer else cos)[: kv.shape[0]],
            (owner_ctx.sin if owner_layer else sin)[: kv.shape[0]],
'''), (
            '''        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_metadata.req_metadata.slot_mapping)
''',
            '''        DeviceOperator.dsa_kv_compress_scatter(
            swa_kv_cache, kv,
            owner_ctx.swa_slots if owner_layer else swa_metadata.req_metadata.slot_mapping,
        )
'''), (
            '''                self._update_indexer_cache(
                    x=hidden_states_cache,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                )
''',
            '''                if owner_layer:
                    owner_ctx.indexer_update(kv_cache)
                else:
                    self._update_indexer_cache(
                        x=hidden_states_cache,
                        kv_cache=kv_cache,
                        attn_metadata=attn_metadata,
                        actual_seq_lengths_query=actual_seq_lengths_query,
                    )
'''), (
            '''            compress_cos, compress_sin, compress_slot_mapping = self._compute_compressor_metadata(
                compressor_attn_metadata.req_metadata,
            )
            compressed_kv = torch.ops._C_ascend.compressor(
''',
            '''            if owner_layer:
                owner_qsl, owner_start, owner_state_bt, compress_cos, compress_sin, compress_slot_mapping = owner_ctx.main_metadata()
            else:
                compress_cos, compress_sin, compress_slot_mapping = self._compute_compressor_metadata(
                    compressor_attn_metadata.req_metadata,
                )
            compressed_kv = torch.ops._C_ascend.compressor(
'''), (
            '''                state_block_table=compressor_kv_state_metadata.req_metadata.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=req_metadata.start_pos,
''',
            '''                state_block_table=owner_state_bt if owner_layer else compressor_kv_state_metadata.req_metadata.block_table,
                cu_seqlens=owner_qsl if owner_layer else actual_seq_lengths_query,
                seqused=None,
                start_pos=owner_start if owner_layer else req_metadata.start_pos,
'''), (
            '''            DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
''',
            '''            DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
            if owner_layer:
                from scripts.loop069_live_owner import record_producer_complete
                record_producer_complete(owner_ctx)
''')]},
    'handoff': {
        'path': ROOT / 'bootstrap/vllm_extreme_handoff.py',
        'base_sha': 'f644bd14ac1cb9c8365ba2464abbff989d7f2c4c2d58279a8e18a5bce918716f',
        'replacements': [(
            '''    target_handoff = DirectTargetHandoff(inputs.target)
''',
            '''    from scripts.loop069_live_owner import verify_handoff
    verify_handoff(inputs, state, config)
    target_handoff = DirectTargetHandoff(inputs.target)
''')],
    },
    'target_handoff': {
        'path': ROOT / 'bootstrap/vllm_target_handoff.py',
        'base_sha': '2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5',
        'replacements': [(
            '''        num_tokens = input_ids.shape[0]
        update_cos_sin(positions)
''',
            '''        num_tokens = input_ids.shape[0]
        from scripts.loop069_live_owner import verify_replay_dispatch
        verify_replay_dispatch(self, num_tokens)
        update_cos_sin(positions)
''')],
    },
    'runner': {
        'path': Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'),
        'base_sha': '004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba',
        'replacements': [(
            '''                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                if logger.isEnabledFor(logging.DEBUG):
''',
            '''                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )
                from scripts.loop069_live_owner import verify_generic_graph_dispatch
                verify_generic_graph_dispatch(
                    batch_desc, cudagraph_mode, num_reqs,
                    num_tokens_unpadded, num_scheduled_tokens_np,
                )

                if logger.isEnabledFor(logging.DEBUG):
''')],
    },
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
        errors = []
        for name, spec in SOURCES.items():
            backup = Path(info[name]['backup']).read_bytes()
            actual = sha(spec['path'].read_bytes())
            if sha(backup) != spec['base_sha']:
                errors.append(f'{name} backup SHA mismatch')
            elif actual == spec['base_sha']:
                print(f'already restored {name} {actual}')
            elif actual == info[name]['patched_sha']:
                spec['path'].write_bytes(backup)
                print(f'restored {name} {sha(backup)}')
            else:
                errors.append(f'{name} source SHA mismatch {actual}')
        if errors:
            raise SystemExit('; '.join(errors))


if __name__ == '__main__':
    main()
