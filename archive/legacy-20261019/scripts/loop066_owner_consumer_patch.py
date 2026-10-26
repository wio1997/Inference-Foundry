#!/usr/bin/env python3
"""Reversible two-source hooks for private layer2 owner state/scatter/QLI gate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
SOURCES={
 'dsa_cp':{
  'path':Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py'),
  'base_sha':'27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e',
  'replacements':[
   ('''            if self.compress_ratio == 4:
                self._update_indexer_cache(
''',
    '''            if self.compress_ratio == 4:
                if not has_prefill and layer_name == "model.layers.2.self_attn.attn":
                    from scripts.loop066_owner_consumer_fixture import prepare_owner_consumer
                    prepare_owner_consumer(
                        self, x=hidden_states_cache, kv_cache=kv_cache,
                        attn_metadata=attn_metadata,
                        qsl=actual_seq_lengths_query, layer_name=layer_name,
                    )
                self._update_indexer_cache(
'''),
   ('''        topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
''',
    '''        if __import__("os").getenv("EXTREME_OWNER_CONSUMER_FIXTURE_DIR"):
            from scripts.loop066_owner_consumer_fixture import complete_owner_consumer
            complete_owner_consumer(
                self, q=q, q_scale=q_scale, weights=weights,
                qli_metadata=qli_metadata, block_table=block_table,
                local_qsl=actual_seq_lengths_query,
                local_ksl=actual_seq_lengths_key,
            )
        topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
'''),
  ],
 },
 'handoff':{
  'path':ROOT/'bootstrap/vllm_target_handoff.py',
  'base_sha':'2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5',
  'replacements':[
   ('''            context = get_forward_context()
            if self.graph_update is not None and self.graph_update_before:
''',
    '''            context = get_forward_context()
            if __import__("os").getenv("EXTREME_OWNER_CONSUMER_FIXTURE_DIR"):
                context.additional_kwargs["extreme_owner_consumer_runtime"] = True
            if self.graph_update is not None and self.graph_update_before:
'''),
  ],
 },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('preview','install','restore'))
    ap.add_argument('--record')
    args=ap.parse_args()
    if args.action in ('preview','install'):
        changed={};original={}
        for name,spec in SOURCES.items():
            raw=spec['path'].read_bytes()
            if sha(raw)!=spec['base_sha']:
                raise SystemExit(f'{name} source SHA changed')
            decoded=raw.decode()
            for old,new in spec['replacements']:
                if decoded.count(old)!=1:
                    raise SystemExit(f'{name} anchor count !=1')
                decoded=decoded.replace(old,new,1)
            target=decoded.encode()
            compile(target,str(spec['path']),'exec')
            original[name]=raw;changed[name]=target
        info={name:{'source':str(spec['path']),'base_sha':spec['base_sha'],
                    'patched_sha':sha(changed[name])} for name,spec in SOURCES.items()}
        print(json.dumps(info))
        if args.action=='preview':
            return
        if not args.record:
            raise SystemExit('--record required')
        record=Path(args.record);record.parent.mkdir(parents=True,exist_ok=True)
        for name in SOURCES:
            backup=record.with_name(record.stem+f'.{name}.original.py')
            backup.write_bytes(original[name]);info[name]['backup']=str(backup)
        record.write_text(json.dumps(info,indent=2))
        for name,spec in SOURCES.items():
            spec['path'].write_bytes(changed[name])
    else:
        if not args.record:
            raise SystemExit('--record required')
        info=json.loads(Path(args.record).read_text())
        for name,spec in SOURCES.items():
            backup=Path(info[name]['backup']).read_bytes()
            if sha(backup)!=spec['base_sha']:
                raise SystemExit(f'{name} backup SHA changed')
            current=spec['path'].read_bytes()
            if sha(current)!=info[name]['patched_sha']:
                raise SystemExit(f'{name} patched source SHA changed, refusing restore')
            spec['path'].write_bytes(backup)
            print(f'restored {name} {sha(backup)}')


if __name__=='__main__':
    main()
