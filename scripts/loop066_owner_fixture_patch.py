#!/usr/bin/env python3
"""Reversible two-source hook for real Runtime-only layer2 private fixture."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
SOURCES={
 'dsa_cp':{
  'path':Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py'),
  'base_sha':'27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e',
  'old':'''            if self.compress_ratio == 4:
                self._update_indexer_cache(
''',
  'new':'''            if self.compress_ratio == 4:
                if not has_prefill and layer_name == "model.layers.2.self_attn.attn":
                    from scripts.loop066_owner_fixture import run_owner_fixture
                    run_owner_fixture(
                        self, x=hidden_states_cache, kv_cache=kv_cache,
                        attn_metadata=attn_metadata,
                        qsl=actual_seq_lengths_query, layer_name=layer_name,
                    )
                self._update_indexer_cache(
''',
 },
 'handoff':{
  'path':ROOT/'bootstrap/vllm_target_handoff.py',
  'base_sha':'2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5',
  'old':'''            context = get_forward_context()
            if self.graph_update is not None and self.graph_update_before:
''',
  'new':'''            context = get_forward_context()
            if __import__("os").getenv("EXTREME_OWNER_FIXTURE_DIR"):
                context.additional_kwargs["extreme_owner_fixture_runtime"] = True
            if self.graph_update is not None and self.graph_update_before:
''',
 },
}

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('preview','install','restore'));ap.add_argument('--record');a=ap.parse_args()
    if a.action in ('preview','install'):
        changed={};original={}
        for name,spec in SOURCES.items():
            raw=spec['path'].read_bytes()
            if sha(raw)!=spec['base_sha']:raise SystemExit(f'{name} source SHA changed')
            decoded=raw.decode()
            if decoded.count(spec['old'])!=1:raise SystemExit(f'{name} anchor count !=1')
            target=decoded.replace(spec['old'],spec['new'],1).encode()
            compile(target,str(spec['path']),'exec')
            original[name]=raw;changed[name]=target
        info={name:{'source':str(spec['path']),'base_sha':spec['base_sha'],
                    'patched_sha':sha(changed[name])} for name,spec in SOURCES.items()}
        print(json.dumps(info))
        if a.action=='preview':return
        if not a.record:raise SystemExit('--record required')
        record=Path(a.record);record.parent.mkdir(parents=True,exist_ok=True)
        for name in SOURCES:
            backup=record.with_name(record.stem+f'.{name}.original.py')
            backup.write_bytes(original[name]);info[name]['backup']=str(backup)
        record.write_text(json.dumps(info,indent=2))
        for name,spec in SOURCES.items():spec['path'].write_bytes(changed[name])
    else:
        if not a.record:raise SystemExit('--record required')
        info=json.loads(Path(a.record).read_text())
        for name,spec in SOURCES.items():
            backup=Path(info[name]['backup']).read_bytes()
            if sha(backup)!=spec['base_sha']:raise SystemExit(f'{name} backup SHA mismatch')
            current=spec['path'].read_bytes()
            if sha(current)==spec['base_sha']:continue
            if sha(current)!=info[name]['patched_sha']:raise SystemExit(f'{name} source changed; refusing overwrite')
            spec['path'].write_bytes(backup)
            print('restored',name,sha(spec['path'].read_bytes()))

if __name__=='__main__':main()
