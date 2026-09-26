#!/usr/bin/env python3
"""Reversible opt-in DSA CP c4 indexer/main-compressor scheduling probe."""
import argparse, hashlib, json, textwrap
from pathlib import Path
SOURCE=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py')
BASE_SHA='27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e'
BEGIN='''        compress_topk_idxs = None
        if self.compress_ratio > 1:
'''
END='''
        notify_kv_cache_written(layer_name)
'''
MAIN_BEGIN='''            coff = 2 if self.compressor_overlap else 1
'''
MAIN_END='''            DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
'''
INDEXER_CALL='''                compress_topk_idxs = self._indexer_select_topk(
                    x=hidden_states_local,
                    qr=qr_local,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    cos=local_cos,
                    sin=local_sin,
                    actual_seq_lengths_query=local_seq_lengths_query,
                    actual_seq_lengths_key=local_seq_lengths_key,
                    qr_pertoken_scale=qr_pertoken_scale_local,
                )
'''

def sha(data):return hashlib.sha256(data).hexdigest()

def render(original):
    if original.count(BEGIN)!=1 or original.count(END)<1:
        raise ValueError('CP forward anchors not unique')
    a=original.index(BEGIN);b=original.index(END,a)
    old=original[a:b]
    if old.count(MAIN_BEGIN)!=1 or old.count(MAIN_END)!=1 or old.count(INDEXER_CALL)!=1:
        raise ValueError('CP compressor/indexer anchors not unique')
    ms=old.index(MAIN_BEGIN);me=old.index(MAIN_END,ms)+len(MAIN_END)
    main=old[ms:me]
    # Keep the original operator calls byte-for-byte; only change their launch order.
    helper='\n'.join('                '+line if line else '' for line in textwrap.dedent(main).rstrip('\n').split('\n'))+'\n'
    helper+='                return compressed_kv, compress_cos, compress_sin, compress_slot_mapping\n'
    indexer='\n'.join('                    '+line if line else '' for line in textwrap.dedent(INDEXER_CALL).rstrip('\n').split('\n'))+'\n'
    prefix=old[len(BEGIN):old.index(INDEXER_CALL)]
    if 'self._update_indexer_cache(' not in prefix:
        raise ValueError('missing original indexer-cache update')
    # prefix starts inside if-compress_ratio>1 and stops after update_indexer_cache.
    new=(BEGIN+
        '            def _extreme_main_compressor_scatter():\n'+helper+'\n'+prefix+
        '                from vllm.forward_context import get_forward_context\n'
        '                _extreme_mode = __import__("os").getenv("EXTREME_CP_FORK_MODE", "off")\n'
        '                _extreme_scope = __import__("os").getenv("EXTREME_CP_FORK_SCOPE", "one")\n'
        '                _extreme_layer_parts = layer_name.split(".")\n'
        '                _extreme_target_layer = (len(_extreme_layer_parts) >= 4\n'
        '                    and _extreme_layer_parts[0:2] == ["model", "layers"]\n'
        '                    and _extreme_layer_parts[2].isdigit()\n'
        '                    and int(_extreme_layer_parts[2]) < 43)\n'
        '                _extreme_fork = (not has_prefill and common_attn_metadata.num_actual_tokens == 96\n'
        '                                 and not getattr(get_forward_context(), "is_draft_model", False)\n'
        '                                 and _extreme_target_layer\n'
        '                                 and _extreme_mode in ("immediate", "overlap")\n'
        '                                 and (_extreme_scope == "all" or\n'
        '                                      (_extreme_scope == "one" and _extreme_layer_parts[2] == "2")))\n'
        '                if _extreme_fork:\n'
        '                    if not getattr(self, "_extreme_cp_fork_logged", False):\n'
        '                        print(f"EXTREME_CP_FORK rank={self.tp_rank} layer={layer_name} mode={_extreme_mode}", flush=True)\n'
        '                        self._extreme_cp_fork_logged = True\n'
        '                    if (__import__("os").getenv("EXTREME_CP_FORK_AUDIT") == "1"\n'
        '                            and not getattr(self, "_extreme_cp_alias_checked", False)):\n'
        '                        from runtime.storage_intervals import audit_disjoint\n'
        '                        _ix_state, _ix_key, _ix_scale, _ix_full = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)\n'
        '                        _ix_live = {"indexer.state": _ix_state, "indexer.key": _ix_key,\n'
        '                                    "indexer.scale": _ix_scale}\n'
        '                        if _ix_full is not None:\n'
        '                            _ix_live["indexer.full"] = _ix_full\n'
        '                        _alias = audit_disjoint({"main.state": state_cache, "main.kv": compress_kv_cache}, _ix_live)\n'
        '                        if not _alias["pass_"]:\n'
        '                            raise RuntimeError("CP fork KV storage alias: " + str(_alias["possible_aliases"]))\n'
        '                        print(f"EXTREME_CP_FORK_ALIAS rank={self.tp_rank} layer={layer_name} pass=1", flush=True)\n'
        '                        self._extreme_cp_alias_checked = True\n'
        '                    from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream\n'
        '                    from vllm_ascend.utils import npu_stream_switch\n'
        '                    _extreme_main_stream = torch.npu.current_stream()\n'
        '                    _extreme_aux_stream = dsv4_dsa_overlap_stream()\n'
        '                    _extreme_aux_stream.wait_stream(_extreme_main_stream)\n'
        '                    with npu_stream_switch(_extreme_aux_stream, enabled=True):\n'
        '                        _extreme_main_keepalive = _extreme_main_compressor_scatter()\n'
        '                    if _extreme_mode == "immediate":\n'
        '                        _extreme_main_stream.wait_stream(_extreme_aux_stream)\n'+
        '                else:\n'+indexer+
        '                    _extreme_main_compressor_scatter()\n'
        '                if _extreme_fork:\n'+indexer+
        '                    if _extreme_mode == "overlap":\n'
        '                        _extreme_main_stream.wait_stream(_extreme_aux_stream)\n'
        '            else:\n'
        '                _extreme_main_compressor_scatter()\n')
    # In the fork path the indexer-cache update remains before the split.
    changed=original[:a]+new+original[b:]
    compile(changed,str(SOURCE),'exec')
    return changed

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preview','install','restore']);p.add_argument('--record');args=p.parse_args()
    raw=SOURCE.read_bytes()
    if args.action in ('preview','install'):
        if sha(raw)!=BASE_SHA:raise SystemExit('borrowed CP source SHA changed; refusing')
        new=render(raw.decode())
        print(json.dumps({'base_sha':BASE_SHA,'patched_sha':sha(new.encode()),'base_bytes':len(raw),'patched_bytes':len(new.encode())}))
        if args.action=='preview':return
        if not args.record:raise SystemExit('--record required')
        record=Path(args.record);record.parent.mkdir(parents=True,exist_ok=True)
        backup=record.with_suffix('.original.py');backup.write_bytes(raw)
        SOURCE.write_text(new)
        record.write_text(json.dumps({'source':str(SOURCE),'backup':str(backup),'base_sha':BASE_SHA,'patched_sha':sha(new.encode())},indent=2))
    else:
        if not args.record:raise SystemExit('--record required')
        info=json.loads(Path(args.record).read_text());backup=Path(info['backup']).read_bytes()
        if sha(backup)!=BASE_SHA:raise SystemExit('backup mismatch')
        if sha(raw)==BASE_SHA:print('already restored');return
        if sha(raw)!=info['patched_sha']:raise SystemExit('patched source changed; refusing overwrite')
        SOURCE.write_bytes(backup);print('restored',sha(SOURCE.read_bytes()))
if __name__=='__main__':main()
