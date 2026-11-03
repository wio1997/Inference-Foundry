"""Pin the specific compressed-cache readers relevant to delayed publication."""
from pathlib import Path
import hashlib,zipfile,json
from finalize_origin_coverage_v9 import O,W,read,write

def receipt(p):
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    assert not(O/'KV-publication-shared-reader-topology-v87.json').exists()
    source=read(O/'KV-publication-source-contract-v87.json')
    member='vllm-ascend/vllm_ascend/models/deepseek_v41/model.py';p=W/'frozen-source-snapshot'/member
    with zipfile.ZipFile(source['original_archive']['path'])as z:assert p.read_bytes()==z.read(member)
    lines=p.read_text().splitlines()
    assert receipt(p)['sha256']=='101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'
    source['source_files'].append(dict(**receipt(p),archive_member=member,
      clauses=[dict(first=a,last=b,source=lines[a-1:b])for a,b in[(506,551),(742,754)]]))
    config=read(O/'attention-indexer-native-numerical-contract-v60.json')['model_config'];p=Path(config['path'])
    assert receipt(p)['sha256']==config['sha256'];text=read(p)['text_config']
    chosen={k:text[k]for k in ['num_hidden_layers','compress_ratios','kv_source_layer_ids','index_source_layer_ids','candidate_source_layer_id']}
    source['source_files'].append(dict(**receipt(p),prior_binding='attention-indexer-native-numerical-contract-v60.json',selected_json_fields=chosen))
    assert chosen['num_hidden_layers']==40 and chosen['compress_ratios'][39]==1
    kv=max(s for s in chosen['kv_source_layer_ids']if s<=39)
    index=max(s for s in chosen['index_source_layer_ids']if s<=39)
    assert(kv,index)==(20,36)
    source['operator_facts']+=['Pinned source/config give final layer39 ratio1,compressed-KV and index-key/scale source20,and latest query/index selection source36. Absence of a compressor in the35node layer39 schema does NOT imply no compressed current-data reader.',
      'For layer39 index_k_source_prefix is formed from KVsource20,not from selection source36. Delaying source20 longKV or index-key publication must still serve intervening/current consumers before final acceptance.']
    write(O/'KV-publication-source-contract-v87.json',source)
    write(O/'KV-publication-shared-reader-topology-v87.json',dict(input_receipts=[receipt(Path(r['path']))for r in source['source_files'][-2:]],
      target_layers=40,final_layer=39,final_compress_ratio=1,compressed_KV_source=20,index_key_scale_source=20,
      latest_index_query_selection_source=36,source20_longKV_consumer_layers=list(range(20,40)),
      namespace_warning='index_source_layer36 identifies query/index selection production,not indexK cache-bank owner;the latter prefix uses KVsource20.',
      consequence='Current following layers still read source20 compressed data before final head/acceptance. SWA-only private pages cannot silently substitute for these current values or for coherent tuple-alias maps.',
      prior_topology_rules_reused_not_new_architecture_discovery=True,loaded_runtime_pointer_identity_proved=False,overall_complete=False))
    state=read(O/'KV-publication-state-plane-conditions-v87.json')
    state['current_final_layer_readers']=dict(layer=39,compressed_KV_source=20,index_key_scale_source=20,latest_index_query_selection_source=36,
      evidence='KV-publication-shared-reader-topology-v87.json')
    write(O/'KV-publication-state-plane-conditions-v87.json',state)
    print(dict(source_files=5,final_layer=39,compressed_KV_and_indexK_source=20,index_query_selection_source=36,all_readers_closed=False))

if __name__=='__main__':main()
