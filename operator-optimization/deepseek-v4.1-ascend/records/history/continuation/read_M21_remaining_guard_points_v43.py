"""Only profile-raised guard gaps; reuse prior point source rather than reread it."""
import ast
import hashlib
from datetime import datetime, timezone
from finalize_origin_coverage_v9 import N, O, W, read, write
from finalize_Engram_whole_frontier_v30 import packet


def main():
    assert not (O/'REVIEW-CHECKPOINT-v43.json').exists()
    questions = [
        dict(profile='v27 positive index-key/long-cache publications and v37 QLI/SMLA metadata; roles/optional length bank observers are not runtime attributes in profile.',
             question='Exactly which role/empty/filter/length guards select the indexer input, query quantization and shared sparse-index bank publications?'),
        dict(profile='v27 saved main/aux stream order plus v35 disjoint local-query/global-cache rows.',
             question='What input-equivalence/quant-method/communication guards and final cache-stream join constrain quantization sharing?'),
        dict(profile='v35 positive CP override fullcallers, cache-only work and AllToAll/output projection.',
             question='How are source-owner global metadata and reused index candidates selected, and which output padding branches already have direct proof?'),
    ]
    write(O/'PROFILE-CODE-QUESTION-GATE-v43.json', dict(
        status='PROFILE_FIRST_FINITE_36_GUARD_RECONCILIATION', previous_turn_classification='PROGRESS:v42fixedcase',
        questions=questions, allowed_new_source=[
            'CP:_ReplicatedCacheImpl._global_layer_metadata; AscendDSAV41CPImpl._select_sparse_indices',
            'DSA:pad_sparse_indices,_write_swa_cache_on_main_stream,_prepare_indexer_inputs,_should_quantize_indexer,_quantize_indexer_query,_select_sparse_indices',
            'DSA:multistream_preprocess493-495 only; rest of method reused from v16 exact excerpts'],
        followup_allowed_source=[
            'DSA/CP:_indexer_hidden_states exact2/3line row-slice methods because shared indices must match local query rows',
            'model.py:DeepseekV41SharedAttentionState.__init__/reset only to distinguish no-op reset from value initialization'],
        reuse_only=['F33-source166-v27','F33-index-consumer166-v27','M19-CP166-v35-v2','M19-owner166-v35-v2',
                    'profile-raised-source-contract-v16.json','M21-adjacent-guard-crosswalk-v42.json'],
        no_repository_wide_audit=True, no_model_device_service_health_work=True, whole_module_closed=False, overall_complete=False))
    source_base = W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/attention'
    specs = [
        ('dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',[
            ('','pad_sparse_indices'),('AscendDSAV41Impl','_write_swa_cache_on_main_stream'),
            ('AscendDSAV41Impl','_prepare_indexer_inputs'),('AscendDSAV41Impl','_should_quantize_indexer'),
            ('AscendDSAV41Impl','_quantize_indexer_query'),('AscendDSAV41Impl','_select_sparse_indices'),
            ('AscendDSAV41Impl','_indexer_hidden_states')]),
        ('context_parallel/dsa_v41_cp.py','039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e',[
            ('_ReplicatedCacheImpl','_global_layer_metadata'),('AscendDSAV41CPImpl','_select_sparse_indices'),
            ('AscendDSAV41CPImpl','_indexer_hidden_states')]),
        ('../models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[
            ('DeepseekV41SharedAttentionState','__init__'),('DeepseekV41SharedAttentionState','reset')]),
    ]
    files=[]
    for rel,sha,selectors in specs:
        p=(source_base/rel).resolve();b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
        txt=b.decode();lines=txt.splitlines();tree=ast.parse(txt);functions=[]
        for owner,name in selectors:
            scope=tree.body if not owner else next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==owner).body
            node=next(x for x in scope if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))and x.name==name)
            assert node.end_lineno-node.lineno+1<=60
            functions.append(dict(owner=owner,name=name,start=node.lineno,end=node.end_lineno,
                                  source=[f'{i+1}: {lines[i]}'for i in range(node.lineno-1,node.end_lineno)]))
        f=dict(path=str(p),sha256=sha,functions=functions)
        if rel=='dsa_v41.py':
            f['statements']=[dict(start=493,end=495,source=[f'{i+1}: {lines[i]}'for i in range(492,495)])]
        files.append(f)
    write(O/'M21-remaining-guard-point-source-v43.json',dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),files=files,
        no_model_or_torch_import=True,scope='12 exact small methods and3line tail; no new reads of previously reviewed cache/C1/C2/full multistream/output methods.'))
    reuse = {}
    for label in ['F33-source166-v27','F33-index-consumer166-v27','M19-CP166-v35-v2','M19-owner166-v35-v2']:
        p=packet(label,True)
        reuse[label]=dict(receipt=str(N/(label+'.json')),full_result=p['full_result'],sha256=p['result_sha256'],result=p['result'])
    write(O/'M21-reused-point-source-receipts-v43.json',dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),receipts=reuse,
        multistream_excerpt='profile-raised-source-contract-v16.json/excerpts:DSA440-492',
        no_remote_query=True,whole_module_closed=False,overall_complete=False))
    print('12 exact methods/3line tail extracted;4prior embedded source receipt hashes verified; no remote/model/device work.')


if __name__=='__main__':
    main()
