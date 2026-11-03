"""Analyze conditional KV publication graphs, current reads and accepted-prefix semantics."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,zipfile,math
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

def receipt(path):
    data=path.read_bytes();return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def main():
    assert not(O/'KV-publication-source-contract-v87.json').exists()
    prior_names=['C2-state-opportunity-contract-v24.json','C2-cache-object-and-allocation-contract-v25.json',
      'F31-rejection-count-state-remap-and-index-proof-v38.json','state-engram-operator-contract-v62.json',
      'ring-state-layout-and-lifetime-proof-v79.json','tail-shared-memory-budget-v85.json',
      'terminal-layer-native-frontier-v86.json','terminal-layer-demand-cut-v86.json',
      'candidate-relation-clauses-v86.json']
    write(O/'KV-publication-prior-evidence-v87.json',dict(input_receipts=[receipt(O/n)for n in prior_names],
      previously_established=['v24 C2 pool/read-before-update and norm/update fork; not new here',
        'v25 tuple alias/backing/global-ID allocation and independent state-observer limits',
        'v38 exact rejection count/selector,old-state remap and retained padded queries',
        'v83 SWA cache publication is not a private attention temporary',
        'v86 final output pruning does not delete full required KV/state inputs'],
      new_scope='Compare acceptance-delayed publication across retained-bank overlay, copied-bank overlay, existing-PA page COW, on-chip retention and recomputation domains; rederive bytes/memory/dependencies against direct producer-to-cache fusion.',
      new_all_reader_or_loaded_binary_proof=False,overall_complete=False))
    archive=read(O/'DSpark-tail-source-index-v85.json')['original_archive'];source=[]
    wanted={
      'vllm-ascend/vllm_ascend/attention/dsa_v41.py':[(227,242),(296,317),(327,385),(440,495),(613,682),(690,716)],
      'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py':[(189,211)]}
    with zipfile.ZipFile(archive['path'])as z:
        for member,ranges in wanted.items():
            path=W/'frozen-source-snapshot'/member;data=path.read_bytes();assert data==z.read(member)
            lines=data.decode().splitlines();source.append(dict(**receipt(path),archive_member=member,
              clauses=[dict(first=a,last=b,source=lines[a-1:b])for a,b in ranges]))
    assert source[0]['sha256']=='3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd'
    old=read(O/'attention-indexer-native-numerical-contract-v60.json')
    part=next(r for r in old['parts']if r['path'].endswith('sparse_flash_mla_torch_adpt.h'))
    path=Path(part['path']);assert receipt(path)['sha256']==part['sha256'];lines=path.read_text().splitlines()
    source.append(dict(**receipt(path),prior_binding='attention-indexer-native-numerical-contract-v60.json',
      clauses=[dict(first=148,last=min(179,len(lines)),source=lines[147:179])]))
    write(O/'KV-publication-source-contract-v87.json',dict(source_files=source,original_archive=archive,
      operator_facts=[
        'scatter_cache_sk preserves physical page stride,casts to cache dtype and skips builder[-1,-1] slots. Typed input capacity is not actual valid-slot traffic.',
        'Both normal and multistream preprocessing write SWA cache before attention; multistream join makes writes visible.',
        'Current SparseFlashMla ori_kv argument is that SWA cache. No independent original-current-token overlay bank is passed. cmp_residual_kv belongs compressed metadata and is not an interchangeable original-KV overlay.',
        'CP slices local Q while passing global hidden/cache metadata for KV. A local empty query is not permission to omit required KV writers.',
        'An external retained KV bank needs new reader/address-resolution semantics. A COW page allocated INSIDE the existing PA pool can instead use the existing cache pointer/block-table interface,with new ownership/publication/lifetime proof; do not assert every variant requires a new attention argument.',
        'All current verification queries must see their required candidate keys before acceptance is known. Delaying the existing scatter without another read path changes results.'],
      new_target_source_files_obtained=0,new_remote_source_read=False,current_loaded_native_binary_identity_proved=False,
      all_KV_or_external_readers_closed=False,broader_framework_audit=False,implementation=False,overall_complete=False))
    path=CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json'
    assignments=read(path)['execution_assignments'];bydb=defaultdict(list)
    for a in assignments:
        if a[2]=='GRAPH'and a[5]in['SWA_KV_PROJECTION','SWA_KV_OR_C1_PROJECTION']:bydb[a[0]].append(a)
    strict=[];ambiguous=[];groups=defaultdict(set);inputs=[receipt(path)]
    for db,ars in bydb.items():
        path=CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json';native=read(path);inputs.append(receipt(path))
        tasks={t['task_record_rowid']:t for t in native['TASK_records']};defs={d['globalTaskId']:d for d in native['COMPUTE_definitions']}
        for a in ars:
            t=tasks[a[1]];d=defs[t['globalTaskId']]
            row=dict(database_index=db,TASK_rowid=a[1],globalTaskId=t['globalTaskId'],graph_ordinal=a[8],block_ordinal=a[9],
              existing_role=a[5],existing_modules=a[3],input_shapes=d['inputShapes_text'],output_shapes=d['outputShapes_text'],
              input_types=d['inputDataTypes_text'],output_types=d['outputDataTypes_text'],root_bound=a[6]!='GRAPH:PHASE_UNBOUND')
            assert dimensions(row['input_shapes'])==[[24,5120],[512,5120]]and dimensions(row['output_shapes'])==[[24,512]]
            assert row['input_types']=='DT_BF16;DT_BF16'and row['output_types']=='DT_BF16'
            if a[5]=='SWA_KV_PROJECTION':strict.append(row);groups[(db,a[8])].add(a[9])
            else:ambiguous.append(row)
    assert len(strict)==3120 and len(ambiguous)==160 and len(groups)==80
    assert all(layers==set(range(40))-{20}for layers in groups.values())
    fronts=[]
    for g in read(O/'terminal-layer-native-frontier-v86.json')['graphs']:
        scatter=next(t for t in g['stages']if t['operator']=='ScatterNdUpdateSk')
        attn=next(t for t in g['stages']if t['operator']=='SparseFlashMla')
        assert dimensions(scatter['input_shapes'])[2]==[24,512]
        assert dimensions(attn['input_shapes'])[1][-1]==512
        fronts.append(dict(database_index=g['database_index'],graph_ordinal=g['graph_ordinal'],root_bound=g['root_bound'],
          layer=39,physical_value_rows=24,width=512,cache_scatter_TASK=scatter['TASK_rowid'],
          attention_TASK=attn['TASK_rowid'],scatter_dtype=scatter['input_types'],attention_dtype=attn['input_types'],
          actual_valid_slots_duplicate_order_pointer_alias_or_acceptance_values_attested=False))
    write(O/'KV-publication-native-scope-v87.json',dict(strict_graph_SWA_projection_rows=strict,
      existing_layer20_SWA_C1_ambiguous_projection_rows=ambiguous,strict_projection_executions=3120,
      ambiguous_projection_executions_retained=160,graphs=80,strict_role_layers_per_graph=39,
      layer20_ambiguity_is_not_missing_SWA_or_zero_work=True,terminal_scatter_attention_frontiers=fronts,
      root_bound_frontiers=64,phase_unbound_frontiers=16,input_receipts=inputs,
      source_schema_not_native_tensor_edge_or_critical_path_proof=True,new_regimes_or_executions_promoted=0,overall_complete=False))
    cases=[dict(q=10,d=0,c=0),dict(q=8,d=5,c=2),dict(q=6,d=5,c=2),dict(q=6,d=5,c=6)]
    for c in cases:
        c['baseline_rejected']=c['d']+1-c['c']if c['d']>0 else 0
        c['conditional_retained_input_prefix_rows']=c['q']-c['baseline_rejected']
    write(O/'KV-publication-prefix-and-dependency-proof-v87.json',dict(prefix_cases=cases,
      count_origin='Reuse exact v38 rejection formula: r=d+1-c ifd>0 else0. Only IF retained current-input publication is this baseline prefix,commit a=q-r. This does not prove future-cache required rows from count alone.',
      source_conditions=['Same request/order/QSL/input-token/position/group/cache-slot and accepted-prefix epoch.',
        'Baseline valid sampler prefix/range/error domain;do not replace exact count by rightmost valid index or clamp its predicate.',
        'No-draft or partial-prefill rows can require all scheduled KV even when sampled-output count is0.',
        'When q>d+1,additional prefill rows remain: a=q-d-1+c,not simply c.',
        'Physical padded rows and invalid[-1,-1] slots have independent masks;Ncapacity is notRvalid.',
        'Actual future-reader union,rollback and external/transfer/alias observers may retain more than a nominal accepted prefix.'],
      shifted_identity_witness=dict(current_input_ids=[11,22,33,44,55,66],sampled_output_ids=[22,99],
        d=5,c=2,retained_input_ids=[11,22],incorrect_output_id_substitution=[22,99],
        scope='Semantic seed+draft versus accepted+replacement shift example,not frozen runtime IDs.'),
      stale_current_read_witness=dict(query=1,old_cached_key_value=0,new_current_key_value=1,
        single_current_key_attention_expected=1,omit_scatter_without_overlay_result=0,
        scope='Finite scalar attention with one required current key,no sink;embed in a larger latent row. Not native execution or actual weights.'),
      baseline_edges=['KVready->cachewrite->current_attention->target_logits->acceptance',
        'acceptance->future logical prefix metadata'],
      invalid_delay_cycle='Keeping cachewrite->current_attention while adding acceptance->cachewrite creates current_attention->acceptance->cachewrite->current_attention.',
      overlay_edges=['KVready->current_overlay_attention->target_logits->acceptance',
        'acceptance+KVbankready+slotmapready->commit->future_cache_reader'],
      scheduling_scope='Edges are per dependent request/tile/value. Do not impose a whole-operator or all-request barrier unless proved;commit may overlap other work subject to real readiness/resources.',
      commit_guards=['Publish complete data and page/map/generation before each future reader.',
        'Keep all old readers AND queued writers before owner/slot/page reuse;ABA generation checks alone cannot satisfy an old reader needing old data.',
        'COW may retain rejected suffix bytes if logically unobservable;no physical-zero requirement is inferred. All external/raw/debug/transfer observers remain conditions.',
        'Shared tuple backing aliases/multi-plane maps must remain coherent;page adoption is not an isolated SWA-pointer swap.'],
      all_readers_or_native_rewrite_legal_proved=False,overall_complete=False))
    print(dict(graphs=80,strict_SWA=3120,ambiguous_SWA_C1=160,terminal_frontiers=[80,64,16],source_files=3,
      new_scope='KV acceptance-delayed publication graphs and shifted prefix semantics',all_readers_closed=False))

if __name__=='__main__':main()
