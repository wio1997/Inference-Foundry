"""Reconcile operator interfaces and joint cuts, not framework behavior or implementation."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write


INPUTS=[
    'candidate-profile-ledger-v82.json','candidate-record-index-v82.json',
    'module-profile-ledger-v82.json','candidate-relation-clauses-v82.json',
    'attention-indexer-native-numerical-contract-v60.json','attention-indexer-work-traffic-models-v60.json',
    'attention-indexer-fusion-reorganization-variants-v60.json',
    'projection-operator-numerical-contract-v61.json','projection-fusion-variants-and-interactions-v61.json',
    'state-engram-operator-contract-v62.json','C2-state-and-pool-costs-v62.json',
    'index-context-operator-contract-v64.json','index-context-fusion-variants-v64.json',
    'index-context-fusion-costs-v64.json','V41-query-operator-path-contract-v75.json',
    'V41-QB-RoPE-work-and-bound-models-v75.json','Q_B-index-epilogue-extension-v82.json',
    'A3-datapath-primary-evidence-v80.json','C2-physical-ring-storage-models-v79.json']


def main():
    assert not(O/'attention-composition-clauses-v83.json').exists()
    # Only source/operator contracts already acquired. No target code scan.
    source_paths=[O/x for x in INPUTS if (O/x).exists()]+[R/'fusion-scan.json',R/'graphs/source-templates-G5.json']
    receipts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in source_paths]
    write(O/'operator-composition-inputs-v83.json',dict(files=receipts,
        missing_optional_names=[x for x in INPUTS if not(O/x).exists()],
        new_target_source_read=False,new_remote_source_read=False,new_device_work=False,
        grade='REUSED_SAVED_PROFILE_AND_PINNED_OPERATOR_CONTRACTS;HISTORICAL_GRAPH_IS_STRUCTURAL_NAVIGATION',overall_complete=False))
    graph=read(R/'graphs/source-templates-G5.json')
    write(O/'source-template-applicability-overlay-v83.json',dict(parent=str(R/'graphs/source-templates-G5.json'),
        templates=len(graph['templates']),nodes=sum(len(t['nodes'])for t in graph['templates']),edges=sum(len(t['edges'])for t in graph['templates']),
        corrections=[
            dict(id='V41-QUERY',contract='V41-query-operator-path-contract-v75.json',rule='Active V4.1 QR RMS1280 precedes Q_B. There is no post-QB head512 RMS in current target/draft query entry; legacyCP reference stays outside it.'),
            dict(id='INDEX-INPUT-DQ',contract='index-context-fusion-variants-v64.json',rule='The q_norm->iq composite edge hides ordinary input DynamicQuant. Current Q_B and IndexQ have separate native quant calls in observed graphs; possible PreparedLinearInput reuse is conditional, not already one quantization.'),
            dict(id='QUANT-OUTPUT-OWNER',contract='quant-epilogue-source-contract-v82.json',rule='INT8->BF16 quantized projection may finish in a Vector epilogue after INT32 GM. A matrix label does not establish Cube ownership of final BF16 or all-onchip fusion.'),
            dict(id='ROW-GENERATION',contract='projection-operator-numerical-contract-v61.json',rule='Old global_or_restored/query_local labels are symbolic roles, not equal row sets or pointer identities. Bind actual query/KV/CP-SP/pad maps before any joint computation.'),
            dict(id='OPAQUE-BASELINE',contract='attention-indexer-native-numerical-contract-v60.json',rule='QLI already fuses INT8 QK, FP16 dequant/ReLU, weighted head reduce, keyscale/membership/TopK. SMLA already fuses SWA+selected compressed attention. Do not register these baseline stages as new fusion gains.'),
            dict(id='PERSISTENT-FANOUT',contract='state-engram-operator-contract-v62.json',rule='IndexKey consumes the pre-RoPE BF16 latent; compressedKV and SWA caches and index/candidate banks retain future readers. A current consumer join does not close future cache/state readers.')],
        original_graph_modified=False,full_primitive_or_runtime_graph_proved=False,overall_complete=False))
    atoms={
      'F01':['mhc.attn_pre'],
      'F02':['token.embedding','mhc.attn_pre','norm.attn_input'],
      'F03':['mhc.attn_post','mhc.ffn_post','mhc.next_pre'],
      'F04':['norm.ffn_input'],
      'F05':['attn.qr.norm1280','attn.qr.quant_for_QB','attn.QB'],
      'F06':['attn.QA','attn.SWA_KV_projection','index.Q_projection','index.W_projection'],
      'F07':['attn.QB','attn.query.tail64_RoPE'],
      'F08':['attn.SWA_KV_projection','attn.SWA_KV.norm512','attn.SWA_KV.tail64_RoPE','state.SWA_publish'],
      'F09':['attn.qr.quant_for_IndexQ','index.Q_projection','index.Q.tail64_RoPE','index.Q.quant128'],
      'F10':['index.W_projection','index.W.cast_scale_cast'],
      'F11':['index.K_projection','index.K.norm128','index.K.tail64_RoPE','index.K.FLOAT_scale_quant','state.index_K_scale_publish'],
      'F12':['index.QLI','index.prepare_indices','state.selected_indices_publish','state.C1_candidates_publish'],
      'F13':['attn.SMLA'],
      'F14':['attn.SMLA','attn.output.inverse_tail64_RoPE','attn.output.pad_pack','attn.output.restore_heads','attn.OA'],
      'F15':['attn.output.pad_pack','attn.output.restore_heads','attn.OA'],
      'F16':['attn.OA','attn.OB','attn.output.reduce_scatter'],
      'F17':['compress.C2.FLOAT_input','compress.C2.Wkv','compress.C2.Wscore'],
      'F18':['compress.C2.pool','compress.latent.norm512'],
      'F19':['compress.latent.norm512','index.K_projection','index.K.norm128','index.K.tail64_RoPE','compress.latent.tail64_RoPE','state.compressed_KV_publish'],
      'F20':['moe.router','moe.TopK','moe.dispatch_FFN_combine'],
      'F21':['moe.dispatch','moe.GMM1','moe.Swiglu','moe.GMM2','moe.combine'],
      'F22':['moe.shared.DQ','moe.shared.GateUp','moe.shared.DSQ','moe.shared.Down'],
      'F23':['moe.shared.Down','moe.dispatch_FFN_combine','moe.shared_routed_merge','mhc.ffn_post'],
      'F24':['engram.hash','engram.UVA_lookup','engram.ready','engram.row_partition'],
      'F25':['engram.projection','engram.gate_residual'],
      'F26':['target.final_HC_collapse','target.final_norm','target.head'],
      'F27':['target.aux_HC_mean','context.main_projection','context.main_norm'],
      'F28':['draft.embedding','draft.HC_repeat','draft.decoder0'],
      'F29':['context.KV_projection','context.KV_norm','context.KV_RoPE','context.assembly','context.slot_publish'],
      'F30':['draft.HC_collapse','draft.norm','draft.head','draft.Markov','draft.confidence','draft.accept_verify'],
      'F31':['metadata.attention','metadata.index','metadata.CP_SP_rows','metadata.accept_reject','metadata.state_generation'],
      'F32':['attn.input.gather','attn.QA','attn.SWA_KV_projection','compress.C2.Wkv','compress.C2.Wscore'],
      'F33':['compress.C1.projection','attn.SWA_KV_projection','compress.latent.norm512','index.K_projection'],
      'F34':['compress.C2.pool','state.C2.old_ring_read','state.C2.tail_update','state.C2.physical_owner'],
      'F35':['target.head','target.score_processors','target.greedy','target.raw_scores','target.reject_output']}
    candidate_ledger=read(O/'candidate-profile-ledger-v82.json')['candidates']
    modules=read(O/'module-profile-ledger-v82.json')['modules']
    registry={r['id']:r for r in read(O/'candidate-record-index-v82.json')['model_candidates']}
    footprints=[]
    for c in candidate_ledger:
        cid=c['id'];master_path=R/'candidates'/(cid+'.json');blob=master_path.read_bytes()
        assert hashlib.sha256(blob).hexdigest()==registry[cid]['master_sha256']
        master=read(master_path)
        footprints.append(dict(id=cid,name=c['name'],operator_atom_union=atoms[cid],
            modules=[m['id']for m in modules if cid in m['candidate_links']],
            captured_master_sha256=registry[cid]['master_sha256'],
            original_template_graph_id=master.get('original_template_graph_id'),
            original_node_mapping=master.get('original_node_mapping'),
            simultaneous_variant_or_runtime_binding_proved=False,
            union_scope='Navigation union of named operator variants. Same atom requires same invocation/row/position/weights/config/value generation and compatible partition; target and draft namespaces stay distinct.'))
    write(O/'candidate-operator-footprints-v83.json',dict(candidates=footprints,count=35,
        source='Exact v82 master hashes checked against candidate-record-index-v82 before v83 mutation; immutable operator-origin projection captured here.',
        no_new_maincandidate=True,not_full_primitive_graph=True,overall_complete=False))
    # Every row names a concrete operator interface, a counterfactual change and its gates.
    rows=[
      ('F08','F13','SWA_PUBLISH_IS_NOT_A_PRIVATE_ATTENTION_TEMP',
       'Compose current SWA KV preparation and SMLA readiness, or schedule SWA partial attention before QLI finishes.',
       'Current-query bypass cannot delete future SWA cache publication. KV row set can be global/restored while query rows are local. A3 Vector RoPE->Cube attention still needs GM unless ownership/algorithm changes. Split attention needs original online state and sink once; no free overlap.', 'PERSIST'),
      ('F09','F10','DISTINCT_PROJECTION_INPUTS_AND_FP16_POSTPROCESS_JOIN',
       'Meet IndexQ head scales and separately projected IndexWeight at the QLI weight-times-scale step.',
       'IndexQ uses normalized QR1280; IndexWeight uses hidden5120, so equal output M does not permit one shared-input projection. Preserve BF16 W projection->FP32*2^-6->FP16 and independently rounded Q head scales. Both ready branches, exact FP16 product, liveness and acyclic joins required.', 'SCALES'),
      ('F09','F11','QUERY_KEY_QUANTIZERS_HAVE_DISTINCT_PRECISION',
       'Reuse scheduling/code only where separate query/key value domains and scalar contracts remain unchanged.',
       'Query scale is rounded FP16 BEFORE quant; key DynamicQuant uses FLOAT before FP16 cache-scale conversion. 95 versus96 witness forbids unification by common scale dtype. Different amax rows, weights, positions, row/rank maps and cache readers remain.', 'SCALES'),
      ('F09','F12','QUERY_EPILOGUE_TO_QLI_RETAINS_VC_BRIDGE',
       'Extend post-BF16 IndexQ RoPE/head-quant epilogue through query-scale/weight preparation for QLI.',
       'Keep BF16 RoPE and rounded FP16 head scales/div_rn/nearbyint. INT8 query values still cross AIV->AIC via GM in stated A3 ownership. QLI already computes native dot/dequant/ReLU/headreduce/TopK; no new gain for those baseline stages. Tiling and future query-scale readers constrain scale-interface deletion.', 'SCALES'),
      ('F10','F12','WEIGHT_SCALE_TO_NATIVE_HEAD_REDUCE',
       'Move current BF16->FP32 scale->FP16 weight epilogue into the QLI preparation boundary.',
       'Preserve weight FP16 rounding before FP16 product with Q scales. Casting weight to FP16 before *2^-6 can overflow. Native M16 head-reduce/broadcast representation and its Vector->Cube GM interface remain unless independently rewritten; no replacement with attention softmax scale.', 'SCALES'),
      ('F11','F12','PERSISTENT_KEY_SCALE_PUBLICATION_AND_NATIVE_QLI_READERS',
       'Joint key/scale publish and current QLI consumption, including TP owner-preparation alternative.',
       'INT8 key bank and FP16 metadata must publish coherent generations for all current/future queries. The persistent cache can itself serve as A3 Vector->Cube GM bridge; fusing calls does not erase this raw-key handoff. Preserve physical slots/strides/duplicates and history keys; only contained private copies or scale readers are eligible.', 'PERSIST'),
      ('F11','F13','INDEX_KEY_IS_NOT_ATTENTION_KV',
       'Coordinate shared latent consumers and their cache publication, including canonical TP key preparation.',
       'IndexK128 INT8+FP16scale and compressedKV512 BF16 are different projections/roles. Neither cache substitutes for the other. Keep the pre-RoPE latent IndexKey read before in-place compressed RoPE. Query sharding remains, so reducing replicated key-prep compute does not divide QLI/SMLA work again.', 'PERSIST'),
      ('F12','F13','QLI_PREPARE_STABLE_INDEX_AND_SMLA_JOIN',
       'Compose QLI->chronological prepare->SMLA with direct tile handoff while retaining required banks.',
       'SMLA expects valid sorted positive prefix and -1 suffix, not score order or filter-only holes. Stable selected indices and C1 candidate banks have other/future layer readers; preserve publication and exact row/position/mask/offset generation. Existing self-copy elision is already baseline candidate credit.', 'SELECT'),
      ('F12','F19','INDEX_SELECTION_AND_COMPRESSED_KV_ARE_SEPARATE_OUTPUTS',
       'Move compressor/key/index preparation closer to the selected compressed-attention join.',
       'QLI produces logical positions and C1 candidates; compressor produces BF16 KV values. Both are needed, with old historical KV and current valid slots. Do not equate selected index bank with compressed data or suppress unselected cache writes without all future readers/rollback proof.', 'SELECT'),
      ('F13','F19','COMPRESSED_CURRENT_TILE_AND_FUTURE_CACHE_FANOUT',
       'Pass a ready compressed current tile toward SMLA while preserving IndexKey pre-RoPE fork and long cache.',
       'Only current visible produced subranges can bypass redundant copies; history and other selected positions still come from cache. The Vector-RoPE->Cube boundary uses GM under fixed A3 ownership. Compressed cache future readers, BF16 round, position/group/slot generation and original softmax order survive.', 'PERSIST'),
      ('F13','F14','ATTENTION_EPILOGUE_INVERSE_ROPE_AND_LAYOUT',
       'Append inverse tail64 RoPE and send-layout production to current SMLA output epilogue.',
       'Keep SMLA BF16 round then inverse-RoPE BF16 round. Full512 output is still live for output projection/communication; only tail64 rotates. Layout/pad/CP-SP/head restore ownership and any other output readers must be contained. Source score/exp GM queues are not eliminated by output-only fusion.', 'OUTPUT'),
      ('F14','F16','OUTPUT_EPILOGUE_AND_TWO_PROJECTIONS_OVERLAP_AT_OA',
       'Compose inverse-RoPE/pack/OA with OA->OB->reduce-scatter using one chosen row/head/feature partition.',
       'OA belongs to both candidates and is computed once. Current rounded OA output/full reduction domains cannot be reassociated across OB/collective. Pack/restore/reduce frontiers depend on partition; A3 CC L1 handoff is conditional on ownership, tile and BF16 boundary. Count the same temporary interface once.', 'OUTPUT'),
      ('F17','F19','C2_PROJECTION_POOL_NORM_LATENT_FORK_CHAIN',
       'Compose C2 Wkv/Wscore projections through pair pool/norm to the key/long-KV latent fork.',
       'F18 pool/norm is a mandatory semantic stage between these footprints, not a removable missing node. Keep FP32 projection/score, stable pool math, pooled BF16 before norm and normalized BF16 before two forks. Include old-ring residual input and future tail updates; tile streaming does not remove pair dependencies.', 'STATE'),
      ('F17','F34','PROJECTED_ROW_DEMAND_INCLUDES_OLD_POOL_AND_NEW_RING',
       'Restrict or reorder C2 projection rows only after closing current pool and future ring demand.',
       'Even-only ring maintenance cannot blindly halve current pair-pool projection: both rows of each completed pair contribute. Old residual pair and tail32 update consumers use different row sets. Preserve invalid-page/slot behavior, read-old-before-write, rollback and 64-bit addressing.', 'STATE'),
      ('F19','F34','LATENT_OBSERVABILITY_DOES_NOT_CLOSE_RING_LIFETIME',
       'Combine latent fork analysis with C2 even-state/private-bank representation alternatives.',
       'Pre-RoPE IndexKey and post-RoPE compressedKV readers both remain. Current latent completion does not retire old ring readers/writers or release SWA-anchored global tuple IDs. Private/paired representation needs budget, owner halves, ABA/generation and rollback proof.', 'STATE'),
      ('F08','F19','SWA_AND_COMPRESSED_BANKS_HAVE_DISTINCT_VALUES',
       'Coordinate two KV preparation branches, preserving shared projection alternatives only in proven C1 domain.',
       'C2 pooled latent is not the current SWA token vector even at equal width512. Separate norms/positions/cache banks remain; C1 identical projection equivalence is a separate F33 scope. Do not cancel SWA and compressed publication or merge RMS domains from width alone.', 'PERSIST'),
      ('F08','F11','DISTINCT_NORM_ROPE_QUANT_DOMAINS',
       'Schedule SWA KV preparation alongside index-key preparation and share only proven input loads.',
       'SWA RMS512/RoPE and IndexKey projection/RMS128/RoPE/FLOAT-scale quant are distinct values and row domains. Common C/S may be read-shared only under same position/dtype/generation, not via replacing one result or normalizer. Preserve all cache pairs and scope C1 projection sharing separately.', 'PERSIST'),
      ('F09','F19','QUERY_KEY_LATENT_JOIN_NEEDS_QUERY_AND_CACHE_GENERATIONS',
       'Coordinate IndexQ query epilogue with compressed latent->IndexKey/long-KV branches before QLI.',
       'Query QR1280 is not compressor normalized latent512. Query and key quantization policies differ. Existing keys unchanged plus cache append does not justify old query scores/TopK reuse; new query can reverse ordering. Cross-layer candidate reuse is positional bank membership, not shared Q/K values.', 'SELECT'),
      ('F11','F18','TWO_DIFFERENT_RMS_BOUNDARIES_ON_KEY_PATH',
       'Compose completed pool output/norm through IndexKey projection/norm/RoPE/quant.',
       'Pool latent RMS512 precedes WK512->128; key RMS128 follows it. Cannot fold the two RMS operations through projection. Preserve pooled BF16 before first norm, normalized pre-RoPE latent fork, key BF16 and key quantizer boundaries plus ring/long-KV consumers.', 'STATE'),
      ('F12','F34','RING_REPRESENTATION_CHANGE_IS_NOT_INDEX_SCORE_REUSE',
       'Compose C2 state pruning/private representation with later QLI/candidate publication.',
       'The even-state observability proof concerns ring maintenance, not query/Kcache value invariance or stable TopK. Preserve accepted prefix, current/rollback positions, physical owner and all subsequent normalized latent/cache generations. Query-change counterexample forbids updating only newly appended QLI scores.', 'STATE'),
      ('F13','F34','SWA_PARTIAL_ATTENTION_AND_C2_POOL_STATE_DEPENDENCIES',
       'Schedule ready SWA attention separately from not-yet-ready C2 compressed branch, or compose at one join.',
       'SWA can precede QLI only with valid current/historical cache readiness and finite resource progress. Original integrated SMLA already combines SWA+CMP; splitting needs compatible unnormalized state and sink exactly once, not BF16 normalized outputs with absent LSE. C2 ring ownership and current compressed completion remain.', 'SPLIT'),
      ('F09','F13','INDEX_QUERY_FRONTEND_THROUGH_QLI_TO_ATTENTION',
       'Analyze the whole IndexQ->QLI->prepare->selected SMLA subgraph rather than treating two pair savings independently.',
       'Include F10 weights, F11 key+scale, persistent selected/candidate banks, F19 longKV and F31 metadata as external demands or absorbed stages. INT8 query VC, QLI scalar score CV and SMLA score/exp CV queues remain in current ownership; exact numerical, chronological-prefix and generation contracts required.', 'JOIN')]
    existing=read(O/'candidate-relation-clauses-v82.json')
    existing_pairs={tuple(sorted([c['a'],c['b']]))for c in existing['clauses']}
    clauses=[]
    evidence={
      'PERSIST':['attention-indexer-native-numerical-contract-v60.json','index-context-operator-contract-v64.json','state-engram-operator-contract-v62.json','A3-datapath-primary-evidence-v80.json'],
      'SCALES':['attention-indexer-native-numerical-contract-v60.json','index-context-operator-contract-v64.json','Q_B-index-epilogue-extension-v82.json'],
      'SELECT':['attention-indexer-native-numerical-contract-v60.json','attention-indexer-fusion-reorganization-variants-v60.json','index-context-operator-contract-v64.json'],
      'OUTPUT':['attention-indexer-native-numerical-contract-v60.json','projection-fusion-variants-and-interactions-v61.json','A3-datapath-primary-evidence-v80.json'],
      'STATE':['state-engram-operator-contract-v62.json','C2-state-and-pool-costs-v62.json','attention-indexer-native-numerical-contract-v60.json'],
      'SPLIT':['attention-indexer-fusion-reorganization-variants-v60.json','C2-state-and-pool-costs-v62.json'],
      'JOIN':['attention-indexer-native-numerical-contract-v60.json','attention-indexer-work-traffic-models-v60.json','index-context-operator-contract-v64.json','Q_B-index-epilogue-extension-v82.json']}
    for i,(a,b,relation,change,gates,kind)in enumerate(rows,1):
        assert tuple(sorted([a,b])) not in existing_pairs
        clauses.append(dict(id=f'REL83-{i:02d}',a=a,b=b,relation=relation,operator_claim=change,
            applicable_domain=gates,evidence=evidence[kind]+['attention-joint-interface-models-v83.json'],
            conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    assert len(clauses)==22
    groups=[dict(id='HG83-INDEX-ATTENTION',members=['F09','F10','F11','F12','F13','F19','F31'],
        contract='Choose one legal query/key scale policy, complete persistent-reader frontier and query/cache/indices/metadata generation before joint costing. QLI internal fusion is baseline; INT8 VC and score/exp CV transfers remain under current owners. Exact TopK fillers/prepare order, bank outputs and SMLA numerical state preserved.'),
      dict(id='HG83-COMPRESS-PERSISTENCE',members=['F08','F11','F13','F17','F18','F19','F33','F34'],
        contract='C1 identical-projection and C2 pair-pool domains are alternatives, not interchangeable. Preserve pair-demand union, old-state read before update, BF16 latent pre-RoPE fork and independent SWA/index/longKV persistent publication. Physical state capacity/lifetime is separate from current attention IO.'),
      dict(id='HG83-ATTENTION-OUTPUT',members=['F07','F12','F13','F14','F15','F16','F31'],
        contract='Compose query readiness, selected attention and output inverse-RoPE/layout/OA/OB/collective using one row/head partition. Distinct forward/inverse rotated values and all BF16 rounds remain. Shared OA and each removable transfer counted once; true tile dependencies and resource cuts replace sums of standalone gains.')]
    for group in groups:group.update(evidence=['attention-composition-clauses-v83.json','attention-joint-interface-models-v83.json'],
        scope_state='SCOPED_OPERATOR_COMPOSITION_AND_RESOURCE_CONSTRAINTS_REVIEWED',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'attention-composition-clauses-v83.json',dict(clauses=clauses,higher_order_groups=groups,
        new_pair_clauses=22,new_maincandidates=0,whole_pair_all_variants_closed=False,overall_complete=False))
    allclauses=existing['clauses']+clauses
    graph_templates={t['id']:t for t in graph['templates']}
    pair_navigation=[];byid={r['id']:r for r in footprints}
    for a,b in combinations(sorted(byid),2):
        left,right=byid[a],byid[b];template_links=[]
        tid=left['original_template_graph_id']
        if tid and tid==right['original_template_graph_id'] and tid in graph_templates:
            leftnodes=set(left['original_node_mapping'] or {});rightnodes=set(right['original_node_mapping'] or {})
            template_links=[e for e in graph_templates[tid]['edges'] if (e['producer']in leftnodes and e['consumer']in rightnodes)or(e['producer']in rightnodes and e['consumer']in leftnodes)]
        matching=[c['id']for c in allclauses if {c['a'],c['b']}=={a,b}]
        pair_navigation.append(dict(a=a,b=b,shared_operator_atom_union=sorted(set(atoms[a])&set(atoms[b])),
            common_module_associations=sorted(set(left['modules'])&set(right['modules'])),
            historical_template_interface_edges=template_links,conditional_clause_ids=matching,
            semantic_pair_reconciliation_done_in_stated_scope=bool(matching),whole_pair_all_variants_closed=False,
            no_independence_or_no_opportunity_inference_from_empty_fields=True))
    assert len(pair_navigation)==595 and sum(bool(p['conditional_clause_ids'])for p in pair_navigation)==69
    write(O/'candidate-pair-navigation-v83.json',dict(pairs=pair_navigation,model_candidates=35,pair_universe=595,
        scoped_conditional_pairs=69,remaining_central_pair_reconciliation=526,
        navigation_is_not_semantic_or_runtime_proof=True,overall_complete=False))
    # Interface models are scoped references; no speculative physical bandwidth binding.
    scale_models=[];output_models=[];publish_models=[]
    for count in [2,3,800,816,830,834]:
        scale_models.append(dict(M=count,index_heads=32,index_width=128,
            private_query_FP16_scale_store_read_reference_bytes=4*count*32,
            private_weight_FP16_store_read_reference_bytes=4*count*32,
            both_private_FP16_plane_union_reference_bytes=8*count*32,
            retained_INT8_query_VC_GM_reference_bytes=2*count*32*128,
            one_native_M16_broadcast_product_GM_plane_store_read_reference_bytes=4*count*32*16,
            qualification='Each scalar plane removed only if all readers absorbed with same rounded scale/weight. Product GM figure is one declared materialized M16 broadcast copy, not measured per-call repetition or mandatory all-algorithm IO.',
            actual_HBM_or_gain=None))
        elems=count*64*512;tail=count*64*64
        output_models.append(dict(M=count,full_output_elements=elems,tail_elements=tail,
            inverse_RoPE_tail_private_interface_reference_bytes=4*tail,
            separate_unpacked_full_output_private_store_read_reference_bytes=4*elems,
            distinct_transfer_union_reference_bytes=4*elems+4*tail,
            final_packed_BF16_output_write_bytes=2*elems,
            qualification='Counterfactual baseline has an unpacked SMLA BF16 GM result, in-place inverse tail RoPE and separate read/pack. Joint epilogue emits final packed BF16 once; communication/OA GM input remains. If actual views/packing differ rebuild transfer set; not native measured bytes.',
            actual_gain=None))
    for dirty_index,dirty_cmp,dirty_swa in [(1,1,1),(3,3,3),(24,12,24)]:
        publish_models.append(dict(dirty_live_unique_index_rows=dirty_index,dirty_live_unique_cmp_rows=dirty_cmp,
            dirty_live_unique_SWA_rows=dirty_swa,index_key_and_scale_publication_bytes=130*dirty_index,
            compressed_KV_publication_bytes=1024*dirty_cmp,SWA_publication_bytes=1024*dirty_swa,
            distinct_bank_publication_total_bytes=130*dirty_index+1024*(dirty_cmp+dirty_swa),
            qualification='Fixed uncompressed disjoint GM-bank ABI, distinct dirty live rows at publication boundary. U counts actual unique dirty demanded rows after duplicate-write semantics, not descriptor M or allocation. Not an all-representation/HBM byte lower bound.'))
    split_witness=dict(partition_A_normalized_output=0,partition_B_normalized_output=1,
        first_possible_normalizers=[1,1],first_merged_output=0.5,
        second_possible_normalizers=[3,1],second_merged_output=0.25,
        implication='Identical normalized outputs do not determine the merged result without normalization mass/compatible online state. This real-valued information-loss example is not a native BF16 numerical test; sink and rounding add further conditions.')
    demand_examples=[]
    for start,length in [(0,6),(1,6),(0,3),(1,3)]:
        present=set(range(start,start+length));current=[];history=set()
        for last in sorted(present):
            if last%2==1:
                pair={last-1,last};current.append(sorted(pair));history|=pair-present
        ring_even={p for p in present if p%2==0}
        demand_examples.append(dict(start=start,length=length,completed_pairs=current,
            current_pair_projection_rows=sorted(set().union(*(set(p)&present for p in current))if current else set()),
            old_history_rows=sorted(history),even_only_future_ring_rows=sorted(ring_even),
            pool_projection_can_be_replaced_by_even_rows=False))
    joint=dict(scale_models=scale_models,output_models=output_models,dirty_persistent_publication_models=publish_models,
        split_normalized_output_information_loss_witness=split_witness,completed_pair_projection_demand_examples=demand_examples,
        symbolic_joint_work=dict(
            QLI='In conventional fullscan domain Q_INT8=2*32*128*sum_q(Lq), useful FP16 headreduce work=2*32*sum_q(Lq); native M16 extra lanes are current implementation, not universal necessary work. Prefilter/membership/bitmap are different domains.',
            SMLA='Q_BF16=4*64*512*sum_q(nq) in conventional full selected QK/PV domain; add exact Vector/softmax/state work. Query changes prohibit unconditional previous-score reuse.',
            current_GM_frontiers='INT8 query VC plane2*M*32*128; QLI scalar-score CV8*sum(Lq); SMLA FP32-score CV8*64*sum(nq)+BF16-exp VC4*64*sum(nq), in unchanged materialized source queues. Output-only fusion does not eliminate these.',
            joint_time='T>=max(Q_INT8/C_INT8,Q_half/C_half,Q_GMread/Bread,Q_GMwrite/Bwrite,Q_shared/Bshared_if_proved,L_Vector_or_sort,L_true_tile_dependencies,L_link_cuts). Same resource necessary work accumulated once, disjoint transfer sets unioned once; no sums of standalone times/savings.',
            physical_numeric_capacity_binding='Unknown Vector/FP32/HF32/Sort/nonlinear/actual GM-L2-HBM/network/fmax terms remain symbolic. Reference published half/INT8/HBM rates are not actual capacity proof.'),
        crosscard_repartition_identity=dict(P=8,local_queries=3,keys_reference=1024,
            current_per_rank_query_key_pairs=3*1024,
            repartition_per_rank_query_key_pairs=(8*3)*(1024//8),
            query_key_work_ratio=1,
            key_owner_prep='A separate identical key-prep ownership rewrite may reduce replicated preparation but needs130B/row distribution plus per-rank physical cache publication; not another8x QLI/SMLA compute division.'),
        transfer_credit_rule='Identify original transfer IDs and value generations. E.g. inverse tail4R and unpacked-plane4A can sum only for the declared distinct native-like baseline; reuse the same IDs across F13/F14/F15/F16 so the joint union counts them once. Persistent/cache/communication outputs are not private interfaces.',
        conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,actual_speedup=None,overall_complete=False)
    write(O/'attention-joint-interface-models-v83.json',joint)
    print(dict(operator_footprints=35,pair_navigation=595,new_conditional_pair_clauses=22,
        scoped_pairs=69,central_pending=526,new_higher_order_groups=3,
        scale_M3_private_union_bytes=8*3*32,query_INT8_VC_M3_bytes=2*3*32*128,
        no_new_target_code=True,no_device_execution=True,overall_complete=False))


if __name__=='__main__':main()
