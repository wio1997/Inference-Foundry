"""Semantic review of all old900 navigation sites in operator-effect domains."""
import ast
import collections
import json
import re
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def write(n,d):(H/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
d=read(H/'remaining900-exact-operator-effects.json')

# These are semantic contracts reviewed against all extracted guard alternatives,
# their calls/writes and early-return continuations. The surrounding selectors
# remain unchanged unless a named candidate explicitly replaces them.
domains=[
('SETUP', ['__init__'], ['M00','M04','M21','M22','M23'], ['F20','F24','F31','F32','H06','H07'],
'Constructor selectors establish cache/scale dtype,backend,DCP/offload,EPLB,async valid-count buffers and capture buckets. Repeated ones/zeros are input/bank setup,not repeated model MACs.',
'Share or fold initialization only under the same mode/shape/dtype/owner/config generation. Preserve selected backend and every dependent buffer or process. ENPU/C8/graph flags cannot be inferred from a kernel name or driver generation.'),
('DRAFTER_SELECTION',['_set_up_drafter','_eagle3_uses_aux_hidden_state','_get_eagle3_aux_layers_from_config','_draft_uses_qwen3_gqa_dspark'],['M00','M15','M18','M19','M20'],['F27','F28','F29','F30','F31'],
'Method/class selection determines draft width,auxiliary layers and target-hidden demand. DSpark,Eagle3,extract,other-model branches have distinct consumers; nonempty per-method lists and PP rank matter.',
'Target-tail pruning or shared target/draft work is conditional on actual demand and parameter/input generations. Preserve alternative selector bodies and errors; a text label DSpark alone is not a Qwen3 layout or an auxiliary-reader certificate.'),
('GRAPH_WRAPPERS',['_use_aclgraph','get_model','_get_gpu_model_runner_module_name','_torch_cuda_wrapper','_replace_gpu_model_runner_function_wrapper','update_pass_config'],['M00','M22'],['F07','F31','H06','H07'],
'Graph dispatch combines compile/breakable/enforce-eager modes; unwrap follows actual wrapper type. Temporary cuda/event/stream/weakref and SP pass bindings are restored in finally,including errors.',
'Keep wrapper dispatch and restore boundaries. A proposed body fusion may preserve the existing public interface; an address/update/offload rewrite needs its exact captured-bank/native contract. Stable descriptor or weak reference does not establish generation/lifetime.'),
('PD_ROLE',['_is_pd_prefill_worker'],['M02','M03','M23'],['F08','F27','F31','F34'],
'Producer-only PD is producer and not consumer; no-forward connector work may still transfer/publish cache state.',
'No local model rows does not imply no connector/peer/cache work. Preserve role-dependent completion and ownership before any producer/defer/reuse transformation.'),
('PP_ACCEPTED_IDS',['_apply_pp_sampled_tokens_from_scheduler_output'],['M04','M23','M24'],['F31','F34','F35','H02'],
'Async PP consumers remap previous request IDs,discarded rows,valid output lengths and new sampled IDs before appending state. Empty-token and missing-previous-request cases differ.',
'Fuse row remap/gather/copy only against one immutable old snapshot or an explicit dependency-safe schedule. Equal batch size cannot identify request rows; keep output-token,token-is-ID and no-spec-count updates.'),
('PRE_FORWARD_STATE',['_update_states'],['M02','M04','M24'],['F31','F34','H02'],
'State update may copy KV blocks,clear newly assigned blocks,repair draft lengths on rollback and incorporate PP sampled IDs. These precede input/model work.',
'A combined copy/clear kernel must preserve original ordering,overlap/source snapshots and block generations. Elide a zero only if overwritten before every read; no independence from different field names or one absent trace connection.'),
('POST_FORWARD_ACCEPTED_STATE',['_update_states_after_model_execute','_sync_num_accepted_tokens'],['M04','M17','M24'],['F31','F34','F35','H02'],
'Accepted counts,previous-to-current request maps and async events are separate from optional hybrid/Mamba align/all state copies. New requests receive explicit defaults; valid output IDs use the original -1 predicate.',
'Combine summaries/remaps only with original signed comparison,cast/overflow,bounds and source epochs. Temp0 or shape equality does not remove accepted-count events. Current exported mamba mode none narrows its own comparisons,not all hidden state effects.'),
('PREPARE_INPUTS',['_prepare_inputs','set_active_loras'],['M04','M05','M21','M24'],['F20','F24','F31','F35','H02','H06','H07'],
'Input preparation derives exact ordered request/token counts,old positions/accepted corrections,logits indices,discard masks,slot maps and optional embeds/LoRA. CP rebuilt/positions-ready flags can suppress only the corresponding producers.',
'Fuse snapshot-to-bank construction and small copies under final consumer strides/dtypes,valid/padded extents and copy-ready edges. Keep actual CP cache-hit generation; do not execute a cached dead producer merely for fusion,or infer cache equality from stable addresses.'),
('ATTENTION_STATE_AND_SANITIZE',['_build_attn_state','_sanitize_placeholder_input_ids_for_forward','_pad_query_start_loc_for_fia'],['M04','M10','M21','M24'],['F20','F31','H02','H06'],
'Decode/prefill/speculative state depends on actual scheduled and valid counts. Placeholder -1 sanitization occurs on the guarded continuation. FIA start-location padding may append a real final endpoint and copy the bank.',
'Fold ID sanitization into the final input producer only if all Engram/router/model/raw-ID readers see the same cleaned prefix. Preserve ordered per-request offsets and padded-tail endpoints; bucket reduction needs a new packing contract,not just fewer allocated tokens.'),
('SPEC_METADATA_TRANSFER',['_copy_spec_decode_metadata_to_device','_calc_spec_decode_metadata','_correct_optimistic_seq_lens_cpu','_copy_valid_sampled_token_count'],['M04','M17','M21','M24'],['F31','F34','F35','H02'],
'Speculative cumulative counts/logits-index planes are distinct arrays; CPU/device branches,pinned owners,pending-copy queues,events and optimistic sequence-length correction define their current epoch.',
'One producer or transfer may emit their combined final representation only with matching consumer ABI and per-field dtype/extent. Source staging owners survive asynchronous DMA; old reader AND queued writer retirement are required. Do not remove a synchronize without the equivalent readiness proof.'),
('DRAFT_DISPATCH_AND_SKIP',['propose_draft_token_ids','_num_drafter_query_tokens','_input_fits_in_drafter','_drafter_runs_model_forward','_skip_drafting'],['M15','M18','M19','M20','M21','M23','M24'],['F27','F28','F29','F30','F31','F35'],
'Drafter dispatch selects tensor/list,packed/padded,method-specific ID/hidden/position inputs and optional probability planes. Fits/skip branches can still prepare next IDs,copy counts or run peer-required dummy forward.',
'Keep method-specific query width and request namespace,all auxiliary/KV/state demands and DP participation. A skipped local proposal is not global zero receiver work. Confidence/probability or target-hidden pruning is a separate observer guard.'),
('DEBUG_OBSERVERS',['_log_propose_draft_token_ids_entry','_start_dump_data','_finalize_dump_data'],['M00','M22','M25'],['F26','F30','F31','F35','H01','H05'],
'Logging/debugger guards affect type/shape/data observation and start/step/stop state; disabled guards and active dump sessions are different domains.',
'Retain the original observer guard and cleanup. Body fusion does not require research of arbitrary logging internals; deleting a content bank requires any enabled observer in its read set. No gain is credited for an already disabled observer.'),
('DRAFT_RESULT_TRANSPORT',['_copy_draft_token_ids_to_cpu','take_draft_token_ids'],['M17','M20','M24','M25'],['F30','F31','F35','H02','H05'],
'Host draft-ID transport depends on structured output/token-history/PP need,zeros_only,available streams/events,valid widths and dynamic per-request K clipping.',
'Batch transfers/notifications only with original per-request ID order and valid prefixes. A zero-only shortcut cannot silently remove grammar/history/PP consumers. Retain staging/event lifetime and parent-result normalization.'),
('EXECUTE_FORWARD',['execute_model'],['M02','M04','M05','M15','M16','M18','M21','M22','M23','M24'],['F08','F24','F26','F27','F28','F29','F30','F31','F32','F34','H02','H06','H07'],
'Execute owns no-token/connector/encoder paths,authoritative state snapshot and CP/PP/cascade/padded mode choices; it then prepares metadata/model inputs,forwards,selects demanded hidden rows/heads and releases metadata/KV publication. Deferred corrections and external-launch alignment remain ordered.',
'A rewritten subgraph must preserve these operator release/read/write effects. No-scheduled-token branches may still do connector/dummy work. Tail demand cannot drop prompt logprobs,auxiliary,KV/EPLB or intermediate PP outputs. Do not introduce whole-operator waits from a chosen tile or claim host gaps are free device overlap.'),
('SAMPLING_AND_GRAMMAR',['sample_tokens','sample_tokens.propose_draft_token_ids','_sample'],['M16','M17','M18','M20','M21','M23','M24','M25'],['F26','F30','F31','F35','H02','H05'],
'Sampling guards select grammar processing,logits conversion/transport,rejection versus ordinary sampler,padded drafting and PP output,accepted state and stream events. Existing rejection and processor order precedes result publication.',
'Fuse public score selection after the same final processed score values; retain earlier raw/entropy/logprob/RNG/error readers. Replacing a grammar CPU round trip needs exact bitmask/dtype/rounding and observer semantics. Cross-iteration next-ID preparation cannot precede its accepted/current snapshot.'),
('NAN_OBSERVERS',['_get_nans_in_logits'],['M16','M17','M25'],['F30','F35'],
'Optional NaN counting/raise paths can read logits and synchronize even when ordinary returned logprobs are absent.',
'If enabled,retain the same stage/dtype count and error behavior or fold that exact check into its producer. Low ratio/temperature/request-logprob absence does not prove this observer disabled.'),
('RESULT_BOOKKEEPING',['_bookkeeping_sync'],['M17','M24','M25'],['F31','F34','F35','H02','H05'],
'Bookkeeping distinguishes async,one-token versus rejection output,invalid/discarded requests,logprobs,prompt logprobs,NaN diagnostics and generator-offset rollback; it appends real IDs and publishes the next previous-request map.',
'A result/summary kernel may share a scan but must preserve valid-ID predicates,per-request ID order,RNG offsets and all old/new owner fields. Empty IDs and skipped requests have state semantics; no unqualified summary reuse across iterations.'),
('ENGRAM_INPUT_OWNER',['_get_engram_device_inputs'],['M14','M21','M24'],['F24','F25','F31','F34'],
'Engram inputs use selected layer/group and current forward metadata/block table; missing layer/metadata returns a different input domain.',
'Lookup/prefetch/reuse keys include selected IDs,positions,layer/table/parameter/cache generations and owner. Stable slot or same head shapes do not establish identity; preserve current lookback/rollback state.'),
('METADATA_READY',['_prepare_device_metadata_for_forward'],['M21','M22','M24'],['F31','H02','H06'],
'A metadata executor is relevant only when submission is in flight; FULL graph execution requires the supported external-event protocol and may otherwise raise.',
'Producer/consumer fusion may replace this boundary only with equivalent readiness and the declared successful ABI domain. Never replace unknown in-flight state by ready merely because buffers exist.'),
('PP_INTERMEDIATE_COPY',['sync_and_slice_intermediate_tensors'],['M04','M05','M22','M23'],['F02','F03','F28','F29','F31','F32','H02','H07'],
'Intermediate banks are per key,may be allocated when missing,and copy/view different extents under SP and sync_self.',
'Share/fold copy with a producer only after exact row-owner/stride/source-generation and original overlap rules. Distinct dictionary keys can alias; pointer stability is not the read-ready or lifetime proof.'),
('OFFLOAD_TAIL_RESTORE',['_maybe_eager_restore_copy_sfa_tails'],['M02','M10','M21','M22','M24'],['F08','F13','F31','F34'],
'Fused-copy offload metadata may require eager tail restoration across dict/list groups before later consumers.',
'Cache-state and tail-restoration work cannot disappear just because the main producer is fused or a local query set is empty. Any deferred/COW path retains every needed tail and queue/publication epoch.'),
('ATTENTION_METADATA_ROOT',['_build_attention_metadata'],['M04','M08','M09','M10','M18','M21','M22','M23','M24'],['F08','F12','F19','F27','F29','F31','F32','F33','F34','H02','H06'],
'Metadata construction spans request/token padding,CPU snapshots or views,per-group block/slot tables,DCP/GDN,common V41/C2 data,offload/tail restore,vision/audio ranges,microbatches and device-task submission. Drafter common metadata must be selected from its own group.',
'Fuse/share equivalent fields from one authoritative snapshot using the exact group/ratio/layer/owner/state/config key and compatible output banks. Existing common-metadata caches are baseline,not new work savings. A dtype/shape-only key or final scalar count misses row/position/table/RNG epochs. Preserve active device-task producers and release events.'),
('DCP_METADATA',['_build_attention_metadata._get_dcp_metadata'],['M21','M23','M24'],['F27','F29','F31','F32','H06'],
'DCP metadata selects disabled,dummy,async corrected or CPU-length domains,with distinct local/global row coordinates.',
'Do not equate PCP,DCP,TP or SP groups by similar names. Repartition requires exact global-to-local map,valid padded requests and current sequence-length epoch; unchanged selectors preserve alternative branches.'),
('BLOCK_AND_SLOT_METADATA',['_build_attention_metadata._get_block_table_and_slot_mapping'],['M02','M21','M23','M24'],['F08','F27','F29','F31','F34','H02','H06'],
'Encoder-only creates its own slot representation; ordinary groups use current block-table and slot banks and fill request/token padding with their original constants.',
'A combined final writer may fold clears only if it fully defines every consumed tail in the correct dtype and map. Different fields/padding constants are not a common zero plane; preserve table validity and cache generations.'),
('BUILDER_DISPATCH',['_build_attention_metadata._build_attn_group_metadata'],['M08','M09','M10','M18','M21','M22','M23'],['F08','F12','F19','F27','F29','F31','F32','F33'],
'Builder class,capture mode,group/microbatch and actual request extents choose build versus capture-build,common V41 ratio metadata,per-layer aliases and optional device metadata tasks. GDN no-op changes the valid domain.',
'Same-ratio reuse must include every effective builder input and output-generation/alias contract,not ratio alone. Fold a public builder pipeline under an unchanged descriptor rather than editing unidentified native helpers; all publication tasks remain represented.'),
('EPLB_HEAT',['update_eplb_heat_collection_status'],['M12','M21','M23','M24'],['F20','F21','F22','F31'],
'Heat collection uses configured stage and maximum cross-DP padded tokens where present; local row zero is not the phase decision.',
'Preserve expert/load/parameter ownership updates when scheduling/fusing dispatch. Do not reuse counters across layers or steps solely because launch capacity is unchanged.'),
('LOAD_AND_BIND',['load_model'],['M00','M06','M09','M14','M15','M18','M19','M22'],['F01','F03','F12','F24','F27','F28','F29','F31','H07'],
'Model/draft/LoRA loading,aux-layer selection,lookback buffers,lim-indexer owner binding,wrapper capture options and update streams establish identities for later sharing. Shared parameter/data_ptr ownership is a construction fact,not all-reader equivalence.',
'Reuse only under unchanged parameter/layout/quantization/config generations and selected model ABI. Startup warmup/weight preparation is not repeated serving MAC work. All temporary selection patches and errors remain outside the rewritten kernel body.'),
('CACHE_BIND_AND_RING',['initialize_kv_cache','initialize_kv_cache_tensors'],['M02','M08','M09','M10','M18','M21','M22','M23','M24'],['F08','F11','F17','F18','F19','F27','F29','F31','F33','F34'],
'KV initialization applies the layerwise plan,binds physical views/aliases and registration,selects per-group kernel block sizes,validates draft group compatibility and prepares ratio2 compressor rings.',
'A shared/fused cache producer keeps all consumer physical layouts and bank epochs. Changing a layout or accepted-input publication requires updating every bound native consumer; allocation identity or sorted layer name does not establish independent storage.'),
('CACHE_SPEC_AND_DIMS',['get_kv_cache_spec','_get_layer_kv_cache_specs','_uses_page_strided_kv_layout','_get_attention_kv_cache_dims'],['M02','M07','M08','M09','M10','M21','M22'],['F08','F11','F13','F17','F18','F19','F31','F33','F34'],
'Spec selection covers V41/cache-only/MLA/indexer/full/hybrid,compressed or page-strided layouts,sparse C4/C8/MXFP/TQ scale planes,sharing and block padding. dtype auto is unresolved until its selected spec/backend.',
'Treat the complete descriptor as a typed contract:base owner,byte offsets,layer/block/page strides,capacity,logical positions,all value/scale types and aliases. No universal necessary bytes from a table/checkpoint/workspace size,and no INT8 KV or BF16 mode from W8A8 model naming.'),
('QUANTIZED_CACHE_INIT',['_is_c8_mxfp_kv_cache','_fill_c8_mxfp_v_scale_caches','_zero_block_ids','_allocate_int8_cache_tensor','_allocate_sparse_c8_indexer_tensors'],['M02','M09','M10','M21','M24'],['F08','F11','F19','F31','F34','H02'],
'Quantized allocation requires positive sizes,alignment/contiguity and proper scale views. MXFP value-scale initialization and block reset can encode a scale different from raw all-zero bytes; KV-transfer presence changes allocation behavior.',
'Fuse initialization or reset only with a final full writer preserving data AND encoded scale planes,all touched blocks and original copy/read order. Reinterpreting INT8 backing as scale dtype requires byte alignment/extent proofs; row zeros do not imply dead scale initialization.'),
('CACHE_LAYOUT_STORAGE',['_allocate_kv_cache_tensors','_reshape_kv_cache_tensors','_adjust_kv_layout'],['M02','M07','M08','M09','M10','M18','M21','M22','M24'],['F08','F11','F13','F17','F18','F19','F27','F29','F31','F33','F34'],
'Descriptor regions and reused layer slots,combined or separate K/V/scale/tuple buffers,page/byte padding,strided backing,hybrid state and packed/offload modes produce different physical views. Bounds/alignment/type/extent guards and raw-object alias tests constrain construction.',
'A legal relayout is a consistent mapping of all live views and observers. Preserve output types,byte/element unit conversions,offset/page/layer strides,tuple ordering,scale encoding and valid padding; physical memory/traffic counts union shared backing once. Shared backing never licenses merging unrelated epochs,or treating allocated padding as useful work. Prefix/COW/reuse needs complete selected view generations and queued-reader/writer retirement.'),
('BATCH_REINITIALIZE',['may_reinitialize_input_batch'],['M04','M21','M22','M24'],['F20','F24','F31','F34','F35','H02','H06','H07'],
'Multiple cache groups/kernel block sizes and offload/speculation settings may replace the input-batch object and its per-request bank capacities.',
'Treat a replacement as a new ownership/config generation even if allocator addresses recur. Cached metadata/graph updates must preserve current bank bindings; no pointer-only ABA reuse.'),
('ATTENTION_BACKEND_SETUP',['initialize_attn_backend','initialize_attn_backend.get_attn_backends_for_group.backend_supports_kernel_block_size','initialize_attn_backend.get_attn_backends_for_group','initialize_attn_backend.create_attn_groups'],['M00','M09','M10','M21','M22'],['F08','F12','F13','F19','F31','F32','F33','H06'],
'Per-layer backend/spec keys and integer-versus-MultipleOf kernel block support choose groups/builders and provider IDs. Device metadata executor enables actual providers,not every vaguely similar backend.',
'Group fusion/reuse requires compatible builder outputs and exact block/rope/cache layout,not class-name similarity. Preserve unsupported-size errors and mode compatibility; provider identity is not a ready/event certificate.'),
('ENCODER_CACHE_LIFETIME',['reset_encoder_cache','_track_tmp_encoder_cache_refs','_on_request_state_removed','_process_encoder_cache_scheduler_output','_get_encoder_output_from_cache','_clear_finished_encoder_cache_copies','_copy_cpu_encoder_cache_to_device','_cache_encoder_output','_prepare_multimodal_fields'],['M02','M04','M05','M22','M24'],['F24','F25','F31','F34','H02','H04','H07'],
'Optional encoder/CPU/offload caches have request reference sets,temporary/promoted ownership,pinned copies,pending events and field-specific CPU/device placement. Early-return guards still control later unconditional copy or cleanup effects.',
'Preserve actual text-versus-multimodal input contract and all request/hash/parameter keys. Batch transfers may share submission but retain source objects until DMA completes and all refcount/event retirement. Missing MM work in a profile is not global unreachability; no reuse across distinct normalizations/hash namespaces.'),
('MODULE_TYPE_IMPORT',['<module>'],['M00','M22'],[],
'TYPE_CHECKING chooses type-only imports versus runtime lazy-loader setup.',
'Keep module import/selector behavior unchanged; it adds no claimed model-kernel fusion or runtime MAC savings.'),
]

owner_map={}
for family,names,mods,cands,effect,transform in domains:
    for short in names:
        full=short if short.startswith('_torch_') or short.startswith('_replace_') or short.startswith('_get_gpu_') or short=='update_pass_config' or short=='<module>' else 'NPUModelRunner.'+short
        assert full not in owner_map,full
        owner_map[full]=family
owners={x['owner'] for x in d['rows']};assert set(owner_map)==owners,(owners-set(owner_map),set(owner_map)-owners)

# Current exporter is a positive config witness, not retroactive attestation of
# a prior cohort or every live Python attribute/native implementation.
metric=read(H/'current-service-config-metric-witness.json');cfg=[]
for line in metric['selected_config_info_lines']:
    labels={k:json.loads('"'+v+'"') for k,v in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"',line)};cfg.append(labels)
assert len(cfg)==2 and {x['engine'] for x in cfg}=={'0','1'}
shared={k:v for k,v in cfg[0].items() if k!='engine' and cfg[1].get(k)==v}
assert shared['kv_sharing_fast_prefill']=='False' and shared['mamba_cache_mode']=='none'
known={'self.cache_config.kv_sharing_fast_prefill':False,'self.cache_config.mamba_cache_mode':'none'}
UNKNOWN=object()
def abstract(n):
    if isinstance(n,ast.Constant):return n.value
    if isinstance(n,ast.Attribute):return known.get(ast.unparse(n),UNKNOWN)
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.Not):
        v=abstract(n.operand);return not bool(v) if v is not UNKNOWN else UNKNOWN
    if isinstance(n,ast.BoolOp):
        vs=[abstract(x) for x in n.values]
        if isinstance(n.op,ast.And):
            if any(v is not UNKNOWN and not bool(v) for v in vs):return False
            return all(bool(v) for v in vs) if all(v is not UNKNOWN for v in vs) else UNKNOWN
        if any(v is not UNKNOWN and bool(v) for v in vs):return True
        return any(bool(v) for v in vs) if all(v is not UNKNOWN for v in vs) else UNKNOWN
    if isinstance(n,ast.Compare) and len(n.ops)==1 and isinstance(n.ops[0],(ast.Eq,ast.NotEq)):
        a,b=abstract(n.left),abstract(n.comparators[0])
        if a is UNKNOWN or b is UNKNOWN:return UNKNOWN
        return a==b if isinstance(n.ops[0],ast.Eq) else a!=b
    return UNKNOWN
rows=[];bounds=[]
for r in d['rows']:
    value=UNKNOWN
    if 'self.cache_config.kv_sharing_fast_prefill' in r['condition'] or 'self.cache_config.mamba_cache_mode' in r['condition']:
        try:value=abstract(ast.parse(r['condition'],mode='eval').body)
        except SyntaxError:pass
    if value is not UNKNOWN:bounds.append({'site_id':r['site_id'],'condition':r['condition'],'source_expression_under_current_exported_config':value,'scope':'Current exported server configuration plus unchanged source selector;not historical runtime attribute/native branch attestation.'})
    rows.append({'site_id':r['site_id'],'owner':r['owner'],'line':r['line'],'family':owner_map[r['owner']],'condition':r['condition'],'enclosing_guards':r['enclosing_guards'],'analysis_state':'CONDITIONAL_SOURCE_OPERATOR_EFFECT_AND_TRANSFORM_DOMAIN_REVIEWED','positive_profile_owner_frames':r['observed_owner_frames'],'runtime_branch_taken_proved':False})
counts=collections.Counter(x['family'] for x in rows)
write('remaining900-operator-dataflow-review.json',{'status':'ALL900_EXACT_GUARDS_AND_CONTINUATION_EFFECTS_REVIEWED_IN35_DOMAINS','source_effects':str(H/'remaining900-exact-operator-effects.json'),'function_continuations':str(H/'remaining900-owner-full-effect-index.json'),'domains':[{'id':f,'owners':[x for x in owner_map if owner_map[x]==f],'operator_modules':m,'candidate_families':c,'guarded_operator_effect':e,'fusion_split_reorganization_answer_and_constraints':t,'site_count':counts[f]} for f,n,m,c,e,t in domains],'site_reviews':rows,'domain_count':len(domains),'owner_count':len(owners),'site_count':len(rows),'current_config_expression_witnesses':bounds,'analysis_limit':'Source-conditional effects and legal transformation domains reviewed,not all live flags/native operands/aliases/modes proved. Opaque named callee behavior remains governed by existing operator contracts and unchanged control flow;no silent no-op or independence inference. This one-file frontier is not the entire model graph completeness proof.','new_device_work':False,'whole_module_fusion_scan_closures':0,'overall_complete':False})
write('current-config-branch-routing.json',{'status':'CURRENT_TWO_ENGINE_POSITIVE_CONFIG_WITNESSES_ROUTED','source':str(H/'current-service-config-metric-witness.json'),'shared_labels':shared,'expression_results':bounds,'specific_conclusions':['Current exported prefix caching enabled,hash_algoSHA256,block_size128;prefix reuse keys still require the full original namespace and immutable token/parameter identity.','Current exported kv_sharing_fast_prefill=False;source shortcuts guarded by this flag receive no current-path saving credit.','Current exported mamba_cache_mode=none rules out the align/all equality checks under this configuration,not every speculative state update.','cache_dtype=auto and kv_cache_layout=None are unresolved selections,not BF16/INT8 or a physical alias/layout proof.','No config label is a memory-bandwidth bound,availablefree-memory value,or evidence of historical flag equality.'],'new_device_work':False,'overall_complete':False})
checks={'900sites_retained_exactly_once':len(rows)==900 and len({x['site_id'] for x in rows})==900,'all80owners_exactly_one_review_domain':len(owners)==80 and owners==set(owner_map),'all_guard_domains_have_effect_and_transform_answers':all(e and t for f,n,m,c,e,t in domains),'all_domain_counts_account900':sum(counts.values())==900,'two_current_engine_configs':len(cfg)==2,'no_runtime_branch_promotion':all(not x['runtime_branch_taken_proved'] for x in rows),'full_earlyreturn_continuations_present':(H/'unguarded-continuation-call-frontier.json').is_file()}
assert all(checks.values()),checks
write('remaining900-review-checks.json',{'status':'PASS','checks':checks,'site_count':len(rows),'owner_count':len(owners),'domain_count':len(domains),'actual_native_or_scope_exhaustiveness_proof':False,'overall_complete':False})
print(json.dumps({'status':'PASS','sites':len(rows),'owners':len(owners),'domains':len(domains),'current_config_expression_results':len(bounds),'overall_complete':False}))
