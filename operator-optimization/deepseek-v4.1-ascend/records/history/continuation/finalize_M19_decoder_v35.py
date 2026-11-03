from copy import deepcopy
from datetime import datetime, timezone
from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet, graph
from persist_profile_refinement import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v35.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    specs = [
        ('profile', 'pass', 'M19-profile166-v35', 'query_M19_forward_scope_v35_166.py', False,
         'All32 saved native fullcallers/typed TASKs/complete COMM under literal DSpark forward; shared body ancestry is not an exclusive M19 PMU bucket.'),
        ('profile', 'pass', 'M19-pattern166-v35', 'audit_M19_forward_patterns_v35_166.py', False,
         '104 complete repeat-to-exit source-order patterns;10/20 global and2/3 local rows,6HCpre/6HCpost each,69liveQ and19cache-only proposal patterns; retain all1560 pre-repeat TASKs.'),
        ('review', 'pass', 'M19-source166-v35', 'read_M19_decoder_and_embedding_owners_v35_166.py', True,
         'Selected decoder tail/ReplicatedQ_a/embedding TP fields/DSA forward read. Upstream AllReduce helper and base same-name methods documented but not assumed current owners.'),
        ('review', 'error', 'M19-owner166-v35', 'read_M19_actual_embedding_plan_and_attention_owners_v35_166.py', False,
         'Owner reader stopped on absent directly-declared DSpark decoder __init__; real rc1 retained, no result/target mutation.'),
        ('review', 'pass', 'M19-owner166-v35-v2', 'read_M19_actual_embedding_plan_and_attention_owners_v35_166_v2.py', True,
         'Actual embedding output_reduce_group plan and base branch contracts captured; absent-constructor expectation corrected to tiny class header/binding localization.'),
        ('profile', 'pass', 'M19-callers166-v35', 'audit_M19_exact_attention_callers_v35_166.py', True,
         'Five complete native call chains identify current CP input/output overrides and restored-head/output-projection dispatch; do not use base method contracts as active code.'),
        ('review', 'error', 'M19-CP166-v35', 'read_M19_actual_CP_overrides_v35_166.py', False,
         'Initial CP read stopped because same method name is not unique in the file; real rc1 retained. No result/target mutation.'),
        ('review', 'pass', 'M19-CP166-v35-v2', 'read_M19_actual_CP_overrides_v35_166_v2.py', True,
         'Profile frame-line192/224/1795/90 selects exact CP owners; binds query/global-cache row split, mandatory all-rank output exchange, DSpark SWA class and decoder tail.'),
    ]
    packets = {}
    for _, status, label, _, embedded, _ in specs:
        if status == 'error':
            assert read(N / (label + '.json'))['returncode'] == 1
        else:
            packets[label] = packet(label, embedded)
    p = packets['M19-profile166-v35']; a = packets['M19-pattern166-v35']
    ps, audit = p['summary'], a['summary']
    assert (ps['native_APIs'], ps['native_TASKs'], ps['native_chains'], ps['source_bound_COMM']) == (12943, 14191, 91, 1096)
    assert (audit['complete_forward_patterns'], audit['pattern_issues'], audit['covered_after_repeat_TASKs'], audit['outside_pattern_TASKs']) == (104, 0, 12631, 1560)
    assert all(x['all_same_opaque_group_name'] for x in audit['groups']) and len(audit['groups']) == 32
    regimes = audit['pattern_regimes']
    assert sum(x['patterns'] for x in regimes if x['phase'] == 'DUMMY_ANCESTRY') == 16
    assert sum(x['patterns'] for x in regimes if x['cache_only_KV_per_pattern'] == 3) == 19
    assert sum(x['patterns'] * x['sparse_attention_per_pattern'] for x in regimes) == 207
    refs = [dict(label=k, receipt=str(N/(k+'.json')), full_result=v['full_result'], sha256=v['result_sha256']) for k,v in packets.items()]
    source_files = {}
    for label in ['M19-source166-v35', 'M19-owner166-v35-v2', 'M19-CP166-v35-v2']:
        for f in packets[label]['result']['files']:
            prev = source_files.setdefault(f['path'], f['sha256']); assert prev == f['sha256']

    profile_file = 'M19-current-forward-patterns-and-dispatch-v35.json'
    contract_file = 'M19-embedding-HC-CP-row-and-consumer-contract-v35.json'
    algebra_file = 'M19-layout-ownership-and-context-query-reorganization-v35.json'
    conditions_file = 'M19-optimization-condition-ledger-v35.json'
    branches_file = 'M19-phase-and-branch-ledger-v35.json'
    graphs_file = 'M19-source-and-conditional-decoder-graphs-v35.json'
    write(O/profile_file, dict(generated_utc=stamp, evidence_grade='MEASURED_SAVED_PROFILE', profile=ps,
        source_order_audit=audit, compact_chain_headers=p['compact_chain_headers'],
        focus_typed_headers=a['focus_chain_headers'], exact_fullcallers=packets['M19-callers166-v35']['result'],
        evidence_refs=refs, population_notes=[
            '104 complete source-order patterns are88proposal/16dummy. Each has one repeat,6HCpre,6HCpost,3DispatchFFNCombine and one finalcollapse ReduceSum. This is not a Python invocation/request/pointer epoch claim.',
            '12631 selected TASKs lie within repeat-to-exit patterns;1560 pre-repeat embedding TASKs are retained. Their sum14191 is complete for this ancestry selector, not a new exclusive M19 functional-resource assignment.',
            'Proposal global repeat rows10x32 and20x56; local physical HC rows2/3. These differ from F27 context rows12/24/6400/6528/6668 and target auxiliary rows24/6400/6528/6672.',
            '69liveQ patterns have207SparseFlashMla;19cache-only patterns have57globalKV updates:12patterns N10 and7patterns N20. Empty localQ does not eliminate globalKV/output work.',
            '16dummy patterns have48positive ZerosLike at DSAforward698, binding the no-attn-metadata zero-output branch. Dummy ancestry is not proof of startup-only/removability.',
            '1096 complete COMM =104embedding AllReduce +416SP AllGather +312attention ReduceScatter +264head AllToAll; API/ComputeInfo/COMM populations remain distinct.',
            'All four communication roles have the same opaque group name within each of32DBs,rankSize8. This supports group binding, not physical topology, transport cost or new reduction numerical equality.'
        ], no_new_model_NPU_service_or_health_work=True, whole_module_closed=False, overall_complete=False))

    facts = [
        ('DOCUMENTED_PLUS_MEASURED','DSpark159 constructs VocabParallelEmbedding;249 embeds then repeats four HC channels before SP. Source258 shards hidden and259 input IDs,260 creates FP32 pre_mix zeros and261 sets column0 to1.104positive repeats and source-order first-HC entries bind the draft initialization domain; not all312attention-HCpre calls are first entries.'),
        ('DOCUMENTED','v7 sp_shard zero-pads only dim0 to TP multiple and takes TP-rank chunk, cloning a no-pad slice. Repeating channels commutes with this copy map. sp_padding_mask uses hidden only for global row count; it also preserves supplied mask values. Do not replace the supplied mask by all-false merely because rows are within N.'),
        ('DOCUMENTED_PLUS_MEASURED','Actual embedding standard path429-435 invokes torch.ops.vllm.all_reduce(output_parallel, output_reduce_group.unique_name). It does not call the newly inspected upstream communication_op helper. Constructor190 takes group from plan; STANDARD and PCPxTP token-embedding plans use get_tp_group(). Fine-grained token exchange, PCP exchange and replicated/TP1 branches remain distinct.'),
        ('DOCUMENTED','DSpark decoder class140 attention_cls=DeepseekV41DSparkAttention; that class derives DeepseekV41SWAAttention. Base decoder797 selects self.attention_cls. SWA constructor608 sets compress_ratio0. No directly declared DSpark decoder __init__ exists; absence is not absence of inherited construction.'),
        ('DOCUMENTED_PLUS_MEASURED','Active CP input owner is context_parallel/dsa_v41_cp189-192: global_metadata=_global_layer_metadata(...); start=metadata.swa.cp_token_range[0]; _preprocess(hidden_states[start:],hidden_states,metadata,global_metadata). Base dsa_v41.py347 passes identical metadata and is another branch, not the current owner.'),
        ('DOCUMENTED_PLUS_MEASURED','Base _preprocess351/352 truncates query/cache using their own num_actual_tokens. No local query updates global caches then returnsNone; current57KV rows10/20 are positive. Live Q_a rows2/3 differ from globalKV rows10/20, so naive same-M Q/KV concatenation is excluded for the existing layout.'),
        ('DOCUMENTED_PLUS_MEASURED','model397 declares Wq_a ReplicatedLinear; current BF16 outputs1280. Wkv ReplicatedLinear512 is pinned in v34; current global WKV264 is207multistream+57cache-only. Input is the already BF16-rounded input_layernorm output before SP gather, not raw HC state. Replicated declarations do not measure live parameter equality.'),
        ('DOCUMENTED_PLUS_MEASURED','Current CP output209-225 pads local attention output to metadata per_rank, restores heads using TP AllToAll, restores any DP/FlashComm suffix, then projects full hidden_states extent. _forward_o_proj1754-1799 applies active grouped Wo_a then Wo_b, writing into caller output.264Wo_a/264Wo_b positive calls include noQ ranks; full output direct publication is already implemented.'),
        ('DOCUMENTED','restore_tp_heads maps [local tokens,all heads] to [all tokens,local heads] by reshape/permute/contiguous then all_to_all_single on TP device_group. A noQ rank can receive nonzero valid-token head shards and must contribute its output weights and required ReduceScatter participation.'),
        ('DOCUMENTED_PLUS_MEASURED','Decoder900-920 does attention ReduceScatter then HCpost, retains hidden as next residual, computes FFN HCpre using attn_pre, existing dual-output RMSNormCast, MLP, HCpost, and returns hidden plus ffn_pre. Next layer needs both state and pre_mix; final collapse is a distinct consumer. Current624HCpre/624HCpost bind the shared decoder structure.'),
        ('DOCUMENTED','Existing RMSNormCast already supplies BF16 norm and FP32 widening of rounded BF16 under v29. Any post/pre/norm fusion retains that boundary, routed/shared hooks and per-layer residual, not an unrounded FP32 norm substitution.'),
        ('DOCUMENTED','Context precompute v34 and query decoder both use that layer attn.wkv, but consume different5120D inputs and position/slot metadata. Same layer weight permits a conditional row-batch packing; same position or slot alone does not make context/query KV interchangeable across acceptance or rollback.'),
        ('UNKNOWN','Exact live CP-start versus SP-owner row relation, prefix mask values, slot/position identities, buffer aliases/capture/last-reader events, native math order, all external observers and causal gain are not recorded by these selected packets.')
    ]
    write(O/contract_file, dict(generated_utc=stamp, facts=[dict(grade=g,statement=s) for g,s in facts], source_files=source_files,
        evidence_refs=refs, reuse=['moe-draft-reconciliation-v7.json','profile-raised-source-contract-v16.json','cache-only-KV-branch-reconciliation-v18.json',
            'M15-normalized-content-consumer-and-precision-contract-v31.json','M16-head-partition-postprocess-and-layout-contract-v32.json','M18-partition-bank-RoPE-and-slot-contract-v34.json'],
        local_questions_answered=['Actual embedding reduce-plan group versus SP','Draft initial repeat/premix/firstHC domain',
            'Active CP override rather than identically-named base method','Global cache-only work and mandatory output exchange on noQ ranks',
            'DSpark SWA class selection and inherited decoder post/pre/residual/premix fanout'],
        stop_rule='These local contracts are now answered. Record runtime/numerical conditions and move to M20/M21, without broad source/SDK audits or requiring an optimization implementation.',
        implementation=False,whole_module_closed=False,overall_complete=False))

    variants = [
        ('F28-v35-LOCAL-REPEAT','F28','Shard the single-channel embedding before four-way repeat; preserve global-mask input and local ID/premix alignment.',
         'S_p pad(E repeated4)=repeat4(S_p pad(E)); identical value/copy map including padding zeros. Source still needs a full initial four-channel residual at its eventual HCpost consumer.',
         'Maintain no-pad clone/custom-op alias contract, contiguous local layout, original mask values and all bank lifetimes. Logical output-byte reduction is not measured traffic or peak memory.'),
        ('F28-v35-EMBED-RS','F28','Replace full-row embedding SUM AllReduce followed by SP shard with a correctly padded SUM ReduceScatter, then local repeat.',
         'For masked vocab contributions E_r, S_p pad(sum_r E_r)=sum_r S_p pad(E_r). Current plan output reduction and SP use the TP group in the reviewed domains; required PCP/token exchange stays before this transformation.',
         'Same rank order/group and token rows, zero padding, dtype/reduction numerical contract and all embedding observers required. Padding can offset communication savings for small N.'),
        ('F28-v35-ID-MASK-PUBLISH','F28','Publish local IDs and padding mask together with existing input metadata or local embedding publication.',
         'SP IDs/mask are deterministic slices/pad of the same original global generation. A shared producer may supply those views rather than independent private pad/cast/copy stages.',
         'Preserve incoming non-None mask values, global outputs still needed by embedding/KV, rejected/anchor/DCP metadata effects and original captured banks. Linked to existing F31, not a new candidate.'),
        ('F28-v35-BROADCAST-RESIDUAL','F28','Delay materialization of identical initial HC channels through first-pre consumers and reconstruct the required residual at first-post or keep a suitable view.',
         'The first HC state is a logical four-channel broadcast of one embedding row. Layout representation can be specialized if every first-pre and first-post read is represented.',
         'Full initial residual is still required. Existing operator may require contiguous storage; new view/tiling/alias interface and all numerical paths remain conditional, overlapping local-repeat and F02.'),
        ('F28-v35-PADDING-ROWS','F28','Prune local row-local HC/FFN arithmetic for provably dead padding while retaining original collective ABI and global model-parallel work.',
         'N10 pads to16 physical rows and N20 to24 under TP8 shard; selected noQ patterns show reachable empty local query domains.',
         'NoQ does not prove an entire rank dead. Keep globalKV, head AllToAll, Wo_a/Wo_b, collective participation, MC2 uniform domains, output padding and all possible observers. No row count alone establishes safe pruning.'),
        ('F02-v35-FIRST-PRE','F02','Use the draft initial broadcast/onehot pre_mix contract to specialize first-pre y and/or coefficient input projection.',
         'Current104first entries are source-consistent initial repeats plus FP32 onehot premix. A real-algebra coefficient prefold or y shortcut can be considered only within a specified numerical domain.',
         'Do not delete data-dependent coefficients needed by HCpost/next premix. Infinity times zero, NaN payload, signed zero, accumulation and BF16/native rounding can defeat an unconditional y=E shortcut.'),
        ('F03-v35-DECODER-LINK','F03','Tile HCpost into the next HCpre/norm consumer, with explicit residual and premix outputs at both attention/FFN and layer boundaries.',
         'Current common decoder returns hidden and ffn_pre and retains post state as a later residual; producer/consumer stages may share loads or tiles while emitting required state.',
         'The complete HC residual is not dead merely because next pre consumes it. Preserve current scaling, rounded add, collective finalization, separate pre_mix and last-layer collapse. F23/F04 effects are not additive.'),
        ('F32-v35-KV-OWNER','F32','Compute512D KV/norm/RoPE at SP row owners before full5120D hidden gather; exchange KV and keep local Q_a where CP-query ownership matches.',
         'KV is a row-local replicated-weight map. Under exact SP-to-CP row ownership, global cache consumers can receive512D values while Q reads its own normalized rows.',
         'CP start/num_actual/per_rank and FlashComm suffix are real metadata, not inferred solely from ceil(N/8). Preserve global cache writes on noQ ranks, positions/slots, parameter generations and all full-hidden observers.'),
        ('F32-v35-DUAL-PROJECT','F32','Compute local BF16 Q_a1280 and KV512 before gather, optionally as a1792-output joint contraction; route/select Q_a by the CP query map and publish global KV.',
         'Both Wq_a and Wkv are declared ReplicatedLinear. A packed local row map can publish separate1280D Q_a and512D KV results instead of5120D hidden; Qnorm/quant/Q_b and each KV norm remain separate.',
         'This changes current unequal-M layout rather than claiming naive existing-input concatenation. Preserve BF16 contraction barriers, exact row mapping/duplicates, collective/new-launch costs and live parameters. Linked to F06; alternative to the512-only aligned-owner case.'),
        ('F29-v35-CONTEXT-QUERY','F29','Prepare context KV lazily per layer and pack its rows with that layer query KV rows into one same-weight WKV contraction when both inputs are ready.',
         'Context bank main_norm rows and query input_layernorm rows are different inputs to the same layer attn.wkv. Row concatenation computes the two independent maps; no input-value equality is required.',
         'Keep distinct row/position/slot metadata and512D norms/RoPE, context-before-query duplicate-write order, noncausal cache readiness and dummy/no-context domains. Changed-M tiling needs a precision contract. Joint preparation may delay the first ready layer; overlap/savings are not free or additive.')
    ]
    variant_rows = [dict(id=i,parent=c,transformation=t,argument=p,conditions=q,implementation=False,native_validation=False,critical_path_gain_us=None) for i,c,t,p,q in variants]
    logical=[]
    for Nrows,C in [(10,2),(20,3)]:
        padded=8*C;D=5120;s=2
        logical.append(dict(global_rows=Nrows,TP=8,physical_rows=padded,local_rows=C,
            original_repeat_output_bytes=Nrows*4*D*s,local_repeat_output_bytes=C*4*D*s,
            original_full_HC_pad_output_bytes=padded*4*D*s,single_channel_pad_output_bytes=padded*D*s,
            embedding_allreduce_ring_payload_bytes=2*7*Nrows*D*s//8,
            padded_embedding_reducescatter_ring_payload_bytes=7*padded*D*s//8,
            current_decoder_input_gather_logical_send_bytes=7*C*D*s,
            aligned_KV512_gather_logical_send_bytes=7*C*512*s,
            dual_Qa1280_KV512_gather_logical_send_bytes=7*C*1792*s,
            note='Illustrative ring/allgather payload convention and logical tensor output extents; not physical traffic, peak allocation, causal benefit or per-request pairing.'))
    write(O/algebra_file,dict(generated_utc=stamp,variants=variant_rows,logical_domain_examples=logical,
        invalid_shortcuts=[
            'Native N10/20 query rows are not F27 context12/24 or target aux24 rows. Never transfer row plans by equal counts.',
            'No localQ does not imply no globalKV or no output rank contribution: restore_tp_heads can deliver valid-token head shards to that rank.',
            'Existing localQ_a M2/3 and globalKV M10/20 cannot be naively concatenated as identical input domains. F32 alternatives explicitly change ownership/interfaces.',
            'Replacing full embedding AllReduce with padded ReduceScatter has payload ratio ceil(N/P)*P/(2N) under the illustrative ring convention; for N1,P8 it is4, not universally a reduction.',
            'Initial onehot premix does not make all HCpre outputs constant or prove floating-point y identity for every special value.',
            'Context and query KV share a layer weight but differ in hidden-state provenance. Equal slot/position does not justify reusing one for the other across accepted/rolled-back iterations.',
            'Current output path already writes projected result into the caller buffer. Do not count a hypothetical second full hidden output copy as a new removal.'
        ],relationships=[
            'F28 local repeat, ReduceScatter/local embedding and broadcast residual overlap the same repeat/pad materialization; jointly count once.',
            'F02 draft first-pre specialization and F28 broadcast representation are dependent variants, not independent full-repeat savings.',
            'F32 aligned512-only and dual1792 projected exchange are ownership alternatives. Nominal feature ratios10 and20/7 are not speedups.',
            'Context/query WKV row packing can compose only after resolving row ownership/epochs; keeping full-hidden gather removes the corresponding F32 communication claim.',
            'F03 decoder boundary composition retains F23 merge/finalization and existing F04 dual-output rounding; avoid double-counting their intermediates.',
            'Padding row pruning preserves all collective participation and global model-parallel work; zero-query branch is already implemented.'
        ], evidence_grade='DERIVED_CONDITIONAL_ALGEBRA',implementation=False,whole_module_closed=False,overall_complete=False))

    requirements=[
        ('GROUP','Embedding output_reduce_group and SP group use the same required ranks/order and per-rank token map; keep PCP/token exchange and different fine-grained branches.'),
        ('VOCAB','Preserve masked original/added vocabulary ownership, local embedding primitive, valid IDs and invalid/padded rows.'),
        ('REDUCTION','SUM dtype/rounding/NaN/signed-zero contract and backend numerical equivalence for AllReduce-to-ReduceScatter.'),
        ('PADPAYLOAD','Use true padded extent before ReduceScatter; more padding can offset payload reduction, especially tiny N.'),
        ('SPCOPY','Retain zero-pad rank slice, no-pad clone/custom-op functional alias rules and required contiguous output.'),
        ('MASK','Original global mask values and generation preserved; hidden argument supplies N only, not mask truth values.'),
        ('IDMETA','Local IDs and mask publication preserve all global input/query/position/slot/token-index outputs and rejection/anchor/DCP effects.'),
        ('FIRSTENTRY','Initial same-value four-channel state and onehot FP32 premix apply only to the first of six HCpre stages per source-order pattern.'),
        ('SPECIALMATH','First-y bypass or coefficient prefold requires explicit finite/special-value/signed-zero/FP32/native reduction and BF16 rounding contract.'),
        ('RESIDUAL','First HCpost and later residual readers still receive the full equivalent four-channel state; logical broadcast does not imply dead storage.'),
        ('COEFFICIENTS','HCpre data-dependent coefficients and next pre_mix remain available; do not delete whole HCpre from a y shortcut.'),
        ('POSTPRE','Cross-boundary composition emits both state and ffn_pre/attn_pre, preserves residual lifetimes and last-layer collapse.'),
        ('NORMBARRIER','Input norm and existing RMSNormCast keep BF16 output and FP32 widening of rounded BF16, not unrounded norm substitution.'),
        ('PARAMS','Wq_a/Wkv/norm/RoPE intended replicas, exact parameter versions and native math retained; declaration is not live equality attestation.'),
        ('CPMAP','Exact CP start/num_actual/per_rank versus SP-owned row mapping, including FlashComm/DP suffix, known or explicitly routed.'),
        ('QKVABI','Q_a1280 and KV512 output barriers/row plans/duplicates and Qnorm/quant/Q_b interface preserved when changing M/layout.'),
        ('KVGLOBAL','Keep global_metadata cache rows/positions/slots and required KV writes even on zero-query ranks.'),
        ('COLLECTIVES','Every required group participant, send/receive count, padding and uniform MC2/DP collective domain remains.'),
        ('OUTPUTHEADS','Keep local-zero padding, head AllToAll, suffix restore, grouped Wo_a/Wo_b and ReduceScatter contributions for all valid global tokens.'),
        ('CACHEWRITE','Preserve cache dtype/layout/negative skip/duplicate-write order and global current-query publication.'),
        ('CTXPROVENANCE','Context main_norm and query input_layernorm rows have separate values/epochs despite shared layer WKV; no same-position reuse assumption.'),
        ('CTXPACK','Same-layer joint WKV row pack preserves separate512norm/RoPE/slot metadata and changed-M numerical contract.'),
        ('CTXORDER','Lazy context schedule preserves context-before-query conflicting cache writes and required noncausal same-request readiness.'),
        ('DUMMY','Preserve actual no-attn-metadata zero output and dummy/profile allocation/communication purposes; dummy ancestry is not startup-only.'),
        ('BRANCHES','No-SP/TP1/PCP/fine-grained/quantized-output/base nonCP/compressed roles remain separate domains from selected SWA CP path.'),
        ('PRUNING','Padding math pruning needs proof of no live row observer and exact sentinel outputs, not shape or noQ count alone.'),
        ('BANKABI','All captured/input/output/premix/residual bank addresses, strides, aliases, readiness and last-reader epochs preserved.'),
        ('EPOCH','Cross-iteration keys include request/context-vs-query provenance/accepted-rollback/slot/position/weight/config generations.'),
        ('COST','Price new launches, packing, collectives, buffering and interference jointly; counts/logical bytes are not time or speedup.'),
        ('OBSERVERS','Keep external hooks, all raw/local/global state observers and valid-row consumers unless independently closed.')]
    write(O/conditions_file,dict(generated_utc=stamp,conditions=[dict(id='M19-v35-'+k,requirement=v,actual_runtime_values='UNKNOWN_UNLESS_EXPLICIT_SOURCE_FACT',current_legality_proved=False)for k,v in requirements],implementation=False,whole_module_closed=False,overall_complete=False))
    branch_rows=[
        ('Proposal liveQ','MEASURED','69patterns/207attention executions; Q_a local2/3 versus KV global10/20.'),
        ('Proposal cache-only','MEASURED_PLUS_DOCUMENTED','19patterns/57globalKV; CP local query slice is empty but output exchange and global cache participation remain.'),
        ('Dummy no-attn-metadata','MEASURED_PLUS_DOCUMENTED','16patterns/48ZerosLike at forward698; still6HCpre/6HCpost and3MoE per pattern.'),
        ('Active CP vs base','MEASURED_PLUS_DOCUMENTED','Actual CP192/224 dispatch bound by fullcaller; base347/432 remains another documented branch.'),
        ('Global/SP/CP row domains','SCOPED_MEASURED_ACTUAL_MAP_UNKNOWN','Global repeat N10/20, local physical C2/3, live query R2/3 and cache N10/20 distinct; CPstart exact values not recorded.'),
        ('Embedding plan branches','DOCUMENTED','Replicated/disableTP, STANDARD, PCPxTP, fine-grained token/head plans and PCP+fine-grained errors preserved.'),
        ('Initial vs later HC','SOURCE_CONSISTENT_BOUND','104initial entries, then later decoder-state/premix inputs;624pre/624post total. First broadcast/onehot not projected onto all entries.'),
        ('SWA vs compressed role','DOCUMENTED_CURRENT_CLASS_BOUND','DSpark attention derives SWA, whose constructor sets compress_ratio0; this source class binding is not a live-field census. Preserve any full-hidden observers in new protocols; targetC1/C2/indexer paths remain separate.'),
        ('Output quantization branches','DOCUMENTED_CURRENT_BF16_PATH_MEASURED','Current TransposeBatchMatMul branch plus Wo_b positive; FP8 weight-scale branch remains separate source domain.'),
        ('NoSP/TP1/empty/PCP/future workload','CONDITIONAL_OUTSIDE_SELECTED_DOMAIN','No selected absence converts a future/current-unrecorded guard to false.'),
        ('Context/query/lazy/rollback','CONDITIONAL','Shared WKV permits row packing, not KV value identity; slot/order/readiness and input provenance remain.')]
    write(O/branches_file,dict(generated_utc=stamp,branches=[dict(branch=b,state=s,evidence=e,whole_branch_closed=False)for b,s,e in branch_rows],stop_rule='Point-local group/CP/current decoder contracts answered. Advance M20/M21; do not audit SDK/native implementation merely to register conditional analysis.',whole_module_closed=False,overall_complete=False))

    baseline=graph(['masked_vocab_parts','embedding_AR','global_repeat','SP_hidden','SP_mask_IDs','initial_premix','HCpre','norm_BF16','SP_input_gather','CPquery_slice','Qa_Qb','global_KV','attention','local_output_pad','head_A2A','Wo_a_Wo_b','SP_output_RS','HCpost','FFN_pre_norm_MoE_post','next_layer_state_premix','collapse_exit_gather'],
        [('masked_vocab_parts','embedding_AR','SUM'),('embedding_AR','global_repeat','BF16'),('global_repeat','SP_hidden','PAD_SHARD'),('SP_hidden','HCpre','FOUR_CHANNEL_STATE'),('initial_premix','HCpre','ONEHOT_FIRST_ONLY'),('SP_mask_IDs','FFN_pre_norm_MoE_post','ROW_METADATA'),('HCpre','norm_BF16','BF16_BARRIER'),('norm_BF16','SP_input_gather','LOCAL_NORMALIZED_ROWS'),('SP_input_gather','CPquery_slice','METADATA_START'),('CPquery_slice','Qa_Qb','LOCAL_VALID_QUERY'),('SP_input_gather','global_KV','GLOBAL_CACHE_ROWS'),('global_KV','attention','CACHE_READY'),('Qa_Qb','attention','QUERY'),('attention','local_output_pad','LOCAL_VALID_RESULT'),('local_output_pad','head_A2A','FIXED_LOCAL_EXTENT'),('head_A2A','Wo_a_Wo_b','GLOBAL_TOKENS_LOCAL_HEADS'),('Wo_a_Wo_b','SP_output_RS','PARTIAL_OUTPUT'),('SP_output_RS','HCpost','LOCAL_RESULT'),('SP_hidden','HCpost','RESIDUAL'),('HCpost','FFN_pre_norm_MoE_post','STATE_AND_PREMIX'),('FFN_pre_norm_MoE_post','next_layer_state_premix','BOTH_OUTPUTS'),('next_layer_state_premix','collapse_exit_gather','LAST_LAYER_STATE')])
    owner=graph(['local_norm_BF16','local_Qa1280_KV512','projected_row_exchange','CPquery_selection','global_KV_cache','Qnorm_quant_Qb','attention_and_output_pipeline'],
        [('local_norm_BF16','local_Qa1280_KV512','REPLICATED_WEIGHT_MAPS'),('local_Qa1280_KV512','projected_row_exchange','BF16_BARRIERS'),('projected_row_exchange','CPquery_selection','EXACT_CP_ROW_MAP'),('projected_row_exchange','global_KV_cache','GLOBAL_KV_PUBLICATION'),('CPquery_selection','Qnorm_quant_Qb','Q_A_ROWS'),('global_KV_cache','attention_and_output_pipeline','CACHE_READY'),('Qnorm_quant_Qb','attention_and_output_pipeline','QUERY')])
    packed=graph(['context_bank_rows','current_query_norm_rows','same_layer_WKV_pack','separate512norm_and_RoPE_maps','context_cache_writes','query_cache_writes','same_layer_attention'],
        [('context_bank_rows','same_layer_WKV_pack','CONTEXT_PROVENANCE'),('current_query_norm_rows','same_layer_WKV_pack','QUERY_PROVENANCE'),('same_layer_WKV_pack','separate512norm_and_RoPE_maps','BF16_PER_ROW'),('separate512norm_and_RoPE_maps','context_cache_writes','CONTEXT_SLOTS'),('separate512norm_and_RoPE_maps','query_cache_writes','QUERY_SLOTS'),('context_cache_writes','query_cache_writes','PRESERVE_CONFLICT_ORDER'),('query_cache_writes','same_layer_attention','NONCAUSAL_READY')])
    write(O/graphs_file,dict(generated_utc=stamp,source_consistent_liveQ_baseline=baseline,conditional_projected_exchange=owner,conditional_context_query_pack=packed,
        branch_note='LiveQ skeleton only: cache-only replaces local Q/attention with empty output preparation while retaining global KV/output collectives. Dummy follows zero-metadata branch. Graphs are not new schedules or fabricated invocation pairings.',
        graphs_acyclic=True,implementation=False,whole_module_closed=False,overall_complete=False))

    evidence=[profile_file,contract_file,algebra_file,conditions_file,branches_file,graphs_file]
    common=dict(contract=str(O/contract_file),algebra=str(O/algebra_file),conditions=str(O/conditions_file),graphs=str(O/graphs_file),implementation=False,new_performance_result=False)
    candidate_updates={}
    for cid in ['F28','F02','F03','F32','F29']:
        vv=[v for v in variant_rows if v['parent']==cid]
        candidate_updates[cid]=dict(review_v35=dict(common,variants=vv),variants_v35=vv,target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v35.json'),
            profile_review_state='M19_ALLRANK_PATTERNS_AND_ACTUAL_CP_GROUP_CONSUMER_CONTRACTS_REVIEWED;CONDITIONAL_UNPRICED',
            conclusion_changing_gap='Actual row maps/mask values/alias/capture/parameter epochs and native numerical/causal-gain conditions remain; current base-versus-override and group/consumer local questions are answered.',
            next_profile_or_contract_action='Advance M20/M21 finite profile coverage; do not repeat whole SDK/model analysis merely for an already conditional opportunity.',all_consumers_closed=False)
    candidate_updates['F06']=dict(review_v35=dict(common,relation='F32 local1792-output Qa/KV contraction is conditional after explicitly changing unequal current row domains; two separate BF16 outputs and norms remain. Not naive same-M current concat.'))
    candidate_updates['F31']=dict(review_v35=dict(common,relation='F28 local ID/mask publication can share the existing metadata producer only with incoming-mask/global-output/rejection/anchor/DCP/capture effects retained.'))
    findings={
        'M00':'DSpark attention_cls140 and base797 bind inherited SWA attention class; active CP overrides resolved from actual fullcallers, not same-name base methods.',
        'M04':'Global query10/20, local physical2/3 and context12/24/long domains distinct. Local ID/mask generation preserves supplied mask truth values and global metadata outputs.',
        'M05':'104 masked-vocab embedding AllReduce/repeat entries; actual output reduction from plan.get_tp_group, PCP/fine-grained/TP1 branches retained.',
        'M06':'624HCpre/624HCpost;104source-consistent initial broadcast/onehot entries and explicit later residual/premix fanout. F02/F03 refine without deleting full state.',
        'M07':'Qa ReplicatedLinear1280,KV512; current local query/global cache M domains differ. F32 projected exchange has512-only or1792dual-route alternatives, preserving BF16 norm/projection barriers.',
        'M10':'57cache-only globalKV updates and207liveQ attention executions separated. Same-position context/query inputs are different provenance, not interchangeable cached values.',
        'M11':'Active CP pad/headAllToAll/suffix restore/Wo_a/Wo_b/outputRS retained even noQ; existing direct caller-output publication not counted as new copy removal.',
        'M12':'Padding-row pruning does not remove uniform MC2/collective participation or model-parallel work for valid global tokens; existing F04/F23 semantics reused.',
        'M18':'Same-layer context/query WKV row packing and lazy preparation added conditionally; distinct main_norm vs query input_norm rows, slots, rounding and context-before-query write order retained.',
        'M19':'104complete forward patterns,88proposal/16dummy,69liveQ/19cache-only,12631inside+1560pre-repeatTASK.10conditional variants/30conditions across existing candidates; no whole-module closure.',
        'M21':'CPstart/num_actual/per_rank and DP/FlashComm suffix are exact row-map obligations. Incoming global mask values and metadata reject/anchor/DCP outputs not inferred from row counts.',
        'M22':'SP no-pad clone/alias, full residual/premix banks and CP output buffer lifetimes retained. New views/packing/exchange require explicit capture readiness/last-reader protocol.',
        'M23':'1096COMM and32same-opaque-group audits bound; actual embedding reduce plan vsSPgetTP contract answered. Logical payload ratios are illustrative, not transport speedup.',
        'M24':'Context/query hidden provenance plus accept/rollback/position/slot/weight/config epochs required for cross-iteration reuse; no pointer/position-only KV identity.'}
    module_updates={mid:dict(finding=text,evidence=evidence,candidates=['F28','F32','F29','F02','F03']+(['F31']if mid in ['M04','M21']else[]),gap='Finite local profile/dispatch/group/consumer questions answered; remaining whole-scope branches and actual-value/native-precision/causal-gain claims are distinct obligations.')for mid,text in findings.items()}
    remaining=[s for s in deepcopy(read(O/'module-coverage-gaps-v34.json')['remaining'])if not s.startswith('NextwholecoveragefrontierM19')]
    remaining.insert(12,'M19v35localcontractsanswered:104completeforwardpatterns/14191ancestryTASK,actualCP192/224row split/outputalltoall,embeddingreduce-planTPgroupandDSparkSWAclass,initial104HCentriesanddecoderresidual/premixfanout.19noQpatternsretain57globalKVandalloutputcollectives;10conditionalexisting-candidatevariants/30conditionsregistered. Base347/432notactiveowner;do notrepeatthemasactualorreadSDKforconditionalregistration. LiveCP-SPmap/mask/alias/epoch/nativeprecision/causalgainremainconditional.')
    remaining.insert(13,'NextwholecoveragefrontierM20/F30Markovconfidence/dynamic/paralleloutputsandM21metadata. Startallrankfullcallers/typedTASK/COMM plus saved frontend/requestflags;reusev32head/Markov/v33probability/v34context/v35decodercontracts. Reviewconfidence readers/step-boundary/empty-ragged/accepted-rejectedmetadata only where profile cannot decide;do notreplaywholeproposer/model/source. ThencontinueM22-M25andfiniteglobalcoverageaudit;35candidateinventoryisnotwhole-goalcompletion.')
    summary=dict(filename='M19-decoder-ownership-and-HC-summary-v35.json',previous_turn_classification='PROGRESS:v34aux/contextpartition/publisher/RoPEcontracts',
        native_APIs=12943,native_TASKs=14191,native_chains=91,source_bound_COMM=1096,complete_forward_patterns=104,proposal_patterns=88,dummy_patterns=16,
        liveQ_patterns=69,cache_only_patterns=19,cache_only_globalKV=57,live_sparse_attention=207,pattern_issues=0,first_entry_source_consistent=104,
        refined_model_candidates=['F28','F02','F03','F32','F29'],related_model_candidates=['F06','F31'],new_model_candidates=[],conditional_variants=len(variant_rows),condition_rows=len(requirements),
        remote_jobs=8,remote_pass=6,remote_retained_errors=2,
        remote_job_status='8remoteCPUjobs terminal:6pass/2retained extraction-guard errors, each corrected in a new explicitly changed reader. One local preparation failure chain retained separately. Main semantic review passes; no model/torch/NPU/service/new-health work.',
        evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report='''本轮按profile→定点源码推进M19草稿decoder。32个保存分片中，DSpark forward ancestry覆盖12943 API、14191 TASK、91条链和1096条完整COMM。已核对104条从repeat到出口gather的完整源码顺序模式，均有6次HCpre、6次HCpost和3次MoE；88条proposal、16条dummy。模式内12631 TASK，模式前embedding1560 TASK完整保留；这些数量不替换v20的功能PMU分桶，也不证明Python invocation、请求或alias代际。

query全局行是10或20，SP物理本地行2或3，不能套用上一轮context12/24或辅助状态24的行计划。69条proposal模式实际运行207次SparseFlashMla；19条本卡无query模式仍有57次全局KV准备，分别36次N10、21次N20。dummy的48次forward698 ZerosLike与无attention metadata的源码返回分支相符，不能称为只在启动发生或可直接删除。

沿完整调用栈纠正了基类归属。当前实际输入准备是context_parallel/dsa_v41_cp192：query按cp_token_range.start切片并用本地metadata截断，cache用全局hidden/global_metadata。当前输出是该文件224，先补本地per_rank、TP head AllToAll，再补DP/FlashComm后缀，执行Wo_a/Wo_b。无本卡query的rank仍可接收到其他有效token的head分片，所以不能跳过其输出权重计算和collective。源码已有直接写caller输出buffer的实现，不另计虚构copy收益。相同名字的base347/432仅保留为其他分支合同。

embedding AllReduce与SP在32个DB内使用相同opaque group，rankSize8。实际standard归约来自output_reduce_group，不走另读的upstream all_reduce小wrapper。plan的STANDARD和PCPxTP token embedding归约组均为get_tp_group；PCP/token exchange、fine-grained、TP1等分支保留。这样补齐了F28原先的局部分组问题。先shard单通道embedding再repeat的复制映射成立；用带正确padding的ReduceScatter代替AllReduce再shard也是条件代数重组。通信逻辑量随padding变化：N10时示例比率0.8，N20为0.6，N1/TP8反而为4。都不是实测网络流量或加速比。mask的hidden参数只提供N，已有mask内容仍必须保留，不能把全前缀默认设False。

当前104个首HC入口已与repeat及onehot premix源码关系绑定；其余HC入口不是同一初始域。F02首pre y/系数投影特化和F28逻辑广播表示仍保留特殊值、舍入和完整首post residual要求。decoder尾部明确返回hidden与ffn_pre，并把post状态作为后续residual。F03跨post/pre/norm重组可共享tile/读取，但不能把完整HC状态视为死数据；已有RMSNormCast FP32平面仍是舍入BF16的widening。

F32新增当前草稿域的两种所有权方案：在精确SP→CP query行关系下，本地算KV512后交换；或者本地先算Q_a1280和KV512，再按CP行计划选择/路由投影结果。后者可研究1792输出的联合GEMM，但它显式改变现有Q局部M与KV全局M的布局，绝不是把当前两次GEMM直接concat。复制权重声明不是实测参数相等，CPstart/num_actual/per_rank与SP chunk的值也仍需保留为条件。各自BF16、norm、RoPE、cache写及无query rank的职责不能省。

另补F29跨context/query的同层WKV行打包和按层延后准备。两种输入来自main_norm与query input_norm，虽然使用同层attn.wkv，也不是同一hidden值；可以条件地拼接行批次，不能仅因slot/position相同就复用KV。必须保留不同positions/slot计划、512维norm、BF16边界、context先于query的冲突写顺序，以及同请求非因果attention的就绪要求。联合和流水、F28多种repeat/pad消除都重叠，未相加计收益。

本轮细化5个既有主候选并关联F06/F31，共10个条件变体、30项约束、分支台账和3张关系图。8次远端CPU作业6pass、2次提取guard错误保留：先是类无直接__init__，再是文件内同名方法不唯一；分别改为小class绑定定位和profile确切frame行选择，均完成取证。本地快照缺文件/缺task的准备链及一次诊断打印语法错误另保留，未做远端操作；主审通过。没有导入模型/torch、启动NPU、改服务或做新健康检查，最后健康仍v12post。35项模型候选、7项框架候选和0整模块关闭保持；26项理论比较INCOMPLETE、3136candidate资源行、849PMU缺失、16个phase-unbound图不变。下一批M20置信度/step消费者及M21元数据，继续充分使用profile。整体Goal active。
'''
    jobs=[dict(kind=kind,status=status,description=desc,artifacts=[N/script,N/(label+'.json')])for kind,status,label,script,_,desc in specs]
    jobs.append(dict(kind='review',status='error',description='Local preparation chain stopped on a missing selected frozen-source file and task-not-found; diagnostic retry had syntax typo. No remote execution. Retained guard record; corrected preparation reuses prior SHA and reads exact stage source.',artifacts=[N/'M19-source-preparation-guard-v35.json',N/'prepare_M19_source_v35.py']))
    jobs.append(dict(kind='review',status='pass',description='Main review binds current CP/group/decoder consumers, derives10conditional refinements/30conditions and excludes noQ/global-work, first-HC/special-value and same-slot context-query shortcuts; no implementation/gain.',artifacts=[O/q for q in evidence]))
    action=('继续Loop004，v35实质进展。先读M19-decoder-ownership-and-HC-summary-v35.json、M19-embedding-HC-CP-row-and-consumer-contract-v35.json、M19-layout-ownership-and-context-query-reorganization-v35.json、module-coverage-gaps-v35.json。'
        'M19全32保存分片12943API/14191TASK/91chains/1096COMM，104完整source-order模式=88proposal+16dummy，每条6HCpre6HCpost3MoE，模式内12631+pre-repeat1560全覆盖，不是request/alias epoch或v20新分桶。'
        'Query N10x32/N20x56与context12/24/long不同，SP物理C2/3；69liveQ->207attention，19noQ->57globalKV(36N10+21N20)，dummy48ZerosLike@dsa698。不得noQ整rank跳过或称dummy只启动。'
        '实际CP dsa_v41_cp192用hidden[start:]本地metadata和fullhidden/globalmetadata；base347/432不是当前owner。CP209-225补per_rank→restore_tp_heads AllToAll→补FlashComm suffix→Wo_a/Wo_b；无Q仍需全局KV/输出贡献。DSpark140attention_cls→DSparkAttention继承SWA，model797类选择，608compress0。'
        'Embedding实际429-435 torch.ops.all_reduce(output_reduce_group.unique_name)，plan STANDARD/PCPxTP输出getTP；非upstreamwrapper12。32DB所有角色sameopaquegroup TP8支持但非物理拓扑/数值证明。SP前repeat条件复制identity，maskhidden只用N但保留输入mask内容。RS逻辑payloadratio Npad/(2N)，N10=.8,N20=.6,N1TP8=4，不是普遍省通信/性能。'
        'F02首104重复/onehot绑定，y bypass/weight fold保留inf*0/NaN/zero/舍入及数据依赖coeff，fullresidual不死。F03tail900-920保留hidden+ffn_pre/residual，RmsNormCast FP32是BF16widening。'
        'F32两方案：精确SP-CP对齐本地KV512交换；或本地Qa1280+KV512可joint1792再按CP路由，改变当前localQ M2/3 vsKVglobal10/20域，非naiveconcat。复制参数/CPstart/numactual/perrank/positionslot/allobservers/新numeric和成本条件；keep所有noQ全局cache/collectives/WO工作。'
        'F29新增同layer context/query WKV rowpack/lazy准备，同weight但hidden来自mainnorm与queryinputnorm不同，保留各自positions/slot/512norm/BF16/context-before-querywrite/noncausalready；不可sameposition复用。'
        '8remote6pass2提取guarderror保留并已修正，另localprepguard链1error+mainreviewpass，共10runs7pass3error；35model7framework0wholemodule，10variants30conditions，无实现/收益/NPU/model/torch/service/newhealth,lastv12post，26theoryINCOMPLETE/3136candidate资源/849missingPMU/16graphunbound不变。'
        '下一批M20/F30 confidence/dynamic/parallel step输出和M21metadata，从allrankprofile/typedTASK/COMM和savedfrontendflag入，复用v32/v33/v34/v35，不重读整proposer/模型/SDK。随后M22-M25和全范围有限coverageaudit；Goalactive。')
    persist_refinement(version=35,previous=34,title='v35 M19 current decoder row ownership and HC/CP consumers',summary=summary,
        changes=['104complete source-order forward patterns and all14191 ancestry TASKs retained with69liveQ/19cache-only/16dummy domains.',
            'Actual CP frame-line dispatch, embedding reduce-planTP group, SWA class and decoder residual/premix fanout bound; base-method scope corrected.',
            '10conditional existing-candidate refinements/30conditions and logical payload counterexamples registered without implementation/performance/whole-module closure.'],
        candidate_updates=candidate_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[contract_file,algebra_file,conditions_file],publication_rows={
            'F28':'| F28 | DSpark embedding、HC 初始化与 decoder 布局 | 条件 | v35绑定104完整forward模式及TP归约来源；先shard后repeat、正确padding的ReduceScatter、本地ID/mask发布与广播residual表示有条件。19组noQ仍需globalKV/输出交换；mask输入值、首post residual和统一collective域保留，逻辑字节不是收益。 |',
            'F32':'| F32 | SP 输入 gather 与 Q_a/KV 所有权重组 | 条件 | v35实际CP192区分本地query与全局cache；草稿当前Q M2/3、KV M10/20。精确行映射下可研究KV512先算后汇聚，或Qa1280+KV512联合本地投影/1792维交换。保持BF16/CP metadata/参数代际及无Q rank的globalKV和Wo贡献，非现有同M直接concat。 |'},
        gate_updates=dict(status='PROFILE_AND_NECESSARY_POINT_CONTRACTS_REVIEWED',answered_by=evidence,
            closed_scope='Current forward populations/initial HC binding/embedding reduce-planTP source/actual CP input-output dispatch/SWA class/decoder residual-premix consumer graph.',
            not_closed='Live CP-SP maps/masks/slots/aliases/epochs/all external observers/native numerical and causal gain/other workload branches or whole module.',
            stop_rule='Move to M20/M21 profile coverage. Do not read SDK or implement candidates merely to establish conditional discovery.'),
        scripts=[N/script for _,_,_,script,_,_ in specs]+[N/'prepare_M19_source_v35.py'],executor_prefix='zcode-M19-',main_script=N/'finalize_M19_decoder_v35.py')


if __name__ == '__main__':
    main()
