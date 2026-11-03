from copy import deepcopy
from datetime import datetime, timezone
import json

from finalize_origin_coverage_v9 import N, O, R, read, write
from finalize_Engram_whole_frontier_v30 import packet, graph
from persist_profile_refinement import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v34.json').exists(), 'refuse duplicate finalization'
    stamp = datetime.now(timezone.utc).isoformat()
    specifications = [
        ('profile', 'M18-profile166-v34', 'query_M18_aux_context_v34_166.py', False,
         'All32 native fullcallers/typed TASKs/complete COMM and 80 graph auxiliary gaps; retain ancestry and functional PMU populations separately.'),
        ('review', 'M18-source166-v34', 'read_M18_context_partition_and_consumers_v34_166.py', True,
         '169 selected source lines bind column-sharded main projection, full norm, context handoff and existing RoPE query interfaces.'),
        ('review', 'M18-fields166-v34', 'read_M18_context_fields_and_rope_proxy_v34_166.py', True,
         'Follow object fields after Name-only zero matches; bind inherited bank reader, metadata publication, readonly Proxy selection and rotary copy effect.'),
        ('profile', 'M18-audit166-v34', 'audit_M18_rows_cycles_and_rope_queries_v34_166.py', False,
         'Exact target-root containment and 312 complete same-thread source-order KV patterns; no equal-count/request/pointer pairing.'),
        ('profile', 'M18-publication166-v34', 'query_M18_context_publication_v34_166.py', False,
         'Exact DFlash base122/132 profile query validly returns zero; retained as scope correction, not evidence of absent publication.'),
        ('review', 'M18-bank-source166-v34', 'read_M18_context_bank_lifetime_v34_166.py', True,
         'Exact inherited persistent-context field uses, DSpark forwarding wrapper and ReplicatedLinear KV owner assignments; base implementation not assumed active.'),
        ('profile', 'M18-actual-publication166-v34', 'query_M18_actual_DSpark_publication_v34_166.py', False,
         'Actual DSpark firstpass: 480 ComputeInfo TASKs/7 chains, including 24 positive TensorMove and 88 metadata kernels; no missing-copy inference.'),
        ('review', 'M18-actual-bank166-v34', 'read_M18_actual_DSpark_bank_v34_166.py', True,
         'Only actual DSpark subclass persistent-context fields/guards/class bases: source prefix bank copy without enclosing guard, per-layer group slot list and inherited consumer.'),
    ]
    packets = {label: packet(label, embedded) for _, label, _, embedded, _ in specifications}
    p = packets['M18-profile166-v34']; a = packets['M18-audit166-v34']
    pub = packets['M18-actual-publication166-v34']
    ps, audit, pubs = p['summary'], a['summary'], pub['summary']
    assert (ps['native_APIs'], ps['native_TASKs'], ps['native_chains'], ps['source_bound_COMM']) == (6632, 10280, 33, 160)
    assert (ps['graph_aux_segments'], ps['graph_aux_types']) == (240, {'ReduceMean': 240, 'hcom_allGather_': 240})
    assert (audit['source_order_complete_KV_cycles'], audit['covered_F29_TASKs'], audit['unclosed_cycle_anchors'], audit['unassigned_F29_TASKs']) == (312, 9360, 0, 0)
    assert audit['F27_row_regimes'] == {'6400': 8, '12': 16, '24': 48, '6528': 8, '6668': 8}
    assert (pubs['native_APIs'], pubs['native_TASKs'], pubs['native_chains']) == (480, 480, 7)
    assert packets['M18-publication166-v34']['summary']['native_TASKs'] == 0
    type_counts = {row['op']: row['TASKs'] for row in pubs['roles']}
    assert type_counts['TensorMove'] == 24 and type_counts['copy_and_expand_dflash_and_dspark_inputs_kernel'] == 88
    refs = [dict(label=label, receipt=str(N / (label + '.json')), full_result=x['full_result'], sha256=x['result_sha256']) for label, x in packets.items()]
    source_files = {}
    source_scopes = []
    for _, label, _, embedded, _ in specifications:
        if not embedded:
            continue
        for f in packets[label]['result']['files']:
            old = source_files.setdefault(f['path'], f['sha256']); assert old == f['sha256']
            source_scopes.append(dict(receipt=label, path=f['path'], sha256=f['sha256'],
                                      functions=[{k:q[k] for k in ['name','start','end','mode','defined_here','locations'] if k in q} for q in f.get('functions', [])],
                                      selected_statement_lines=[q['start'] for q in f.get('statements', [])]))

    profile_file = 'M18-current-aux-context-and-publication-profile-v34.json'
    contract_file = 'M18-partition-bank-RoPE-and-slot-contract-v34.json'
    algebra_file = 'F27-F29-row-feature-and-KV-reorganization-v34.json'
    condition_file = 'M18-optimization-condition-ledger-v34.json'
    branch_file = 'M18-phase-and-branch-ledger-v34.json'
    graph_file = 'M18-source-and-conditional-reorganization-graphs-v34.json'
    write(O / profile_file, dict(generated_utc=stamp, evidence_grade='MEASURED_SAVED_PROFILE', profile=ps,
        root_and_source_order_audit=audit, actual_DSpark_firstpass=pubs,
        native_chain_headers=p['native_chain_headers'], firstpass_chain_headers=pub['chain_headers'], evidence_refs=refs,
        populations=dict(native_selected_TASKs=10280, graph_aux_compute_TASKs=480, selected_ancestry_TASKs=10760,
                         separate_actual_firstpass_ComputeInfo_TASKs=480, prior_v20_M18_functional_PMU_bucket=9936),
        hierarchy_notes=[
            'F27 has 88 proposal GEMM/Transpose/full5120RmsNorm pipelines; no dummy F27 in this selector. 88 complete COMM records differ from 64 ComputeInfo hcom executions.',
            'F29 has 264 proposal and 48 dummy KV cycles; each has 30 TASKs. The source-order pattern covers all 9360 selected TASKs, but is not a Python invocation, request, pointer or slot-generation identifier.',
            'Target native auxiliary has 72 means and 72 COMM; graph auxiliary has 240 means +240 hcom and240 COMM. Do not merge ComputeInfo and COMM counts.',
            '64 uniquely root-bound graphs account for192 auxiliary segments;16 unbound graphs account for48. Native target mean roots and F27 proposal cohorts must not be paired by equal counts or nearest time.',
            '24 positive TensorMove are the long rows6400/6528/6668,8 each. The source bank copy is not proved absent for small rows: this is a ComputeInfo/caller selection, not an exhaustive memcpy/API census.',
            '10760 ancestry-selected compute TASKs and480 firstpass compute TASKs do not replace v20 functional M18=9936, and their sums are not latency/gain/physical traffic.'
        ], no_new_model_NPU_service_or_health_work=True, whole_module_closed=False, overall_complete=False))

    facts = [
        ('DOCUMENTED_PLUS_MEASURED', 'main_proj is ColumnParallelLinear(15360,5120,gather_output=True,bias=False). Native output is BF16[M,640], TP8 gather layout [8,M,640], Transpose [M,8,640], then full5120D RMSNorm. first_layer.main_proj/main_norm alias the same objects, not separate weights.'),
        ('MEASURED', 'Three target auxiliary states are HC means. Long local rows800/816/834 gather to6400/6528/6672; graph local3 gathers to24. Indexed auxiliary domains24->12,24->24,6672->6668 are positive. Equal row counts do not prove identity row plans or no duplicates.'),
        ('DOCUMENTED', '_propose invokes combine_hidden_states then set_inputs_first_pass. Active AscendDSparkProposer inherits AscendDflashProposer but overrides the firstpass setter. DFlash base122/132 has zero selected profile hits and cannot be used as the current publisher.'),
        ('DOCUMENTED_PLUS_MEASURED', 'Actual dspark_proposer.py259 has a prefix copy outside any enclosing conditional, copying target_hidden_states[:_dflash_num_context] to the persistent bank prefix; _dflash_num_context is int(cad.query_start_loc_cpu[batch_size]), not inferred from a tensor shape. There are24 positive native TensorMove. All bank-tail/alias/readiness and other copy domains remain separate.'),
        ('DOCUMENTED', 'DSpark309 builds the per-layer context-slot list by selecting _per_group_context_slot_mapping_buffers[gidx] for _layer_group_idx. It is not the DFlash base int32 Tensor indexed as a layer list. Actual group identities/values are unrecorded; equal group index can establish same source bank under stable mapping, not equal shape alone.'),
        ('DOCUMENTED_PLUS_MEASURED', 'Actual DSpark274 calls the input copy/expand metadata kernel per draft attention group;88 native executions are positive. It publishes query/input/position/context-slot/query-slot/token-index outputs, with rejected-token, SAMPLE_FROM_ANCHOR, DCP_SIZE/DCP_RANK/CP_INTERLEAVE_SIZE inputs. Avoiding hidden-state copy does not delete this work.'),
        ('DOCUMENTED', 'Inherited build_model_inputs_first_pass315 passes the bank prefix, context-position-buffer prefix and per-layer slots to the small forwarding wrapper455 and then DSpark295-309. Empty states/None global slots return; empty positions return inside the layer loop. A per-layer None reaches store after its projection.'),
        ('DOCUMENTED_PLUS_MEASURED', 'Each layer uses its own attn.wkv/kv_norm/cache. model.py417 constructs wkv as ReplicatedLinear(dim,head_dim,bias=False),425 constructs RMSNorm(head_dim,eps). Native wkv is BF16[M,5120] x BF16[512,5120] -> BF16[M,512]. Replicated declaration is not a loaded parameter-equality or binary-attestation measurement.'),
        ('DOCUMENTED_PLUS_MEASURED', 'KV BF16 output -> full512D norm BF16 ->448-nope/64-RoPE split -> interleaved rotation -> ViewCopy of original tail view -> concat -> cache scatter. Preserve BF16 boundaries, each512D normalization, layer weights and per-layer caches; do not norm a combined1536D row.'),
        ('DOCUMENTED_PLUS_MEASURED', '_apply_dsv4_rope calls get_cos_and_sin_dsa(positions) with use_cache=False,layer_names=None. The same source context_positions expression is passed to all loop iterations. Default query freshly indexes every registered applicable config/group and returns proxies; source185/186 produces FP32 lookup results.'),
        ('DOCUMENTED', 'Existing layer_names filters config keys, then still applies registered group/position-map rules. RopeDataProxy string lookup selects the layer configuration/required groups without writing global query buffers. Under stable registry/tables/positions and normal valid names, filtering/hoisting preserves the selected entries, even if layers use different configs.'),
        ('DOCUMENTED_PLUS_MEASURED', 'ComplexExpRotaryEmbedding.forward400 calls torch_npu.npu_rotary_mul(x,cos,sin), with no explicit source cast;402 copies result into original x view. Two Casts per cycle attributed under400 are native lowering, not two .to(BF16) statements. Sharing FP32 lookup does not prove removal of624 internal casts.'),
        ('DOCUMENTED', 'use_cache=True is an existing API that gathers into mutable global runtime/spec buffers, optionally draft_index-1; missing buffers skip entries. It changes alias/capacity/lifetime/readiness requirements and is not interchangeable with the pure default by a flag alone.'),
        ('DOCUMENTED_PLUS_MEASURED', 'For linear 1D slots the current standard cache writer computes valid=s>=0, physical=max(s,0), (physical//block_size,physical%block_size) INT32, masks invalid pairs to-1, then ScatterNdUpdateSk. All312 selected cycles have this path; native cache input shape[17306,128,512] does not prove valid-slot values, allocation size, duplicate-write semantics or spare capacity.'),
        ('UNKNOWN', 'Actual position/row-index/slot values, _layer_group_idx contents, live pointers/aliases, registry config identities, capture bank observers, weight generations, all cross-iteration reuse keys and physical critical-path gain are not recorded by the selected profile/source packets.')
    ]
    write(O / contract_file, dict(generated_utc=stamp, evidence_grades=['MEASURED','DOCUMENTED','DERIVED','UNKNOWN'],
        source_files=source_files, source_scopes=source_scopes, evidence_refs=refs,
        facts=[dict(grade=g, statement=s) for g,s in facts],
        local_questions_answered=['Current F27 feature partition and full-row norm boundary',
            'Actual DSpark subclass publisher and inherited bank consumer; Name-only zero is not no consumer',
            'Per-layer list routed from group banks, not base Tensor slot indexing',
            'Replicated KV declaration with separate layer weights/norms/caches',
            'Existing RoPE config filter, pure default lookup, Proxy selection and mutable cached-query alternative',
            'Source has rotary input call and in-place tail copy, not explicit C/S dtype casts'],
        lookup_stop_rule='These local questions are answered. Retain actual-value/ABI/numeric conditions; do not read whole proposer/SDK or implement a kernel merely to register conditional opportunities.',
        target_implementation_changed=False, loaded_native_binary_attested=False, whole_module_closed=False, overall_complete=False))

    alternatives = [
        dict(id='F27-v34-INDIRECT', parent='F27', transformation='One main-projection contraction reads three auxiliary sources through the ordered row plan directly, avoiding private row-selected copies/concat where unobserved.',
             argument='[S_I A0 | S_I A1 | S_I A2] = S_I [A0|A1|A2], including repeated valid indices and original order. Preserve a single15360D contraction and BF16 output boundary.',
             limits='Not three separately rounded BF16 GEMMs plus adds. Bounds/negative-index behavior, readonly generations and all raw auxiliary observers remain conditions.'),
        dict(id='F27-v34-GATHER-NORM', parent='F27', transformation='Let full-row norm read rank-major gather shards directly, or have collective layout publication supply the norm layout; avoid a private transpose where legal.',
             argument='Read value[p,m,c] as feature p*640+c of row m. Reduce all8 shards for the5120D norm and apply matching gamma/eps before publishing BF16 output.',
             limits='For M>1 [8,M,640] is not a contiguous [M,5120] view. Full row readiness/global reduction is still required; local640 norms are invalid.'),
        dict(id='F27-v34-BANK-OUT', parent='F27', transformation='Publish final norm prefix directly into the existing persistent context bank, or redesign the explicit precompute input ABI to consume the norm output.',
             argument='For original context count Nc, an equivalent normal write of norm_output[:Nc] to the same bank prefix can replace a separate prefix copy if no intermediate observer or alias sees the temporary.',
             limits='Keep original source Nc, unchanged tail, captured address/stride, readiness, all bank consumers and metadata274. Only24 TensorMove are positively counted here; no claim88 copy kernels removed.'),
        dict(id='F27-v34-ROW-OWNER', parent='F27', transformation='Replicate complete main weight; project each rank physical auxiliary rows to full5120 features, normalize locally, then exchange rows and apply the original ordered selector.',
             argument='Projection and full-feature RMSNorm are row-local. With complete W and same per-row arithmetic, S_I f(A)=f(S_I A).',
             limits='A640-output shard cannot do this alone. Complete weight150MiB/rank versus18.75MiB shard, extra131.25MiB logical weight. Computing N physical rows before an M-row selector costs N/M projection work; padding/extra rows/earlier overlap and quantization metadata matter.'),
        dict(id='F27-v34-FEATURE-CONSUMER', parent='F27', transformation='Retain640-feature partitions and use a global norm-statistic reduction plus partition-aware downstream KV consumers.',
             argument='For already rounded BF16 projection y, s=sum_p sum_c float(y_p,c)^2 determines full-row denominator; local gamma can be applied after global s. KV partial outputs then need their own global sum/consumer protocol.',
             limits='FP32 reduction/native RMS order and BF16 outputs must be preserved or numerically validated. Gathering full context afterward adds a collective without reducing activation volume; three FP32 KV AllReduces may exceed the current BF16 feature AllGather.'),
        dict(id='F29-v34-KV-EPILOGUE', parent='F29', transformation='Fuse private KV512 norm, 64-tail rotary result assembly and standard cache row publication, retaining original numerical stages.',
             argument='Each512-row normalization and BF16 barrier precedes the same64D interleaved rotary; private448+64 assembly can be represented as the same cache-row stores.',
             limits='Keep ViewCopy effects unless the original tail is proved private. Packed-cache writer, negative slots, duplicate-slot input order, current-query writers and SWA readiness remain separate domains.'),
        dict(id='F29-v34-JOINT-OR-STREAM', parent='F29', transformation='Pack three distinct KV weights into one1536-output contraction with three512D norms, or pipeline per-layer preparation/consumption.',
             argument='A block concatenation of three output-channel weight matrices represents the three independent linear maps in real algebra; common context loads can be shared.',
             limits='Distinct matrices remain distinct. BF16 contraction outputs and separate512D norms/configs/caches remain. Joint output readiness may delay first-layer consumption; joint and streaming gains overlap and are not additive.'),
        dict(id='F29-v34-LOOKUP', parent='F29', transformation='Use existing required-config filtering and/or hoist one FP32 RoPE query over the context-KV loop, selecting each layer entry afterward.',
             argument='Default query185/186 is a pure lookup of stable tables at stable position values. A query for the union of requested configs contains each selected layer result; layers need not share one config.',
             limits='Source same Tensor argument is not mutation/readiness proof. Preserve layer/group rules, bounds/errors for the chosen normal domain and readonly C/S. Filtering and hoisting overlap. No automatic internal Cast removal or cross-iteration address-only cache.'),
        dict(id='F29-v34-SLOT-CSE', parent='F29', related='F31', transformation='Compute standard INT32 slot pairs once per shared group bank/block-size/value generation and reuse them across eligible layer caches.',
             argument='f_b(s)=(-1,-1) for s<0, else(floor(s/b),s%b) is independent of layer values; same group bank and stable b/s epoch gives identical metadata.',
             limits='Actual _layer_group_idx values are unknown. Different groups/block sizes/slot generations cannot be merged by shape. Per-layer cache writes remain; packed-cache route and updates to query slots are not deleted.'),
        dict(id='F29-v34-ROW-OWNER-KV', parent='F29', related='F27', transformation='After complete-weight row-local main projection/full5120 norm, compute the three replicated KV maps at row owners and gather per-layer KV rows instead of full context rows.',
             argument='Each row-local KV/norm/RoPE can commute with an ordered row exchange when every owner uses the same per-layer parameters and correct post-metadata position for that row. Receivers then execute the original slot-write input sequence.',
             limits='ReplicatedLinear declaration is supporting source evidence, not measured equality. Requires complete main W, positions/rejection/DCP row mapping, all context-bank consumers closed or retained, and unchanged cache contracts. Direct remote cache writes are a stronger different protocol.'),
    ]
    communication_examples=[]
    P,D,s=8,5120,2
    for physical_rows, selected_rows in [(24,12),(24,24),(6400,6400),(6528,6528),(6672,6668)]:
        baseline=(P-1)*D*s*(3*physical_rows+selected_rows)//P
        row_context=(P-1)*D*s*physical_rows//P
        row_kv=(P-1)*1536*s*physical_rows//P
        communication_examples.append(dict(N=physical_rows,M=selected_rows,
            baseline_three_aux_plus_feature_gather_logical_send_bytes_per_rank=baseline,
            complete_weight_row_context_gather_logical_send_bytes_per_rank=row_context,
            complete_weight_row_KV_gathers_logical_send_bytes_per_rank=row_kv,
            main_projection_physical_row_work_ratio=physical_rows/selected_rows,
            accounting='Illustrative minimal allgather payload convention (P-1)/P. No topology/protocol/physical traffic/overlap/cost measurement. Not a paired invocation assertion.'))
    write(O / algebra_file, dict(generated_utc=stamp, evidence_grade='DERIVED_CONDITIONAL_ALGEBRA_NOT_IMPLEMENTATION',
        alternatives=alternatives,
        invalid_shortcuts=[
            dict(claim='Move current640-output shard before SP auxiliary gather, then simply gather diagonal outputs.', reason='Rank p owns row block A_p and output-feature weight W_p. A_p W_p supplies only diagonal row-owner/feature-owner blocks; all A_q W_p for q!=p are missing. Need full-weight replication or explicit new input/weight/output exchange.'),
            dict(claim='Replace full5120D norm with8 independent640D norms.', reason='Denominators differ. Toy full row[1,2],gamma1,eps0 yields[1/sqrt(2.5),2/sqrt(2.5)] whereas independent one-feature norms yield[1,1]. Toy is not an actual epsilon/domain assertion.'),
            dict(claim='Split15360D contraction into three separately BF16-rounded contributions and add without a precision proof.', reason='Rounding is nonlinear. Under BF16 nearest-even toy contributions1+2^-8,-1,0, separately rounded contributions sum0, while BF16 of the full real sum is2^-8. This illustrates the barrier, not loaded native accumulation order.'),
            dict(claim='Reuse one KV result or normalize concatenated1536 features.', reason='Three layers have distinct weights/norm parameters/caches. Joint contraction preserves three results and three512D normalization domains.'),
            dict(claim='Smaller KV dimension guarantees lower communication.', reason='Under a ring payload illustration three512D FP32 AllReduces send10752 bytes/row/rank, while a5120D BF16 AllGather sends8960; ratio1.2 before any metadata/latency. Actual algorithm and physical performance unmeasured.'),
            dict(claim='Use same pointer/shape to cache across iterations, or skip stores when slots look unchanged.', reason='Position values, request/accept/reject/slot generation, layer weight/config generation, last readers and current-query writes can change at the same address. Existing KV caching is not a new optimization.')],
        exact_boundaries=['Target HC mean BF16','Main15360D contraction output BF16','Full5120D main RMSNorm output BF16',
            'Each KV512 contraction BF16','Each full512D KV norm BF16','Actual rotary input/output dtype and64D interleave','Original per-layer BF16 cache-row/INT32 slot publication'],
        illustrative_payloads=communication_examples,
        logical_weight_bytes=dict(full_main=5120*15360*2, current_shard=640*15360*2, extra_per_rank=(5120-640)*15360*2),
        relationships=[
            'Indirect main input overlaps row-selection/concat materialization; do not add separate savings twice.',
            'Gather-aware norm and direct bank output can compose, but the same transpose/copy/output temporary is counted once.',
            'Complete-weight row owner and feature-sharded global-statistic protocols are ownership alternatives, not additive.',
            'Row-owner-KV subsumes row-context gather only if no required full-context bank reader remains; keeping that gather defeats the corresponding claimed communication removal.',
            'Config filtering, within-invocation query hoisting and mutable preallocated query buffers are overlapping choices with different alias contracts.',
            'Joint1536-output GEMM and per-layer streaming trade common-input reuse against earlier readiness. An altered launch schedule is not free overlap.',
            'F31 group slot-pair metadata sharing is linked to F29 rather than counted as a new model candidate.'
        ], native_numerical_parity=False, critical_path_gain_us=None, implemented=False, whole_module_closed=False, overall_complete=False))

    conditions = [
        ('ROWPLAN','F27','Ordered valid selector, original bounds/negative-index/duplicate semantics; raw aux and I generation readonly through last use.'),
        ('CONTRACTION','F27','Preserve one main15360D dot and output BF16 barrier; changed tiling/native reduction order requires a separate precision contract.'),
        ('RANKLAYOUT','F27','TP rank-to-feature weight order and [P,M,C]->[M,P*C] map, padding and valid-row extents retained.'),
        ('FULLNORM','F27','Complete5120D denominator, matching gamma/epsilon and BF16 output; local640 norm excluded.'),
        ('BANKPREFIX','F27','Original Nc from query_start_loc_cpu and original persistent destination extent/stride/tail; no shape-based Nc substitution.'),
        ('BANKOBSERVERS','F27','Explicit inherited bank reader retained; external/dynamic aliases, capture addresses, other read epochs and readiness protected.'),
        ('PUBLISHMETADATA','F27','DSpark274 input/query/context position/slot/token-index outputs and rejection/anchor/DCP effects remain.'),
        ('FULLWEIGHT','F27','Row-owner scheme has complete main weight plus valid quantization metadata/version; logical150MiB capacity/load cost assessed, not assumed free.'),
        ('ROWEXPANSION','F27','N physical versus M selected rows, padding, owner routing, duplicate IDs and selection-availability timing retained; N24->M12 can double main row work.'),
        ('AUXOBSERVERS','F27','All raw auxiliary gather consumers retained or independently closed before removing target-side gathers.'),
        ('GLOBALSTATS','F27','Feature-sharded RMS has globally correct rounded-y statistics, FP32 reduction/eps/gamma and BF16 boundaries; new collective measured only if implemented later.'),
        ('DOWNSTREAMSHARD','F27','Feature-sharded downstream KV supplies all partials and consumer layouts; no assumed gain if original full context gather remains.'),
        ('DISTINCTKV','F29','Three separate KV weight/norm/config/cache domains; a joint output keeps three512D norms and individual BF16 results.'),
        ('ROPEINPUT','F29','Preserve64D interleaved rotary and actual native input dtype; lookup FP32 reuse alone does not remove internal Casts.'),
        ('TAILALIAS','F29','Original rotary tail view write is retained or proved private with no intervening observer.'),
        ('POSITIONEPOCH','F29','Same context position values/extent/readiness over loop; metadata/rejection/DCP/anchor mapping retained.'),
        ('CONFIGEPOCH','F29','Stable layer_info/registry/full tables and valid layer names/groups; union filtering does not require all layer configs equal.'),
        ('COSSINREADONLY','F29','Hoisted C/S consumers do not modify shared lookup output; lifetime covers all layer last reads.'),
        ('RUNTIMECACHE','F29','If existing use_cache=True is chosen: registered runtime/spec buffers, capacity, draft_index and mutable-buffer readiness/last-reader contract verified.'),
        ('SLOTGROUP','F29','Slot-pair reuse only within same source group-bank/value generation and block-size/layout policy, not same shapes.'),
        ('SLOTSEMANTICS','F29','Invalid -1 pair convention, true page stride, bounds and original duplicate-slot input/write order preserved.'),
        ('CACHECONSUMERS','F29','All per-layer caches/current-query writes and same-request noncausal attention readiness preserved; no scatter deletion from index CSE.'),
        ('PACKEDBRANCH','F29','Packed-cache route has its own native writer contract; standard pair-scatter proof does not cover it.'),
        ('OWNERPARAMS','F29','Before computing KV at row owners, all owners have the same intended per-layer WKV/norm/rotary parameters and versions; ReplicatedLinear source is not a live-value measurement.'),
        ('OWNERCONSUMERS','F29','All full-context bank observers retained or closed before substituting KV-row gathers; original receiver slot sequence maintained.'),
        ('EMPTYDUMMY','F29','Empty global context/positions/slots, per-layer None, dummy/is_profile allocation and warmup branches retained; no model/profile work deleted solely as dummy.'),
        ('STATEGEN','F29','Cross-iteration reuse keys include request/accepted context/position/slot/layer-weight/RoPE-config/rollback epochs and last readers; no address-only cache.'),
        ('CRITICALPATH','F27/F29','New weight traffic, launches, collectives, synchronization and overlap interference priced jointly; TASK sums/logical bytes are not speedup.')
    ]
    write(O / condition_file, dict(generated_utc=stamp, conditions=[dict(id='M18-v34-'+key,parent=parent,requirement=req,
        actual_condition_values='UNKNOWN_UNLESS_EXPLICITLY_IN_PINNED_SOURCE',new_variant_current_legality_proved=False) for key,parent,req in conditions],
        implementation=False, whole_module_closed=False, overall_complete=False))
    branches = [
        ('Current native long target aux','MEASURED','24 roots,3 native means/COMM each;800/816/834 local rows. No F27 root pairing by ordinal/equal counts.'),
        ('Current graph target aux','MEASURED','240 graph gaps,3 per80 graphs.192 bound/48 unbound segments retained; graph ancestry is not native Python stack proof.'),
        ('Current F27 proposal','MEASURED','88 GEMM/Transpose/fullnorm; rows6400/12/24/6528/6668; no selected dummy F27.'),
        ('Current context KV proposal/dummy','MEASURED','264 proposal+48 dummy source-order cycles,30TASK each; no unassigned selected TASKs. Pattern not invocation/slot epoch proof.'),
        ('Prefix vs indexed aux','MEASURED_PLUS_DOCUMENTED','16 prefix Cat,72 indexed Cat;216 Index+216 Check. N24->12/24 and6672->6668. Same M does not imply identity I.'),
        ('Actual subclass vs inherited base','DOCUMENTED_PLUS_MEASURED','Actual DSpark setter480ComputeInfoTASK; DFlash setter0selectedhits; inherited DFlash build315 remains actual KV consumer.'),
        ('Group slots / DCP / rejection / anchor','DOCUMENTED_ACTUAL_VALUES_UNKNOWN','Per-layer list mapped from group banks; actual metadata274 has explicit scalar flags. No shape-derived flag/group-value claim.'),
        ('Empty states/positions/globalNone/per-layerNone','DOCUMENTED_ACTUAL_SELECTION_UNKNOWN','Global early exits and projection-before-layerNone store are pinned. All selected312cycles positive standard scatters; do not infer other branches impossible.'),
        ('Standard vs packed cache','STANDARD_MEASURED_PACKED_CONDITIONAL','Selected path uses full pair preparation and ScatterNdUpdateSk; packed writer path not replaced by standard proof.'),
        ('RoPE default/filter/mutable runtime buffer','DEFAULT_MEASURED_OTHERS_DOCUMENTED','Default false/None at source62 with185/186 Index; filtering is existing config interface, mutable runtime/spec buffer alternative has separate ABI.'),
        ('Quantized/nonSP/TP1/other framework paths','UNKNOWN_OUTSIDE_SELECTED_PATH','Current shapes/source ordinary main gather positive; alternative framework branches and future workloads retained, no broad dispatcher audit.'),
        ('Capture / accepted rollback / cross-iteration state','CONDITIONAL_NOT_MEASURED','Same persistent bank/positions/slots can mutate. Versioned reuse and all readiness/lifetime obligations remain; existing cache reuse is not a new hit-rate gain.')
    ]
    write(O / branch_file, dict(generated_utc=stamp, branches=[dict(branch=b,state=s,evidence=e,whole_branch_closed=False) for b,s,e in branches],
        stop_rule='Finite local profile/consumer/partition questions answered; proceed to M19 rather than recursively auditing all runtime values/SDK for already conditional candidates.',
        whole_module_closed=False, overall_complete=False))

    nodes=['aux0','aux1','aux2','ordered_rowplan','selected_concat','main640','feature_gather_rankmajor','transpose','full5120norm','context_bank_copy','metadata274','bank_prefix','context_positions','layer_slot_banks']
    edges=[('aux'+str(i),'selected_concat','RAW_AUX') for i in range(3)]
    edges += [('ordered_rowplan','selected_concat','ORDERED_ROW_MAP'),('selected_concat','main640','RAW'),('main640','feature_gather_rankmajor','TP_FEATURE_PUBLISH'),('feature_gather_rankmajor','transpose','LAYOUT'),('transpose','full5120norm','FULL_ROW_RAW'),('full5120norm','context_bank_copy','RAW'),('context_bank_copy','bank_prefix','PREFIX_WRITE'),('context_bank_copy','metadata274','CURRENT_SOURCE_CONTROL'),('metadata274','context_positions','PUBLISH'),('metadata274','layer_slot_banks','PUBLISH')]
    for i in range(3):
        names=[str(i)+q for q in ['_WKV512','_norm512','_lookup','_rotary64','_assemble','_slotpair','_scatter','_attention']];nodes+=names
        w,n,l,r,c,s,t,att=names
        edges += [('bank_prefix',w,'RAW'),(w,n,'BF16_BARRIER'),('context_positions',l,'POSITION_EPOCH'),(l,r,'COS_SIN'),(n,r,'KV_TAIL_VIEW'),(r,c,'ROTARY_RESULT_AND_COPY_EFFECT'),(n,c,'NOPE_PREFIX'),('layer_slot_banks',s,'SLOT_GROUP'),(s,t,'INT32_PAIRS'),(c,t,'BF16_CACHE_VALUES'),(t,att,'CACHE_READINESS')]
    baseline=graph(nodes,edges)
    indirect=graph(['aux0','aux1','aux2','I','indirect_main640','TP_rankmajor','gather_aware_fullnorm_to_bank','metadata','KV_consumers'],
        [('aux'+str(i),'indirect_main640','RAW') for i in range(3)]+[('I','indirect_main640','ORDERED_SELECTOR'),('indirect_main640','TP_rankmajor','FEATURE_COLLECTIVE'),('TP_rankmajor','gather_aware_fullnorm_to_bank','ALL_FEATURES_READY'),('gather_aware_fullnorm_to_bank','KV_consumers','BF16_BANK_PREFIX'),('metadata','KV_consumers','POSITION_AND_SLOTS')])
    owner=graph(['local_aux3','complete_main_W','full5120_row_projection','full5120_row_norm','metadata_rowmap','three_distinct_KV_norm_RoPE','KV_row_gathers','original_receiver_slot_sequence','layer_caches'],
        [('local_aux3','full5120_row_projection','RAW'),('complete_main_W','full5120_row_projection','COMPLETE_WEIGHT'),('full5120_row_projection','full5120_row_norm','BF16_BARRIER'),('full5120_row_norm','three_distinct_KV_norm_RoPE','FULL_ROW'),('metadata_rowmap','three_distinct_KV_norm_RoPE','CORRECT_ROW_POSITION'),('three_distinct_KV_norm_RoPE','KV_row_gathers','PER_LAYER_BF16'),('metadata_rowmap','original_receiver_slot_sequence','ORIGINAL_SLOT_ORDER'),('KV_row_gathers','layer_caches','VALUES'),('original_receiver_slot_sequence','layer_caches','CACHE_UPDATE_ORDER')])
    write(O / graph_file, dict(generated_utc=stamp, source_consistent_baseline=baseline, conditional_indirect_and_bank_output=indirect,
        conditional_complete_weight_row_owner_KV=owner, scope='Three layer nodes describe source-consistent structure, not an inferred pairing of all native chains/root counts.',
        omitted_obligations='Attention/current-query/other-bank observers and alias/WAR/WAW/capture epochs are explicit conditions, not assumed absent by these DAGs.',
        graphs_acyclic=True, implementation=False, whole_module_closed=False, overall_complete=False))

    evidence=[profile_file,contract_file,algebra_file,condition_file,branch_file,graph_file]
    f27vars=[v for v in alternatives if v['parent']=='F27'];f29vars=[v for v in alternatives if v['parent']=='F29']
    common=dict(contract=str(O/contract_file),algebra=str(O/algebra_file),conditions=str(O/condition_file),graphs=str(O/graph_file),implementation=False,new_performance_result=False)
    candidate_updates={
        'F27':dict(profile_review_state='ALLRANK_AUX_ROWS_COLUMN_PARTITION_FULLNORM_ACTUAL_BANK_CONTRACT_REVIEWED;CONDITIONAL_UNPRICED',
            variants_v34=f27vars, review_v34=dict(common,variants=f27vars), target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v34.json'),
            conclusion_changing_gap='Actual row/group/alias/capture/weight epochs and all readers remain conditions. Diagonal-shard shortcut excluded; complete-weight/feature-partition alternatives distinguished. No need whole source to register these.',
            next_profile_or_contract_action='Advance M19 finite profile coverage; revisit only a stronger current-value/numerical/performance claim.', all_consumers_closed=False),
        'F29':dict(profile_review_state='312_FULL_SOURCE_ORDER_KV_PATTERNS_AND_ROPE_PROXY_GROUP_SLOT_REPLICATION_CONTRACT_REVIEWED;CONDITIONAL_UNPRICED',
            variants_v34=f29vars, review_v34=dict(common,variants=f29vars), target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v34.json'),
            conclusion_changing_gap='Actual group/config/value/alias/capture/parameter epochs, packed/other branches and physical causal benefit remain conditional; no automatic624Cast removal or cache write deletion.',
            next_profile_or_contract_action='Do not repeat answered pure-query/filter/Proxy/bank/slot routing questions; move to M19 decoder profile.', all_consumers_closed=False),
        'F31':dict(review_v34=dict(common,relation='F29 per-group slot-pair metadata CSE; keep actual DSpark274 query/position/rejection/anchor/DCP outputs. Scope refines existing F31, not new candidate.'))
    }
    findings={
        'M02':'Per-layer standard cache slot mapping and negative-skip publication remain; slot CSE does not remove cache writes, alter duplicate order or prove physical allocation savings.',
        'M04':'Ordered aux row domains24->12/24 and6672->6668; actual DSpark context bank prefix and metadata publisher localized, original Nc comes from CPU query_start_loc.',
        'M07':'Each context KV has BF16 WKV/full512 norm/interleaved64 rotary/tail-copy/cache boundaries; ReplicatedLinear declaration and pure FP32 lookup/filter contracts pinned.',
        'M15':'Three HC-mean aux checkpoints:72native plus240graph means; 88unique target roots withwork,48graphsegmentsunbound. Raw aux other consumers retained for any pre-gather move.',
        'M18':'Allrank F27 proposal88 and F29 312source-orderKVcycles/9360TASK; actual DSpark firstpass480TASK. Five F27 and five F29 conditional refinements,28 conditions, no implementation/gain.',
        'M19':'Existing decoder/cache consumers constrain joint-KV vs layer streaming. Three distinct layer weights/norms/caches retained; current-query noncausal readiness is not replaced by causal-prefix readiness.',
        'M21':'Actual subclass per-layer slot list selects per-group banks;274metadata/rejection/SAMPLE_FROM_ANCHOR/DCP outputs retained. Equal shapes/counts do not establish actual group identity.',
        'M22':'Norm direct-to-bank and mutable RoPE runtime buffers have different capture/alias/tail/last-reader contracts. Small-copy native absence not inferred from24positiveTensorMove.',
        'M23':'Column640 sharding blocks naive pre-gather projection. Complete-main-weight row owner and feature-global-stat/KV-partial protocols are distinct alternatives with explicit logical communication/weight costs.',
        'M24':'Same context positions expression supports conditional within-call lookup CSE; cross-iteration reuse requires position/request/accepted-slot/weight/config generations, not pointers or shape.'
    }
    module_updates={mid:dict(finding=text,evidence=evidence,candidates=(['F27','F29','F31'] if mid in ['M04','M18','M21','M22'] else ['F27','F29']),
        gap='Finite local source questions answered; full branch/observer universe and actual runtime-value/ABI/precision/causal-gain conditions remain open.') for mid,text in findings.items()}
    remaining=[x for x in deepcopy(read(O/'module-coverage-gaps-v33.json')['remaining']) if not x.startswith('NextwholecoveragefrontierM18')]
    remaining.insert(10,'M18v34currentaux/column640/full5120norm/actualDSparkpublisher/per-layergroupslotlist/ReplicatedKV/pureRoPEquery/filter/Proxyquestionsanswered. 312KVsource-orderpatternsarenotrequest/pointerepochs;24TensorMoveonlypositiveComputeInfosurface,nosmall-copyabsence. F27/F29conditionalrow-feature/KV/filter/hoist/metadatareorganizationsregistered;actualguards/generations/allobservers/nativeprecision/causalgainremainconditional. Do notrepeattheseaswhollyunlocatedorreadSDKmerelytorecordcandidate.')
    remaining.insert(11,'NextwholecoveragefrontierM19/F28DSparkembedding/HCinitialization/decoder/sharedcomputation. Startsavedallranknativefullcallers/typedTASK/COMM/graphsections;separateproposal/dummy,SProwdomainandcontext-KVvsquery-KV. Reusev7decoder/context/v31collapse/v32heads/v34KVcontracts;pointreadonlyacandidate-criticalunknownconsumer/branch,notallmodelcode. M20MarkovandM21metadatawillfollowwithoutcountingsameworktwice.')
    summary=dict(filename='M18-aux-context-reorganization-summary-v34.json',previous_turn_classification='PROGRESS:v33probability-consumer/RNG/selected-recovery',
        selected_native_APIs=6632,selected_native_TASKs=10280,selected_native_chains=33,graph_aux_segments=240,
        F27_proposal_pipelines=88,F29_source_order_KV_cycles=312,F29_proposal_cycles=264,F29_dummy_cycles=48,
        actual_DSpark_firstpass_TASKs=480,positive_long_context_TensorMove=24,actual_metadata_kernels=88,
        exact_base_DFlash_profile_hits=0,per_layer_group_slot_list_contract=True,within_call_RoPE_query_CSE_conditional=True,
        naive_pre_gather_column_shard_move_invalid=True,condition_rows=len(conditions),refined_model_candidates=['F27','F29'],related_model_candidates=['F31'],
        new_model_candidates=[],remote_jobs=8,remote_pass=8,remote_retained_errors=0,
        remote_job_status='8 CPU-only remote jobs terminal and pass. Base-scope zero query retained; main semantic review corrects actual subclass and excludes absence/alias/count/gain inferences. Initial local finalizer schema KeyError retained separately; stopped before shared mutation and corrected. No NPU/model/service/new-health work.',
        evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report='''本轮继续以保存的 profile 为主线，定点源码只补分区、消费者和生命周期。覆盖32个分片的6632原生API、10280个TASK、33条链，以及80个图中的240段辅助状态准备。F27实际proposal有88条投影/重建/norm路径；F29有312条完整的30-TASK源码顺序模式，其中264条proposal、48条dummy。它们不是同一组，未按相同计数、最近时间或顺序强行配对请求。16个图仍没有唯一phase/root；其48段辅助准备继续保留。

行域已明确。辅助状态先在4路HC维做mean，长行800/816/834在TP8汇聚成6400/6528/6672；图内为3到24。后续实测有24到12、24到24、6672到6668的索引选择。索引长度相同不说明索引是恒等映射，也不排除重复行。可用三源间接读取投影代替私有选择副本和concat，但要保留原行序、重复、边界和单次15360维点积的BF16输出；分成三次各自BF16舍入的GEMM再相加没有等价保证。

main_proj确实按输出列分片，每卡640列，随后TP8重建，再对完整5120维做norm。不能把现有分片直接移到辅助状态汇聚前：那只算到了“本卡行×本卡输出列”的对角块，其余行/列组合缺失。完整权重复制后的行分区方案是另一种有条件的重组：每卡先算自己的行和完整5120维norm，再交换行。完整权重逻辑150MiB/卡，当前分片18.75MiB，增加131.25MiB；若先算物理24行再选12行，主投影行工作量可变成两倍。激活通信逻辑量、权重代价、目标层期间已发生的通信重叠要共同衡量。

另一个方向是让完整行norm直接读取[8,M,640]的rank-major结果，或在发布时给出其所需布局，并将norm输出直接写到持久context bank的原prefix。M大于1时原gather不能直接当连续[M,5120]看待。只在本地640维做norm不等于完整5120维norm；若保留feature分区，必须增加正确的全局统计和下游分区协议。三路512维FP32 AllReduce也不保证比一路5120维BF16 AllGather省通信，文档已列出逻辑负载反例，未估算实测收益。

publisher定位做了必要纠正。最初DFlash基类122/132查询为0，这是有效的范围结果。实际profile命中的是DSpark子类set_inputs_first_pass，共480个ComputeInfo TASK、7条链，其中24次长context TensorMove和88次输入metadata内核。子类259源码在无外层if包围的语句中把target_hidden_states的原Nc前缀复制到持久bank；Nc来自query_start_loc_cpu。24次只是当前可见大复制，不能推论小尺寸没有其他复制记录或88次均可删除。原bank消费者通过继承的build_model_inputs_first_pass315读入KV预计算。可研究norm直接发布到同一bank或改变显式消费接口，但必须保持prefix/tail、捕获地址、别名、读者和ready代际。metadata274还要产生query/input/position/slot/token-index并处理拒绝、anchor与DCP，不能一起删掉。

每层slot来自_layer_group_idx映射的group bank列表，并非DFlash基类Tensor直接按layer切片。相同group bank、block_size及slot值代际时，可以共用标准INT32 pair准备；不同组或相同shape不能据此合并。原每层cache写仍保留负slot、重复slot输入顺序和query写入的依赖。当前312条都是标准pair+scatter路径，packed writer仍是独立条件分支。

KV构造声明使用ReplicatedLinear，支持研究在完整权重的行owner处先计算三层KV，再交换三路512维值而非完整5120维context；这仍要求实际各卡层参数一致、正确的metadata后position行映射，以及所有完整context bank读者被保留或替代。三层权重、norm、配置和cache分别保留。联合1536输出GEMM可以共享context读取，但必须保留三次512维norm及BF16边界；它与按层流水是有准备时刻取舍的方案，收益不能相加。

RoPE局部合同也已补齐。get_cos_and_sin_dsa已有layer_names配置过滤接口；当前调用只传positions，走use_cache=False的全适用配置FP32查表，Proxy按层名只读选择。稳定positions和registry/table代际下，可以按所需配置过滤，或在同次KV循环外查一次并让各层选择自己的配置；无需假设三层使用相同配置。过滤和hoist的重复消除范围重叠。use_cache=True是写入全局预分配buffer的另一接口，有容量/alias/last-reader要求。forward400源码直接调用npu_rotary_mul，没有显式C/S cast，所以不能把每条链内部两个Cast、全体624次Cast直接算成查表复用收益。尾部copy写回原view也必须保留或证明私有。

本轮仍细化F27/F29并关联F31，记录10个条件变体、28项约束、分支台账和三张无环关系图，没有增加主候选，也没有修改或验证优化实现。8次远端CPU只读作业均pass；本地成稿脚本首版因一个定位记录使用locations而非start/end触发KeyError，在共享状态修改前停止，修正字段适配后主审通过，错误记录保留；基类零命中和Name-only范围纠正都完整保留。没有导入模型/torch、启动NPU负载、改服务或做新健康检查。最后健康仍v12post，当前状态不能由旧记录推断。全局35项模型候选、7项框架候选，0个整模块关闭；26项理论比较仍INCOMPLETE，3136个candidate-set资源行、849个PMU缺失和16个phase-unbound图保持。下一批是M19的实际embedding/HC/decoder路径，继续从profile进入，已回答的局部问题不重复查整仓源码。整体Goal继续active。
'''
    jobs=[dict(kind=kind,status='pass',description=description,artifacts=[N/script,N/(label+'.json')]) for kind,label,script,_,description in specifications]
    jobs.append(dict(kind='review',status='error',description='Initial local finalizer stopped before shared-state mutation: a source localization record has locations rather than start/end. Error retained; corrected schema-normalization then main semantic review passed.',artifacts=[N/'M18-finalizer-integration-error-v34.json']))
    jobs.append(dict(kind='review',status='pass',description='Main semantic review: row-selector identity, diagonal-shard counterexample, fullnorm/rounding barriers, actual subclass publication, pure-query and group metadata conditions;10 refinements/28conditions, no implementation/gain.',artifacts=[O/q for q in evidence]))
    action=('继续Loop004，v34为实质进展。先读M18-aux-context-reorganization-summary-v34.json、M18-partition-bank-RoPE-and-slot-contract-v34.json、F27-F29-row-feature-and-KV-reorganization-v34.json、module-coverage-gaps-v34.json。'
        'M18全32保存分片6632API/10280TASK/33链+240graphaux段，F27proposal88；F29完整source-order312cycle=264proposal+48dummy，每条30TASK共9360，不是invocation/alias/slot epoch。'
        '原aux local800/816/834->6400/6528/6672，graph3->24；Index24->12/24、6672->6668，不能按相同行数认恒等I。F27Column640->TP8rankmajor[8,M,640]->Transpose->full5120RMS；直接前移本地W只得行/feature对角块缺offdiagonal，无效。'
        'F27条件variant:三源间接同15360D投影、gather-aware fullnorm/directprefixbank、完整W行owner、featureglobalstats+新consumer；BF16barriers/行序/所有aux读者保留。完整W150MiB vs shard18.75MiB是逻辑权重，N24->M12前移可多算2x，非实测成本收益。'
        'baseDFlash122/132profile0是范围纠正，实际dspark_proposer242/243/259/274/315/317/318有480ComputeInfoTASK/7chains，24longTensorMove和88metadata。source259无外层if的prefixcopy，Nc=query_start_loc_cpu[batch]，小尺寸无ComputeInfo不推没copy；保留banktail/capture/alias/readiness。'
        'actualDSpark309按_layer_group_idx映射group contextslots列表，baseDFlashbuild315继承消费bank/posprefix/slots，wrapper455直转；不能Tensor标量索引误判。metadata274带rejection/anchor/DCP并写query/input/position/slots/tokenindex，不因bankcopy省掉。'
        'wkv sourceReplicatedLinear，3层distinct512norm/Rope/cache；条件完整mainW行owner可先三KV再gather，前提同层跨卡参数一致、正确metadata后positions和全contextbankobserverclosure/保留。joint1536仍3x512norm，和layerstreaming非加算。'
        'Rope默认getCos185/186纯FP32查表，已有layer_names过滤config；Proxy只读选择，stablepos/table/registry可同次循环hoist或filterunion，不需要同config。use_cacheTrue写runtime/specbuffer有独立alias/capacity/draft_index合同。forward400只是npu_rotary_mul而非source.to，624internalCast不得直接算可删；402tailcopyeffects保留/证明私有。'
        'slotpairCSE需同groupbank/blocksize/epoch，保留-1skip/duplicateinputorder/每层cache/query依赖；packedroute独立，不能同shape合并或删除312scatter。跨迭代用position/request/accepted-slot/weight/configgeneration而非address。'
        '8remoteCPUjobs均pass+主审1pass，首版本地finalizer字段schema错误1保留(共享修改前停止，已修正)，checkpoint PROGRESS；35model7framework0wholemodule，F27/F29共10refinements/28conditions并linkF31，无newcandidate/实现/性能。无model/torch/NPU/service/newhealth,lastv12post；26theoryINCOMPLETE/3136candidate资源/849missingPMU/16graphphaseunbound不变。'
        '下一批M19/F28从savedallranknativecallers/typedTASK/COMM/graphsections分析embedding/HC初始化/decoder共享，分proposal/dummy、SProwdomain、contextKV/queryKV；复用v7/v31/v32/v34，只有profile不足才定点代码，不重复已答SDK/proposer。Goalactive，仍有整体未分析范围。')
    persist_refinement(version=34,previous=33,title='v34 M18 aux/context row-feature ownership and publisher contracts',summary=summary,
        changes=['Allrank auxiliary/F27/F29 scopes and complete312KVsource-orderpatterns bound without fabricated root/request pairing.',
                 'Actual DSpark subclass publisher/per-layer group-slot list/ReplicatedKV/pureRoPE filter-and-Proxy contracts distinguished from base paths and native lowering.',
                 '10conditional row-feature/KV/publication/lookup/metadata refinements with28conditions; naive diagonal-shard/localnorm/cast-removal/communication-gain shortcuts excluded.'],
        candidate_updates=candidate_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[contract_file,algebra_file,condition_file],
        publication_rows={
            'F27':'| F27 | DSpark 辅助状态投影、完整行 norm 与 context 发布 | 条件 | v34绑定88条实际proposal及列分片640→TP8→完整5120维norm；间接三源投影、gather布局/norm直写bank、完整权重行分区和feature统计协议为条件变体。当前分片直接前移会漏off-diagonal计算；保留行序、BF16边界和所有bank/aux读者，逻辑通信不作收益。 |',
            'F29':'| F29 | DSpark context KV、norm/RoPE、缓存写与组索引复用 | 条件 | v34覆盖312条完整source-order KV模式，分别264proposal/48dummy；ReplicatedKV声明、实际group-slot列表、纯FP32 RoPE查表/过滤及Proxy已定点核对。联合/流水、查表hoist、同组slot-pair和owner-KV交换均有独立条件；不自动删除624内部Cast或每层cache写。 |'},
        gate_updates=dict(status='PROFILE_AND_NECESSARY_POINT_CONTRACTS_REVIEWED',answered_by=evidence,
            closed_scope='Current aux row domain/column partition/fullnorm/actual DSpark publisher/inherited bank reader/group slot routing/ReplicatedKV/RoPE query and Proxy local contracts.',
            not_closed='Actual values/aliases/epochs/all observers/capture/native precision/physical gain/other branches or whole module.',
            stop_rule='Advance M19 saved-profile coverage. Conditional discovery does not require an SDK audit, live-value census or implementation merely to record an opportunity.'),
        scripts=[N/script for _,_,script,_,_ in specifications],executor_prefix='zcode-M18-',main_script=N/'finalize_M18_aux_context_v34.py')


if __name__ == '__main__':
    main()
