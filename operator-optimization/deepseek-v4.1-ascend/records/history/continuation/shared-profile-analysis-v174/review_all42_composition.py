"""Cover all registered families using shared subgraphs and joint constraints."""
import collections
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def write(n,d):(H/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
model=read(O/'candidate-profile-ledger-v173.json')['candidates'];framework=read(O/'framework-candidate-ledger-v154.json')['candidates']
prior=read(N/'shared-profile-analysis-v173/joint-operator-graph-and-resource-review.json');first={c for n in prior['nodes'] for c in n['candidates']}
remaining=read(H/'remaining23-candidate-frontier.json')['remaining'];oldclauses=read(O/'candidate-relation-clauses-v154.json')

# (id, compound opportunity, nonadditive/alternative interaction, legality and
# state conditions, work/limit comparison route). No measured gain is invented.
answers=[
('F05','RMS→exact rounded norm output→row quantization→Q projection can share reads,scale production and tile handoff.',
'F05/F06/F07/F09 overlap the same input/norm/quant/projection surfaces;count each once. A shared quantized input is conditional on the exact PreparedLinearInput/quantizer contract.',
'Retain BF16 norm rounding before quantization,original full-row reduction and all norm consumers. Scaling before rounding or canceling RMS against a downstream linear map is not automatically equivalent.',
'Keep necessary norm/quant work plus classical projection MACs;max(Cube,Vector,necessary traffic) with shared capacity,not the sum of standalone limits.'),
('F06','Joint Q/KV/input preparation may reuse an identical source scan or jointly tile compatible projections.',
'Same input does not merge distinct weights/MACs. Combining with F17/F33 may introduce mixed FLOAT/BF16 kernels,new output packing and different communication demands.',
'Match row ownership,value/weight generations,dtypes and output rounding. Retain any earlier KV/cache or coefficient deadline;price replicated weights and producer-before-collective alternatives.',
'Sum only distinct necessary MACs within one mode/pool;keep mode-specific maxima until an issue-sharing proof exists. Derive necessary input/weight bytes from residency and chosen algorithm.'),
('F07','Q projection output can feed tail RoPE directly in a tile-preserving producer epilogue.',
'This consumes the same projection/rope intermediate also used by Q/KV/index variants. Legacy CP head-RMS afterQB is outside the frozen V41 query path and remains a separate reference.',
'Keep the original BF16 projection output rounding,positions,head partition,rotated tail and unchanged dimensions. An on-chip interface still honors all other readers and original capture/output banks.',
'Only an otherwise materialized interface is a candidate byte budget;not all HBM traffic. Tile pipeline readiness may be finer than whole-operator completion.'),
('F08','KV projection→norm→RoPE→SWA store can become one public producer,or current query consumers can use an exact bypass while future cache publication is retained.',
'Direct-store fusion and delayed/COW/shadow/recompute publication are alternative graphs. F11/F19/F27/F29/F34 share cache planes and must not double-count stores or observer demand.',
'Current verify can need newly produced KV before acceptance. Keep page/slot/position/owner generations,all current/future readers and queued writers;prefix acceptance count is not token identity.',
'Compare nonduplicate transform work and necessary persistent writes for the chosen initial state. Shadow/COW/recompute pays its own reads,writes,extra work and live memory.'),
('F09','Indexer query projection→tailRoPE→per-head quantization can share exact qR input preparation or a tile handoff.',
'Reuse of Q/index quantized input is distinct from sharing projection weights or deleting query-specific quantization. Weight/RoPE overlap already present in the baseline earns no new credit.',
'Retain BF16 RoPE boundary,FP16 scale rounding,minimum2^-24,div_rn and nearbyint source rules. Separate original qR/parameter/row epochs and all future readers.',
'Typed BF16/INT8 work and Vector conversion/quantization remain. SplitK or upstream panel representation has its own reductions,publication and numeric cost.'),
('F10','Weight projection→BF16→FP32 scale→FP16 product may fuse conversion/scaling into the same typed producer.',
'F09/F10/F12 compete for auxiliary stream/Vector/Cube capacity. One early/late stream interval is not guaranteed overlap or extra hardware.',
'Preserve every intermediate round and original reduction ordering;head-weight values/scales must match the query quantizer generation. A mathematically combined real product is not IEEE equivalence.',
'Use mode-specific necessary work and required interfaces after transformation;keep native FLOAT/HF32 modes and cache traffic conditional.'),
('F11','Index key norm→RoPE→quantization→key/scale cache store can fuse under one persistent-output contract.',
'The compressor latent fork F19 and SWA/shared KV F08/F33 may share a source but not automatically gamma,quantizer,scale dtype or cache layout.',
'Retain key/scale pair coherence,original quantizer precision,valid rows,slots,causal positions and all query generations. A shared pointer is not equality of key content/epoch.',
'Count both required typed key and scale outputs;selected cache residency and persistent observation determine traffic floors,not the whole cache allocation.'),
('F12','Score/TopK→visibility/chronological repair→index publication can share passes;one produced C1 candidate map can serve its exact four consumers with a list or bitmap representation.',
'C1 candidate-producer/consumer and C2 full-scan modes differ. Membership penalties can leave finite-score fillers outside the allowed set;list/bitmap,early pruning,localTopK merge and certified speculative tiles are alternative contracts.',
'Preserve comparator,ties,NaNs/finite-sentinel behavior,valid multiplicity,chronological order,-1 tail and all map/position/forward epochs. A changed query forbids blanket cross-layer/iteration result reuse.',
'Keep required score/selection work unless a complete demand certificate permits pruning. New bitmap construction,merge,summary certification,publication and fallback/reread work are paid once.'),
('F13','Already-fused sparse attention has opportunities at input/cache/index/output interfaces and in a legally ordered SWA-before-index/CSA continuation.',
'Splitting SWA/CSA and later joining online state is an alternative to the existing fused kernel. Duplicate KV elimination,projected state or rounded summaries each require their own exact domain.',
'Preserve selected sparse edges,sink exactly once,BF16 exponent/PV and accumulation order,all original masks/visibility/page generations. Real-arithmetic softmax state merge alone does not prove floating point equivalence.',
'Necessary work depends on actual sparse edges and allowed algorithm. Independent Cube/Vector/cache/network lower bounds combine by max;no new internal-fusion or fullN² work credit.'),
('F14','Attention→inverse tailRoPE→layout/head restoration→output projection can share tile materialization or move a projection across communication under new ownership.',
'F14/F15/F16 describe overlapping interfaces. WA-before-AlltoAll,full-group weight replication,query-owner projection and current group-specialized paths are mutually competing schedules.',
'Preserve rotated64-tail/unchanged448-dim source domain,all BF16 boundaries,head/group permutation and original sum order. Keep partial-output readers and the exact collective membership/protocol.',
'Price changed per-rank MACs,all necessary weight replicas,padding and card ingress jointly. Fewer interface bytes is not a latency prediction.'),
('F15','Padding/restoration and projection can use a virtual-valid-row representation or a final direct producer instead of a separate copy/pad.',
'Current compact G1 and larger all-group output paths are distinct. A late row-demand cut may require max(U_p) padding rather than ceil(totalU/TP).',
'Retain bias and zero/nonfinite semantics,logical/physical extents and every consumer. Zero padding is not automatically zero projection when nonfinite weights or observed tails matter.',
'Only required outputs/rows can be removed;changed padding/tile utilization,ownership and metadata costs remain in the same work/traffic ledger.'),
('F16','Two output projections and reduce-scatter may pipeline tiles while retaining the original intermediate numeric surface,or use a separately justified repartition.',
'A precomposed WA*WB matrix or one unrounded accumulator is a new numeric domain. RawINT32 reduction,post-dequant BF16 reduction and selected-owner schedules are alternatives.',
'Preserve intermediate BF16 round,dequant/scale order,full dot products,collective sum/tie/protocol and owner-visible outputs. Do not commute a reduction across nonlinear quantization by real linear algebra alone.',
'Keep distinct necessary per-mode projection work and derive the new payload cut. Shared Cube/Vector/communication budgets and changed weight memory prevent adding ideal speedups.'),
('F17','Two compressor input projections may share exact source conversions or be jointly tiled,including the SWA/C2 mixed projection frontier.',
'C1/BF16 and C2/FLOAT branches are not a single BF16 GEMM by default. F06/F33/projection-before-gather change weight replication,packing and input availability.',
'Preserve FLOAT/HF32 and BF16 rounding domains and both consumers. BF16-representable input does not prove FLOAT weights/results reducible to BF16;segmented/mixed representations have explicit bridges.',
'Mode-wise MAC/cycle cases remain separate;extra conversion/dequant/packing/readiness and cold/warm weight traffic are included. No automatic savings of one of two different matrices.'),
('F18','Compressor state pooling/gating→norm may combine a state-aware kernel or operate incrementally on the exact changed ring window.',
'F18/F19/F34 overlap ring/latent/cache state. Full pooling,incremental update,recompute and output memoization are alternative schedules and storage choices.',
'Keep ring tuple/half/slot/position/generation,old-state reads before writes,gate/softmax/norm rounds and wraparound/partial-window behavior. A stable ring pointer or append count is not content identity.',
'Necessary state work depends on the selected algorithm and proven unchanged region. Pay retained or new state,publication,recompute and live capacity once.'),
('F19','A compressor latent can fork to index-key and compressed-KV consumers with a shared exact producer or shared norm/transform when value contracts match.',
'Do not conflate shared latent input with identical gamma/rotation/quantization or output banks. Source ownership,context/KV publication and index consumers overlap F08/F11/F18/F27/F29/F33/F34.',
'Retain every consumer and persistent key/scale/KV generation;cross-layer/iteration reuse requires actual value/parameter/position equality. Changed field aliases or publication order need their own contract.',
'Count common necessary work once only when identity is proved;keep separate typed branch work,required stores and extended lifetime otherwise.'),
('F33','C1 compressor/SWA same-domain projections may share a producer or use projection-before-communication with a consistent new row and weight ownership.',
'Combine with F06/F08/F17/F19 using one selected representation. Existing HC local collapse and compact baseline paths are not new savings;C2FLOAT remains its own mode.',
'Match source rows/weights/config generations,all rounding and valid/padded dimensions. New projected payload or group placement must supply every original consumer and bank epoch.',
'Re-derive joint required raw/projected information and card ingress;price weight replicas,extra arithmetic/packing,finite Vector/Cube and old/new live buffers. Do not add overlapping input-byte budgets.'),
('H01','Polling/wait completion can become an ordered event/notification or share a wait pipeline when outputs are actually ready.',
'H01/H02/H05 share host submission/result boundaries. CPU delay can postpone peer arrivals inside a resident device kernel;it is not universally irrelevant to low device arithmetic activity.',
'Preserve cancellation/error/result ordering and producer-ready/consumer-retire semantics. Historical thread evidence is not a current171 onCPU cost or a causal wait-time fraction.',
'Host/controller work and queue capacity remain symbolic/current-unmeasured;do not subtract wall waits as CPU work or add an ideal host gain to the full device gain.'),
('H02','Snapshot-derived indices/counts/positions/masks and small pinned copies may share a final producer or transfer,including the exact CP-cache-hit dead-producer domain.',
'H02/F20/F24/F31/F35/H06/H07 overlap the same banks. Merge submission is different from changing data representation,cache identity or eliminating a required copy.',
'Keep all field dtypes/strides/valid tails,source snapshot/alias rules,queued DMA ownership and per-stream/rank completion. Use actual CP cache-hit keys;no independent buffers from different names.',
'Count necessary per-field work and unique materialization once;source buffers live through DMA,consumer banks through old readers and new writers. No HBM/time bound from logical interface bytes alone.'),
('H03','Exact immutable token-prefix block-hash chains can be memoized/shared within the original namespace before suffix work.',
'Current prefix caching with SHA256/block128 is already enabled. H03 avoids repeated hash work only;existing KV-cache hits are not new wins. H04 text-tokenization reuse is a different key domain.',
'Keep parent hash,token IDs,block boundaries,salt/adapter/model/config and all original namespace inputs. Same text prefix or buffer address alone is not a valid hash-result reuse key.',
'Compare repeated hash work with one certified prefix plus each suffix and cache/lookup overhead. It is host work,not repeated NPU model compute or fullKV traffic.'),
('H04','Tokenization can batch work or exactly reuse an identical normalized prompt/tokenizer input;prefix reuse needs a certified tokenizer boundary/carry state.',
'H04/H03 may share an immutable admitted-input object,but token IDs and text are not one interchangeable hash namespace. Partial text prefix reuse can fail at a token merge across the boundary.',
'Retain tokenizer revision/options,normalization/formatting,added tokens and exact IDs. Keep input length/error and request ownership semantics;do not infer identical tokenization from equal character count.',
'Use the actual host function/transduction work and current queue constraints. Saved function time alone is not a request critical-path gain or device-utilization improvement.'),
('H05','Result-ready transport,decode/serialization and notifications may pipeline or batch under an explicit unchanged response contract.',
'H05/H01/H02 share staging/events and H06/speculative acceptance controls result sizes. First nonempty client text is a response boundary,not a device prefill completion timestamp.',
'Keep request/token order,accepted-prefix contents,logprob/usage/finish/error fields and feedback readiness. Changing chunk granularity/timing is a different contract;no premature publication of a partially ready ID bank.',
'Current causal ready/onCPU/transport costs stay unmeasured where absent. A long main-thread wall interval is not entirely removable CPU work;no free overlap or duplicate ideal gain.'),
('H06','Uniform speculative width6 and TP8 give the existing minimum compatible24-token descriptor;logical/physical row-domain splitting can define a new packing/bucket ABI.',
'Drop-in16 is excluded in that unchanged uniform-width/TP domain. Smaller logical counts,nonuniform/segmented packing,multiple descriptors or row-owner shifts are alternative implementations,not one shape edit.',
'Retain request boundaries,valid masks,global/local counts,slots,all target/draft metadata,peer collective agreement and captured bank lifetime. Every affected kernel/consumer must understand the new logical versus physical domain.',
'Reprice padding work,extra packing/launches,per-card maximum owner demand and receiver expert work jointly. Fewer allocated rows need not reduce the bottleneck cut or full MoE receiver demand.'),
('H07','Raw-token graph input and graph-external embedding can become one selected authoritative production path when duplicate values/consumers are proved.',
'H07/H02/H06/F02/F24/F31 overlap raw-ID,embedding,graph-bank and Engram readers. Existing64replay associations are provenance,not independent field-deadness proof.',
'Preserve all raw-ID/embedding users,parameter/row/layout epochs,capture addresses and optional renorm or weight mutations. max_norm/renorm needs its original observable effect or an independent idempotence certificate.',
'Count one exact duplicate producer only after all reads/writes are reconciled;no automatic embedding-table HBM credit. Bank live memory and producer/collective readiness remain priced.'),
]
assert {x[0] for x in answers}=={x['id'] for x in remaining}

rows=[]
allc={x['id']:x for x in model+framework}
for cid,op,interaction,guards,cost in answers:
    cs=[x['id'] for x in oldclauses['clauses'] if cid in [x.get('a'),x.get('b')]]
    hs=[x['id'] for x in oldclauses['higher_order_groups'] if cid in x.get('members',[])]
    rows.append({'id':cid,'name':allc[cid].get('name',allc[cid].get('title')),'composed_opportunity':op,'mutual_overlap_and_alternative_graphs':interaction,'semantic_numerical_ownership_and_state_contract':guards,'necessary_work_and_joint_limit_route':cost,'reused_pair_clauses':cs,'reused_higher_order_groups':hs,'state':'SCOPED_COMPOSED_ANALYSIS_WITH_EXPLICIT_RUNTIME_PREMISES','measured_gain':None,'new_implementation':False})
write('remaining23-composed-family-review.json',{'status':'ALL23_REMAINING_REGISTERED_FAMILIES_COMPOSED_IN_SHARED_CONTRACTS','families':rows,'count':len(rows),'existing19_joint_graph':str(N/'shared-profile-analysis-v173/joint-operator-graph-and-resource-review.json'),'all42_registered_family_coverage':True,'candidate_universe_exhaustiveness_not_implied':True,'all_runtime_gates_or_all_implementations_legal_not_proved':True,'overall_complete':False})

# Reuse actual existing source templates instead of inventing a falsely ordered
# full runtime graph. Their typed edges remain conditional and hierarchical.
g5=read(O/'operator-graph-G5-v88.json');attn_ids={'ATTN_SWA','ATTN_C2_SOURCE','ATTN_C2_CONSUMER','ATTN_C1_SOURCE','ATTN_C1_INDEX_SOURCE','ATTN_C1_CONSUMER'}
templates=[]
for t in g5['templates']:
    templates.append({'id':t['id'],'source':t.get('source'),'candidate_bindings':[],'graph_source':str(O/'operator-graph-G5-v88.json'),'original_source_graph':t,'runtime_instance':False,'phase_layer_rank_iteration_namespaces_must_remain_distinct':True})
for gid,modules in [('METADATA_AND_STATE_PUBLICATION',['M04','M21','M22','M24']),('SCORE_SELECTION_AND_REJECTION',['M16','M17','M20'])]:
    templates.append({'id':gid,'candidate_bindings':[],'source_contract_modules':modules,'source_contract':str(O/'module-phase-branch-matrix-v173.json'),'primitive_graph_complete':False,'runtime_instance':False,'phase_layer_rank_iteration_namespaces_must_remain_distinct':True})
footprints={x['id']:x for x in read(O/'candidate-operator-footprints-v154.json')['candidates']}
bound={};gids={x['id'] for x in templates}
attn_by_id={x['id']:x for x in g5['templates'] if x['id'] in attn_ids}
anchors={'F05':['q_norm','q_quant','q_b'],'F06':['q_a','kv'],'F07':['q_b','q_rope'],'F08':['kv','kv_norm','kv_rope','swa_store'],'F09':['iq','iq_rope','iq_quant'],'F10':['iw','iw_scale'],'F11':['ik','ik_norm','ik_quant','ik_store'],'F13':['attention'],'F14':['inverse_rope','o_a'],'F15':['pad','restore_heads','o_a'],'F16':['o_a','o_b','reduce_scatter'],'F17':['compress_proj','score_proj'],'F18':['compress_norm'],'F19':['compress_proj','ik','cmp_store'],'F32':['q_a','kv']}
for c in model:
    cid=c['id'];fp=footprints[cid];gid=fp['original_template_graph_id']
    # Some earlier mappings name a supertemplate; bind exact named atoms to the
    # same declared source template where available,retaining the original map.
    matches=[x['id'] for x in templates if x['id']==gid]
    if cid in anchors:matches=[k for k,t in attn_by_id.items() if set(anchors[cid])<={x['id'] for x in t['nodes']}]
    elif cid=='F12':matches=[k for k,t in attn_by_id.items() if {'qli','indices_in'}&{x['id'] for x in t['nodes']}]
    elif cid=='F33':matches=['ATTN_C1_SOURCE']
    elif cid=='F31':matches=['METADATA_AND_STATE_PUBLICATION','DSPARK_BOUNDARIES','DSPARK_SHARED_KV']
    elif cid=='F35':matches=['SCORE_SELECTION_AND_REJECTION','DSPARK_BOUNDARIES']
    if not matches and gid and gid.startswith('ATTN'):matches=sorted(attn_ids)
    if not matches:
        assert gid is not None,('missing explicit semantic template',cid)
        matches=[gid]
        templates.append({'id':gid,'candidate_bindings':[],'original_source_graph':None,'original_footprint_supertemplate':True,'runtime_instance':False,'phase_layer_rank_iteration_namespaces_must_remain_distinct':True});gids.add(gid)
    for t in templates:
        if t['id'] in matches:t['candidate_bindings'].append(cid)
    bound[cid]={'candidate':cid,'templates':matches,'original_template_graph_id':gid,'semantic_template_binding_is_not_native_instance_attribution':True,'original_atom_union':fp['operator_atom_union'],'original_node_mapping':fp.get('original_node_mapping'),'atom_union_is_navigation_not_same_invocation_proof':True,'composition_contract':str(N/'shared-profile-analysis-v173/joint-operator-graph-and-resource-review.json') if cid in first else str(H/'remaining23-composed-family-review.json')}
host_bind={
'H01':['HOST_COMPLETION_NOTIFY'],'H02':['HOST_INPUT_PACK','GRAPH_INPUT_BANK'],'H03':['HOST_PREFIX_HASH'],'H04':['HOST_TOKENIZE'],'H05':['HOST_RESPONSE_READY'],'H06':['GRAPH_INPUT_BANK','PACKING_BUCKET_DOMAIN'],'H07':['GRAPH_INPUT_BANK','EMBEDDING_ALTERNATIVE']}
for c in framework:
    cid=c['id'];bound[cid]={'candidate':cid,'templates':host_bind[cid],'original_framework_modules':c['modules'],'composition_contract':str(H/'remaining23-composed-family-review.json')}
    for gid in host_bind[cid]:
        t=next((x for x in templates if x['id']==gid),None)
        if t is None:t={'id':gid,'candidate_bindings':[],'runtime_instance':False,'conditional_host_or_public_boundary':True};templates.append(t)
        t['candidate_bindings'].append(cid)

ports=[
{'from':'HOST_TOKENIZE','to':'HOST_PREFIX_HASH','value':'exact admitted token IDs plus original hash namespace','full_operator_barrier_required':False},
{'from':'HOST_PREFIX_HASH','to':'HOST_INPUT_PACK','value':'current scheduled cache/slot/request snapshot,after original admission/lookup/allocation','full_operator_barrier_required':False},
{'from':'HOST_INPUT_PACK','to':'GRAPH_INPUT_BANK','value':'typed ordered fields,valid/padded extents,staging and copy-ready events','full_operator_barrier_required':False},
{'from':'GRAPH_INPUT_BANK','to':'EMBEDDING_ALTERNATIVE','value':'raw IDs or the selected exact precomputed embedding path,all other raw-ID users retained','full_operator_barrier_required':False},
{'from':'EMBEDDING_ALTERNATIVE','to':'MHC_BLOCK_AND_CROSS_LAYER','value':'first target residual bank and owner namespace','full_operator_barrier_required':False},
{'from':'ENGRAM','to':'MHC_BLOCK_AND_CROSS_LAYER','value':'current layer residual plus ready/mask event;existingG1placesEngram before its selected layer HC-attention pre','full_operator_barrier_required':False},
{'from':'MHC_BLOCK_AND_CROSS_LAYER','to':'ATTN_*','value':'exact normalized/collapsed input and coefficient/residual dependencies','full_operator_barrier_required':False},
{'from':'ATTN_*','to':'MHC_BLOCK_AND_CROSS_LAYER','value':'restored/projected output with original rounds and all HC/aux/KV observers','full_operator_barrier_required':False},
{'from':'MHC_BLOCK_AND_CROSS_LAYER','to':'MOE_WRAPPER','value':'exact FFN norm dual BF16/FLOAT outputs and router/expert/merge dependencies','full_operator_barrier_required':False},
{'from':'MOE_WRAPPER','to':'MC2_INTERNAL_KNOWN_STAGES','value':'IDs/maps/scales/masks and current receiver work under preserved peer protocol','full_operator_barrier_required':False},
{'from':'MOE_WRAPPER','to':'MHC_BLOCK_AND_CROSS_LAYER','value':'original routed/shared scale/add and HC-post result;next layer is a new layer namespace','full_operator_barrier_required':False},
{'from':'MHC_BLOCK_AND_CROSS_LAYER','to':'DSPARK_BOUNDARIES','value':'all demanded raw/normalized/head/aux/KV rows,not sampled rows alone','full_operator_barrier_required':False},
{'from':'DSPARK_BOUNDARIES','to':'DSPARK_SHARED_KV','value':'current selected context/row/position/cache generations under actual accepted/proposed-state readiness','full_operator_barrier_required':False},
{'from':'DSPARK_BOUNDARIES','to':'HOST_RESPONSE_READY','value':'accepted output IDs,all required fields/errors/finish state and actual readiness','full_operator_barrier_required':False},
{'from':'HOST_RESPONSE_READY','to':'HOST_COMPLETION_NOTIFY','value':'per-request output completion/queue ownership,not just a global tensor pointer','full_operator_barrier_required':False},
]
global_rules=[
{'id':'IDENTITY','rule':'For shared work or memoization,match actual ordered row/value/parameter/config/position/state generations and compatible numeric contracts. A source atom union,shape,pointer or candidate name is not a same-invocation identity proof.'},
{'id':'OVERLAP','rule':'Interfaces and useful work touched by multiple candidates are charged once in the selected transformed graph. F05/F09 quant, F06/F17/F33 input/projection, F08/F11/F19 KV and F14/F15/F16 output surfaces are explicit examples.'},
{'id':'NUMERICS','rule':'Retain every original round/reduction/tie/NaN/scale/RNG/error effect unless a declared equivalence proof permits changing it. Real algebra or final BF16 dtype alone does not prove IEEE equality.'},
{'id':'ALTERNATIVES','rule':'Choose one consistent representation/partition/publication schedule at each interface. Full versus sharded weights,early projection versus raw gather,fused attention versus split continuation,direct cache versus COW/recompute,and row-worker versus expert migration are alternative domains,not additive gains.'},
{'id':'STATE_AND_LIFETIME','rule':'Union all read and write effects,including pending writers,captured/native/aux/host observers,rollback and future cache readers. Preserve every producer-ready and final-retirement edge;no whole-operator barrier unless necessary.'},
{'id':'RESOURCE','rule':'All target/draft/layer/iteration instances share actual finite pools. Current24Cube/48Vector per exposed device,paired-card/TP/EP cuts and AICPU-sharing caveats apply globally;host capacity is a separate symbolic resource.'},
{'id':'BOUNDS','rule':'Necessary transformed work and genuine capacity uppers yield lower bounds. Combine independent resources by max;accumulate only proved nonduplicate necessary work on the same resource. Do not invert a feasible schedule,fastest measurement or rated reference into a physical universal upper.'},
{'id':'NETWORK','rule':'Re-derive required information per card/owner after fusion/repartition/replication. Possible forwarding/reuse prevents blind sum of rank-pair traffic or TP/EP cuts;prove disjoint necessary bytes before a shared-ingress sum.'},
{'id':'ITERATION','rule':'Cross-iteration edges carry an iteration offset and accepted/proposal/state generation. Prefetch depending on a new Markov proposal waits for that ID or pays speculation;A alone cannot release a proposal-dependent prefetch.'},
{'id':'PARENT_CHILD','rule':'Hierarchical macro/children are one work population. M19 or fullattention/MoE parents must not be added to their already counted primitive children. The source templates are interfaces and alternative branches,not simultaneous launches.'},
{'id':'OBSERVATION_SCOPE','rule':'Do not mix v164PMU times with v168receiver work,old166host timings with current171 causal savings,or missing/ambiguous observations with zero work. Current positive config labels narrow only their exact source conditions.'},
{'id':'COMPLETENESS','rule':'Every registered35+7family is routed here. This does not prove all possiblemodel/sourcevariants or all implementations enumerated;that requires an independent source/graph/obligation audit rather than a pair count or a finished report.'},
]
write('all42-composition-coverage.json',{'status':'ALL_REGISTERED35_PLUS7_HAVE_SHARED_SUBGRAPH_AND_GLOBAL_INTERACTION_CONTRACTS','candidate_bindings':list(bound.values()),'templates':templates,'cross_template_ports':ports,'global_higher_order_constraints':global_rules,'reused19_joint_graph':prior,'reused212_pair_clauses':str(O/'candidate-relation-clauses-v154.json'),'all42_registered_families_routed':set(bound)==set(allc),'all_possible_pairs_or_subsets_not_enumerated':True,'no_automatic_independence_for_nonadjacent_or_disjoint_atoms':True,'candidate_universe_complete':False,'overall_complete':False})

checks={'42candidateIDs_exact':set(bound)==set(allc) and len(bound)==42,'23previously_missing_contracts':len(rows)==23 and {x['id'] for x in rows}==set(allc)-first,'all35originalatom_unions_preserved':all(bound[k]['original_atom_union']==footprints[k]['operator_atom_union'] for k in footprints),'all42have_composition_contract':all(x['composition_contract'] for x in bound.values()),'all12existing_source_templates_retained':{t['id'] for t in g5['templates']}<={t['id'] for t in templates},'no_new_measured_gain_claim':all(x['measured_gain'] is None and not x['new_implementation'] for x in rows),'shared_resource_lifetime_iteration_rules_present':{'RESOURCE','STATE_AND_LIFETIME','ITERATION','PARENT_CHILD'}<={x['id'] for x in global_rules}}
assert all(checks.values()),checks
write('all42-composition-checks.json',{'status':'PASS','checks':checks,'registered_family_coverage':42,'scope_exhaustiveness_or_native_legality_proved':False,'overall_complete':False})
print(json.dumps({'status':'PASS','registered_families':len(bound),'new_composed_contracts':len(rows),'source_and_host_templates':len(templates),'joint_constraints':len(global_rules),'overall_complete':False}))
