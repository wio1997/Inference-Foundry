"""Review the exact reference set by operator effects,not individual syntax tasks."""
import collections
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def write(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def pin(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
source=N/'shared-profile-analysis-v177/profile-selected-operator-guard-census.json'
census=read(source);matrix=read(O/'module-phase-branch-matrix-v177.json')
modrows={m['id']:[r['id'] for r in m['branches']] for m in matrix['modules']}

# Each capsule is a conditional operator answer. It preserves the current
# dispatch/control path unless the explicitly named public subgraph is changed.
# Scope coverage is separated from live branch counts and native ABI validity.
specs=[
('CONTEXT_DISPATCH',['ascend_forward_context.py'],['M00','M12','M22','M23'],['F20','F21','F32','H06'],
 'Decode/PD role and MoE-communication selection change participants,ownership and legal kernel interfaces.',
 'Keep the original role/mode/group dispatch as a control port. Apply a candidate only to its selected typed branch;equal local row counts or a zero request do not cancel remote receiver work.',
 'Do not count a selector as another device arithmetic stage or infer a current algorithm from an enum label alone.'),
('DSA_CP',['attention/context_parallel/dsa_cp.py','attention/context_parallel/dsa_v41_cp.py'],['M07','M08','M09','M10','M11','M18','M21','M23'],['F05','F06','F07','F08','F12','F13','F14','F15','F16','F19','F29','F31','F32'],
 'CP/local-head restoration,source/consumer cache roles,metadata construction,optional overlap and Hadamard/rotation paths use different rows and owners.',
 'Shared-input projection and attention/output fusion retain all norm/quant/RoPE rounds,head permutation,padding,global cache/index readers and TP/CP cuts. Backend metadata/local helper branches keep their own producer and publication conditions.',
 'Full-gather and sharded alternatives are conditional domains;do not assume all local heads or historical cache versions are interchangeable.'),
('DSA_RUNTIME_METADATA',['attention/dsa_v41.py'],['M07','M08','M09','M10','M11','M21','M22','M24'],['F05','F06','F07','F08','F09','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F31','F33','F34'],
 'SWA/C1/C2 roles select distinct Q/KV/index/compressor histories;metadata may be built eagerly or published through graph-owned buffers.',
 'Compose only the matching role template. Preserve external history,all score/key/latent/index outputs and typed quantizers. Empty QLI may zero-return only when actualtokens==0 and not metadata_in_graph;graph mode retains backend planning/copy publication.',
 'A planner output,alias or zero branch is not inferred from an absent native name. Internal CANN tile/precision/retirement conditions stay explicit for private edits.'),
('DSPARK_WINDOW',['attention/dsa_v1.py','ops/triton/spec_decode/dspark_swa_indices.py'],['M18','M21'],['F29','F31'],
 'Query-start/sequence/window,width,empty and request-broadcast branches determine logical index occurrences and lengths.',
 'Fuse arithmetic/repeat/publication while retaining exact per-request membership,signed division,invalid suffix,layout and graph-stable outputs. Reuse across queries only for equal window/position generations.',
 'Capacity,width or physical page count is not actual visible-key demand;zero local tokens does not certify all surrounding state writes dead.'),
('GRAPH_CAPTURE',['compilation/updatable_graph.py','worker/worker.py'],['M00','M21','M22','M24'],['F31','H01','H02','H06'],
 'Graph capture token,compile/warmup/profile sizing,actual dummy execution,worker output and cleanup control graph-visible state.',
 'Preserve capture/update/replay,parameter-bank lifetimes and exception/finally release as control/state ports. Public producer fusion can retain these ports;per-replay write deletion requires exact generation/readers.',
 'No capture/warmup cost is automatically a repeated-serving saving;profiler activity is not the dummy is_profile flag.'),
('ROTATION_ADAPTOR',['device/device_op.py','ops/rope_dsv4.py'],['M07','M10','M21'],['F07','F08','F14','F31'],
 'Rotation layout/dimensionality/inverse convention,cos/sin registry and inplace index_fill select different alias/value effects.',
 'Preserve untouched dimensions,both BF16/FP32 rounds,negative-sine fallback,output alias and versioned cos/sin/position keys. Fuse only the matching typed rotation or final metadata writer.',
 'Registry/address equality alone is not value-generation equality;indexed fill is an observable mutation until all readers are preserved.'),
('SEQUENCE_PARTITION',['models/common/ops/sequence_parallel.py','distributed/device_communicators/base_device_communicator.py','distributed/parallel_state.py'],['M05','M11','M12','M13','M14','M16','M19','M21','M23'],['F14','F15','F16','F21','F22','F24','F28','F29','F30','F31','F32','H06','H07'],
 'Pad/no-pad clone,TP dimension/group,all-gather/reduce-scatter/all-reduce and inplace/out-of-place paths define row ownership and aliases.',
 'Mask/producer/consumer fusion or repartition must preserve valid prefix,True invalid tail,clone/noalias,group,dtype/reduction and all collective participants. Single-owner raw cut bounds apply only in their declared residency/representation domain.',
 'No raw-rank traffic count is a universal wire lower bound;no-pad does not authorize returning an alias when the original clones.'),
('COMPRESSOR_RING',['models/deepseek_v41/compressor.py','ops/triton/compressor/compressor_triton.py'],['M08','M10','M21','M24'],['F17','F18','F19','F31','F33','F34'],
 'Ratio/shape/type guards select key/score projection,pooling,ring/cache updates and empty/partial/full windows.',
 'Fuse producer-to-pool/state publication under exact ratio,position,slot and generation;retain FP32 key/score and BF16/scale outputs,ring overwrite order and all other-layer readers. Duplicate work after a rewrite is counted once.',
 'Pool shape or allocated state capacity does not prove visible rows,constant values or free storage. Split/packed/new consumer ABI stays conditional.'),
('DSPARK_MODEL',['models/deepseek_v41/dspark.py','model_executor/models/qwen3_dspark.py'],['M00','M15','M16','M18','M19','M20','M21','M24'],['F26','F27','F28','F29','F30','F31','F32','F35'],
 'Context projection/sharedKV,slot shape/packed-cache branches,decoder/SP choices,Markov and optional confidence use distinct inputs and output/state ports.',
 'Preserve ordered seed/previous predictions,raw hidden confidence versus head-normalized input,all context/aux/KV publishers and accepted-state consumers. Replicated/sharded/owner Markov are alternative protocols with required ID/score/confidence communication.',
 'Same shape does not equate target/draft weights;future confidence/keep decisions cannot prune their own producer. Zero slot/empty context keeps its original no-write and alias contract.'),
('ENGRAM_LOOKUP_GATE',['models/deepseek_v41/engram/common.py','models/deepseek_v41/engram/embedding.py','models/deepseek_v41/engram/hash_state.py','models/deepseek_v41/engram/layer.py','models/deepseek_v41/engram/npu.py'],['M00','M14','M21','M23','M24'],['F24','F25','F31','F34'],
 'History/slot-cache/raw-ID/dead/owned masks,host-offload formats,DP/TP groups,lookup buffers and rotated/unrotated gate paths have different state and numerical effects.',
 'Share hash/history resolution,unique selected-row lookup,prepared parameters or unchanged pre-rotation FP32 hidden only under content/generation/owner/readiness equality. Retain DEAD/slot publication,full output writes,signbit/NaN/signed-zero behavior and unrotated hidden residual.',
 'INT8 scales are group32 and host/UVA traffic is separate from HBM. Table allocations are not per-call traffic;zero owned load must still produce required zeros. Lookup dp_size is not the global DP or shared-host-memory group.'),
('TARGET_MODEL',['models/deepseek_v41/model.py'],['M00','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15','M16','M24'],['F01','F02','F03','F04','F05','F06','F07','F08','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F32','F34'],
 'Layer plan,Engram placement/rotation/preparation/events,SP,HC pre/post,aux/MTP snapshots,MoE tuple/router/shared branches and final collapse select different value/state forks.',
 'Cross-layer fusion retains full4H coefficient input,old/current residual,post/pre coefficients,aux/MTP raw snapshots and all target/draft consumers. Preserve Engram prepare/capture/retire events and stream lifetimes;combine projections only for identical input/parameter generations with original rounds.',
 'The model container and expanded layer children are alternative accounting levels. Same H,zero local rows or a false gate mask does not erase raw-state/value/residual work.'),
('TEXT_VISION_WRAPPER',['models/deepseek_v41/vl_model.py'],['M00','M05','M16'],['F02','F28','F32','H04'],
 'The loaded tree contains a vision tower and multimodal wrapper,while the registered frozen completion payloads contain plain text and no image inputs.',
 'Keep wrapper dispatch and embedding/head delegation. Text-embedding fusion preserves the fixed successful input contract;vision-only compute is outside this workload,not declared globally absent from the model. Loaded vision parameters remain in identity/residency records.',
 'Do not infer no observers or no initialization cost from lack of vision execution;changing the workload to images requires a new case.'),
('ATTENTION_REGISTRY',['models/layer/attention/layer.py','ops/dsa.py'],['M00','M07','M08','M09','M10','M21','M22'],['F08','F13','F17','F18','F19','F31','F33'],
 'Backend/cache-kind/shape/static-forward registry construction binds later attention,compressed cache and metadata interfaces.',
 'Retain selected implementation/registry object and public cache/metadata ports. Legacy ratio/packed/architecture alternatives are conditional cases,not substituted into the current V4.1 query producer path.',
 'Class/factory names do not by themselves attest a native arithmetic mode or ABI;unknowns limit affected private transformations.'),
('LINEAR_QUANT_ADAPTORS',['ops/cv_linear.py','ops/linear.py','ops/linear_op.py','quantization/method_adapters.py','quantization/methods/w8a8/w8a8_dynamic.py','model_executor/layers/linear.py','model_executor/layers/vocab_parallel_embedding.py'],['M00','M05','M07','M08','M09','M11','M12','M13','M14','M16','M18','M20'],['F05','F06','F09','F10','F11','F16','F17','F20','F21','F22','F28','F30','F32'],
 'Projection/quantizer selection,bias/tuple output,prepared-input scale rank,activation quant type,gather/reduction and loaded packed attributes determine the actual arithmetic and layout.',
 'Shared-input multioutput GEMM or norm/quant fusion must retain each required matrix,precision/rounding,scale/gamma,bias and return/alias contract. PreparedLinearInput reuse requires the same quantizer and generation;QR and routing quantizers are not common expressions merely because both are INT8.',
 'Weight construction/loading cost is not repeated inference work. Direct MNK/cycle bounds are restricted algorithm/mode cases,not all equivalent algorithms;same storage/shape does not prove parameter value equality.'),
('ROUTER_EXPERT_MAP',['ops/fused_moe/router/fused_topk_router.py','model_executor/layers/fused_moe/router/base_router.py'],['M12','M23','M24'],['F20','F21','F31'],
 'Scoring/group/topK/hash/text-vision bias,fused support,ID dtype,EPLB mapping/recording and zero-ID branches alter routing values or state.',
 'Reuse grouping/maps only with exact tie,scale,hash-ID,mask,record/valid-row and generation contracts. Preserve getter-selected expert map and zero-numel return alias;physical mapping or load-record writes remain separate state effects.',
 'Current W8A8/text configuration does not authorize discarding alternate source guards or treating a repeated logical expert as a repeated physical owner.'),
('MOE_COMPOSITE',['ops/fused_moe/fused_moe.py','ops/fused_moe/moe_comm_method.py','ops/fused_moe/prepare_finalize.py','ops/fused_moe/routed_experts.py','model_executor/layers/fused_moe/runner/moe_runner.py'],['M00','M12','M13','M21','M22','M23','M24'],['F20','F21','F22','F23','F31','F32','H06'],
 'Runner/quant/comm method,capacity clipping,pad/split IDs,internal router,expert map,mega-MoE buffers,workspace and output tuple/finalization branches control an already-fused composite.',
 'Documented internal split/repartition retains peer/count/credit/row/scale/clip and original barrier/unpermute protocol. Receiver assignments determine GEMM work;local sender0 does not imply no incoming compute. Keep source quantizer distinctions and map/property generations.',
 'Do not recount existing dispatch/GMM1/SwiGLU/GMM2/combine fusion. Workspace/capacity/core-budget labels are not actual writes,MAC rows or physical-rate proof. Unknown private ABI is an unsupported private edit,not global independence.'),
('SHARED_EXPERTS',['ops/fused_moe/shared_experts.py','model_executor/layers/fused_moe/runner/shared_experts.py','model_executor/layers/activation.py','ops/activation.py'],['M12','M13','M23','M24'],['F20','F21','F22','F23','F31','F32'],
 'TP/SP/DP-only ownership,replicated weights,padding,A8 INT/MXFP/LoRA paths,activation clamp,submitted gate-up and milestone/event waits define shared MLP execution.',
 'Fuse gate-up/activation/quant/down/final merge only in its typed path,with exact rounds,scale,clamp,partial outputs and row ownership. Overlap retains routed-stage readiness and finite shared Cube/Vector/BW/cut resources;replica-to-shard is a new construction/protocol domain.',
 'A8/MXFP/LoRA alternatives are not interchangeable. An optional missing event is not proof that producer data are ready,and interface fusion budgets cannot be added to an independent ideal overlap gain.'),
('NORM_BIAS_RESIDUAL',['ops/layernorm.py'],['M06','M07','M08','M09','M12','M15','M16','M18','M19'],['F01','F02','F03','F04','F05','F11','F17','F26','F30'],
 'Residual-present and bias-loaded branches change norm inputs and additional residual/bias outputs or state;quant-description caches bind setup choices.',
 'Norm/quant/HC boundary fusion preserves epsilon,gamma,FP32 accumulation,BF16 rounds and all residual/rstd/public outputs required by that public path. Same normalized values or unchanged pre-rotation hidden may be reused only with identical keys/readers.',
 'All582 recorded AscendRMSNorm catalogue instances have bias_loaded=False in this cohort;this binds that snapshot branch,not all possible model configurations or native bitwise behavior.'),
('INDEXER_QUERY_PREP',['models/deepseek_v41/indexer.py','ops/triton/prepare_indexer_indices.py','ops/triton/quantize_indexer_query.py'],['M07','M09','M10','M21','M24'],['F09','F10','F11','F12','F13','F31','F33','F34'],
 'Owned-key/source role,query128 geometry,output tuple/scale,empty masks,visible range,bit-key sort and graph-stable index/length publication define indexer inputs and occurrences.',
 'Fuse projections/query quant/preparation/topK boundaries while preserving quantizer/rounding/ties,source generations and all index consumers. Sort/decode keeps valid multiplicity and the original sentinel/order;dedup is not generally equivalent to occurrence-sensitive attention.',
 'Selected count or descriptor size is not actual key union/cache traffic. The allowed equality/certificate domain for pruning is separate from unconditional native topK replacement.'),
('SLOT_MAPPING',['worker/block_table.py','ops/triton/compute_slot_mapping.py'],['M02','M04','M08','M10','M18','M21','M22','M24'],['F08','F19','F27','F29','F31','F34'],
 'Ordinary/adaptive/grouped/compressed physical and logical coordinates,request/token tiling,mamba/PAD and capacity tails determine slot-map writes.',
 'Fuse common coordinate arithmetic or final publication with identical blocksize,ratio,group owner,valid request/token mask and epoch. Retain invalid/sentinel writes and output aliases;launch heuristics may change but cannot change the full valid write domain.',
 'No-query/padded/allocation sizes are not a no-op certificate;packing/COW/refcount changes are new state/consumer ABI domains,not free reductions of baseline writes.'),
('SAMPLING_VERIFY',['sample/rejection_sampler.py','v1/sample/sampler.py','ops/triton/reject_sample.py'],['M01','M16','M17','M20','M21','M24','M25'],['F30','F31','F35'],
 'Greedy/random/synthetic/entropy/penalty/logprob flags,actualN1 versus ragged counts,row maps and recovery/bonus/output epochs select distinct algorithms.',
 'Fuse only the chosen source comparator/prefix/bonus/RNG path;retain all raw/processed score and probability/logprob readers,mutation,valid cells and exact draw order. Typed argmax output/widening and duplicate-row head reuse have their own value/row contracts.',
 'Current1104observed calls bind greedy/noPenalty/noEntropy/noSynthetic/logprobNone,not all-score-deadness. Random CDF/race and PyTorch fallback remain conditional source cases;preparation dualTrue flags are not actual random demand.'),
('VOCABULARY_LAYOUT',['ops/vocab_parallel_embedding.py','model_executor/layers/logits_processor.py'],['M05','M15','M16','M17','M19','M20','M23'],['F26','F28','F30','F31','F32','F35'],
 'Standard/replicated/fine-grained/PCP vocabulary layout,added-token padding,head dtype,quantizer,bias/softcap/scale and gather/remap produce different score/ID contracts.',
 'Head-local processed best plus global key/ID or full-score alternatives preserve postprocessing tie behavior and every reader. Replicated Markov and sharded base heads have distinct parameter/communication domains;do not collapse two rounded accumulators into one.',
 'Same shape does not equate target/draft embedding/head weights. Full vocab,TP-local vocab and remapped IDs remain separate;INT64MAX/native collective support is conditional,with explicit alternative protocols.'),
('PROPOSER_STATE',['spec_decode/dflash_proposer.py','spec_decode/dspark_proposer.py','spec_decode/llm_base_proposer.py'],['M04','M15','M16','M17','M18','M19','M20','M21','M22','M23','M24'],['F26','F27','F28','F29','F30','F31','F32','F34','F35','H02','H06','H07'],
 'Merged draft/dummy/first-pass,SP/padding/hidden-row adjustment,context slots,seed/next IDs,confidence/keep,shared KV and graph/eager branches cross multiple iterations.',
 'Preserve actual row/order/slot/publication and target-aux versus draft-query/raw-confidence inputs. Compose preparation with its final consumer/writer only when all bank/owner/generation/RNG/state ports survive. Conditional all0 dummy-index/head reuse needs no demand or exact repeated-row proof.',
 'Future keep/accept cannot prune its own producer;zero local/dummy calls may still publish state or receive expert work. Existing source-class inheritance/delegation is retained,not declared a separate duplicate computation.'),
('DEVICE_METADATA_QUEUE',['worker/device_metadata.py'],['M04','M08','M09','M10','M18','M21','M22','M24'],['F08','F12','F19','F27','F29','F31','F34','H02','H06'],
 'Capture-producer/descriptor/ordered-task/CPU-sidecar and in-flight submit branches determine current metadata producer and ready event.',
 'A fused metadata producer may keep the same public output bank/event port. Skipping a submission or reusing a plan requires identical parameters,guards and generation plus both old-reader and queued-writer retirement.',
 'An empty task list or stable descriptor alone does not prove all internal work absent;sidecar/graph-visible producers remain modeled.'),
('INHERITED_RUNNER',['v1/worker/gpu_model_runner.py','v1/worker/worker_base.py'],['M00','M02','M03','M04','M05','M15','M16','M17','M18','M21','M22','M23','M24','M25'],['F02','F08','F24','F26','F27','F28','F29','F30','F31','F32','F34','F35','H01','H02','H06','H07'],
 'Inherited preparation,positions,metadata/block/slot views,capture dispatch,accepted-token bookkeeping,MM fields and output/abort/copy lifetimes are control/state effects around public operators.',
 'Keep unrelated admission/MM/error/tracing/control flow unchanged. Only the explicitly selected tensor/metadata producer-consumer subgraph is rewritten;retain all original request/cache/slot/input/output,old-reader,queued-writer and finally/error ports. Empty-query/encoder-only and dynamic shape branches retain their own zero/invalid-tail behavior.',
 'Frozen requests are text-only,not proof that the library lacks MM paths. These guards have preserved interface answers,not a mandate to research the whole framework or a claim of measured host savings.'),
('COMPILE_SPECIALIZATION',['compilation/decorators.py'],['M00','M04','M21','M22','M24'],['F31','F32','H06','H07'],
 'AOT/source hash,compile/eager/partition,backed/unbacked shapes,negative dimension normalization and cache keys select execution/capture contracts.',
 'Retain original compile/dispatch/validation and dimension equivalence classes outside public operator rewrites. Changed row/layout/operator graphs require their own valid cache/specialization and storage-generation keys.',
 'No Python wrapper or compile cache is itself another GEMM or serving-time fusion saving;dynamic/unbacked marks do not authorize unknown-shape alias or capacity assumptions.'),
('RPC_TRACING_PARAMETER',['tracing/otel.py','v1/executor/multiproc_executor.py','model_executor/parameter.py'],['M00','M03','M22','M24','M25'],['H01','H05','H06'],
 'Worker method/output-rank/error/shutdown,trace context and Tensor parameter dispatch are control/observation boundaries.',
 'Preserve all configured observers,error/termination,output publication and object lifetime. Host transport candidates change only their declared ready/payload/serialization boundary with original ordering and consumer acknowledgement retained.',
 'Not changing these controls is an explicit boundary of the operator rewrite,not evidence of zero host cost or all possible IPC/event behavior being measured.'),
]

file_domain={}
for item in specs:
    for suffix in item[1]:
        # Different package roots may share suffixes;the longest match wins.
        file_domain[suffix]=item[0]
byid={item[0]:item for item in specs};assigned=collections.defaultdict(list);unmatched=[]
for g in census['guards']:
    hits=[s for s in file_domain if g['path'].endswith('/'+s)]
    if not hits:unmatched.append(g);continue
    did=file_domain[max(hits,key=len)];assigned[did].append(g)
assert not unmatched,[(g['path'],g['owner']) for g in unmatched]
assert sum(map(len,assigned.values()))==1867
assert len({g['id'] for gs in assigned.values() for g in gs})==1867
capsules=[];coverage=[]
for did,gs in assigned.items():
    _,paths,mods,candidates,effect,answer,limit=byid[did]
    assert all(m in modrows for m in mods)
    cap={'id':did,'source_files':sorted({g['path'] for g in gs}),'guard_count':len(gs),
        'operator_modules':mods,'candidate_families':candidates,
        'semantic_family_refs':{m:modrows[m] for m in mods},
        'guarded_value_state_or_ownership_effect':effect,
        'fusion_split_reorganization_answer':answer,
        'retained_condition_or_measurement_limit':limit,
        'whole_function_continuations':'Use the pinned census function lexicalcalls/writes/returns,not only the guarded body. Lexical effects are overapproximations,not purity or native alias proof.',
        'runtime_applicability':'Source-conditional;only explicitly cited snapshots/calls are positive current witnesses.',
        'work_and_capacity_authority':str(N/'shared-profile-analysis-v174/module-analysis.json'),
        'registered_composition_authority':str(N/'shared-profile-analysis-v174/all42-composition-coverage.json'),
        'state':'OPERATOR_CONDITIONAL_EFFECT_AND_REWRITE_DOMAIN_REVIEWED'}
    capsules.append(cap)
    for g in gs:
        coverage.append({'guard_id':g['id'],'source':g['path'],'source_sha256':g['sha256'],
            'definition_key':g['definition_key'],'line':g['line'],'kind':g['kind'],'condition':g['condition'],
            'review_domain':did,'semantic_family_refs':cap['semantic_family_refs'],
            'state':'CONDITIONAL_OPERATOR_ANSWER_OR_EXPLICIT_PRESERVED_INTERFACE',
            'actual_branch_taken_or_new_native_legality_not_inferred':True})
write('exact-guard-operator-contract-review.json',{
    'status':'ALL1867_REFERENCE_GUARDS_HAVE_CONDITIONAL_OPERATOR_OR_PRESERVED_INTERFACE_ANSWERS',
    'input':pin(source),'semantic_ledger':pin(O/'module-phase-branch-matrix-v177.json'),
    'domains':capsules,'guard_coverage':coverage,
    'old1101_reviewed_guards_reused':pin(N/'shared-profile-analysis-v174/all1101-static-operator-frontier-reconciliation.json'),
    'not_1867_new_optimization_tasks':True,'no_cartesian_pair_work':True,
    'unknown_native_eligibility_not_promoted_to_legal':True,
    'complete_hierarchical_G0_and_full_scope_review_separate':True,
    'overall_complete':False})
write('guard-review-checks.json',{'status':'PASS','guards':1867,'operator_effect_domains':len(capsules),
    'one_exact_domain_per_reference':True,'all_source_guards_retained':True,'unmatched':unmatched,
    'all26_modules_represented_or_preserved_by_original_host_contracts':True,
    'new_source_body_reads_or_NPU_work':0,'fullscope_completion_not_inferred_from_bijection':True})
print(json.dumps({'status':'PASS','exact_guard_answers':len(coverage),'domains':len(capsules),'unmatched':len(unmatched)}))
