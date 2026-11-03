"""One semantic rule round over existing source templates,not pair enumeration."""
import collections
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def pin(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
np=O/'operator-graph-node-coverage-v124.json';ep=O/'operator-graph-edge-coverage-v124.json'
nodes=read(np)['rows'];edges=read(ep)['rows']
groups=collections.defaultdict(list)
for r in nodes:groups[(r['graph'],r['template'])].append(r)

# Scope-specific answers preserve the public graph's state and numerical
# boundaries. These are analysis contracts,not native lowering validations.
contracts={
('G1',''):('M00-M25/F01-F35/H01-H07',
 'Target40layers and DSpark3blocks remain distinct invocations and parameters;include frontend/prefix/input/output ports from the26module ledger. Auxiliary and Engram branches remain live where guarded. A decoder macro and its expanded children are alternative accounting levels.',
 'Cross-layer/iteration fusion can change scheduling but cannot eliminate residual/aux/state outputs,accepted/rollback transitions or next-invocation RAW. Same shape does not equate weights or generation;all42composed contracts retain shared resource cuts.'),
('G5','ATTN_SWA'):('M07/M10/M11;F05-F08/F13-F16/F32',
 'SWA Q/KV paths have different row domains;retain norm,INT8 rounding,RoPE and persistent KV writes. Output inverseRoPE/headrestore/WoA/WoB and collective consumers remain.',
 'Sibling shared-input projections may share input loads under common ownership;pair/triple/output fusion preserves all rounding and required publication. Collective movement must recompute owner/cut and padding domains.'),
('G5','ATTN_C2_SOURCE'):('M07-M11/M21/M24;F05-F19/F31-F34',
 'QR and QS quantizers are distinct. Compressor key/score FP32 branches,ring state,index score/topk/fixup/publication and historical cache consumers remain. C2complete mask has a persistent8192B backing with24element views in current observations.',
 'Reuse only same row/version/rotation/ring epoch;keep score/key fanout and native index occurrence semantics. Whole public fusion can preserve existing banks;private initializer deletion still needs exact writer/reader ranges and epochs.'),
('G5','ATTN_C2_CONSUMER'):('M07/M10/M11/M24;F05-F08/F13-F16/F34',
 'Consumer-layer reads of prior compressed cache/index state are external generation-sensitive inputs,not missing producers or free recomputation.',
 'Fuse local projection/attention/output boundaries conditionally;cross-layer sharing must retain source publication and consumer layer/query state,not substitute a source layer cache merely because shape matches.'),
('G5','ATTN_C1_SOURCE'):('M07-M11;F05-F19/F33',
 'C1 compressor/index-source contracts differ from C2 ring pooling;keep FP32 key/score projections and selected C1 state consumers.',
 'F33 belongs to its C1 source branch. Shared input/project/quant/tile extensions cannot carry C2 masks or query counts into this branch;all source quantization/rounding and shared-history readers remain.'),
('G5','ATTN_C1_INDEX_SOURCE'):('M07-M11;F05-F19/F33',
 'Indexer-source and KV-source layer roles may differ;the query/index projections do not prove identical cache state or all-reader ownership.',
 'Shared input reuse is conditional on identical value generation and output contracts;index preparation can fuse under F12/F31 while preserving ordered duplicate occurrences and separate compressor publishers.'),
('G5','ATTN_C1_CONSUMER'):('M07/M10/M11;F05-F08/F13-F16',
 'Prior C1 index/cache state is consumed under its own layer/query/row domain and persistent generation.',
 'Preserve all external state reads during local fusion/split/repartition;projected attention work and necessary network cut change only after specifying exact ownership.'),
('G5','MHC_BLOCK_AND_CROSS_LAYER'):('M06/M15;F01-F04/F26',
 'Current coefficient path reads full4H residual while delayed mixing can use prior pre_mix;old residual,current coefficients,post residual and next pre_mix are different values.',
 'Do not substitute the mean or weightedY for full coefficient input. Multioutput coefficient/post/pre and auxiliary observers survive cross-boundary fusion;rounding and coefficient deadlines constrain tile extension.'),
('G5','MOE_WRAPPER'):('M12/M13;F20-F23',
 'Routing/shared-expert branches use distinct quantized inputs and have correction/merge/HC consumers;receiver demand can be positive when local sender rows are zero.',
 'Reuse routing/preprocessing only with identical semantic IDs/rounding;merge/shared/cross-card variants retain counts,credit,overflow,scale and synchronization. Existing dispatch_ffn_combine is already fused.'),
('G5','MC2_INTERNAL_KNOWN_STAGES'):('M12/M23;F21',
 'Counts,dispatch,GMM1,SwiGLUquant,GMM2 and return are known stages inside an existing composite,not new standalone launches. Allocated128row capacity is not necessary MAC rows.',
 'Stage splitting/repartitioning requires matching mailbox,credit,route-row and ownership protocol. Work uses actual receiver assignments;cut/compute/live expert-weight costs are charged jointly,no sender-zero global noop.'),
('G5','ENGRAM'):('M14/M24;F24/F25',
 'Hash/host-UVA lookup/row selection/rotation/gate/masks/events have distinct outputs. Selected host rows and scales matter;table allocation size is not per-call link traffic.',
 'Share unchanged pre-rotation hidden.float under F25 only while retaining unrotated residual and exact rounding. Hash/lookback/DP/TP row reorganization keeps sequence history,mask,owner and event epochs;prefetch adds live storage.'),
('G5','DSPARK_BOUNDARIES'):('M15-M21/M24;F27-F31/F35',
 'Main context projection/sharedKV,seed and embedding,decoder blocks,base logits and ordered Markov additions/confidence precede acceptance. Dynamic policy keep lengths and retrieved output are guarded consumers.',
 'Base-row argmax narrowing and repeated-input head variants retain score banks/readers;Markov steps and RNG states cannot reorder. A dummy call can still publish real state or receive expert rows;allGreedy and no-demand must be separately bound.'),
('G5','DSPARK_SHARED_KV'):('M18/M24;F27/F29',
 'Context concat/project/norm/slot publication and query-context positions determine persistent sharedKV writes.',
 'Cross-target/draft or cross-iteration sharing needs identical loaded parameters,context rows,positions,slots and epochs. Existing consumers/rollback constrain elision;shape or shared buffer identity alone is insufficient.'),
('G5','MHC_ATTENTION_COEFFICIENT_FORK'):('M06-M12;F01-F08/F14-F16',
 'Full4H coefficient path and delayedY path fork from one residual;both join current Post and subsequent FFNPre. Projection,KV publication,attention,output reduction retain actual owners.',
 'Coefficient load sharing is conditional;QR/QS and HC rounds remain separate. Cross-boundary extension must meet coefficient deadline and preserve every fork output. Do not sum macro and primitive lower bounds.'),
('G6','SAMPLER_GREEDY_FALLBACK_ROUTE'):('M17;F35',
 'Actual-count/backend guard chooses N1 or ragged body. Synthetic greedy draws can advance RNG even if greedy IDs are used;later random writer and final output epochs remain separate.',
 'Fused selection/verification retains exact target IDs,all valid output cells and prescribed RNG sequence. Observed nonempty calls have allGreedy=True/allRandom=False;preparation-state dualTrue flags are not a live random-demand witness.'),
('G6','SAMPLER_GREEDY_GENERAL_BODY'):('M17;F35',
 'Ragged starts/counts map draft and target IDs into per-request mismatch,prefix,bonus and retained output cells.',
 'Fuse map/compare/first-failure/prefix/bonus without changing source order,synthetic predicates,ties or masked cells. Required argmax output can be INT32 then widened,while preserving other score readers.'),
('G6','SAMPLER_GREEDY_ACTUAL_N1_BODY'):('M17;F35',
 'N1 eligibility depends on actual counts and source guard,not padded capacity;one-step mismatch and bonus still preserve old output cells.',
 'Specialized pair/triple body is conditional on actualN1;do not apply to ragged requests or move the bonus index. RNG/synthetic acceptance semantics remain the original contract.'),
('G7','ASCEND_DEFAULT_PROBABILITY_CUTOFF_AND_RACE'):('M17;F35',
 'Probability-space K/P cutoffs,original CDF,masks,returned score observers,FP32 softmax and seeded/default exponential draws define this backend.',
 'Fusion preserves threshold tie/mask rules,draw order,record_stream/wait and mixed/allRandom choice. An equivalent ID-only path requires no other score demand;upstream filter is a distinct numerical implementation.'),
('G7','UPSTREAM_MONOLITHIC_PIVOT_FILTER'):('M17;F35',
 'Pivot/quota scans,normalization/workspace generations and final ordered-vocab stores constitute the upstream filter;K/P branches have distinct rounding and ties.',
 'Split only documented stages with retained quota/normalizer/live workspace. Do not replace its distribution by the default Ascend filter based only on equal K/P values or remove returned logits consumers.'),
('G8','RECOVERY_PYTORCH_OWNER_MAP'):('M17/M21;F31/F35',
 'Tzero is a guarded early return;interval comparison,argmax-first-match and unmatched owner0 retain the original output map contract.',
 'Owner-map reuse needs identical cumulative counts,output capacity and epoch. Sharing with metadata preparation preserves empty/unmatched handling and reader lifetime;not any same-sized row map.'),
('G8','RECOVERY_PYTORCH_VALUES'):('M17;F35',
 'Ngram exclusion or P-D clamp,Q-owner gather,zero/Inf guards and final vocab argmax define recovered IDs.',
 'Fuse recovery without reordering rounded subtraction/division or altering bad-Q sentinel/tie rules. Preserve RNG bank and retained P/D/score readers;no blanket full-vocab-bank deletion.'),
('G8','VERIFY_PYTORCH_ORDINARY'):('M17;F35',
 'Selected probabilities and Float64 uniforms drive ordinary per-step acceptance;optional entropy and eager recovery have separate observers.',
 'Fuse compare/valid/first failure/prefix/bonus with original arithmetic and output cells. Entropy=False is observed in current config,but alternate entropy branch remains a conditional contract.'),
('G8','VERIFY_PYTORCH_BLOCK'):('M17;F35',
 'Clamped ratio cumprod and uniform cumprod select last legal accept;block entropy/recovery rules differ from ordinary verification.',
 'Do not commute prefix multiplication or substitute first-rejection logic. Fused scans preserve order,valid/nonplaceholder,last-legal rule and next-recovery/bonus placement.'),
('G8','RECOVERY_OPTIONAL_BLOCKWISE'):('M17;F35',
 'Owner map,host maxN read,sequential capped ratio prefix and guarded piP-D recovery are ordered dependencies.',
 'Fuse only with the same prefix arithmetic and original request Q;host read removal needs a proved bound or exact device substitute. Branches cannot share a normalizer merely from identical shape.'),
('G9','PREPARE_VISIBLE_MULTISET'):('M09/M21;F12/F31',
 'Visible range,sentinel/bit-key sort and decoding preserve sorted occurrences including duplicates;optional length counts valid slots,not unique IDs.',
 'Index-preparation fusion may avoid intermediate casts/sort buffers under the declared typed contract. Deduplication or length changes are not legal for occurrence-sensitive SMLA unless the full consumer algorithm is changed and re-proved.'),
('G9','DSPARK_LOGICAL_WINDOW_BROADCAST'):('M18/M21;F29/F31',
 'Per-request query-length/prefix/window calculations produce repeated logical index rows and visible lengths in graph-stable banks.',
 'Fuse arithmetic/repeat/publication under identical request membership/query counts/window and signed division. Across queries reuse rows only where equal;retain per-query native readers and publication epochs.'),
('G9','SMLA_PA_INDEX_OCCURRENCES'):('M10/M21;F12/F13/F31',
 'Slot division,topk occurrence read,page lookup/range check and KV gather preserve duplicate occurrence multiplicity and logical-to-physical translation.',
 'Fuse gather with attention under the same cache/page generation and bounds. Index width,block size and address ownership are actual inputs;one duplicate index is not redundant attention mass by default.'),
}
assert set(contracts)==set(groups),(set(groups)-set(contracts),set(contracts)-set(groups))
rule_answers={
 'PAIR_PRODUCER_CONSUMER':'Conditional public boundary composition;preserve all source rounds and output aliases/readers. Native single-kernel feasibility and gain remain separate.',
 'TRIPLE_CHAIN_OR_FORK_JOIN':'Extend through a chain/fork only with all side outputs,branch predicates,owner and state epochs retained. The template-specific fork contract applies.',
 'MULTI_EXTEND_ONE_BOUNDARY':'Use the42family joint nonduplicate-work/resource/lifetime contracts. No sum/product of independently ideal speedups;extend only while all public outputs survive.',
 'SPLIT_COMPOSITE_AT_DOCUMENTED_BOUNDARY':'Only explicit source/public stages are split;opaque private stages are conditional unknowns rather than silently invented interfaces. Preserve mailbox/RNG/cache/workspace protocols.',
 'SHARED_INPUT_REUSE':'Same source value,rounding,parameter and iteration generation plus lifetime are required. Different quantizers and target/draft weights remain distinct.',
 'CROSS_MODULE_BOUNDARY':'Preserve named external ports,layer/card/iteration namespaces,owner cuts and published state. Next-invocation dependencies remain explicit offsets.',
 'MULTICONSUMER_PRESERVE_ALL_USES':'Retain every modeled side consumer and alias/state observer. Missing actual identity is a guarded rewrite premise,never an independence/noop conclusion.'}
capsules=[]
for key,rs in groups.items():
    owner,semantics,answer=contracts[key]
    capsules.append({'id':key[0]+'/'+(key[1] or 'WHOLE_MODEL_MACRO'),
        'graph':key[0],'template':key[1],'operators_and_families':owner,
        'original_value_state_and_consumer_contract':semantics,
        'fusion_split_reorganization_answer':answer,
        'rules':{k:{'state':'CONDITIONAL_SOURCE_SEMANTICS_REVIEWED','answer':v} for k,v in rule_answers.items()},
        'node_ids':[r['node_id'] for r in rs],
        'edge_indices':[r['edge_index'] for r in edges if (r['graph'],r['template'])==key],
        'new_non_equivalent_family_in_this_round':None,
        'live_native_allocation_or_complete_scope_universe_proof':False})
node_rows=[{'graph':r['graph'],'template':r['template'],'node_id':r['node_id'],
    'review_capsule':r['graph']+'/'+(r['template'] or 'WHOLE_MODEL_MACRO'),
    'status':'CONDITIONAL_OPERATOR_VALUE_AND_ALL_MODELED_USES_REVIEWED',
    'original_structural_row_preserved_by':str(np),'no_native_epoch_completion_inference':True} for r in nodes]
edge_rows=[{'graph':r['graph'],'template':r['template'],'edge_index':r['edge_index'],'edge':r['edge'],
    'review_capsule':r['graph']+'/'+(r['template'] or 'WHOLE_MODEL_MACRO'),
    'status':'DEPENDENCY_PORT_AND_CONDITIONAL_REWRITE_EFFECT_REVIEWED',
    'iteration_or_non_RAW_not_erased':True,'original_structural_row_preserved_by':str(ep)} for r in edges]
assert len(node_rows)==854 and len(edge_rows)==1346 and len(capsules)==27
assert len({(r['graph'],r['template'],r['node_id']) for r in node_rows})==854
assert len({(r['graph'],r['template'],r['edge_index']) for r in edge_rows})==1346
write('declared-source-template-rule-review.json',{
    'status':'DECLARED27_SOURCE_TEMPLATES_HAVE_ONE_COMPLETE_CONDITIONAL_RULE_REVIEW_ROUND',
    'input_pins':[pin(np),pin(ep),pin(N/'shared-profile-analysis-v174/all42-composition-coverage.json')],
    'capsules':capsules,'node_coverage':node_rows,'edge_coverage':edge_rows,
    'rules':list(rule_answers),'new_non_equivalent_candidate_families':[],
    'registered42_composition_constraints_reused':True,
    'no_cartesian_candidate_pairs_or_new_target_code_reads':True,
    'full_whole_scope_scan_complete':False,
    'not_SCAN_COMPLETE_IN_SCOPE_reason':'This round covers the declared source graphs. Complete observed G0/all effective branch/callee and value-generation coverage still requires explicit reconciliation;the partial native ET is not that proof.',
    'overall_complete':False})
print(json.dumps({'status':'PASS_CONDITIONAL_SOURCE_RULE_ROUND','templates':27,'nodes':854,'edges':1346,'rules':7,'new_non_equivalent_families':0,'whole_scope_complete':False}))
