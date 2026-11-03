import collections
import hashlib
import json
from pathlib import Path
N=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030');O=N/'profile-first-20261031';H=N/'shared-profile-analysis-v178'
def read(p):return json.loads(p.read_text())
def save(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

g=read(H/'hierarchical-G0-with-boundary-and-state-ports.json')
old=read(O/'operator-graph-G1-v88.json')
guards=read(N/'shared-profile-analysis-v177/profile-selected-operator-guard-census.json')['guards']
answers=read(H/'exact-guard-operator-contract-review.json')
bindings=read(H/'candidate-to-public-hierarchy-bindings.json')['rows']
ports=read(H/'host-score-and-state-port-contracts.json')
ids=[n['id'] for n in g['nodes']];ids_set=set(ids)
state_ids={s['id'] for s in ports['state_envelopes']}
checks={
 'unique_hierarchy_node_ids':len(ids)==len(ids_set),
 'all_original338_macro_nodes_retained':{n['id'] for n in old['nodes']}<=ids_set,
 'all_source_and_boundary_edge_endpoints_resolve':all(e[k] in ids_set for e in g['edges'] for k in ['producer','consumer']),
 'exact1867_guard_reference_bijection':collections.Counter(x['id'] for x in guards)==collections.Counter(x['guard_id'] for x in answers['guard_coverage']),
 'all42_candidates_have_explicit_state_envelopes':len(bindings)==42 and all(b['state_envelope_refs'] and set(b['state_envelope_refs'])<=state_ids for b in bindings),
 'all8_source_epochs_present':[e['source_layer'] for e in g['shared_topk_epochs']]==[2,8,14,20,24,28,32,36],
 'range_WAR_including_next_iteration_present':any(e['dependency_kind']=='BANK_RANGE_WAR' and e.get('iteration_offset')==1 for e in g['edges']),
 'no_stale_context_to_query_macro_edge':not any(e['producer']=='dspark.main' and e['consumer']=='dspark.blocks.0' for e in g['edges']),
 'unknown_private_effects_remain_preserved':all(s['actual_all_native_consumers_or_complete_retirement_proved'] is False for s in ports['state_envelopes']),
 'no_endpoint_count_as_full_scope_proof':g['whole_scope_complete'] is False}
assert all(checks.values()),checks
rules=[
 ('PAIR_PRODUCER_CONSUMER',
  'TopK->fixup->publication can fuse locally, but the resulting epoch remains visible to the source and subsequent reuse layers. Dropping publication/register-only forwarding is conditional on preserving every original read. Engram lookup/projection keeps both layer slots.',
  ['F12','F13','F24','F25','F31']),
 ('TRIPLE_CHAIN_OR_FORK_JOIN',
  'Extend the local producer chain with slot/position/valid-range and rounded side outputs intact. Target aux/context, draft query embedding, and raw confidence hidden are different branches; a common macro name cannot merge them.',
  ['F08','F11','F18','F19','F27','F28','F29','F30']),
 ('MULTI_EXTEND_ONE_BOUNDARY',
  'Cross-layer TopK/attention/metadata composition shares one value epoch per source, with last-reader range retirement before overwrites. Whole-range version storage is R_live*512*4 only if choosing that scheme; not a universal16MiB requirement or a latency claim. Retain original collective participants.',
  ['F12','F13','F29','F31','F32','F33','F34']),
 ('SPLIT_COMPOSITE_AT_DOCUMENTED_BOUNDARY',
  'Public index publication/attention and shared-KV publication/query consumption have source boundaries. Private MC2/MemSet/workspace protocols do not gain a legal split from this graph. Retain existing native composites unless separately proved.',
  ['F01','F13','F21','F24','F29','F34']),
 ('SHARED_INPUT_REUSE',
  'Source layer24 index with source layer20 compressed KV is a concrete different-source case. Shared TopK backing is not shared values across source epochs. Global Engram preparation is not charged twice for two consumers. Target/draft head storage identities and replicated Markov layout remain distinct.',
  ['F06','F12','F17','F24','F26','F27','F30','F32','F33','F35','H07']),
 ('CROSS_MODULE_BOUNDARY',
  'Request/row/page/slot generations and graph-bank readiness cross host/model/metadata/cache/output boundaries. Next accepted IDs/lengths feed next iteration, not their own producer. Remote receiver roles survive sender0. No missing profiler record is treated as independence.',
  ['F20','F21','F29','F31','H01','H02','H03','H05','H06']),
 ('MULTICONSUMER_PRESERVE_ALL_USES',
  'Retain index-layer readers, C2 mask views, HC/aux/Engram branches, score/logprob/entropy/RNG interfaces, ordered Markov seed+samples0..3 and output/state readers. Preserve-score argmax+ID conversion remains a separate alternative from stronger dead-score/head claims.',
  ['F01','F03','F12','F19','F23','F25','F26','F30','F31','F35'])]
save('delta-rule-and-evidence-sufficiency-review.json',{
 'status':'CHANGED_DEPENDENCY_RULE_ROUND_REVIEWED;FULL_SCOPE_SEMANTIC_ACCEPTANCE_PENDING',
 'inputs':[pin(H/'hierarchical-G0-with-boundary-and-state-ports.json'),pin(H/'exact-guard-operator-contract-review.json')],
 'rules':[{'rule':r,'answer':a,'affected_families':f,'new_non_equivalent_family':None} for r,a,f in rules],
 'not_cartesian_pair_enumeration':True,
 'new_positive_unconditional_native_eligibility':[],
 'new_performance_or_physical_upper_bound_claims':[],
 'independent_structural_checks':checks,
 'remaining_scope_acceptance_question':
   'Do the27 guarded effect-domain answers and15 conservative state envelopes cover every conclusion-changing public consumer/effect of the retained360semantic rows and the selected hierarchy? Check semantic adequacy,not more reference counts. Unchanged opaque-native interfaces may stay conditional; do not require every private alias or physical mode globally.',
 'evidence_policy':
   'Reuse profile/source contracts. Only a named uncovered operator effect or a condition that changes a candidate conclusion justifies targeted source/probe work; broad current-container authorization already exists.',
 'whole_scope_complete':False})
save('independent-overlay-checks.json',{'status':'PASS','checks':checks,'nodes':len(ids),'edges':len(g['edges']),
    'new_NPU_context_model_or_restart_calls':0,'overall_complete':False})
print(json.dumps({'status':'PASS','checks':len(checks),'nodes':len(ids),'edges':len(g['edges']),'delta_rules':len(rules),'overall_complete':False}))
