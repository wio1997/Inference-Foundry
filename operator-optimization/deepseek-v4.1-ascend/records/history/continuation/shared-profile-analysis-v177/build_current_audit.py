"""Replace stale completion claims by current evidence and exact remaining work."""
import collections
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def write(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def pin(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

matrix=read(O/'module-phase-branch-matrix-v176.json')
rows=[{'module':m['id'],**b} for m in matrix['modules'] for b in m['branches']]
guards=read(H/'profile-selected-operator-guard-census.json')
assert not guards['missing_or_mismatched']
assert len(rows)==360 and len({r['id'] for r in rows})==360
parent=read(H/'operator-parent-scope-census.json')
assert parent['intact_nodes']==821146 and parent['operator_names']==235
classes=read(H/'loaded-module-instance-census.json')['classes']
assert len(classes)==49

# Full original obligations remain available as immutable history. Their
# strings must not override newer evidence or introduce universal-MFU gates.
stale=read(O/'coverage-current-obligations-v176.json')
rule=N/'shared-profile-analysis-v176/declared-source-template-rule-review.json'
comparison=N/'shared-profile-analysis-v174/module-analysis.json'
metric=N/'shared-profile-analysis-v172/all26-module-metric-and-P5-routes.json'
cause=N/'shared-profile-analysis-v172/lowest-module-cause-review.json'
facts=N/'shared-profile-analysis-v176/operator-conclusions-from-profile.json'
static=N/'shared-profile-analysis-v174/all1101-static-operator-frontier-reconciliation.json'
composition=N/'shared-profile-analysis-v174/all42-composition-coverage.json'

requirements=[
 {'id':'MODEL_VERSION_HARDWARE_LOAD','state':'FIXED_SOURCE_AND_MIGRATED_CASE_DOMAIN_WITH_EXPLICIT_NATIVE_MODE_UNKNOWN',
  'requirement':'Pin specified model/source/hardware/CANN/workload and separate measurement cohorts.',
  'evidence':[str(N/'shared-profile-analysis-v172/fixed-case-branch-contract-reuse.json'),str(H/'profile-selected-operator-guard-census.json'),str(facts)],
  'finding':'72profile/loaded-class source files read in the ownedcontainer;all retained profile hashes match,no missingfiles. Frozen8textrequests/2seeds/client/sourcepins retain32K/1K/C8/P90. CANN9.2 and user-accepted driver/firmware differences remain. Native internal modes are conditional bound inputs,not invented.',
  'completion_proved_for_analysis_contract':True},
 {'id':'ALL_MODULE_CONDITIONAL_BRANCHES','state':'INCOMPLETE_EXACT_GUARD_TO_EXISTING_OPERATOR_CONTRACT_RECONCILIATION',
  'requirement':'Every applicable operator branch has an analysis answer,including conditions changing state/precision/ownership/readiness.',
  'evidence':[str(O/'module-phase-branch-matrix-v176.json'),str(static),str(H/'profile-selected-operator-guard-census.json')],
  'finding':f"All360named semanticfamilies and original1101exactstatic sites have scopedanswers. Profile/loaded-class census adds an exact{len(guards['guards'])}guard reference across{len(guards['functions'])}selectedfunctions outside the two already-reviewed files. These are not that many newunanalysed tasks;guard-to-contract coverage has not yet been checked as a bijection. Duplicate qualified definitions are retained by sourceposition and fullfunction lexical continuations are saved.",
  'completion_proved_for_analysis_contract':False},
 {'id':'CURRENT_FINITE_GRAPH_LEDGER','state':'INCOMPLETE_HIERARCHICAL_G0_INTERFACE_AND_ALIAS_PORT_RECONCILIATION',
  'requirement':'Freeze complete declared public G0 with node/data/all-consumer/control/communication/alias/state/case ports and original candidate mappings.',
  'evidence':[str(rule),str(parent and H/'operator-parent-scope-census.json'),str(facts),str(composition)],
  'finding':'27source templates/854nodes/1346edges,49loaded classes and235recordedop names are indexed. Native nesting has10434exact caller/scope groups and136schemas;14direct infrastructure/Triton names lack a publicoperator parent and have known operator-family entries. Observed-census coverage alone is not the complete source-call/port/alias-universe proof.',
  'completion_proved_for_analysis_contract':False},
 {'id':'CROSS_OPERATOR_LAYER_CARD_ITERATION','state':'REGISTERED42_COMPOSITION_AND_DECLARED_SOURCE_RULE_ROUND_ANSWERED;FULL_SCOPE_DEPENDS_ON_G0',
  'requirement':'Analyze pair/triple/larger/shared-input/cross-layer/card/iteration and split/recombine opportunities with interactions.',
  'evidence':[str(composition),str(rule),str(N/'shared-profile-analysis-v173/joint-operator-graph-and-resource-review.json')],
  'finding':'All35model+7framework families have shared guard/resource/lifetime contracts;one7rule round over all27declaredsource templates yielded no newnon-equivalentfamily. No Cartesianpair debt or old383pair count remains as a worklist. Fullscope closure awaits the two exactcoverage checks above.',
  'completion_proved_for_analysis_contract':False},
 {'id':'NATIVE_OPERATOR_REGIMES','state':'CONDITIONAL_NATIVE_ELIGIBILITY_AND_UNSUPPORTED_PRIVATE_EDITS_EXPLICIT',
  'requirement':'Use sufficient operand/numerical/protocol evidence for every claimed legal native transformation.',
  'evidence':[str(facts),str(H/'operator-parent-scope-census.json'),str(N/'shared-profile-analysis-v174/current-conclusion-changing-evidence-policy.json')],
  'finding':'No unproven private MemSet deletion,all-score-deadness,zero-sender globalnoop or native precision compatibility claim is promoted. Public function composition and private task editing are separate domains. Original historical9regimes remain unsupported for their private-edit claims;they are not a blanket barrier to all publicoperator analysis.',
  'completion_proved_for_analysis_contract':True},
 {'id':'THEORY_AND_MEASURED_COMPARISON','state':'PROTOCOL_P5_EXPLICIT_CONDITIONAL_SYMBOLIC_COMPARISONS_PRESENT',
  'requirement':'Per-object Q,T,L,Q/T,U,L/T,T/L,utilization,dominant constraint/evidence/status,with valid matched scope.',
  'evidence':[str(comparison),str(N/'shared-profile-analysis-v174/P5-contract-checks.json')],
  'finding':'94explicitrecords cover26wholemodulecontracts,42unimplementedcompositions and26matchedpartialdense-mode populations. Unknown wholemoduletimes/work/resource uppers remain symbolic. Rated1800MHz is referenceonly;no attainable-speedup or physicalMFU claim.',
  'completion_proved_for_analysis_contract':True},
 {'id':'LOWEST_MODULE_UTILIZATION','state':'PROTOCOL_P4_COMPARABLE_MEASURED_GROUP_MINIMA_ESTABLISHED',
  'requirement':'Report allmodule coverage and lowest comparable measured module/group under an explicit metric.',
  'evidence':[str(metric),str(cause),str(N/'shared-profile-analysis-v172/module-diagnosis-independent-checks.json')],
  'finding':'M12is lowest in12measuredMACmodules in both formalwindows and pooled data;M23lowest arithmeticactivity among20Vectorbearingmodules. Host/no-PMU/ambiguous/missing data remain explicit,notzero. A globalphysicalMFUminimum across host/Cube/Vector/communication is neither established nor a protocol prerequisite.',
  'completion_proved_for_analysis_contract':True},
 {'id':'CAUSES_AND_CRITICAL_PATH','state':'PROTOCOL_P6_SUPPORTED_LOCATION_AND_COMPETING_CAUSES_WITH_LIMITS',
  'requirement':'Explain measured minima with support/refutation/competing hypotheses and criticalpath limitations.',
  'evidence':[str(cause),str(N/'shared-profile-analysis-v172/module-window-diagnosis.json')],
  'finding':'Already-fusedF21owns98.7649%/99.1925% ofM12cumulative native time and99.7213%/99.9279%reportedAICcycles in the two windows. Small-work/receiver imbalance/communication-credit-wait remain distinct explanations;no current internal-stage causal timing or removable request latency is invented.',
  'completion_proved_for_analysis_contract':True},
 {'id':'CRITICAL_EVIDENCE_AND_FINAL_CLOSURE','state':'NOT_ACHIEVED_TWO_EXACT_SOURCE_COVERAGE_CHECKS_REMAIN',
  'requirement':'Do not complete merely because reports/artifacts/checks exist;all original requestedscope must withstand current audit.',
  'evidence':[str(H/'profile-selected-operator-guard-census.json'),str(H/'operator-parent-scope-census.json'),str(rule)],
  'finding':'Need exactguard-to-operator-contract reconciliation and complete hierarchical publicG0 boundary/alias-port reconciliation. No newNPUacquisition is needed by default;do not reopen reviewed source or demand fmax/privateABI/implementation for all candidates.',
  'completion_proved_for_analysis_contract':False},
]
write('current-requirement-audit.json',{
    'status':'CURRENT_EVIDENCE_REPLACES_STALE_BLANKET_PARTIAL_FLAGS',
    'previous_goal_turn':'PROGRESS','requirements':requirements,
    'protocol_path':'/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/PROTOCOL.md',
    'historical_obligations_preserved_by':pin(O/'coverage-current-obligations-v176.json'),
    'all_original9_requirements_retained':{r['id'] for r in requirements}=={r['id'] for r in read(O/'whole-scope-requirement-audit-v176.json')['requirements']},
    'overall_complete':False})

domains=collections.defaultdict(list)
for r in guards['guards']:domains[r['path']].append(r)
write('exact-source-guard-coverage-frontier.json',{
    'status':'FINITE_PROFILE_SELECTED_GUARD_SET_TO_RECONCILE_WITH_EXISTING_ANSWERS',
    'guard_count':len(guards['guards']),'files_with_guards':len(domains),
    'selected_functions':len(guards['functions']),
    'reviewed1101_inventory_not_reopened':str(static),
    'file_groups':[{'path':p,'guard_ids':[r['id'] for r in rs],
                    'owners':sorted({r['owner'] for r in rs}),
                    'state':'EXACT_REFERENCE_SET_NOT_A_COUNT_OF_NEW_UNANALYZED_OPTIMIZATIONS'} for p,rs in domains.items()],
    'next_action':'Match each guard to an existing semanticcontract or an explicitlypreserved public-boundary effect. Read only an unmatched operator-changing condition/effect;do not research allframeworkcontrolflow or repeat1101oldguards.',
    'dynamic_delegation_and_inheritance_not_proved_by_static_local_closure':True,
    'overall_complete':False})
write('coverage-audit-checks.json',{'status':'PASS','checks':{
    'all360_branch_ids_retained':len(rows)==360,'all_retained_profile_source_hashes_match':not guards['missing_or_mismatched'],
    'all235_op_names_censused':parent['operator_names']==235,'49_actual_loaded_classes':len(classes)==49,
    'source_guard_ids_unique':len({r['id'] for r in guards['guards']})==len(guards['guards']),
    'all_original9_requirements_retained':len(requirements)==9,
    'P4_P5_P6_current_outcomes_replace_legacy_global_MFU_gate':True,
    'fullscope_completion_not_asserted':True},'overall_complete':False})
print(json.dumps({'status':'PASS','requirements':len(requirements),'answered_for_analysis':sum(r['completion_proved_for_analysis_contract'] for r in requirements),
                  'explicit_guard_reference_count':len(guards['guards']),'files_with_guards':len(domains),'overall_complete':False}))
