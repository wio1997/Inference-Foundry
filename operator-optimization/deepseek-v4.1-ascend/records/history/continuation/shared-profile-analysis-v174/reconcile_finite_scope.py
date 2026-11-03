"""Advance exact static frontier coverage without turning it into live execution proof."""
import collections
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent
N=H.parent
O=N/'profile-first-20261031'

def read(p):return json.loads(p.read_text())
def pin(p):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(name,data):(H/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

inventory_path=O/'M22-semantic-branch-inventory-v46.json'
inv=read(inventory_path)['static_branch_sites']
prior={r['id']:r for r in inv if r['review_status']=='EXACT_GUARD_LOCAL_DATAFLOW_RECONCILED'}
front900_path=H/'remaining900-operator-dataflow-review.json'
front91_path=N/'shared-profile-analysis-v173/91-guard-operator-dataflow-review.json'
new900={r['site_id']:r for r in read(front900_path)['site_reviews']}
new91={r['site_id']:r for r in read(front91_path)['site_reviews']}
assert len(prior)==110 and len(new900)==900 and len(new91)==91
assert not (set(prior)&set(new900) or set(prior)&set(new91) or set(new900)&set(new91))
assert set(prior)|set(new900)|set(new91)=={r['id'] for r in inv}
rows=[]
for r in inv:
    sid=r['id']
    if sid in prior:
        authority=[str(O/e) for e in r['semantic_evidence']]
        assert all(Path(e).is_file() for e in authority)
        review='PREVIOUS_EXACT_GUARD_LOCAL_OPERATOR_DATAFLOW_REUSED'
    elif sid in new900:
        authority=[str(front900_path)+'#'+sid]
        review='V174_GUARDED_AND_CONTINUATION_OPERATOR_EFFECT_REVIEW'
    else:
        authority=[str(front91_path)+'#'+sid]
        review='V173_LIFECYCLE_OPERATOR_EFFECT_REVIEW'
    rows.append({'id':sid,'source':r['source'],'source_sha256':r['source_sha256'],
                 'owner':r['owner'],'condition':r['condition'],'review':review,
                 'authority':authority,'live_branch_taken_proved':False})
write('all1101-static-operator-frontier-reconciliation.json',{
    'status':'EXACT_ORIGINAL1101_SITE_BIJECTION_WITH_OPERATOR_EFFECT_REVIEWS',
    'source_inventory':pin(inventory_path),'counts':dict(collections.Counter(r['review'] for r in rows)),
    'source_counts':dict(collections.Counter(r['source'] for r in rows)),
    'rows':rows,'original_unreviewed_static_frontier_remaining':0,
    'scope':'Only this fixed1101site inventory across acl_graph and model_runner;not a repository-wide or live graph alias proof.',
    'public_boundary_rewrite_and_private_native_task_edit_are_distinct':True,
    'runtime_unknowns_are_conditional_premises_not_independence_or_noop_proofs':True,
    'whole_model_scan_complete':False})

partial_path=N/'shared-profile-analysis-v173/13-partial-labels-current-routing.json'
partial=read(partial_path)
write('current-conclusion-changing-evidence-policy.json',{
    'status':'DECISION_GATES_SEPARATED_FROM_ANALYSIS_DELIVERY',
    'static_frontier_reconciled':str(H/'all1101-static-operator-frontier-reconciliation.json'),
    'conditional_answers_for13_old_partial_labels':partial['routes'],
    'registered_composition_coverage':str(H/'all42-composition-coverage.json'),
    'all26_and42_P5_contracts':str(H/'module-analysis.json'),
    'no_additional_work_for':[
        'universal physical MFU or unavailable physical fmax',
        'revalidating known absent columns in immutable profiler exports',
        'private MemSet packet decoding as a prerequisite for whole public Pad replacement',
        'every pair or subset of candidate families',
        'actual speedup or implementation parity for an unimplemented conditional candidate',
        'retroactively repairing old166 identities using new171 observations'],
    'genuine_remaining_work':[
        {'id':'OBSERVED_G0_BINDING','question':'Bind one current coherent public tensor/bank/control graph and distinguish observable consumers from diagnostic labels.','evidence':'joint176 acquisition or bounded explicit failure;reuse old v164 PMU separately.','excludes':'No universal private allocation-retirement proof is demanded for source-conditional public rewrite.'},
        {'id':'FULL_RULE_PASS','question':'Audit current declared nodes,edges,guards and rewrite rules once using graph coverage and shared composition constraints;record any new non-equivalent opportunity.','evidence':'Existing G1/G5/G6/G7/G8/G9 coverage plus v174 effects and current graph supplement.','excludes':'A count of42 registered families alone is not candidate-universe closure.'}],
    'unknown_numeric_bounds_can_be_reported_symbolically':True,
    'overall_complete':False})
print(json.dumps({'status':'PASS','exact_static_sites':len(rows),'remaining_original_static_frontier':0,
                  'conditional_not_live':True}))
