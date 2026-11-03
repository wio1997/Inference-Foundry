"""Preserve all historical rows while publishing the actual current review."""
import copy
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
screen=read(H/'all358-current-analysis-screen.json');screenmap={x['id']:x for x in screen['rows']}
partial={x['id']:x for x in read(H/'13-partial-labels-current-routing.json')['routes']}
variants=read(H/'two-dummy-demand-variant-cuts.json')['new_semantic_rows']
base=O/'module-phase-branch-matrix-v154.json';original=read(base);b=copy.deepcopy(original)
for m in b['modules']:
    for r in m['branches']:
        x=screenmap[r['id']]
        r['current_operator_review_v173']={'screen_state':x['screen_state'],'original_state_retained':r['state'],'source_or_runtime_conditions':x['named_remaining_conditions'],'superseded_historical_gate_notes':x['current_overlay_notes'],'metric_and_P5_authority':str(N/'shared-profile-analysis-v172/all26-module-metric-and-P5-routes.json'),'whole_module_closed':False}
        if r['id'] in partial:r['current_operator_review_v173']['named_analysis_and_evidence_route']=partial[r['id']]
        if r['id'].startswith('M01-'):r['current_operator_review_v173']['operator_boundary_review']=str(H/'request-operator-boundary-review.json')
        if r['id'] in ['M21-S11','M21-S12','M21-S15','M22-S05','M22-S11','M22-S12']:r['current_operator_review_v173']['lifecycle_effect_review']=str(H/'91-guard-operator-dataflow-review.json')
    for v in variants:
        if v['id'].split('-')[0]==m['id']:
            m['branches'].append({'id':v['id'],'branch':v['boundary'],'state':'CONDITIONAL_OPERATOR_VARIANT_ANALYZED;CURRENT_RUNTIME_SAVINGS_NOT_PROVED','evidence':[str(H/'two-dummy-demand-variant-cuts.json'),str(H/'lifecycle-profile-witnesses.json'),str(H/'lifecycle-operator-effects-source.json')],'scope':'Operator value demand and row work only; preserve communication/capture/state/precision interfaces.','gap':v['required_guard'],'current_operator_review_v173':v,'whole_module_closed':False})
b.update(parent_matrix=pin(base),current_review_checkpoint=173,original_semantic_rows=358,current_semantic_rows=360,scope='All original358 rows preserved,with current review fields and two explicit variants under existingF30/F31. Historical state labels and native gaps are not silently overwritten.',current_screen=str(H/'all358-current-analysis-screen.json'),current_partial_routes=str(H/'13-partial-labels-current-routing.json'),all_phases_closed=False,all_conditions_closed=False,whole_modules_closed=0,overall_complete=False)
bp=O/'module-phase-branch-matrix-v173.json';write(bp,b)

base=O/'module-profile-ledger-v154.json';m=copy.deepcopy(read(base));metric={x['module']:x for x in read(N/'shared-profile-analysis-v172/all26-module-metric-and-P5-routes.json')['modules']}
for row in m['modules']:
    mid=row['id'];bs=next(x for x in b['modules'] if x['id']==mid)['branches'];row['review_v173']={'current_branch_matrix':str(bp),'current_semantic_row_ids':[x['id'] for x in bs],'primary_metric_and_P5_route':metric[mid],'source_conditional_analysis_is_a_result':True,'actual_unknowns_are_not_zero_or_independence':True,'lifecycle_and_joint_review':[str(H/'91-guard-operator-dataflow-review.json'),str(H/'joint-operator-graph-and-resource-review.json')],'whole_module_closed':False}
m.update(parent_ledger=pin(base),current_review_checkpoint=173,current_semantic_rows=360,current_module_count=26,whole_modules_closed=0,overall_complete=False)
mp=O/'module-profile-ledger-v173.json';write(mp,m)

base=O/'candidate-profile-ledger-v154.json';c=copy.deepcopy(read(base));graph=read(H/'joint-operator-graph-and-resource-review.json');selected={i for n in graph['nodes'] for i in n['candidates']}
for row in c['candidates']:
    row['review_v173']={'joint_graph':str(H/'joint-operator-graph-and-resource-review.json'),'included_in_selected_joint_graph':row['id'] in selected,'existing_conditional_relations_not_cartesian_tasks':True,'new_implementation_or_measured_gain':False}
    if row['id'] in ['F30','F31']:row['review_v173']['new_explicit_variants']=[v['id'] for v in variants if row['id'] in v['parent_candidates']]
c.update(parent_ledger=pin(base),current_review_checkpoint=173,current_model_candidates=35,framework_candidates_reused=str(O/'framework-candidate-ledger-v154.json'),current_semantic_rows=360,whole_modules_closed=0,overall_complete=False)
cp=O/'candidate-profile-ledger-v173.json';write(cp,c)

oldrows={r['id']:r for m in original['modules'] for r in m['branches']};newrows={r['id']:r for m in b['modules'] for r in m['branches']}
checks={'26modules_retained':len(b['modules'])==len(m['modules'])==26,'360_unique_rows':len(newrows)==360,'all358_old_rows_retained_without_oldfield_changes':all(all(newrows[k][a]==v for a,v in x.items()) for k,x in oldrows.items()),'only2_new_rows':set(newrows)-set(oldrows)=={'M21-S19','M16-OP-11'},'35candidateIDs_unchanged':{x['id'] for x in c['candidates']}=={x['id'] for x in read(base)['candidates']},'all26_P5_routes_kept':all('primary_metric_and_P5_route' in x['review_v173'] for x in m['modules'])}
assert all(checks.values()),checks
write(H/'current-ledger-preservation-checks.json',{'status':'PASS','checks':checks,'authorities':{'candidate_ledger':str(cp),'module_ledger':str(mp),'branch_matrix':str(bp)},'no_whole_module_closure':True,'overall_complete':False})
print(json.dumps({'status':'PASS','semantic_rows':360,'modules':26,'model_candidates':35,'selected_joint_candidates':len(selected),'overall_complete':False}))
