from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,json

N=Path(__file__).parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT=O/'whole-coverage-frontier-audit-v28.json';assert not OUT.exists()
inputs=[]
def read(p):
    b=p.read_bytes();inputs.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    return json.loads(b)
mods=read(O/'module-profile-ledger-v27.json');matrix=read(O/'module-phase-branch-matrix-v27.json')
models=read(O/'candidate-profile-ledger-v27.json');fw=read(O/'framework-candidate-ledger-v27.json')
old=read(R/'coverage/summary.json')
assert len(mods['modules'])==len(matrix['modules'])==26
assert len(models['candidates'])==34 and len(fw['candidates'])==7
mm={m['id']:m for m in mods['modules']};bm={m['id']:m for m in matrix['modules']}
assert set(mm)==set(bm)=={f'M{i:02d}'for i in range(26)}
candidate_ids={c['id']for c in models['candidates']+fw['candidates']}
module_rows=[];missing_refs=[];missing_links=[]
for key,m in mm.items():
    desired=set(m['candidate_links']);existing=set(bm[key].get('candidate_links',[]))
    assert desired<=candidate_ids,(key,desired-candidate_ids)
    missing_links += [dict(module=key,candidate=c)for c in sorted(desired-existing)]
    module_rows.append(dict(id=key,name=m['name'],module_ledger_candidate_links=sorted(desired),
        phase_matrix_candidate_links=sorted(existing),links_missing_in_phase_matrix=sorted(desired-existing),
        links_only_in_phase_matrix=sorted(existing-desired),phase=bm[key]['phase'],
        branch_records=len(bm[key]['branches']),whole_module_closed=m['whole_module_closed'],
        first_two_specific_branches=[dict(branch=b['branch'],state=b['state'],gap=b.get('gap'))for b in bm[key]['branches'][:2]]))
graphs=[]
for rel in ['graphs/G1.json','graph-revision-aux-context-v2/G1.json','graphs/source-templates-G5.json','graph-revision-aux-context-v2/G5.json']:
    j=read(R/rel)
    if 'templates'in j:
        graphs.append(dict(path=rel,sha256=inputs[-1]['sha256'],templates=[dict(id=t['id'],nodes=len(t['nodes']),edges=len(t['edges']))for t in j['templates']]))
    else:graphs.append(dict(path=rel,sha256=inputs[-1]['sha256'],nodes=len(j.get('nodes',[])),edges=len(j.get('edges',[])),known_gaps=j.get('known_gaps',[]),complete=j.get('complete')))
legacy={}
for name in ['node-structural-coverage.csv','edge-structural-coverage.csv']:
    p=R/'coverage'/name;b=p.read_bytes();inputs.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    rows=list(csv.DictReader(b.decode().splitlines()))
    unknown=[r for r in rows if r['status'].startswith('UNKNOWN')]
    legacy[name]=dict(rows=len(rows),semantic_coverage_counts=dict(Counter(r['semantic_coverage']for r in rows)),
        structural_status_counts=dict(Counter(r['status']for r in rows)),
        structurally_unknown_rows=unknown,
        unknown_by_template=dict(Counter((r['graph']+':'+r['template'])for r in unknown)))
masters=[]
for c in models['candidates']:
    p=R/'candidates'/(c['id']+'.json')
    masters.append(dict(id=c['id'],exists=p.exists(),path=str(p),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None))
    if not p.exists():missing_refs.append(dict(id=c['id'],missing_master=str(p),current_authoritative_record=str(O/'candidate-profile-ledger-v27.json')+'#'+c['id']))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),previous_turn_classification='PROGRESS:v27F33exact80orders/WAR/20conditions',
    model_candidates=34,framework_candidates=7,modules=26,phase_branch_records=sum(r['branch_records']for r in module_rows),
    phase_matrix_modules_with_missing_candidate_links=sum(bool(r['links_missing_in_phase_matrix'])for r in module_rows),
    missing_module_candidate_links=len(missing_links),missing_candidate_master_records=missing_refs,
    legacy_structural_ledger_scope='2026-10-28 G1/G5 partialgraphs;all semantic_coverage fields remainUNKNOWN. Not updated wholesale from later candidate existence.',
    legacy_ledger_matches_graph_revision=dict(baseG1=old['graph_sha256']['G1']==graphs[0]['sha256'],aux_context_G1=old['graph_sha256']['G1']==graphs[1]['sha256']),
    next_profile_frontier=dict(module='M17',reason='Only3phase/branch rows;targetgreedy/rejection andprobabilistic/logprob/RNG alternatives have no fresh v14+ complete nativecaller census bound here. Query saved allrank fullchains/tasks before any sampling source read.'),
    no_target_source_or_device_work=True,whole_modules_closed=0,overall_complete=False)
result=dict(summary=summary,input_manifest=inputs,module_frontier=module_rows,missing_module_candidate_links=missing_links,
    graph_version_surfaces=graphs,legacy_structural_ledgers=legacy,candidate_master_surfaces=masters,
    limits=['Historical graph versions overlap;do notsum asuniquenodes/edges orclaimcandidatecountprovessemanticcoverage.',
            'Branch records include repeated scoped-review rows,not a proved exhaustive condition universe.',
            'Synchronizing candidate links repairs navigation only;no semantic branch is closed automatically.'])
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
