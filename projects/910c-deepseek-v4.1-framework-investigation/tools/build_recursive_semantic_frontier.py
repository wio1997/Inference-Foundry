#!/usr/bin/env python3
"""Combine byte-verified source receipts without converting them to branch closure."""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import json
from build_recursive_source_worklist import MANIFESTS,R
def main():
    workpath=R/'recursive-source-worklist.json';oldpath=R/'recursive-source-disposition.json';work=json.loads(workpath.read_text());old=json.loads(oldpath.read_text());pins={};inputs=[workpath,oldpath]
    extra=['aux-context-dependencies.json','aux-context-registry-dependencies.json','compressor-path-dependencies.json','engram-lookup-path-dependencies.json','engram-hash-path-dependencies.json','engram-hash-kernel-dependencies.json']
    for name in MANIFESTS+extra:
        p=R/name;inputs.append(p)
        for s in json.loads(p.read_text())['sources']:
            if not s['runtime_path'].endswith('.py'):continue
            key=(s['runtime_path'],s.get('runtime_sha256',s.get('sha256')));local=Path(s['local_path']);assert sha256(local.read_bytes()).hexdigest()==key[1];pins[key]=local
    excerpts=[];rejected=[]
    for p in sorted(R.glob('*source-excerpts.json')):
        d=json.loads(p.read_text());rows=d.get('excerpts',d.get('methods',d.get('nodes',[]))) if isinstance(d,dict) else [];used=False
        for e in rows:
            key=(e.get('runtime_path'),e.get('sha256'));text=e.get('lines');lo=e.get('first',e.get('first_line'));hi=e.get('last',e.get('last_line'))
            if key not in pins or not isinstance(text,list) or not all(isinstance(x,str) for x in text) or not isinstance(lo,int) or not isinstance(hi,int):continue
            actual=pins[key].read_text().splitlines()[lo-1:hi]
            if actual!=text:rejected.append(dict(artifact=str(p),first=lo,last=hi,reason='EXCERPT_BYTES_DIFFER_FROM_PINNED_SOURCE'));continue
            excerpts.append(dict(path=key[0],sha256=key[1],first=lo,last=hi,artifact=str(p),name=e.get('name',e.get('method')),qualification=e.get('qualification')));used=True
        if used:inputs.append(p)
    assert not rejected,rejected
    previous={x['node']:x for x in old['current_body_source_walks']};node_rows=[]
    for n in work['nodes']:
        exact=[e for e in excerpts if e['path']==n['runtime_path'] and e['sha256']==n['sha256'] and e['first']<=n['first'] and e['last']>=n['last']]
        prior=previous.get(n['id']);state='EXACT_SHA_WHOLE_BODY_EXCERPT_EXISTS' if exact else 'PRIOR_EXPLICIT_BOUNDED_BODY_WALK_RECEIPT' if prior else 'NO_WHOLE_BODY_EVIDENCE_IN_CURRENT_INDEX'
        node_rows.append(dict(node=n['id'],receiver=n['receiver'],implementation=n['implementation'],runtime_path=n['runtime_path'],sha256=n['sha256'],first=n['first'],last=n['last'],body_evidence_state=state,exact_excerpt_evidence=[dict(artifact=e['artifact'],name=e['name'],first=e['first'],last=e['last']) for e in exact],prior_body_walk_receipt=prior,branch_site_count=len(n['branches']),branch_state='SOURCE_RANGE_EVIDENCE_IS_NOT_PER_LEG_SEMANTIC_DISPOSITION',deferred_definitions=n['deferred_definitions'],all_recursive_callees_closed=False))
    routes=defaultdict(list)
    for name,key,edgekey in [('aux-context-edge-dispositions.json','bindings','original_worklist_edges'),('compressor-path-edge-dispositions.json','routes','original_edges'),('engram-lookup-path-recursive-intersections.json','routes','old_worklist_edges')]:
        p=R/name;inputs.append(p)
        for route in json.loads(p.read_text())[key]:
            for e in route[edgekey]:routes[(e['from_node'],e['line'],e['callee'])].append(dict(artifact=str(p),semantic_disposition=route.get('selected_source_binding',route.get('selected_binding',route.get('disposition'))),native_or_ownership_admission=False))
    unresolved=[]
    for e in work['edges']:
        k=(e['from_node'],e['line'],e['callee']);named=routes.get(k,[])
        if not e.get('to_node'):
            unresolved.append(dict(from_node=e['from_node'],line=e['line'],callee=e['callee'],original_state=e['state'],receiver_semantic_receipts=named,current_state='EXPLICIT_RECEIVER_OR_PRIMITIVE_CONTRACT_RECEIPT' if named else e['state'],all_recursive_callees_closed=False))
    remaining=[n for n in node_rows if n['body_evidence_state']=='NO_WHOLE_BODY_EVIDENCE_IN_CURRENT_INDEX'];unique={}
    prior_full={n['node'] for n in old['prior_exact_SHA_excerpt_intersections'] if any(e['covers_entire_body'] for e in n['ranges'])}
    assert not prior_full & {n['node'] for n in remaining},'Cumulative index dropped an earlier byte-verified whole-body receipt'
    for n in remaining:
        k=(n['runtime_path'],n['sha256'],n['first'],n['last']);v=unique.setdefault(k,dict(runtime_path=n['runtime_path'],sha256=n['sha256'],first=n['first'],last=n['last'],implementation=n['implementation'],receivers=[]));v['receivers'].append(n['receiver'])
    output=dict(status='CUMULATIVE_EXACT_SHA_BODY_AND_SELECTED_RECEIVER_EVIDENCE_INDEX_NOT_COMPLETENESS',old_worklist_node_count=work['node_count'],old_worklist_edge_count=work['edge_count'],source_identity_count=len(pins),byte_verified_excerpt_count=len(excerpts),body_evidence_states=dict(Counter(n['body_evidence_state'] for n in node_rows)),nodes=node_rows,unresolved_edges=unresolved,edges_with_explicit_receiver_receipt=sum(bool(e['receiver_semantic_receipts']) for e in unresolved),remaining_body_evidence_nodes=len(remaining),remaining_unique_body_ranges=len(unique),remaining_ranges=list(unique.values()),original_bound_inputs=122,original_unbound_inputs=34,distinction=['This cumulative index prevents repeatedly treating an exact-SHA prior-viewed body as unseen solely because a late helper was missing from an old index. It is not a new mechanism, native test or proof of all branch legs.','Aliases/inherited receivers can share exact body bytes but still require their own actual selected roles, arguments, state and consumers. Body-range reuse does not auto-prove receiver or branch applicability.','The original319-node overapproximation excludes some field/native/constructor/closure expansion. Added aux/Compressor/Engram receipts are sidecars, not all-callgraph closure.','Remaining ranges are evidence gaps, not necessarily newly unscanned code or a complete list of legal runtime legs. Check earlier exactSHA semantic reports before another read; expand actual field/native boundaries as required.','Python language/control boundaries, Torch primitives and actual native implementation contracts remain distinguishable; an unresolved call string is neither a new fusion candidate nor proof of impossibility.'],next_action='Bind remaining AscendMoERunner routed/shared transform and _maybe_reduce helpers to selected SEDP/DFC output path, then remaining selected host/model helpers and native boundaries with complete delivery prices. Do not repeat Engram tiny stopped variants.',whole_chain_complete=False,whole_recursive_callgraph_closed=False,new_primitive_admissions=0,every_replacement_bitwise=True,inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in dict.fromkeys(inputs)],builder_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'recursive-semantic-frontier.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['body_evidence_states','remaining_body_evidence_nodes','remaining_unique_body_ranges','edges_with_explicit_receiver_receipt','whole_chain_complete']}))
if __name__=='__main__':main()
