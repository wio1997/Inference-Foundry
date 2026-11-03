"""Extract exact effects of the existing900-node frontier, not a new repo scan."""
import ast
import collections
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
d=json.loads((O/'M22-adjacent-operator-boundary-crosswalk-v79.json').read_text())
p=Path(d['worker_snapshot_path']);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==d['worker_snapshot_sha256']
tree=ast.parse(raw.decode());byloc=collections.defaultdict(list);parents={}
for n in ast.walk(tree):
    for ch in ast.iter_child_nodes(n):parents[ch]=n
    if hasattr(n,'lineno'):byloc[(n.lineno,n.col_offset,n.end_lineno,n.end_col_offset,type(n).__name__)].append(n)
ix=json.loads((N/'shared-profile-analysis-v172/profile-frame-symbol-delta.json').read_text());ix=ix.get('review',ix)['all_frame_index']
observed=collections.defaultdict(list)
for f,x in ix.items():
    if x['path']=='/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py':observed[x['enclosing_qualified_symbol']].append({'frame':f,'line':x['line'],'calls':x.get('call_expressions_on_profile_line',[])})

def text(n):return ast.unparse(n)
def subeffects(n):
    calls=[];writes=[];returns=[];raises=[]
    # A lexical over-approximation including nested selectors. It does not
    # claim every effect occurs on every branch or infer dynamic alias safety.
    for z in ast.walk(n):
        if isinstance(z,ast.Call):calls.append({'line':z.lineno,'call':text(z.func),'expression':text(z)})
        elif isinstance(z,ast.Assign):writes.append({'line':z.lineno,'targets':[text(t) for t in z.targets],'value':text(z.value)})
        elif isinstance(z,ast.AnnAssign):writes.append({'line':z.lineno,'targets':[text(z.target)],'value':text(z.value) if z.value else None})
        elif isinstance(z,ast.AugAssign):writes.append({'line':z.lineno,'targets':[text(z.target)],'value':text(z)})
        elif isinstance(z,ast.Return):returns.append({'line':z.lineno,'value':text(z.value) if z.value else None})
        elif isinstance(z,ast.Raise):raises.append({'line':z.lineno,'value':text(z.exc) if z.exc else None})
    return {'calls':calls,'writes':writes,'returns':returns,'raises':raises}

rows=[];unresolved=[];owners=collections.defaultdict(list)
for s in d['records']:
    parts=s['site_id'].split(':');loc=tuple(int(z) for z in parts[1:5])+(parts[5],);matches=byloc.get(loc,[])
    if len(matches)!=1:unresolved.append({'site':s['site_id'],'matches':len(matches)});continue
    n=matches[0];selected=n
    if s['kind']=='comprehension':
        cur=n
        while cur in parents and not isinstance(cur,ast.comprehension):cur=parents[cur]
        if isinstance(cur,ast.comprehension):selected=cur
    effects=subeffects(selected)
    branches={}
    if isinstance(n,ast.If):branches={k:subeffects(ast.Module(body=v,type_ignores=[])) for k,v in [('true',n.body),('false',n.orelse)]}
    elif isinstance(n,ast.IfExp):branches={'true':subeffects(n.body),'false':subeffects(n.orelse)}
    elif isinstance(n,ast.Try):branches={k:subeffects(ast.Module(body=v,type_ignores=[])) for k,v in [('body',n.body),('else',n.orelse),('finally',n.finalbody)]};branches['handlers']=[subeffects(h) for h in n.handlers]
    r={'site_id':s['site_id'],'owner':s['owner'],'source_family':s['source_family'],'line':s['line'],'kind':s['kind'],'condition':s['condition'],'enclosing_guards':s.get('enclosing_guards',[]),'lexical_effects':effects,'branch_effects':branches,'observed_owner_frames':observed.get(s['owner'],[])}
    rows.append(r);owners[s['owner']].append(r)
assert not unresolved,unresolved
allnodes=ast.walk(tree)
functions={}
class FunctionIndex(ast.NodeVisitor):
    def __init__(self):self.stack=[]
    def visit_ClassDef(self,n):self.stack.append(n.name);self.generic_visit(n);self.stack.pop()
    def visit_FunctionDef(self,n):
        name='.'.join(self.stack+[n.name]);functions[name]={'first':n.lineno,'last':n.end_lineno,'arguments':text(n.args),'effects':subeffects(n)}
        self.stack.append(n.name);self.generic_visit(n);self.stack.pop()
FunctionIndex().visit(tree)
summaries=[]
for owner,rs in owners.items():
    cs=collections.Counter(c['call'] for r in rs for c in r['lexical_effects']['calls']);ws=collections.Counter(t for r in rs for w in r['lexical_effects']['writes'] for t in w['targets'])
    summaries.append({'owner':owner,'sites':len(rs),'function':{k:v for k,v in functions.get(owner,{}).items() if k!='effects'},'unique_conditions':sorted(set(r['condition'] for r in rs)),'called_functions':sorted(cs),'assigned_targets':sorted(ws),'observed_owner_frames':observed.get(owner,[])})
(H/'remaining900-owner-full-effect-index.json').write_text(json.dumps({'source_sha256':sha,'scope':'Full lexical function continuations retained for earlyreturn/suffix effects;not a call-purity or native alias proof.','functions':{k:v for k,v in functions.items() if k in owners}},ensure_ascii=False,indent=2)+'\n')
result={'status':'ALL900_EXACT_SOURCE_NODES_EFFECTS_EXTRACTED','source':str(p),'source_sha256':sha,'scope':'Only old900explicitnavigation sites in one pinnedworker file. Node effects are lexical overapproximations,not proof of all paths,call purity,alias identity or runtime reachability. Semantic review follows per operator-effect family.','site_count':len(rows),'owner_count':len(owners),'unresolved':unresolved,'rows':rows,'owner_summaries':summaries,'new_source_files_opened':1,'new_device_work':False,'overall_complete':False}
out=H/'remaining900-exact-operator-effects.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');assert out.stat().st_size<48*1024*1024
(H/'remaining900-owner-summaries.json').write_text(json.dumps({'source':str(out),'source_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'owners':summaries},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'sites':len(rows),'owners':len(owners),'bytes':out.stat().st_size,'unresolved':0,'owners_with_positive_current_profile_frames':sum(bool(x['observed_owner_frames']) for x in summaries)}))
