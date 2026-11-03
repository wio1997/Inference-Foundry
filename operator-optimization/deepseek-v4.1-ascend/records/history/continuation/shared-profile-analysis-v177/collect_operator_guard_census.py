"""Profile-selected function/loaded-class guard census;do not reread old1101 guards."""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'scope-input.json').read_text())
BASE=Path(os.environ.get('WIO_CENSUS_ANALYSIS_ROOT','/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis'))
out=BASE/'census177';out.mkdir(exist_ok=True)
def write(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
roots=collections.defaultdict(set);hashes=collections.defaultdict(set)
for r in spec['frames'].values():
    roots[r['path']].add(r['enclosing_qualified_symbol']);hashes[r['path']].add(r['source_sha256'])
class_roots=collections.defaultdict(set);excluded=[]
for c in spec['classes']:
    qualified=c['qualified_class'];module,name=qualified.rsplit('.',1)
    # Fixed benchmark prompts are strings;vision only has loaded parameters.
    # Retain their existence and exclusion reason,not a no-op inference from trace.
    if module.startswith('vllm_ascend.models.deepseek_v41.vision') or module in ['vllm_ascend.ops.mm_encoder_attention','vllm_ascend.ops.rotary_embedding']:
        excluded.append({'class':qualified,'reason':'Vision-only class in this loaded tree;fixedregisteredtext-onlyrequestdomain.','class_present_not_execution_absence_proof':True});continue
    for q in [qualified]+[v for i in c['instances'] for v in [i['quant_method'],i['impl_class']] if v]:
        mod,cls=q.rsplit('.',1)
        if mod.startswith('torch.'):continue # standard Tensor/module contracts;profile roots retained separately
        prefix='/vllm-workspace/vllm-ascend/' if mod.startswith('vllm_ascend.') else '/vllm-workspace/vllm/' if mod.startswith('vllm.') else None
        if prefix:class_roots[prefix+mod.replace('.','/')+'.py'].add(cls)

rows=[];files=[];missing=[];already=[];functions=[]
for path in sorted(set(roots)|set(class_roots)):
    p=Path(path)
    if not p.exists():missing.append({'path':path,'reason':'MISSING_SOURCE_FILE'});continue
    b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
    if hashes[path] and hashes[path]!={sha}:
        missing.append({'path':path,'reason':'HASH_MISMATCH','expected':sorted(hashes[path]),'actual':sha});continue
    tree=ast.parse(b);owners={};funcs={};qualified=[];parents={};aliases=collections.defaultdict(set);raw_names={};class_names=set()
    def visit(n,owner=''):
        own=owner
        if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            own=(owner+'.' if owner else '')+n.name
            if isinstance(n,ast.ClassDef):class_names.add(own)
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                key=own+'#L'+str(n.lineno);funcs[key]=n;aliases[own].add(key);raw_names[key]=own
        owners[id(n)]=own
        for c in ast.iter_child_nodes(n):parents[id(c)]=n;visit(c,own)
    visit(tree)
    chosen={key for raw in roots[path] for key in aliases.get(raw,[])}
    for name in funcs:
        if any(raw_names[name].startswith(c+'.') for c in class_roots[path]):
            leaf=raw_names[name].rsplit('.',1)[-1]
            if leaf not in ['weight_loader','load_weights','_load_weights','_load_weights_from_checkpoint','_load_weights_iter']:chosen.add(name)
    # Resolve statically named local self/helper callees to a fixpoint,without
    # scanning unrelated functions from large framework files.
    while True:
        more=set()
        for name in chosen:
            raw=raw_names[name];cls=max((c for c in class_names if raw.startswith(c+'.')),key=len,default='')
            for n in ast.walk(funcs[name]):
                if isinstance(n,ast.Call):target_node=n.func
                elif isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load):target_node=n
                elif isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Load) and ast.unparse(n).startswith('self.'):target_node=n
                else:continue
                e=ast.unparse(target_node)
                if e.startswith('self.'):
                    candidates=[cls+'.'+e[5:]]
                elif isinstance(target_node,ast.Name):
                    parts=raw.split('.')
                    candidates=['.'.join(parts[:i]+[e]) for i in range(len(parts),-1,-1)]
                else:candidates=[e]
                for target in candidates:more.update(aliases.get(target,[]))
        if more<=chosen:break
        chosen|=more
    missing_roots=[x for x in roots[path] if x and x not in aliases]
    files.append({'path':path,'sha256':sha,'bytes':len(b),'known_profile_hashes_match':not hashes[path] or hashes[path]=={sha},'chosen_function_count':len(chosen),'missing_profile_symbols':missing_roots,'actual_class_names':sorted(class_roots[path]),'duplicate_qualified_definitions':{k:sorted(v) for k,v in aliases.items() if len(v)>1}})
    if path.endswith('/worker/model_runner_v1.py') or path.endswith('/compilation/acl_graph.py'):
        already.append({'path':path,'sha256':sha,'reason':'Original1101exactguardinventory operator-effectreview already complete;no repeatedeffectextraction.'});continue
    for name in sorted(chosen):
        f=funcs[name]
        functions.append({'path':path,'sha256':sha,'symbol':raw_names[name],'definition_key':name,'first':f.lineno,'last':f.end_lineno,
            'whole_function_lexical_calls':sorted({ast.unparse(z.func) for z in ast.walk(f) if isinstance(z,ast.Call)}),
            'whole_function_return_expressions':[ast.unparse(z.value) if z.value else None for z in ast.walk(f) if isinstance(z,ast.Return)],
            'whole_function_assignment_targets':sorted({ast.unparse(t) for z in ast.walk(f) if isinstance(z,(ast.Assign,ast.AnnAssign,ast.AugAssign)) for t in (z.targets if isinstance(z,ast.Assign) else [z.target])}),
            'lexical_overapproximation_not_purity_or_alias_proof':True})
        for ordinal,n in enumerate(ast.walk(f)):
            # Nested-function guards belong to that function's own row only.
            if owners[id(n)]!=raw_names[name]:continue
            cond=None
            if isinstance(n,(ast.If,ast.IfExp,ast.While)):cond=ast.unparse(n.test)
            elif isinstance(n,ast.Assert):cond=ast.unparse(n.test)
            elif isinstance(n,(ast.For,ast.AsyncFor)):cond=ast.unparse(n.iter)
            elif isinstance(n,(ast.Try,ast.TryStar)):cond='exception_handlers:'+','.join(ast.unparse(h.type) if h.type else 'any' for h in n.handlers)
            elif isinstance(n,(ast.With,ast.AsyncWith)):cond='contexts:'+','.join(ast.unparse(i.context_expr) for i in n.items)
            elif isinstance(n,ast.comprehension):cond=ast.unparse(n.target)+' in '+ast.unparse(n.iter)+';filters:'+','.join(ast.unparse(i) for i in n.ifs)
            if cond is None:continue
            body=list(getattr(n,'body',[])) if isinstance(getattr(n,'body',[]),list) else [getattr(n,'body',None)]
            other=list(getattr(n,'orelse',[])) if isinstance(getattr(n,'orelse',[]),list) else [getattr(n,'orelse',None)]
            def effects(nodes):
                calls=[];writes=[]
                for root in nodes:
                    if root is None:continue
                    for z in ast.walk(root):
                        if isinstance(z,ast.Call):calls.append(ast.unparse(z.func))
                        if isinstance(z,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
                            targets=z.targets if isinstance(z,ast.Assign) else [z.target]
                            writes.extend(ast.unparse(t) for t in targets)
                return {'calls':sorted(set(calls)),'writes':sorted(set(writes)),
                    'returns':any(isinstance(z,ast.Return) for root in nodes if root is not None for z in ast.walk(root))}
            pos=n
            while not hasattr(pos,'lineno'):pos=parents[id(pos)]
            rows.append({'id':hashlib.sha256((path+':'+sha+':'+name+':'+str(pos.lineno)+':'+type(n).__name__+':'+str(ordinal)).encode()).hexdigest()[:20],
                'path':path,'sha256':sha,'owner':raw_names[name],'definition_key':name,'line':pos.lineno,'last_line':pos.end_lineno,'AST_ordinal':ordinal,'position_is_enclosing_comprehension_expression':pos is not n,'kind':type(n).__name__,'condition':cond,
                'body_effects':effects(body),'else_effects':effects(other),'semantic_review_status':'CENSUS_ONLY_NOT_AUTOMATICALLY_ANALYZED'})
write('profile-selected-operator-guard-census.json',{'status':'PROFILE_AND_LOADED_CLASS_SELECTED_AST_GUARDS',
    'files':files,'functions':functions,'guards':rows,'excluded_vision_classes':excluded,'already_reviewed_sources':already,'missing_or_mismatched':missing,
    'scope':'Recorded source functions plus loaded nonvision class methods and statically named local helper/callable-reference closure. Dynamic/delegated callees are represented by existing public operator contracts;this census alone is not semantic completion.',
    'new_NPU_or_model_work':0,'whole_scope_complete':False})
print(json.dumps({'status':'PASS_CENSUS' if not missing else 'GAPS','files':len(files),'selected_functions':len(functions),'guards':len(rows),'already_reviewed_sources':len(already),'missing':len(missing)}))
