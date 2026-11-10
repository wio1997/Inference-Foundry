#!/usr/bin/env python3
"""Conservative callee frontier, independent of frozen predicate counts.

Only statically resolvable direct/same-receiver/MRO calls are expanded. Field
receivers, native methods and missing imports remain explicit edges. AST
expansion is never reported as a source review or a completed fusion search.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter, deque
import ast, json

R=Path('audit/execution-search')
MANIFESTS=['whole-forward-dependencies.json','whole-forward-binding-dependencies.json',
           'draft-output-dependencies.json','moe-output-dependencies.json',
           'target-output-dependencies.json','target-output-aux-dependencies.json']

def module(path):
    parts=path.removeprefix('/vllm-workspace/').split('/')[1:]
    parts[-1]=parts[-1].removesuffix('.py')
    if parts[-1]=='__init__':parts.pop()
    return '.'.join(parts)

def main():
    pins={};provenance={}
    for name in MANIFESTS:
        for row in json.loads((R/name).read_text())['sources']:
            p=row['runtime_path'];h=row.get('runtime_sha256',row.get('sha256'))
            if not p.endswith('.py'):continue
            assert h and sha256(Path(row['local_path']).read_bytes()).hexdigest()==h,p
            if p in pins:assert pins[p]['sha256']==h,('conflicting disk identities',p)
            pins[p]=dict(runtime_path=p,local_path=row['local_path'],sha256=h)
            provenance.setdefault(p,[]).append(name)
    units={};classes={};functions={}
    for row in pins.values():
        m=module(row['runtime_path']);tree=ast.parse(Path(row['local_path']).read_text());imports={}
        for n in tree.body:
            if isinstance(n,ast.Import):
                for a in n.names:imports[a.asname or a.name.split('.')[0]]=a.name if a.asname else a.name.split('.')[0]
            elif isinstance(n,ast.ImportFrom):
                pkg=m if Path(row['local_path']).name=='__init__.py' else m.rsplit('.',1)[0]
                base=('.'.join(pkg.split('.')[:len(pkg.split('.'))-n.level+1])+('.'+n.module if n.module else '')) if n.level else n.module
                for a in n.names:imports[a.asname or a.name]=(base+'.'+a.name).strip('.')
        units[m]=dict(pin=row,imports=imports,tree=tree)
        for n in tree.body:
            if isinstance(n,ast.ClassDef):
                classes[m+'.'+n.name]=(m,n)
                for f in n.body:
                    if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):functions[m+'.'+n.name+'.'+f.name]=(m,f,m+'.'+n.name)
            elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):functions[m+'.'+n.name]=(m,n,None)
    def ref(n,m):
        if isinstance(n,ast.Name):return units[m]['imports'].get(n.id,m+'.'+n.id)
        if isinstance(n,ast.Attribute):
            base=ref(n.value,m)
            return base+'.'+n.attr if base else None
        return None
    mros={}
    def mro(q):
        if q in mros:return mros[q]
        if q not in classes:return [q]
        m,c=classes[q];bases=[ref(b,m) for b in c.bases];assert all(bases)
        seq=[mro(b)[:] for b in bases]+[bases[:]];out=[q]
        while any(seq):
            seq=[s for s in seq if s];heads=[s[0] for s in seq if not any(s[0] in t[1:] for t in seq)]
            assert heads,('C3 unresolved',q)
            h=heads[0];out.append(h)
            for s in seq:
                if s[0]==h:s.pop(0)
        mros[q]=out;return out
    def lookup(receiver,name,after=None):
        order=mro(receiver)
        if after is not None:
            if after not in order:return None
            order=order[order.index(after)+1:]
        for owner in order:
            q=owner+'.'+name
            if q in functions:return q
            if owner in classes:
                m,c=classes[owner]
                for n in c.body:
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
                        target=ref(n.value,m)
                        if target in functions:return target
        return None
    roots=[]
    binding=json.loads((R/'whole-forward-dispatch.json').read_text())
    for r in binding['resolutions']:
        q=r['resolved_owner']+'.'+r['method']
        if q not in functions:q=lookup(r['receiver'],r['method'])
        assert q,(r['receiver'],r['method'])
        roots.append((r['receiver'],q,'prior_source_dispatch_binding'))
    original_entry=json.loads((R/'whole-forward-entry.json').read_text())
    for r in original_entry['roots']:
        if r['owner']=='AscendDSAV41PCPImpl':continue  # frozen legacy CP has its explicit binding above
        receiver=module(r['runtime_path'])+'.'+r['owner']
        q=lookup(receiver,r['method']);assert q,(receiver,r['method'])
        roots.append((receiver,q,'original_root_current_source_binding_required_per_role'))
    extras={
      'vllm_ascend.worker.model_runner_v1.NPUModelRunner':['sample_tokens','_sample','_bookkeeping_sync'],
      'vllm_ascend.sample.sampler.AscendSampler':['forward','sample','apply_logits_processors'],
      'vllm_ascend.sample.rejection_sampler.AscendRejectionSampler':['forward','apply_logits_processors'],
      'vllm.v1.worker.gpu_model_runner.AsyncGPUModelRunnerOutput':['__init__','get_output'],
      'vllm.v1.worker.gpu_input_batch.InputBatch':['set_async_sampled_token_ids','update_async_output_token_ids','update_async_spec_token_ids'],
      'vllm.v1.engine.core.EngineCore':['step','step_with_batch_queue','post_step'],
      'vllm.v1.core.sched.scheduler.Scheduler':['update_from_output','_update_request_with_output','update_draft_token_ids','update_draft_token_ids_in_output'],
      'vllm.v1.core.sched.async_scheduler.AsyncScheduler':['_update_after_schedule','_update_request_with_output'],
      'vllm.v1.executor.multiproc_executor.WorkerProc':['enqueue_output','handle_output','async_output_busy_loop'],
    }
    for receiver,names in extras.items():
        for name in names:
            q=lookup(receiver,name);assert q,(receiver,name)
            roots.append((receiver,q,'current_target_output_delivery'))
    queue=deque((a,b) for a,b,_ in roots);seen=set();nodes=[];edges=[]
    def key(receiver,q):return (receiver or '<module>')+' :: '+q
    while queue:
        receiver,q=queue.popleft()
        if (receiver,q) in seen:continue
        seen.add((receiver,q));m,f,owner=functions[q];pin=units[m]['pin'];nid=key(receiver,q)
        branches=[];calls=[];deferred=[]
        def walk(n,guards):
            if n is not f and isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):
                deferred.append(dict(name=getattr(n,'name','<lambda>'),first=n.lineno,last=n.end_lineno,state='NESTED_DEFINITION_REQUIRES_ACTUAL_CALL_BINDING'))
                return
            if isinstance(n,(ast.If,ast.IfExp,ast.For,ast.While,ast.Try,ast.With,ast.Match)):
                cond=ast.unparse(n.test) if hasattr(n,'test') else ast.unparse(n.iter) if isinstance(n,ast.For) else type(n).__name__
                if isinstance(n,(ast.If,ast.IfExp)):legs=['true','false']
                elif isinstance(n,(ast.For,ast.While)):legs=['body/continue','zero-or-false/normal-exit','break','else']
                elif isinstance(n,ast.Try):legs=['body']+['except:'+ast.unparse(h.type) if h.type else 'except:BaseException' for h in n.handlers]+['else','finally']
                elif isinstance(n,ast.Match):legs=['case:'+ast.unparse(c.pattern) for c in n.cases]+['no-matching-case']
                else:legs=['body','normal-exit','exceptional-exit']
                branches.append(dict(line=n.lineno,kind=type(n).__name__,condition=cond,legs=[dict(leg=l,predicate_binding='NOT_INFERRED_BY_AST',source_leg_review='PENDING_EXPLICIT_REVIEW',observed_population='UNKNOWN') for l in legs]))
                guards=guards+[dict(line=n.lineno,condition=cond)]
            if isinstance(n,ast.Call):calls.append((n,guards))
            for child in ast.iter_child_nodes(n):walk(child,guards)
        for statement in f.body:walk(statement,[])
        nodes.append(dict(id=nid,receiver=receiver,implementation=q,runtime_path=pin['runtime_path'],sha256=pin['sha256'],first=f.lineno,last=f.end_lineno,source_manifests=provenance[pin['runtime_path']],state='AST_EXPANDED_NOT_SOURCE_REVIEWED',branches=branches,deferred_definitions=deferred))
        for c,guards in sorted(calls,key=lambda v:(v[0].lineno,v[0].col_offset)):
            dest=None;dest_receiver=None;state='DYNAMIC_RECEIVER_OR_NATIVE_BOUNDARY_REQUIRES_CLASSIFICATION';func=c.func
            if isinstance(func,ast.Attribute) and isinstance(func.value,ast.Name) and func.value.id=='self' and receiver:
                dest=lookup(receiver,func.attr);dest_receiver=receiver
            elif isinstance(func,ast.Attribute) and isinstance(func.value,ast.Call) and isinstance(func.value.func,ast.Name) and func.value.func.id=='super' and receiver:
                dest=lookup(receiver,func.attr,after=owner);dest_receiver=receiver
            else:
                target=ref(func,m)
                if target in functions:
                    dest=target
                    if functions[target][2]:dest_receiver=receiver if c.args and isinstance(c.args[0],ast.Name) and c.args[0].id=='self' else functions[target][2]
                elif target in classes:state='CONSTRUCTOR_NOT_EXPANDED_INPUT_OR_DISPATCH_FRONTIER'
                elif isinstance(func,ast.Name) and func.id in {'len','range','int','float','bool','list','dict','set','tuple','enumerate','zip','min','max','sum','isinstance','getattr','hasattr','all','any','type','sorted','reversed','print'}:state='PYTHON_LANGUAGE_BOUNDARY_RETAINED'
                elif target and (target.startswith('vllm.') or target.startswith('vllm_ascend.')):state='IMPORTED_OR_LOCAL_SYMBOL_SOURCE_BINDING_REQUIRED'
            if dest:
                state='STATIC_BODY_EXPANDED_SELECTION_AND_LEGS_STILL_PENDING';queue.append((dest_receiver,dest))
            edges.append(dict(from_node=nid,line=c.lineno,column=c.col_offset,callee=ast.unparse(func),to_node=key(dest_receiver,dest) if dest else None,state=state,enclosing_conditions=guards,review='NOT_INFERRED_FROM_RESOLUTION'))
    old=json.loads((R/'model-gate-source-review.json').read_text())
    pending=[b for b in old['branches'] if b.get('condition_state')=='UNRESOLVED_INPUT_OR_RUNTIME'];assert len(pending)==34
    out=dict(status='EXPLICIT_CONSERVATIVE_RECURSIVE_FRONTIER_NOT_COMPLETENESS_PROOF',roots=[dict(receiver=a,implementation=b,origin=c) for a,b,c in roots],source_count=len(pins),node_count=len(nodes),edge_count=len(edges),edge_states=dict(Counter(e['state'] for e in edges)),nodes=nodes,edges=edges,
             original_unbound_input_predicates=pending,original_unbound_count=34,original_source_bindings=122,
             distinction='34 is an old root predicate-input ledger; neither 34 nor recursive AST counts are unreviewed legal-leg counts. This overapproximation exposes source/receiver work, including optional/inactive paths. Static resolution does not assert reachability or reviews. Every frontier must later be classified with actual dispatch/consumer/price evidence.',
             omitted_scope=['Constructors are recorded but not recursively executed. Field receivers, nested closure calls, dynamic dispatch and native/custom op implementations require explicit binding.','Only direct indexed bodies are expanded; no automatic whole-file or prior-range closure. M00–M11 still require explicit reader/ownership and branch-leg disposition.','Current source pins are 27 from this run; older manifests are exact local SHA evidence from their recorded acquisitions, not fresh loaded-code validation.'],
             whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,every_replacement_bitwise=True,
             inputs=[dict(path=str(R/n),sha256=sha256((R/n).read_bytes()).hexdigest()) for n in MANIFESTS+['whole-forward-entry.json','whole-forward-dispatch.json','model-gate-source-review.json']],builder_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'recursive-source-worklist.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['source_count','node_count','edge_count','edge_states','original_unbound_count','whole_chain_complete']}))
if __name__=='__main__':main()
