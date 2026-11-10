#!/usr/bin/env python3
"""Pin actual selected forward sources before any historical-range intersection."""
from pathlib import Path
from hashlib import sha256
import ast,json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    prior={}
    for name in ['common-device-entry.json','engram-joint-entry.json','engram-row-owner-dependencies.json','hc-coefficient-consumer-dependencies.json']:
        p=ROOT/name
        if p.exists():
            for r in json.loads(p.read_text())['sources']:prior[r['runtime_path']]=r
    names=[n for n in prior if '/models/deepseek_v41/' in n or n.endswith(('/attention/dsa_v41.py','/attention/context_parallel/dsa_v41_cp.py','/worker/model_runner_v1.py','/spec_decode/dspark_proposer.py','/spec_decode/llm_base_proposer.py'))]
    spec=[]
    for name in sorted(names):
        r=prior[name];p=Path(r.get('local_path',r.get('path')));spec.append(dict(runtime_path=name,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=115)
    if p.returncode:raise RuntimeError(p.stderr)
    d=json.loads(p.stdout);roots=[]
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'whole-forward-runtime-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);s['local_path']=str(p)
        p=Path(s['local_path']);raw=p.read_bytes();assert sha256(raw).hexdigest()==s['runtime_sha256'];tree=ast.parse(raw)
        for parent in [tree,*[n for n in tree.body if isinstance(n,ast.ClassDef)]]:
            for m in parent.body:
                if not isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) or not (m.name.startswith('forward') or m.name in ['compute_logits','sample','execute_model','propose','_build_attention_metadata']):continue
                calls=[];branches=[]
                for n in ast.walk(m):
                    if isinstance(n,ast.Call):calls.append(dict(line=n.lineno,callee=ast.unparse(n.func)))
                    if isinstance(n,(ast.If,ast.IfExp,ast.For,ast.While,ast.Try,ast.With)):
                        condition=ast.unparse(n.test) if hasattr(n,'test') else ast.unparse(n.iter) if isinstance(n,ast.For) else type(n).__name__
                        branches.append(dict(line=n.lineno,last_line=n.end_lineno,kind=type(n).__name__,condition=condition,legality='UNKNOWN_PENDING_FROZEN_FLAG_AND_CALLER_REVIEW',execution_coverage='UNPROVEN'))
                roots.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],owner=parent.name if isinstance(parent,ast.ClassDef) else '<module>',method=m.name,first_line=m.lineno,last_line=m.end_lineno,calls=sorted(calls,key=lambda n:n['line']),branch_sites=sorted(branches,key=lambda n:n['line']),entry_reachability='STATIC_CANDIDATE_PENDING_SELECTED_DISPATCH_PROOF',source_body_reviewed_this_entry=False))
    d.update(status='CURRENT_SELECTED_FORWARD_SOURCES_IDENTITY_AND_STATIC_ENTRY_INVENTORY_ONLY',reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),roots=roots,root_count=len(roots),callsite_count=sum(len(r['calls']) for r in roots),branch_site_count=sum(len(r['branch_sites']) for r in roots),whole_chain_complete=False,qualification='Source locations seeded from earlier pins but contents freshly SHA-read. AST roots/calls/guards derived before historical range intersection. Static calls include constructors/shape operations and possible inactive classes; not all selected roots, unique operators, runtime coverage, legal outcomes or completed fusion searches. Next bind dispatch/frozen flags, then exact-SHA prior ranges; source-viewed must stay separate from branch/consumer/native evidence.',new_primitive_admissions=0,new_framework_replacement_admissions=0)
    (ROOT/'whole-forward-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),roots=len(roots),calls=d['callsite_count'],branches=d['branch_site_count'],whole_chain_complete=False)))
if __name__=='__main__':main()
