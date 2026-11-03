#!/usr/bin/env python3
"""Source-site coverage denominator. Syntactic indexing never closes a review.

Run with Python >=3.12 to parse the frozen target language version. No target
module is imported or executed. Calls/aliases are potential sites, not dataflow.
"""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
assert sys.version_info>=(3,12), 'Use the available python3.12 parser'
SNAPSHOTS=[ROOT/'frozen-source-snapshot',ROOT/'sampler-source-supplement/frozen-source-snapshot']
OUT=ROOT/'coverage-ledger'
OUT.mkdir(exist_ok=True)

def module_key(path):
    parts=list(Path(path).parts)[1:]
    if not parts or not parts[-1].endswith('.py'):return None
    parts[-1]=parts[-1][:-3]
    if parts[-1]=='__init__':parts.pop()
    return '.'.join(parts)

def expression(x):
    return ast.unparse(x) if x is not None else ''

def family(path):
    if 'models/deepseek_v41/engram/' in path:return 'engram'
    if 'models/deepseek_v41/vision.py' in path or 'models/deepseek_v41/vl_model.py' in path:return 'model_wrapper_and_vision'
    if 'models/deepseek_v41/dspark.py' in path or 'qwen3_dspark.py' in path:return 'draft_model_and_heads'
    if 'models/deepseek_v41/' in path:return 'target_model_topology_projection_mhc_moe'
    if '/sample/' in path or 'reject_sample' in path:return 'sampler_rejection_and_constraints'
    if '/spec_decode/' in path:return 'proposer_and_speculative_state'
    if 'logits_processor.py' in path or 'vocab_parallel_embedding.py' in path:return 'vocabulary_projection_and_communication'
    if '/attention/' in path or '/ops/dsa' in path:return 'attention_metadata_and_cache'
    if '/ops/fused_moe/' in path or '/layers/fused_moe/' in path or '/mc2/' in path:return 'expert_dispatch_compute_combine'
    if '/v1/core/' in path:return 'scheduler_and_kv_cache_management'
    if '/v1/engine/' in path:return 'engine_queues_and_response'
    if '/worker/' in path:return 'worker_execution_and_batch_state'
    if '/compilation/' in path:return 'graph_capture_replay_update'
    if '/distributed/' in path or '/models/common/ops/' in path:return 'parallel_collective_and_partition'
    if '/quantization/' in path:return 'quantization_and_layout'
    if '/patch/' in path or 'platform' in path or path.endswith('models/__init__.py'):return 'registration_and_runtime_dispatch'
    if '/entrypoints/' in path:return 'request_frontend_and_sampling_contract'
    if '/device/' in path:return 'hardware_dispatch_and_capability'
    if '/ops/' in path:return 'primitive_and_composite_operator'
    return 'support_dependency_scope_unresolved'

files={};roots=[];missing=[]
for snapshot in SNAPSHOTS:
    manifest=json.loads((snapshot/'SOURCE-MANIFEST.json').read_text());roots.append(manifest['roots']);missing.extend(manifest['missing'])
    for item in manifest['files']:
        path=snapshot/item['path'];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
        if item['path'] in files:
            assert files[item['path']]['sha256']==item['sha256'],'Conflicting source snapshots'
        else:files[item['path']]=dict(item,local_path=str(path),family=family(item['path']),module=module_key(item['path']))

modules={v['module']:k for k,v in files.items() if v['module']}
units=[];decisions=[];calls=[];imports=[];errors=[]

class Inventory(ast.NodeVisitor):
    def __init__(self,meta,text):
        self.meta=meta;self.text=text;self.scope=[];self.guards=[]
    def base(self,node):
        instance=hashlib.sha256((self.meta['path']+'\0'+self.meta['sha256']).encode()).hexdigest()[:16]
        return {'id':f"{instance}:{node.lineno}:{node.col_offset}:{getattr(node,'end_lineno',node.lineno)}:{getattr(node,'end_col_offset',node.col_offset)}:{type(node).__name__}",
                'source':self.meta['path'],'source_sha256':self.meta['sha256'],
                'line':node.lineno,'end_line':getattr(node,'end_lineno',node.lineno),
                'owner':'.'.join(self.scope) or '<module>','family':self.meta['family'],
                'enclosing_guards':list(self.guards),'review_status':'UNREVIEWED',
                'runtime_binding':'UNKNOWN','semantic_evidence':[]}
    def scoped(self,node,kind):
        row=self.base(node);row.update(kind=kind,name=node.name,qualified_name='.'.join([self.meta['module'],*self.scope,node.name]))
        if isinstance(node,ast.ClassDef):row['bases']=[expression(x) for x in node.bases]
        else:row['arguments']=expression(node.args);row['decorators']=[expression(x) for x in node.decorator_list]
        units.append(row);self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
    def visit_ClassDef(self,node):self.scoped(node,'class')
    def visit_FunctionDef(self,node):self.scoped(node,'function')
    def visit_AsyncFunctionDef(self,node):self.scoped(node,'async_function')
    def decision(self,node,condition,variants):
        row=self.base(node);row.update(kind=type(node).__name__,condition=condition,variants=variants)
        decisions.append(row);return row['id']
    def visit_If(self,node):
        key=self.decision(node,expression(node.test),['true','false'])
        self.visit(node.test)
        for variant,body in [('true',node.body),('false',node.orelse)]:
            self.guards.append({'decision_id':key,'variant':variant})
            for item in body:self.visit(item)
            self.guards.pop()
    def visit_IfExp(self,node):
        key=self.decision(node,expression(node.test),['true','false']);self.visit(node.test)
        for variant,item in [('true',node.body),('false',node.orelse)]:
            self.guards.append({'decision_id':key,'variant':variant});self.visit(item);self.guards.pop()
    def visit_For(self,node):
        self.decision(node,expression(node.target)+' in '+expression(node.iter),['zero_iterations','iteration','normal_else','break_or_exception'])
        self.generic_visit(node)
    visit_AsyncFor=visit_For
    def visit_While(self,node):
        self.decision(node,expression(node.test),['zero_iterations','iteration','normal_else','break_or_exception']);self.generic_visit(node)
    def visit_Try(self,node):
        self.decision(node,'try/exception/finally',['normal',*['except '+expression(h.type) for h in node.handlers],'finally']);self.generic_visit(node)
    visit_TryStar=visit_Try
    def visit_BoolOp(self,node):
        self.decision(node,expression(node),['short_circuit_at_operand','evaluate_next_operand']);self.generic_visit(node)
    def visit_Assert(self,node):
        self.decision(node,expression(node.test),['pass','raise']);self.generic_visit(node)
    def visit_Match(self,node):
        self.decision(node,expression(node.subject),[expression(c.pattern) for c in node.cases]+['no_match']);self.generic_visit(node)
    def visit_comprehension(self,node):
        # Comprehension nodes lack their own line; use their iterable anchor.
        self.decision(node.iter,expression(node.target)+' in '+expression(node.iter)+''.join(' if '+expression(x) for x in node.ifs),['zero_iterations','iteration','filtered_out'])
        decisions[-1]['id']+=':comprehension'
        decisions[-1]['kind']='comprehension'
        self.generic_visit(node)
    def visit_Call(self,node):
        row=self.base(node);callee=expression(node.func)
        attrs=[]
        if isinstance(node.func,ast.Attribute):
            attr=node.func.attr
            if attr.endswith('_') and not attr.endswith('__'):attrs.append('potential_inplace_effect')
            if attr in ('view','reshape','as_strided','detach','narrow','slice','split','chunk','unbind'):attrs.append('potential_alias_or_view')
            if attr in ('wait','wait_stream','wait_event','record','record_stream','synchronize','query'):attrs.append('potential_stream_or_lifetime_effect')
            if any(x in attr.lower() for x in ('all_gather','all_to_all','all_reduce','reduce_scatter','broadcast','send','recv')):attrs.append('potential_collective_or_p2p')
        if 'register' in callee or 'import_module' in callee:attrs.append('runtime_dispatch_or_registration')
        row.update(callee=callee,keyword_names=[x.arg for x in node.keywords],argument_count=len(node.args),potential_effects=attrs,resolution_status='UNRESOLVED_CALLEE_AND_VALUE_FLOW')
        calls.append(row);self.generic_visit(node)
    def visit_Import(self,node):
        for index,name in enumerate(node.names):
            row=self.base(node);row['id']+=f':member{index}';row.update(import_kind='import',module=name.name,member=None,alias=name.asname or name.name.split('.')[0]);imports.append(row)
    def visit_ImportFrom(self,node):
        parts=self.meta['module'].split('.')
        if not self.meta['path'].endswith('/__init__.py'):parts=parts[:-1]
        if node.level:
            base=parts[:len(parts)-(node.level-1)]
            absolute='.'.join([*base,*([node.module] if node.module else [])])
        else:absolute=node.module or ''
        for index,name in enumerate(node.names):
            row=self.base(node);row['id']+=f':member{index}';row.update(import_kind='from',module=absolute,member=name.name,alias=name.asname or name.name);imports.append(row)

for path,meta in sorted(files.items()):
    if not meta['module']:continue
    text=Path(meta['local_path']).read_text()
    try:tree=ast.parse(text,filename=path)
    except SyntaxError as e:errors.append({'source':path,'error':str(e)});continue
    Inventory(meta,text).visit(tree)

for item in imports:
    imported=item['module'];member=item['member'];candidates=[imported]
    if member and member!='*':candidates.insert(0,imported+'.'+member)
    found=[{'module':x,'source':modules[x]} for x in candidates if x in modules]
    item['snapshot_module_candidates']=found
    item['dependency_status']='SOURCE_PRESENT_TARGET_BINDING_UNREVIEWED' if found else ('EXTERNAL_PACKAGE_OR_MISSING_SOURCE' if not imported.startswith(('vllm.','vllm_ascend.')) else 'INTERNAL_SOURCE_MISSING_OR_SYMBOL_ONLY')

def save(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    if isinstance(data,list) and data:
        keys=list(dict.fromkeys(k for row in data for k in row))
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for row in data:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

save('source-files',list(files.values()));save('source-units',units);save('branch-sites',decisions);save('call-sites',calls);save('import-sites',imports)
summary={'status':'STRUCTURE_INDEXED_NOT_SEMANTICS_COVERED','scope_closed':False,'overall_complete':False,
         'source_files':len(files),'python_files':sum(bool(v['module']) for v in files.values()),
         'source_units':len(units),'decision_sites':len(decisions),'call_sites':len(calls),'import_sites':len(imports),
         'parse_errors':errors,'missing_explicit_source_paths':missing,
         'decision_kinds':dict(collections.Counter(x['kind'] for x in decisions)),
         'family_file_counts':dict(collections.Counter(x['family'] for x in files.values())),
         'all_reviews_initially':'UNREVIEWED',
         'limits':['Import presence does not resolve dispatch or module reachability.','AST calls/effect names do not prove producer-consumer identity, aliasing, execution or complete effects.','Loop/exception labels are structural; guarded iteration and exception paths need explicit semantic review.','C++/kernel bodies are inventoried as files only and require a separate primitive/effect expansion.','No finite maximum fusion subgraph size is used as a completion criterion.']}
save('summary',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
