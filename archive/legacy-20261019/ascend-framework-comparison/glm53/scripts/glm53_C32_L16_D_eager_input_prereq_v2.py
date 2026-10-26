"""Read-only L16-D eager input source and initial-environment prerequisite.

Importing this module performs no remote action. plans() reads frozen local
evidence, renders the two node programs and performs no SSH. execute() is the
sole remote entry point, is fixed to RUN_ID, requires an approved actual review
whose module/host-code hashes match exactly, and creates its run directory with
exist_ok=False. Neither program issues HTTP, NPU, SDK, model or profile work.

The evidence is intentionally limited to installed-source qualification plus the
process initial environment (`/proc/<pid>/environ`). It is not a live Python
`get()` measurement, not a flag change and not an optimization dispatch.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shlex
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = '20261019T-L16-sglang-D-eager-input-source-prereq-v1'
REVIEW = 'results/C32-L16-D-eager-input-prereq-actual-review-v1.json'
L15_RUN_ID = '20261019T-L15-sglang-existing-D-timeline-source-v1'
L15_LOOP = 'diagnostics/model-profile/loops/L15-D-existing-native-timeline-source-v1.json'
L15_DECISION = 'results/C32-L15-timeline-source-decision-v1.json'
L15_DECISION_SHA = '8136fdf8171aa9170f3ca7462fb64b907f2b49823ebb43b41d738ee71091b2e8'
L15_RECEIPT = 'runs/20261019T-L15-sglang-existing-D-timeline-source-v1/receipt.json'
L15_RECEIPT_SHA = 'ea0e76dc6b3b778349cde0dded5944c7b811c722ab85ca2d2182202704fa7baa'
L15_ACTUAL_REVIEW = 'results/C32-L15-D-timeline-source-actual-review-v1.json'
L15_ACTUAL_REVIEW_SHA = '70b0de421f85080079ab3548157681ad55d7e99b111571629c1ce19e219dbc4c'
L15_MODULE = 'scripts/glm53_C32_L15_D_timeline_source_v1.py'
L15_MODULE_SHA = 'bdc451ea0927c63cd47d76bd7ec9ab4a46270221db952d48a18e6509a236e3a7'
L15_WRAPPER = 'scripts/inspect_C32_L15_D_timeline_source_v1.py'
L15_WRAPPER_SHA = '8cdcbe25d8cd8e1b7dc9d80c743ea4bab07ea103ad366369b24adc88d6504408'
L15_SOURCE_DIR = 'runs/20261019T-L15-sglang-existing-D-timeline-source-v1/source'
L15_EAGER_RUNNER_LOCAL = L15_SOURCE_DIR + '/node-36/sglang.srt.model_executor.runner.eager_runner.py'
L15_EAGER_RUNNER_SHA = 'a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5'
L13_EVIDENCE = 'results/C32-L13-D-two-step-evidence-v1.json'
L13_EVIDENCE_SHA = '5fccbbafa1072417625192d6f7f904a6815685f507866787e09526ae9bd768bf'
L13_DECISION = 'results/C32-L13-D-diagnostic-acceptance-decision-v1.json'
L13_DECISION_SHA = 'b4f297c2b8bfa5ecdd0691c0fa3a2100c5091aa683627c64389093753514ce6a'
L13_AUDITOR = 'scripts/audit_C32_L13_D_diagnostic_local_v2.py'
L13_AUDITOR_SHA = 'e9754f931295600e24cee438cb811dd04dae82b3e7678691156233f1b8c414ed'
CAPTURE_ID = '20261019T-L13-sglang-C32-D-two-step-native-v1'
NODE_CAPTURES = {
    36: 'runs/20261019T-L13-sglang-C32-D-two-step-native-v1/node-36.json',
    41: 'runs/20261019T-L13-sglang-C32-D-two-step-native-v1/node-41.json',
}
NODE_CAPTURE_SHA = {
    36: '4cee2c44897634fb53b985d37d8f13d358df61445fe356954bb11303e5375b75',
    41: '43ca3b1ef2f920c97751b8377ad97d8b8fcdf542131f5092b390a8ae32df0c32',
}
CAPACITY = 'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json'
CAPACITY_SHA = '19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50'
INVENTORY = 'scripts/inventory_readonly.py'
INVENTORY_SHA = 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
CODEC = 'scripts/glm53_native_output_transport_v1.py'
CODEC_SHA = 'ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5'
IDENTITY = 'scripts/glm53_C32_L12C_mixed_model_identity_v1.py'
IDENTITY_SHA = 'ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98'
RESOLVER_ROOT = '/sgl-workspace/sglang/python'
FIXED_MODULES = [
    'sglang.srt.model_executor.cuda_graph_buffer_registry',
    'sglang.srt.model_executor.runner_utils',
    'sglang.srt.environ',
    'sglang.srt.model_executor.runner.eager_runner',
]
HELPER_PREFIX = 'sglang.srt.model_executor.runner_utils'
HELPER_TOKENS = ['_grouped_foreach_copy_', 'share_input_buffer']
KEY_TOKENS = [
    'EagerRunner',
    'load_batch',
    'build_eager_registry',
    'fill_from',
    'extract_buffer',
    'share_input_buffer',
    '_grouped_foreach_copy_',
    'SGLANG_EAGER_INPUT_NO_COPY',
    'EnvBool',
    'Envs',
]
ENV_KEYS = [
    'SGLANG_EAGER_INPUT_NO_COPY',
    'SGLANG_ENABLE_OVERLAP_PLAN_STREAM',
    'TASK_QUEUE_ENABLE',
    'HCCL_BUFFSIZE',
]
MAX_HELPER = 2
MAX_SOURCES = 6
MAX_SOURCE = 1024 ** 2
MAX_TOTAL_SOURCE = 5 * 1024 ** 2
MAX_NODE_JSON = 8 * 1024 ** 2
MAX_KEY_DEFINITIONS = 32
MAX_IMPORTS = 256

SOURCE_READER_TEMPLATE = r'''import ast,hashlib,json,pathlib
ROOT=pathlib.Path(__RESOLVER_ROOT__)
FIXED=__FIXED_MODULES__
HELPER_PREFIX=__HELPER_PREFIX__
HELPER_TOKENS=__HELPER_TOKENS__
KEY_TOKENS=__KEY_TOKENS__
MAX_HELPER=__MAX_HELPER__
MAX_SOURCE=__MAX_SOURCE__
MAX_TOTAL=__MAX_TOTAL_SOURCE__
EXPECTED_EAGER_RUNNER_SHA=__EXPECTED_EAGER_RUNNER_SHA__
MAX_KEY_DEFINITIONS=__MAX_KEY_DEFINITIONS__
MAX_IMPORTS=__MAX_IMPORTS__
if ROOT.is_symlink():raise RuntimeError('resolver_root_symlink')
BASE=ROOT.resolve(strict=True)
def _guarded(p):
    p=pathlib.Path(p)
    if p.is_symlink():raise RuntimeError('symlink_leaf:'+str(p))
    resolved=p.resolve(strict=True)
    if not resolved.is_relative_to(BASE):raise RuntimeError('outside_root:'+str(p))
    cursor=p
    while cursor!=ROOT:
        if cursor.is_symlink():raise RuntimeError('symlink_component:'+str(cursor))
        parent=cursor.parent
        if parent==cursor:break
        cursor=parent
    if not resolved.is_file():raise RuntimeError('not_a_file:'+str(p))
    return resolved
def _candidates(name):
    stem=ROOT.joinpath(*name.split('.'))
    return [{'path':str(stem/'__init__.py'),'kind':'package_init'},{'path':str(pathlib.Path(str(stem)+'.py')),'kind':'module_py'}]
def _probe(cands):
    rows=[]
    for item in cands:
        p=pathlib.Path(item['path'])
        rows.append({'path':item['path'],'kind':item['kind'],'exists':p.exists(),'is_symlink':p.is_symlink(),'is_file':p.is_file()})
    return rows
def _choose(rows):
    for row in rows:
        if row['exists'] and not row['is_symlink'] and row['is_file']:return row
    return None
def _package(name,kind):
    return name if kind=='package_init' else name.rsplit('.',1)[0]
def _relative_target(package,level,module,alias_name):
    if level<=0:return module,'absolute'
    parts=package.split('.')
    drop=level-1
    if drop>=len(parts):return None,'relative_level_'+str(level)+'_beyond_top_level'
    base='.'.join(parts[:len(parts)-drop])
    leaf=module if module else alias_name
    if not leaf:return None,'relative_level_'+str(level)+'_empty_leaf'
    return base+'.'+leaf,'relative_level_'+str(level)
def _walk_defs(body,prefix=()):
    for node in body:
        if isinstance(node,ast.ClassDef):
            yield node,prefix+(node.name,)
            yield from _walk_defs(node.body,prefix+(node.name,))
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            yield node,prefix+(node.name,)
            yield from _walk_defs(node.body,prefix+(node.name,))
def _index(text,tree):
    classes=[];definitions=[];token_lines=[]
    lines=text.splitlines()
    for node,qual in _walk_defs(tree.body):
        name='.'.join(qual)
        if isinstance(node,ast.ClassDef) and len(classes)<64:
            classes.append({'name':node.name,'qualname':name,'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',None)})
        if any(token in name for token in KEY_TOKENS) and len(definitions)<MAX_KEY_DEFINITIONS:
            end=getattr(node,'end_lineno',None) or node.lineno
            segment='\n'.join(lines[node.lineno-1:end])
            definitions.append({'kind':type(node).__name__,'qualname':name,'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',None),'source':segment[:8192],'truncated':len(segment)>8192,'original_chars':len(segment)})
    for number,line in enumerate(lines,1):
        if any(token in line for token in KEY_TOKENS) and len(token_lines)<96:
            token_lines.append({'lineno':number,'text':line[:512],'truncated':len(line)>512})
    return {'classes':classes,'key_definitions':definitions,'key_token_lines':token_lines}
def _envbool_semantics(tree,text):
    lines=text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node,ast.ClassDef) or node.name!='EnvBool':continue
        method=next((item for item in node.body if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)) and item.name=='parse'),None)
        if method is None:return {'found':True,'semantics_qualified':False,'reason':'parse_method_missing','class_lineno':node.lineno}
        end=getattr(method,'end_lineno',None) or method.lineno
        parse_source='\n'.join(lines[method.lineno-1:end])
        returns=[item for item in ast.walk(method) if isinstance(item,ast.Return)]
        returns_ok=len(returns)==2 and all(isinstance(item.value,ast.Constant) and isinstance(item.value.value,bool) for item in returns)
        lower_applied=False
        for item in ast.walk(method):
            if isinstance(item,ast.Assign) and len(item.targets)==1 and isinstance(item.targets[0],ast.Name) and item.targets[0].id=='value':
                call=item.value
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='lower' and isinstance(call.func.value,ast.Name) and call.func.value.id=='value':
                    lower_applied=True
        true_values=set();false_values=set()
        for item in ast.walk(method):
            if not isinstance(item,ast.If) or not isinstance(item.test,ast.Compare):continue
            compare=item.test
            if len(compare.ops)!=1 or not isinstance(compare.ops[0],ast.In):continue
            if not isinstance(compare.left,ast.Name) or compare.left.id!='value':continue
            container=compare.comparators[0]
            if not isinstance(container,(ast.List,ast.Tuple,ast.Set)):continue
            values=[];values_ok=True
            for element in container.elts:
                if isinstance(element,ast.Constant) and isinstance(element.value,str):values.append(element.value.lower())
                else:values_ok=False
            if not values_ok or len(item.body)!=1:continue
            branch=item.body[0]
            if isinstance(branch,ast.Return) and isinstance(branch.value,ast.Constant) and isinstance(branch.value.value,bool):
                (true_values if branch.value.value else false_values).update(values)
        qualified=bool(lower_applied and returns_ok and true_values and false_values)
        return {'found':True,'class_lineno':node.lineno,'parse_lineno':method.lineno,'parse_end_lineno':getattr(method,'end_lineno',None),'parse_source':parse_source[:8192],'parse_truncated':len(parse_source)>8192,'lower_applied':lower_applied,'returns_ok':returns_ok,'true_values':sorted(true_values),'false_values':sorted(false_values),'semantics_qualified':qualified,'reason':None if qualified else 'EnvBool_parse_pattern_not_statically_qualified'}
    return {'found':False,'semantics_qualified':False,'reason':'EnvBool_class_missing','true_values':[],'false_values':[]}
def _environ_extraction(tree,text):
    lines=text.splitlines()
    assignments=[]
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Envs']
    bodies=[(node.body,node.lineno) for node in classes] if classes else [(tree.body,0)]
    for body,class_lineno in bodies:
        for node in body:
            targets=[];value=None
            if isinstance(node,ast.Assign):targets=node.targets;value=node.value
            elif isinstance(node,ast.AnnAssign):targets=[node.target];value=node.value
            for target in targets:
                if not isinstance(target,ast.Name) or target.id!='SGLANG_EAGER_INPUT_NO_COPY':continue
                end=getattr(node,'end_lineno',None) or node.lineno
                segment='\n'.join(lines[node.lineno-1:end])
                call=value if isinstance(value,ast.Call) else None
                constructor=ast.unparse(call.func) if call is not None else None
                envbool_call=bool(call is not None and ((isinstance(call.func,ast.Name) and call.func.id=='EnvBool') or (isinstance(call.func,ast.Attribute) and call.func.attr=='EnvBool')))
                positional=call.args if call is not None else []
                keyword=next((item for item in call.keywords if item.arg=='default'),None) if call is not None else None
                candidate=positional[0] if positional else (keyword.value if keyword is not None else None)
                literal=None;literal_type=None;literal_qualified=False
                if isinstance(candidate,ast.Constant):
                    literal=candidate.value;literal_type=type(candidate.value).__name__
                    literal_qualified=isinstance(candidate.value,bool)
                default_source_qualified=bool(envbool_call and literal_qualified)
                assignments.append({'target':'SGLANG_EAGER_INPUT_NO_COPY','class_Envs_lineno':class_lineno or None,'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',None),'expression':ast.unparse(value),'source':segment[:8192],'truncated':len(segment)>8192,'constructor':constructor,'envbool_constructor':envbool_call,'positional_count':len(positional),'default_position':'positional_0' if positional else ('keyword_default' if keyword is not None else None),'default_expression':ast.unparse(candidate) if candidate is not None else None,'default_literal':literal,'default_literal_type':literal_type,'default_literal_qualified':literal_qualified,'default_source_qualified':default_source_qualified})
    semantics=_envbool_semantics(tree,text)
    qualified=next((item for item in assignments if item['default_source_qualified'] is True),None)
    return {'found':bool(assignments),'assignments':assignments,'qualified_default':{'present':qualified is not None,'literal':qualified['default_literal'] if qualified else None,'literal_type':qualified['default_literal_type'] if qualified else None,'lineno':qualified['lineno'] if qualified else None,'expression':qualified['expression'] if qualified else None,'default_source_qualified':qualified is not None},'envbool_semantics':semantics,'default_source_qualified':qualified is not None}
def _contains_no_copy(test):
    for node in ast.walk(test):
        if isinstance(node,ast.Attribute) and node.attr=='SGLANG_EAGER_INPUT_NO_COPY':return True
        if isinstance(node,ast.Name) and node.id=='SGLANG_EAGER_INPUT_NO_COPY':return True
    return False
def _segment(lines,body):
    if not body:return None
    start=body[0].lineno;end=getattr(body[-1],'end_lineno',None) or body[-1].lineno
    return '\n'.join(lines[start-1:end])
def _eager_load_batch(tree,text):
    lines=text.splitlines()
    cls=next((node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='EagerRunner'),None)
    if cls is None:return {'found':False,'reason':'EagerRunner_class_missing'}
    method=next((node for node in cls.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='load_batch'),None)
    if method is None:return {'found':False,'reason':'load_batch_method_missing','class_lineno':cls.lineno}
    end=getattr(method,'end_lineno',None) or method.lineno
    segment='\n'.join(lines[method.lineno-1:end])
    branches=[]
    for node in ast.walk(method):
        if isinstance(node,ast.If) and _contains_no_copy(node.test):
            branches.append({'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',None),'test_source':ast.unparse(node.test),'body_source':_segment(lines,node.body),'orelse_source':_segment(lines,node.orelse),'body_returns':[ast.unparse(item.value) if item.value is not None else None for item in node.body if isinstance(item,ast.Return)]})
    calls=[]
    for node in ast.walk(method):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('fill_from','extract_buffer','replace'):
            calls.append({'name':node.func.attr,'lineno':node.lineno,'source':ast.unparse(node)[:4096]})
    calls.sort(key=lambda item:(item['lineno'],item['name']))
    returns=[{'lineno':node.lineno,'expression':ast.unparse(node.value) if node.value is not None else None} for node in ast.walk(method) if isinstance(node,ast.Return)]
    returns.sort(key=lambda item:item['lineno'])
    return {'found':True,'class_lineno':cls.lineno,'lineno':method.lineno,'end_lineno':getattr(method,'end_lineno',None),'source':segment[:32768],'truncated':len(segment)>32768,'original_chars':len(segment),'no_copy_branch_found':bool(branches),'no_copy_branches':branches,'registry_or_replace_calls':calls,'returns':returns}
def _read(name,role,parent=None,basis=None):
    cands=_candidates(name);rows=_probe(cands);chosen=_choose(rows)
    row={'name':name,'role':role,'candidates':rows,'parent':parent,'basis':basis,'selected_type':None,'exists':False}
    if chosen is None:
        row['error']='module_not_found_in_fixed_candidates'
        return row,None,None
    row['selected_type']=chosen['kind'];row['selected_path']=chosen['path'];row['exists']=True
    try:
        resolved=_guarded(chosen['path']);raw=resolved.read_bytes()
        if len(raw)>MAX_SOURCE:raise RuntimeError('source_bytes_exceeds_limit:'+str(len(raw)))
        text=raw.decode('utf-8');tree=ast.parse(text)
        row.update({'resolved':str(resolved),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':text,'ast_parse_ok':True})
        row['package']=_package(name,chosen['kind'])
        row['index']=_index(text,tree)
        if name=='sglang.srt.environ':row['environ_extraction']=_environ_extraction(tree,text)
        if name=='sglang.srt.model_executor.runner.eager_runner':row['eager_load_batch']=_eager_load_batch(tree,text)
        return row,row['package'],tree
    except Exception as error:
        row['error']=repr(error);row['exists']=False
        return row,None,None
sources=[];tree_by_name={};package_by_name={};errors=[]
for name in FIXED:
    row,package,tree=_read(name,'fixed')
    sources.append(row)
    if tree is not None:tree_by_name[name]=tree;package_by_name[name]=package
    if row.get('error'):errors.append({'name':name,'error':row['error']})
imports_observed=[];skipped=[];matched=[]
for name in FIXED:
    tree=tree_by_name.get(name)
    if tree is None:continue
    package=package_by_name[name]
    parent_sha=next((item.get('sha256') for item in sources if item['name']==name),None)
    for node in ast.walk(tree):
        if not isinstance(node,ast.ImportFrom):continue
        level=node.level or 0
        for alias in node.names:
            target,resolution=_relative_target(package,level,node.module,alias.name)
            alias_text=(alias.name or '')+':'+(alias.asname or '')
            info={'parent_source':name,'parent_sha256':parent_sha,'parent_package':package,'lineno':node.lineno,'column':node.col_offset,'level':level,'module':node.module,'aliases':[item.name for item in node.names],'alias_basis':alias_text,'target':target,'resolution':resolution}
            if len(imports_observed)<MAX_IMPORTS:imports_observed.append(info)
            if target is None:
                if len(skipped)<128:skipped.append(info|{'reason':'unresolved_relative_target'})
                continue
            prefix_match=target.startswith(HELPER_PREFIX+'.')
            alias_token_match=any(token in alias_text for token in HELPER_TOKENS)
            target_token_match=any(token in target for token in HELPER_TOKENS)
            if prefix_match and alias_token_match:
                if len(matched)<64:matched.append(info|{'prefix_match':True,'alias_token_match':True,'target_token_match':target_token_match,'reason':'eligible_helper_import'})
            elif len(skipped)<128:
                skipped.append(info|{'prefix_match':prefix_match,'alias_token_match':alias_token_match,'target_token_match':target_token_match,'reason':'prefix_mismatch' if not prefix_match else 'alias_token_mismatch'})
seen=set(FIXED);helpers=[]
for info in sorted(matched,key=lambda item:(FIXED.index(item['parent_source']),item['lineno'],item['column'],item['target'])):
    target=info['target']
    if target in seen:continue
    seen.add(target)
    if len(helpers)>=MAX_HELPER:
        if len(skipped)<128:skipped.append(info|{'reason':'helper_module_cap_reached'})
        continue
    row,package,tree=_read(target,'helper',parent={'source':info['parent_source'],'sha256':info['parent_sha256'],'package':info['parent_package']},basis={'ast_import_from':{'level':info['level'],'module':info['module'],'aliases':info['aliases'],'lineno':info['lineno'],'column':info['column'],'resolution':info['resolution']},'reason':'explicit_ImportFrom_alias_token_under_helper_prefix'})
    row['import_reference']=info
    helpers.append(row)
    if row.get('error'):errors.append({'name':target,'error':row['error']})
sources.extend(helpers)
budget=MAX_TOTAL;included=0
for row in sources:
    text=row.get('text')
    if not isinstance(text,str):continue
    size=len(text.encode('utf-8'))
    if included+size>budget:
        row['text']=None;row['text_omitted']='total_source_text_budget_exceeded';row['text_bytes']=size
    else:
        included+=size;row['text_bytes']=size
eager=next((row for row in sources if row['name']=='sglang.srt.model_executor.runner.eager_runner'),None)
environ=next((row for row in sources if row['name']=='sglang.srt.environ'),None)
load_batch=eager.get('eager_load_batch') if eager else None
out={'project_id':'glm53','resolver_root':str(ROOT),'fixed_modules':FIXED,'helper_prefix':HELPER_PREFIX,'helper_tokens':HELPER_TOKENS,'key_tokens':KEY_TOKENS,'sources':sources,'imports_observed':imports_observed,'skipped_imports':skipped,'errors':errors,'helper_selection':[{'name':row['name'],'selected_type':row.get('selected_type'),'exists':row.get('exists'),'sha256':row.get('sha256'),'import_reference':row.get('import_reference')} for row in helpers],'fixed_resolved':sum(1 for row in sources if row['role']=='fixed' and row.get('exists') is True),'helper_resolved':sum(1 for row in sources if row['role']=='helper' and row.get('exists') is True),'total_sources':len(sources),'total_source_bytes':sum(row.get('bytes',0) for row in sources if isinstance(row.get('bytes'),int)),'included_text_bytes':included,'checks':{'all_fixed_resolved':sum(1 for row in sources if row['role']=='fixed' and row.get('exists') is True)==len(FIXED),'all_ast_parsed':all(row.get('ast_parse_ok') is True for row in sources),'total_sources_within_cap':len(sources)<=MAX_HELPER+len(FIXED),'eager_runner_sha256':eager.get('sha256') if eager else None,'eager_runner_sha_matches_l15':bool(eager and eager.get('sha256')==EXPECTED_EAGER_RUNNER_SHA),'environ_assignments_found':bool(environ and environ.get('environ_extraction',{}).get('found')),'load_batch_found':bool(load_batch and load_batch.get('found')),'no_copy_branch_found':bool(load_batch and load_batch.get('no_copy_branch_found'))},'limits':{'max_helper_modules':MAX_HELPER,'max_total_sources':MAX_HELPER+len(FIXED),'max_source_bytes':MAX_SOURCE,'max_total_source_text_bytes':MAX_TOTAL,'follow_depth':1,'no_recursive_traversal':True,'framework_imported':False,'torch_imported':False,'SDK_imported':False,'path_boundary':str(ROOT),'symlink_allowed':False}}
print(json.dumps(out))
'''


def _build_source_reader(resolver_root=RESOLVER_ROOT):
    reader = SOURCE_READER_TEMPLATE
    for placeholder, value in (
        ('__RESOLVER_ROOT__', repr(resolver_root)),
        ('__FIXED_MODULES__', repr(FIXED_MODULES)),
        ('__HELPER_PREFIX__', repr(HELPER_PREFIX)),
        ('__HELPER_TOKENS__', repr(HELPER_TOKENS)),
        ('__KEY_TOKENS__', repr(KEY_TOKENS)),
        ('__MAX_HELPER__', repr(MAX_HELPER)),
        ('__MAX_SOURCE__', repr(MAX_SOURCE)),
        ('__MAX_TOTAL_SOURCE__', repr(MAX_TOTAL_SOURCE)),
        ('__EXPECTED_EAGER_RUNNER_SHA__', repr(L15_EAGER_RUNNER_SHA)),
        ('__MAX_KEY_DEFINITIONS__', repr(MAX_KEY_DEFINITIONS)),
        ('__MAX_IMPORTS__', repr(MAX_IMPORTS)),
    ):
        assert reader.count(placeholder) == 1, placeholder
        reader = reader.replace(placeholder, value)
    assert not re.search(r'__[A-Z_]+__', reader), 'unrendered source-reader placeholder'
    assert not re.search(r'\b(?:null|true|false)\b', reader), 'JSON literal in Python source reader'
    ast.parse(reader)
    compile(reader, '<L16-source-reader>', 'exec')
    return reader


SOURCE_READER = _build_source_reader()

TEMPLATE = r'''
import base64,collections,gzip,hashlib,json,pathlib,subprocess,time,traceback
out={'project_id':'glm53','run_id':L16_RUN_ID,'mode':'READ_ONLY_EAGER_INPUT_SOURCE_AND_INITIAL_ENV_PREREQUISITE','node':L16_NODE,'read_only':True,'evidence_basis':'initial_process_environ_plus_frozen_installed_source','live_python_get_measured':False,'profile_csv_trace_reread':False,'model_requests':0,'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'new_profiles':0,'remote_writes':0}
started=time.monotonic()
ENV_KEYS=L16_ENV_KEYS
WORKERS=L16_WORKERS
def _proc_path(pid,leaf):return pathlib.Path('/proc')/str(pid)/leaf
def _nspid(pid):
    try:rows=_proc_path(pid,'status').read_text().splitlines()
    except Exception as error:return None,repr(error)
    for row in rows:
        if row.startswith('NSpid:'):
            try:return [int(item) for item in row.split()[1:]],None
            except Exception as error:return None,repr(error)
    return None,'NSpid_missing'
def _selected_environment(pid):
    try:raw=_proc_path(pid,'environ').read_bytes()
    except Exception as error:return {'error':repr(error),'keys':{},'source':'/proc/<pid>/environ read-only','initial_process_environment':True}
    selected={key:{'present':False,'raw_value':None,'raw_chars':0,'truncated':False} for key in ENV_KEYS}
    for item in raw.split(b'\0'):
        if b'=' not in item:continue
        key,value=item.split(b'=',1)
        try:name=key.decode('utf-8')
        except Exception:continue
        if name not in selected:continue
        text=value.decode('utf-8','replace')
        selected[name]={'present':True,'raw_value':text[:512],'raw_chars':len(text),'truncated':len(text)>512}
    return {'error':None,'keys':selected,'source':'/proc/<pid>/environ read-only; only the four allowlisted keys are retained','initial_process_environment':True,'other_environment_retained':False,'full_environment_hashed':False}
def _proc_stat(pid):
    try:text=_proc_path(pid,'stat').read_text()
    except Exception as error:return {'error':repr(error)}
    try:
        rest=text.rsplit(')',1)[1].split()
        return {'state':rest[0],'ppid':int(rest[1]),'starttime':int(rest[19]),'field_note':'field 22 starttime; fields parsed after comm'}
    except Exception as error:return {'error':repr(error),'raw_chars':len(text)}
def _cgroup(pid):
    try:text=_proc_path(pid,'cgroup').read_text()
    except Exception as error:return {'error':repr(error),'contains_current_cid':False}
    return {'error':None,'contains_current_cid':L16_CID in text,'line_count':len(text.splitlines()),'chars':len(text)}
def _alive(pid):
    path=pathlib.Path('/proc')/str(pid)
    if not path.exists():return False,0
    try:argv=[item for item in _proc_path(pid,'cmdline').read_bytes().split(b'\0') if item]
    except Exception:return False,0
    stat=_proc_stat(pid)
    state=stat.get('state')
    return bool(argv and state not in (None,'Z','X')),len(argv)
def _classify(raw,semantics):
    default_qualified=bool(semantics.get('default_source_qualified'))
    default_literal=semantics.get('default_literal')
    if raw is None:
        return {'state':'absent','effective_value_static':default_literal if default_qualified else None,'qualified':default_qualified,'basis':'key_not_present_in_initial_environ; value_from_actual_EnvBool_default_source' if default_qualified else 'key_not_present_in_initial_environ; default_source_not_qualified','source_default_literal':default_literal,'default_source_qualified':default_qualified}
    text=raw.get('raw_value')
    if raw.get('truncated'):return {'state':'value_truncated_not_classified','effective_value_static':None,'qualified':False,'basis':'raw_value_exceeds_512_chars','raw_value':text,'source_default_literal':default_literal,'default_source_qualified':default_qualified}
    if not semantics.get('semantics_qualified'):return {'state':'present_semantics_unqualified','effective_value_static':None,'qualified':False,'basis':'actual_EnvBool_parse_source_not_statically_qualified','raw_value':text,'source_default_literal':default_literal,'default_source_qualified':default_qualified}
    lowered=text.lower()
    if lowered in semantics.get('true_values',[]):return {'state':'explicit_true','effective_value_static':True,'qualified':True,'basis':'actual_EnvBool_parse_source','raw_value':text,'source_default_literal':default_literal,'default_source_qualified':default_qualified}
    if lowered in semantics.get('false_values',[]):return {'state':'explicit_false','effective_value_static':False,'qualified':True,'basis':'actual_EnvBool_parse_source','raw_value':text,'source_default_literal':default_literal,'default_source_qualified':default_qualified}
    return {'state':'invalid_value','effective_value_static':None,'qualified':False,'basis':'raw_value_retained_not_repaired; actual_EnvField_get_catches_ValueError_and_uses_default_only_if_default_source_qualified','raw_value':text,'default_literal_if_ValueError_fallback':default_literal if default_qualified else None,'default_source_qualified':default_qualified}
def _worker_row(expected):
    pid=expected['host_pid'];nspid,nspid_error=_nspid(pid);cgroup=_cgroup(pid);stat=_proc_stat(pid);env=_selected_environment(pid);alive,argc=_alive(pid)
    nspid_ok=bool(nspid and nspid[-1]==expected['namespace_pid'])
    cgroup_ok=bool(cgroup.get('contains_current_cid'))
    no_copy=env.get('keys',{}).get('SGLANG_EAGER_INPUT_NO_COPY')
    classification=(_classify(no_copy if no_copy and no_copy.get('present') else None,semantics) if env.get('error') is None else {'state':'initial_environ_unreadable','effective_value_static':None,'qualified':False,'basis':'initial_environ_read_failed; not treated as absent/default','error':env.get('error')})
    return {'expected':expected,'alive':alive,'cmdline_argc':argc,'namespace_pid_chain':nspid,'namespace_pid_error':nspid_error,'namespace_pid_last':nspid[-1] if nspid else None,'namespace_pid_matches_expected':nspid_ok,'cgroup_contains_current_cid':cgroup_ok,'cgroup_error':cgroup.get('error'),'proc_stat':stat,'selected_environment':env,'no_copy_classification':classification,'identity_qualified':bool(alive and nspid_ok and cgroup_ok)}
try:
    assert model_identity['host_pid']==L16_CAP_PARENT_HOST_PID
    assert model_identity['namespace_pid']==L16_CAP_PARENT_NS_PID
    reader=subprocess.run(['docker','exec',L16_CID,'python3','-S','-c',L16_SOURCE_READER],capture_output=True,text=True,timeout=180)
    if reader.returncode!=0:source={'error':{'rc':reader.returncode,'stderr':reader.stderr[-4000:]},'sources':[],'checks':{}}
    else:source=json.loads(reader.stdout)
    out['source_resolution']=source
    environ_source=next((row for row in source.get('sources',[]) if row.get('name')=='sglang.srt.environ'),None)
    eager_source=next((row for row in source.get('sources',[]) if row.get('name')=='sglang.srt.model_executor.runner.eager_runner'),None)
    env_extraction=environ_source.get('environ_extraction') if environ_source else None
    load_batch=eager_source.get('eager_load_batch') if eager_source else None
    semantics=(env_extraction or {}).get('envbool_semantics') or {'semantics_qualified':False,'true_values':[],'false_values':[]}
    if env_extraction and env_extraction.get('default_source_qualified'):semantics['default_source_qualified']=True;semantics['default_literal']=env_extraction.get('qualified_default',{}).get('literal')
    out['source_read']={'method':'docker exec <CID> python3 -S -c <reader>','imports':['ast','hashlib','json','pathlib'],'framework_imported':False,'torch_imported':False,'SDK_imported':False,'resolver_root':L16_RESOLVER_ROOT,'fixed_modules':L16_FIXED_MODULES,'helper_prefix':L16_HELPER_PREFIX,'helper_alias_tokens':L16_HELPER_TOKENS,'max_helper_modules':L16_MAX_HELPER,'max_total_sources':L16_MAX_SOURCES,'max_source_bytes':L16_MAX_SOURCE,'max_total_source_text_bytes':L16_MAX_TOTAL_SOURCE,'source_reader_sha256':hashlib.sha256(L16_SOURCE_READER.encode()).hexdigest()}
    out['environ_qualification']={'module':'sglang.srt.environ','assignment_found':bool(env_extraction and env_extraction.get('found')),'assignments':(env_extraction or {}).get('assignments'),'default_source_qualified':bool(env_extraction and env_extraction.get('default_source_qualified')),'qualified_default':(env_extraction or {}).get('qualified_default'),'envbool_semantics':semantics,'note':'AST only; the environ module was not imported and no framework, torch or SDK code was executed.'}
    out['eager_load_batch_source']={'module':'sglang.srt.model_executor.runner.eager_runner','sha256':eager_source.get('sha256') if eager_source else None,'load_batch':load_batch,'note':'Source recording only; load_batch was not executed.'}
    workers=[_worker_row(item) for item in WORKERS]
    parent_pid=model_identity['host_pid']
    parent_nspid,parent_nspid_error=_nspid(parent_pid);parent_cgroup=_cgroup(parent_pid);parent_stat=_proc_stat(parent_pid);parent_env=_selected_environment(parent_pid);parent_alive,parent_argc=_alive(parent_pid)
    out['workers']=workers
    out['parent_model']={'host_pid':parent_pid,'namespace_pid_expected':model_identity['namespace_pid'],'namespace_pid_chain':parent_nspid,'namespace_pid_error':parent_nspid_error,'namespace_pid_matches_expected':bool(parent_nspid and parent_nspid[-1]==model_identity['namespace_pid']),'cgroup_contains_current_cid':bool(parent_cgroup.get('contains_current_cid')),'cgroup_error':parent_cgroup.get('error'),'proc_stat':parent_stat,'selected_environment':parent_env,'alive':parent_alive,'cmdline_argc':parent_argc,'identity_from_original_mixed_render':True}
    state_counts=collections.Counter(item['no_copy_classification']['state'] for item in workers)
    out['worker_summary']={'expected':len(WORKERS),'alive':sum(1 for item in workers if item['alive']),'identity_qualified':sum(1 for item in workers if item['identity_qualified']),'project_cgroup_match':sum(1 for item in workers if item['cgroup_contains_current_cid']),'namespace_pid_match':sum(1 for item in workers if item['namespace_pid_matches_expected']),'host_pids_distinct':len({item['expected']['host_pid'] for item in workers})==len(WORKERS),'namespace_pids_distinct':len({item['expected']['namespace_pid'] for item in workers})==len(WORKERS),'no_copy_state_counts':dict(state_counts),'any_explicit_true':state_counts.get('explicit_true',0)>0,'all_absent':state_counts.get('absent',0)==len(WORKERS),'native_option_may_already_be_enabled':state_counts.get('explicit_true',0)>0,'not_a_new_optimization':state_counts.get('explicit_true',0)>0,'collection_is_not_an_optimization':True,'trial_authorized':False,'selected_keys':ENV_KEYS}
    out['interpretation']={'label':'INITIAL_ENV_PLUS_FROZEN_SOURCE_QUALIFIED','live_python_get_measured':False,'note':'/proc environ is the process initial environment. The EagerRunner no-copy branch and EnvBool default are statically parsed from the installed source, not executed. No runtime function return, tensor lifetime, performance gain or trial authorization was measured.','trial_authorized':False,'flag_change_authorized':False,'current_runtime_preserved':True,'graph_taskqueue0_preserved':True,'capacity_preserved':True}
    checks={'source_reader_rc_zero':reader.returncode==0,'fixed_sources_resolved':source.get('fixed_resolved')==len(L16_FIXED_MODULES),'total_sources_within_cap':source.get('total_sources',len(source.get('sources',[])))<=L16_MAX_SOURCES,'sources_ast_parsed':all(row.get('ast_parse_ok') is True for row in source.get('sources',[])),'eager_runner_sha_matches_l15':source.get('checks',{}).get('eager_runner_sha_matches_l15') is True,'environ_assignments_found':source.get('checks',{}).get('environ_assignments_found') is True,'load_batch_found':source.get('checks',{}).get('load_batch_found') is True,'no_copy_branch_found':source.get('checks',{}).get('no_copy_branch_found') is True,'workers_expected_16':len(WORKERS)==16,'workers_host_pids_distinct':len({item['expected']['host_pid'] for item in workers})==len(WORKERS),'workers_namespace_pids_distinct':len({item['expected']['namespace_pid'] for item in workers})==len(WORKERS),'workers_all_alive':all(item['alive'] for item in workers),'workers_all_project_cgroup':all(item['cgroup_contains_current_cid'] for item in workers),'workers_all_namespace_pid':all(item['namespace_pid_matches_expected'] for item in workers),'parent_alive':parent_alive,'parent_cgroup_match':bool(parent_cgroup.get('contains_current_cid')),'parent_namespace_pid_match':bool(parent_nspid and parent_nspid[-1]==model_identity['namespace_pid']),'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'model_requests':0,'new_profiles':0,'remote_writes':0,'profile_csv_trace_reread':False}
    checks.update({'workers_environment_readable':all(item['selected_environment'].get('error') is None for item in workers),'parent_environment_readable':parent_env.get('error') is None,'workers_proc_stats_readable':all(item['proc_stat'].get('error') is None for item in workers),'parent_proc_stat_readable':parent_stat.get('error') is None})
    out['checks']=checks
    zero_keys=('HTTP_requests','NPU_calls','SDK_init_calls','model_requests','new_profiles','remote_writes')
    out['collection_pass']=all(value is True for key,value in checks.items() if key not in zero_keys+('profile_csv_trace_reread',)) and all(checks[key]==0 for key in zero_keys) and checks['profile_csv_trace_reread'] is False
except BaseException as error:
    out['error']=repr(error);out['traceback']=traceback.format_exc();out['collection_pass']=False
out['duration_seconds']=time.monotonic()-started
raw=json.dumps(out).encode();assert len(raw)<=L16_MAX_NODE_JSON,'node context exceeds 8 MiB'
packed=gzip.compress(raw,mtime=0);assert len(packed)<=8*1024**2
print('GLM53_NATIVE_GZIP='+json.dumps({'version':1,'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'packed_bytes':len(packed),'packed_sha256':hashlib.sha256(packed).hexdigest(),'archive':None,'gzip_base64':base64.b64encode(packed).decode()}),flush=True)
'''


def _read(relative, digest):
    raw = (ROOT / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, relative
    return json.loads(raw)


def _sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _load(name, relative, digest):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(template, values):
    pattern = r'\b(?:' + '|'.join(re.escape(key) for key in values) + r')\b'
    code = re.sub(pattern, lambda match: repr(values[match[0]]), template)
    ast.parse(code)
    assert not ({key for key in values if re.search(r'\b' + re.escape(key) + r'\b', code)}), 'unrendered placeholder'
    assert not re.search(r'\b(?:null|true|false)\b', code), 'JSON literal in rendered Python code'
    return code


def _node_capture_identity(node, capacity):
    relative = NODE_CAPTURES[node]
    raw = (ROOT / relative).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == NODE_CAPTURE_SHA[node], relative
    document = json.loads(raw)
    assert document['rc'] == 0
    frame = document['native_output_transport']
    assert frame['version'] == 1 and 0 < frame['raw_bytes'] <= 256 * 1024 ** 2 and frame['packed_bytes'] <= 8 * 1024 ** 2
    payload = json.loads(document['stdout'])
    assert payload['project_id'] == 'glm53' and payload['node'] == node
    assert payload['container_id'] == capacity['providers'][str(node)]['container_id']
    assert payload['provider_phase'] == 'READY'
    identity = payload['actual_model_identity']
    supervisor = payload['supervisor']
    assert supervisor['model_pid'] == identity['namespace_pid']
    assert supervisor['phase'] == 'MODEL_RUNNING' and supervisor['model_started'] is True
    owned = payload['owned_npu_process_rows']
    assert isinstance(owned, list) and len(owned) <= 64
    audit = payload['persistent_audit']
    workers = []
    non_worker_audits = []
    for filename, item in sorted(audit.items()):
        match = re.fullmatch(r'sdma-(\d+)\.jsonl', filename)
        if not match:
            non_worker_audits.append({'file': filename, 'reason': 'name_not_sdk_pid'})
            continue
        audit_pid = int(match.group(1))
        init_rows = [row for row in item.get('rows', []) if row.get('event') == 'initialize_complete']
        if not init_rows:
            non_worker_audits.append({'file': filename, 'sdk_pid': audit_pid, 'reason': 'no_initialize_complete'})
            continue
        assert len(init_rows) == 1, filename
        row = init_rows[0]
        assert row['pid'] == audit_pid and row['project_id'] == 'glm53'
        assert row['protocol'] == 'SDMA' and row.get('protocol_value') == 1
        assert row['rc'] == 0 and row.get('role') == 'Decode'
        if not isinstance(row.get('device_id'), int):
            raise AssertionError(filename + ' missing integer device_id')
        matches = [owned_row for owned_row in owned if owned_row.get('namespace_pid') == audit_pid]
        assert len(matches) == 1, filename + ' owned row match count'
        owned_row = matches[0]
        workers.append({
            'node': node,
            'namespace_pid': audit_pid,
            'host_pid': owned_row['host_pid'],
            'device_id': row['device_id'],
            'card': owned_row['card'],
            'chip': owned_row['chip'],
            'session': row.get('session'),
            'audit_file': filename,
            'audit_bytes': item['bytes'],
            'audit_sha256': item['sha256'],
        })
    assert len(workers) == 16, f'node {node} SDK worker count {len(workers)}'
    assert sorted(worker['device_id'] for worker in workers) == list(range(16))
    assert len({worker['host_pid'] for worker in workers}) == 16
    assert len({worker['namespace_pid'] for worker in workers}) == 16
    profile_workers = payload['native_profile']['workers']
    assert isinstance(profile_workers, dict) and len(profile_workers) == 16
    by_namespace_pid = {}
    for folder, info in profile_workers.items():
        by_namespace_pid.setdefault(info['namespace_pid'], []).append((folder, info))
    for worker in workers:
        matches = by_namespace_pid.get(worker['namespace_pid'], [])
        assert len(matches) == 1, f'node {node} profile worker match'
        folder, info = matches[0]
        assert info['native_device_id'] == worker['device_id']
        worker['folder'] = folder
        worker['TP_rank'] = info['TP_rank']
    expected_ranks = list(range(16)) if node == 36 else list(range(16, 32))
    assert sorted(worker['TP_rank'] for worker in workers) == expected_ranks
    worker_namespace_pids = {worker['namespace_pid'] for worker in workers}
    unmatched_owned_rows = [row for row in owned if row.get('namespace_pid') not in worker_namespace_pids]
    workers.sort(key=lambda worker: worker['device_id'])
    return {
        'node': node,
        'capture_path': relative,
        'capture_sha256': digest,
        'capture_bytes': len(raw),
        'native_output_raw_sha256': frame['raw_sha256'],
        'container_id': payload['container_id'],
        'parent_model': {
            'host_pid': identity['host_pid'],
            'namespace_pid': identity['namespace_pid'],
            'supervisor_model_pid': supervisor['model_pid'],
            'argv_recorded_in_capture': True,
        },
        'persistent_audit': {
            'files': len(audit),
            'initialize_complete_workers': len(workers),
            'non_worker_audits': non_worker_audits,
        },
        'owned_npu_process_rows_total': len(owned),
        'unmatched_owned_rows': unmatched_owned_rows,
        'unmatched_owned_rows_note': 'Recorded for identity provenance only; these rows are not added to the worker whitelist and no new PID is discovered.',
        'workers': workers,
    }


def _inputs():
    l15_decision = _read(L15_DECISION, L15_DECISION_SHA)
    assert l15_decision['project_id'] == 'glm53' and l15_decision['decision'] == 'ACCEPTED'
    assert l15_decision['collection_receipt_sha256'] == L15_RECEIPT_SHA
    assert l15_decision['source_eager_runner_sha256'] == L15_EAGER_RUNNER_SHA
    assert l15_decision['performance_certificate_allowed'] is False
    l15_receipt = _read(L15_RECEIPT, L15_RECEIPT_SHA)
    assert l15_receipt['project_id'] == 'glm53' and l15_receipt['run_id'] == L15_RUN_ID
    assert l15_receipt['pass'] is True and l15_receipt['source_sha256'] == L15_MODULE_SHA
    assert l15_receipt['review']['sha256'] == L15_ACTUAL_REVIEW_SHA
    assert l15_receipt['model_requests'] == 0 and l15_receipt['HTTP_requests'] == 0
    assert l15_receipt['NPU_calls'] == 0 and l15_receipt['SDK_init_calls'] == 0
    assert l15_receipt['remote_writes'] == 0
    l15_review = _read(L15_ACTUAL_REVIEW, L15_ACTUAL_REVIEW_SHA)
    assert l15_review['project_id'] == 'glm53' and l15_review['run_id'] == L15_RUN_ID
    assert l15_review['decision'] == 'APPROVED_READ_ONLY' and l15_review['read_only'] is True
    assert l15_review['module_sha256'] == L15_MODULE_SHA
    assert l15_review['host_codes_sha256']['36'] and l15_review['host_codes_sha256']['41']
    assert _sha(L15_MODULE) == L15_MODULE_SHA
    assert _sha(L15_WRAPPER) == L15_WRAPPER_SHA
    l15_loop = _read(L15_LOOP, _sha(L15_LOOP))
    assert l15_loop['project_id'] == 'glm53' and l15_loop['loop_id'] == 'L15-D-existing-native-timeline-source-v1'
    assert l15_loop['state'] == 'CLOSED_ACCEPTED' and l15_loop['decision'] == 'ACCEPTED'
    assert l15_loop['acceptance']['decision_sha256'] == L15_DECISION_SHA
    assert l15_loop['acceptance']['collection_receipt_sha256'] == L15_RECEIPT_SHA
    assert l15_loop['acceptance']['actual_review_sha256'] == L15_ACTUAL_REVIEW_SHA
    assert _sha(L15_EAGER_RUNNER_LOCAL) == L15_EAGER_RUNNER_SHA
    assert _sha(L15_SOURCE_DIR + '/node-41/sglang.srt.model_executor.runner.eager_runner.py') == L15_EAGER_RUNNER_SHA

    l13_evidence = _read(L13_EVIDENCE, L13_EVIDENCE_SHA)
    l13_decision = _read(L13_DECISION, L13_DECISION_SHA)
    assert l13_evidence['project_id'] == 'glm53' and l13_evidence['capture_id'] == CAPTURE_ID
    assert l13_evidence['diagnostic_evidence_pass'] is True and l13_evidence['diagnostic_only'] is True
    assert l13_evidence['performance_certificate_allowed'] is False
    assert l13_evidence['all32_D_native_profile_workers'] is True
    assert l13_evidence['actual_P_MemCache_read_workers'] == 32
    assert l13_evidence['actual_D_SDMA_receiving_devices'] == 32
    assert l13_evidence['physical_wire_bytes'] is None
    assert l13_decision['project_id'] == 'glm53' and l13_decision['decision'] == 'ACCEPTED'
    assert l13_decision['evidence_sha256'] == L13_EVIDENCE_SHA
    assert l13_decision['auditor_sha256'] == L13_AUDITOR_SHA
    assert _sha(L13_AUDITOR) == L13_AUDITOR_SHA

    capacity = _read(CAPACITY, CAPACITY_SHA)
    assert capacity['project_id'] == 'glm53' and capacity['pass'] is True
    assert capacity['selected_variant'] == 'candidate'
    for node in (36, 41):
        provider = capacity['providers'][str(node)]
        assert provider['container_id'] == l13_evidence['runtime_CIDs'][str(node)], f'node {node} capacity CID differs from L13 runtime'
    capacity_subset = {
        'providers': {str(node): capacity['providers'][str(node)] for node in (36, 41)},
        'selected_variant': capacity['selected_variant'],
        'meta_container_id': capacity['meta_container_id'],
    }
    identities = {str(node): _node_capture_identity(node, capacity) for node in (36, 41)}
    all_workers = [worker for node in (36, 41) for worker in identities[str(node)]['workers']]
    assert len(all_workers) == 32
    assert len({(worker['node'],worker['host_pid']) for worker in all_workers}) == 32
    assert sorted(worker['TP_rank'] for worker in all_workers) == list(range(32))
    assert len({(worker['node'],worker['namespace_pid']) for worker in all_workers}) == 32
    for identity in identities.values():
        assert identity['parent_model']['namespace_pid'] == identity['parent_model']['supervisor_model_pid']
    return {
        'l15_decision': l15_decision,
        'l15_receipt': l15_receipt,
        'l15_review': l15_review,
        'l15_loop': l15_loop,
        'l13_evidence': l13_evidence,
        'l13_decision': l13_decision,
        'capacity': capacity,
        'capacity_subset': capacity_subset,
        'identities': identities,
        'all_workers': all_workers,
        'base': _load('inventory', INVENTORY, INVENTORY_SHA),
        'codec': _load('codec', CODEC, CODEC_SHA),
        'identity_renderer': _load('identity', IDENTITY, IDENTITY_SHA),
    }


def plans():
    inputs = _inputs()
    codes = {}
    for node in (36, 41):
        identity = inputs['identities'][str(node)]
        values = {
            'L16_RUN_ID': RUN_ID,
            'L16_NODE': node,
            'L16_CID': inputs['capacity_subset']['providers'][str(node)]['container_id'],
            'L16_WORKERS': identity['workers'],
            'L16_CAP_PARENT_HOST_PID': identity['parent_model']['host_pid'],
            'L16_CAP_PARENT_NS_PID': identity['parent_model']['namespace_pid'],
            'L16_ENV_KEYS': ENV_KEYS,
            'L16_SOURCE_READER': SOURCE_READER,
            'L16_RESOLVER_ROOT': RESOLVER_ROOT,
            'L16_FIXED_MODULES': FIXED_MODULES,
            'L16_HELPER_PREFIX': HELPER_PREFIX,
            'L16_HELPER_TOKENS': HELPER_TOKENS,
            'L16_MAX_HELPER': MAX_HELPER,
            'L16_MAX_SOURCE': MAX_SOURCE,
            'L16_MAX_SOURCES': MAX_SOURCES,
            'L16_MAX_TOTAL_SOURCE': MAX_TOTAL_SOURCE,
            'L16_MAX_NODE_JSON': MAX_NODE_JSON,
        }
        code = inputs['identity_renderer'].render(node, inputs['capacity_subset']) + '\n' + _render(TEMPLATE, values)
        ast.parse(code)
        codes[str(node)] = code
    meta = {
        'project_id': 'glm53',
        'run_id': RUN_ID,
        'mode': 'READ_ONLY_EAGER_INPUT_SOURCE_AND_INITIAL_ENV_PREREQUISITE',
        'draft_status': 'DRAFT_PENDING_PRIMARY_REVIEW',
        'purpose': 'Qualify the actual installed eager input-registry/helper/environ sources and the current 32 D model workers initial environment before deciding whether a single-variable SGLANG_EAGER_INPUT_NO_COPY False->True test is worth freezing. This draft does not authorize a model experiment, flag change, deployment, reload or optimization.',
        'evidence_basis': 'initial_process_environ_plus_frozen_installed_source',
        'live_python_get_measured': False,
        'optimization_authorized': False,
        'flag_change_authorized': False,
        'runtime_change_authorized': False,
        'trial_authorized': False,
        'basis': {
            'l15_decision': {'path': L15_DECISION, 'sha256': L15_DECISION_SHA, 'decision': 'ACCEPTED'},
            'l15_receipt': {'path': L15_RECEIPT, 'sha256': L15_RECEIPT_SHA, 'pass': True},
            'l15_actual_review': {'path': L15_ACTUAL_REVIEW, 'sha256': L15_ACTUAL_REVIEW_SHA, 'decision': 'APPROVED_READ_ONLY'},
            'l15_loop': {'path': L15_LOOP, 'sha256': _sha(L15_LOOP), 'state': 'CLOSED_ACCEPTED', 'decision': 'ACCEPTED'},
            'l15_module': {'path': L15_MODULE, 'sha256': L15_MODULE_SHA},
            'l15_wrapper': {'path': L15_WRAPPER, 'sha256': L15_WRAPPER_SHA},
            'l15_eager_runner_source': {'path': L15_EAGER_RUNNER_LOCAL, 'sha256': L15_EAGER_RUNNER_SHA, 'cross_node_match': True},
            'l13_evidence': {'path': L13_EVIDENCE, 'sha256': L13_EVIDENCE_SHA},
            'l13_decision': {'path': L13_DECISION, 'sha256': L13_DECISION_SHA},
            'l13_auditor': {'path': L13_AUDITOR, 'sha256': L13_AUDITOR_SHA},
            'capacity': {'path': CAPACITY, 'sha256': CAPACITY_SHA},
        },
        'capture': {'capture_id': CAPTURE_ID, 'node_captures': {str(node): {'path': NODE_CAPTURES[node], 'sha256': NODE_CAPTURE_SHA[node]} for node in (36, 41)}},
        'worker_identity': {
            'basis': 'L13 native capture persistent_audit initialize_complete rows matched to owned_npu_process_rows host_pid and native_profile workers; exact 16 SDK workers per node.',
            'nodes': {
                str(node): {
                    'capture_parent_model': inputs['identities'][str(node)]['parent_model'],
                    'persistent_audit': inputs['identities'][str(node)]['persistent_audit'],
                    'owned_npu_process_rows_total': inputs['identities'][str(node)]['owned_npu_process_rows_total'],
                    'unmatched_owned_rows': inputs['identities'][str(node)]['unmatched_owned_rows'],
                    'workers': inputs['identities'][str(node)]['workers'],
                }
                for node in (36, 41)
            },
            'all32_workers': inputs['all_workers'],
            'all32_workers_distinct_host_pid': len({worker['host_pid'] for worker in inputs['all_workers']}) == 32,
            'all32_workers_distinct_namespace_pid': len({worker['namespace_pid'] for worker in inputs['all_workers']}) == 32,
            'all32_TP_ranks_complete': sorted(worker['TP_rank'] for worker in inputs['all_workers']) == list(range(32)),
            'no_new_pid_discovery': True,
            'no_npu_smi_exploration': True,
        },
        'source_resolution': {
            'resolver_root': RESOLVER_ROOT,
            'fixed_modules': FIXED_MODULES,
            'helper_prefix': HELPER_PREFIX,
            'helper_alias_tokens': HELPER_TOKENS,
            'candidate_rule': 'package/__init__.py first, then module.py; both candidates exists/symlink/is_file flags recorded; no broad find',
            'relative_import_rule': 'Python package semantics: level n resolves against the selected source package with n-1 parent components removed; beyond top-level is unresolved; local fixture validated',
            'max_fixed_modules': len(FIXED_MODULES),
            'max_helper_modules': MAX_HELPER,
            'max_total_sources': MAX_SOURCES,
            'max_source_bytes_each': MAX_SOURCE,
            'max_total_source_text_bytes': MAX_TOTAL_SOURCE,
            'node_json_limit_bytes': MAX_NODE_JSON,
            'follow_depth': 1,
            'no_recursive_traversal': True,
            'framework_or_torch_or_SDK_imported': False,
            'expected_eager_runner_sha256': L15_EAGER_RUNNER_SHA,
            'source_reader_python_literals': True,
            'source_reader_json_null_true_false_tokens': False,
        },
        'source_reader': {
            'sha256': hashlib.sha256(SOURCE_READER.encode()).hexdigest(),
            'bytes': len(SOURCE_READER.encode()),
            'nested_in_each_rendered_node_program': True,
        },
        'environment_read': {
            'selected_keys': ENV_KEYS,
            'source': '/proc/<pid>/environ, initial process environment only',
            'other_environment_retained': False,
            'full_environment_hashed': False,
            'credentials_read_or_written': False,
            'runtime_python_get_measured': False,
            'initial_process_environment_only': True,
            'no_new_pid_discovery': True,
            'no_npu_smi_exploration': True,
            'static_source_qualification': 'Envs.SGLANG_EAGER_INPUT_NO_COPY assignment, EnvBool default literal and EnvBool.parse semantics are parsed from the installed environ source; the module is not imported.',
        },
        'boundaries': {
            'nodes': [36, 41],
            'sequential': True,
            'max_concurrent_SSH': 1,
            'node_invocations': 2,
            'extra_SSH': 0,
            'HTTP_requests': 0,
            'NPU_calls': 0,
            'SDK_init_calls': 0,
            'model_requests': 0,
            'new_profiles': 0,
            'remote_writes': 0,
            'profile_or_CSV_or_trace_reread': False,
            'P_or_Meta_containers_contacted': False,
            'source_text_local_only': True,
            'no_publish': True,
            'flag_service_model_config_change': False,
            'graph_taskqueue0_or_capacity_change': False,
        },
        'limits': {
            'source_files_max': MAX_SOURCES,
            'source_file_bytes_max': MAX_SOURCE,
            'source_total_text_bytes_max': MAX_TOTAL_SOURCE,
            'node_json_bytes_max': MAX_NODE_JSON,
            'workers_per_node': 16,
            'worker_host_pid_whitelist_exact': True,
            'environment_keys_only': ENV_KEYS,
            'workers_per_node': 16,
            'node_invocations_if_approved': 2,
            'max_concurrent_SSH': 1,
        },
        'actual_review_requirement': {
            'path': REVIEW,
            'decision': 'APPROVED_READ_ONLY',
            'read_only': True,
            'module_sha256_must_equal': 'computed at execute time',
            'host_codes_sha256_must_equal': 'computed at execute time',
            'model_requests_authorized': 0,
            'new_profile_authorized': False,
        },
        'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wrapper': {'path': 'scripts/inspect_C32_L16_D_eager_input_prereq_v1.py', 'sha256': _sha('scripts/inspect_C32_L16_D_eager_input_prereq_v1.py') if (ROOT / 'scripts/inspect_C32_L16_D_eager_input_prereq_v1.py').is_file() else None, 'pins_module_sha256': True},
        'host_codes_sha256': {node: hashlib.sha256(code.encode()).hexdigest() for node, code in codes.items()},
        'review_path': REVIEW,
        'review_created': False,
        'actual_review_created': False,
        'execution_run_dir_created': False,
        'executed': False,
        'invoke_called': False,
        'zero_request_scope': True,
        'plans_local_only': True,
        'ssh_sessions_by_this_draft': 0,
        'no_publish': True,
        'performance_certificate_allowed': False,
        'objective_complete': False,
    }
    return codes, meta


def _py_compile(path):
    with tempfile.NamedTemporaryFile(suffix='.pyc', delete=True) as handle:
        import py_compile
        py_compile.compile(str(path), cfile=handle.name, doraise=True)


def _fixture_tree(base):
    import textwrap
    files = {
        'sglang/__init__.py': '',
        'sglang/srt/__init__.py': '',
        'sglang/srt/model_executor/__init__.py': '',
        'sglang/srt/environ.py': textwrap.dedent('''
            class EnvBool:
                def parse(self, value):
                    value = value.lower()
                    if value in ["true", "1", "yes", "y"]:
                        return True
                    if value in ["false", "0", "no", "n"]:
                        return False
                    raise ValueError(value)
            class Envs:
                SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)
        '''),
        'sglang/srt/model_executor/cuda_graph_buffer_registry.py': textwrap.dedent('''
            class EagerInputRegistry:
                def fill_from(self, batch, **kwargs):
                    return batch
                def extract_buffer(self, **kwargs):
                    return None
            def build_eager_registry(**kwargs):
                return EagerInputRegistry()
        '''),
        'sglang/srt/model_executor/runner/__init__.py': 'class BaseRunner:\n    pass\n',
        'sglang/srt/model_executor/runner/eager_runner.py': textwrap.dedent('''
            from sglang.srt.environ import envs
            def replace(value):
                return value
            class EagerRunner:
                def load_batch(self, forward_batch, pp_proxy_tensors=None, **kwargs):
                    if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
                        return replace(forward_batch)
                    registry = self._eager_registry
                    registry.fill_from(forward_batch)
                    return registry.extract_buffer()
        '''),
        'sglang/srt/model_executor/runner_utils/__init__.py': textwrap.dedent('''
            from .copy_helpers import share_input_buffer
            from .grouped_helpers import _grouped_foreach_copy_one
            from ...environ import Envs
        '''),
        'sglang/srt/model_executor/runner_utils/copy_helpers.py': 'def share_input_buffer(value):\n    return value\n',
        'sglang/srt/model_executor/runner_utils/grouped_helpers.py': 'def _grouped_foreach_copy_one(value):\n    return value\n',
    }
    for relative, text in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def local_validation(codes=None, meta=None):
    reader = _build_source_reader()
    outer = Path(__file__).read_text()
    ast.parse(outer)
    _py_compile(Path(__file__))
    ast.parse(reader)
    compile(reader, '<L16-source-reader>', 'exec')
    assert not re.search(r'\b(?:null|true|false)\b', reader)
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / 'python'
        _fixture_tree(base)
        fixture_reader = _build_source_reader(str(base))
        fixture = __import__('subprocess').run(['python3', '-S', '-c', fixture_reader], capture_output=True, text=True, timeout=60)
        assert fixture.returncode == 0, fixture.stderr[-4000:]
        fixture_result = json.loads(fixture.stdout)
        assert fixture_result['fixed_resolved'] == 4
        assert fixture_result['helper_resolved'] == 2
        assert fixture_result['total_sources'] == 6
        assert fixture_result['checks']['all_ast_parsed'] is True
        assert fixture_result['checks']['environ_assignments_found'] is True
        assert fixture_result['checks']['load_batch_found'] is True
        assert fixture_result['checks']['no_copy_branch_found'] is True
        environ = next(row for row in fixture_result['sources'] if row['name'] == 'sglang.srt.environ')
        extraction = environ['environ_extraction']
        assert extraction['default_source_qualified'] is True
        assert extraction['qualified_default']['literal'] is False
        assert extraction['envbool_semantics']['semantics_qualified'] is True
        assert 'true' in extraction['envbool_semantics']['true_values']
        relative_three = [row for row in fixture_result['imports_observed'] if row['level'] == 3]
        assert any(row['target'] == 'sglang.srt.environ' and row['resolution'] == 'relative_level_3' for row in relative_three)
        assert fixture_result['helper_selection'][0]['import_reference']['alias_basis'].startswith('share_input_buffer')
        assert any('_grouped_foreach_copy_' in row['import_reference']['alias_basis'] for row in fixture_result['helper_selection'])
        environ_path = base / 'sglang/srt/environ.py'
        environ_path.write_text(environ_path.read_text().replace('EnvBool(False)', 'EnvBool(_fixture_default())'))
        fixture_reader = _build_source_reader(str(base))
        second = __import__('subprocess').run(['python3', '-S', '-c', fixture_reader], capture_output=True, text=True, timeout=60)
        assert second.returncode == 0, second.stderr[-4000:]
        second_result = json.loads(second.stdout)
        second_environ = next(row for row in second_result['sources'] if row['name'] == 'sglang.srt.environ')
        assert second_environ['environ_extraction']['default_source_qualified'] is False
    if codes is None or meta is None:
        codes, meta = plans()
    assert len(codes) == 2
    for node, code in codes.items():
        ast.parse(code)
        compile(code, '<L16-node-' + node + '>', 'exec')
        tree = ast.parse(code)
        assignment = None
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'WORKERS' for target in item.targets):
                assignment = item
        assert assignment is not None and len(assignment.targets) == 1
        workers = ast.literal_eval(assignment.value)
        expected = meta['worker_identity']['nodes'][node]['workers']
        assert len(workers) == 16 and len({row['host_pid'] for row in workers}) == 16
        assert [row['host_pid'] for row in workers] == [row['host_pid'] for row in expected]
        assert [row['namespace_pid'] for row in workers] == [row['namespace_pid'] for row in expected]
        assert len({row['TP_rank'] for row in workers}) == 16
        assert not re.search(r'\b(?:null|true|false)\b', code)
    return {
        'outer_AST_parse': True,
        'outer_py_compile': True,
        'nested_source_reader_AST_parse': True,
        'nested_source_reader_py_compile': True,
        'source_reader_python_literals': True,
        'resolver_fixture_fixed4_helper2': True,
        'resolver_relative_level_3_fixture': True,
        'envbool_default_false_qualified': True,
        'envbool_nonliteral_default_unqualified': True,
        'eager_load_batch_no_copy_branch_fixture': True,
        'actual_codes_rendered': len(codes),
        'workers_per_node_rendered': 16,
        'actual_codes_ast_parse': True,
        'json_literal_scan': True,
        'SSH_sessions': 0,
        'HTTP_requests': 0,
        'NPU_calls': 0,
        'SDK_init_calls': 0,
        'model_requests': 0,
        'new_profiles': 0,
        'remote_writes': 0,
    }


def execute(run_id, review_path=REVIEW):
    assert run_id == RUN_ID and review_path == REVIEW
    review_abs = ROOT / review_path
    assert review_abs.is_file() and not review_abs.is_symlink()
    assert review_abs.resolve().is_relative_to(ROOT.resolve())
    review_raw = review_abs.read_bytes()
    review = json.loads(review_raw)
    codes, meta = plans()
    expected = {node: hashlib.sha256(code.encode()).hexdigest() for node, code in codes.items()}
    module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['project_id'] == 'glm53' and review['run_id'] == RUN_ID
    assert review['decision'] == 'APPROVED_READ_ONLY' and review['read_only'] is True
    assert review['module_sha256'] == module_sha
    assert review['host_codes_sha256'] == expected
    assert review.get('model_requests_authorized') == 0 and review.get('new_profile_authorized') is False
    base = _load('inventory', INVENTORY, INVENTORY_SHA)
    codec = _load('codec', CODEC, CODEC_SHA)
    run = ROOT / 'runs' / RUN_ID
    run.mkdir(exist_ok=False)
    (run / 'execution-once.lock').write_text(str(os.getpid()))
    receipt = {
        'project_id': 'glm53',
        'run_id': RUN_ID,
        'mode': 'READ_ONLY_EAGER_INPUT_SOURCE_AND_INITIAL_ENV_PREREQUISITE',
        'review': {'path': str(review_path), 'sha256': hashlib.sha256(review_raw).hexdigest()},
        'source_sha256': module_sha,
        'host_codes_sha256': expected,
        'nodes': {},
        'model_requests': 0,
        'HTTP_requests': 0,
        'NPU_calls': 0,
        'SDK_init_calls': 0,
        'new_profiles': 0,
        'max_concurrent_SSH': 1,
        'node_invocations': 0,
        'extra_SSH': 0,
        'remote_writes': 0,
        'profile_or_CSV_or_trace_reread': False,
        'initial_process_environment_only': True,
        'live_python_get_measured': False,
        'raw_wire_written_before_decode': True,
        'source_text_written_local_only': True,
        'no_publish': True,
        'performance_certificate_allowed': False,
    }
    results = {}
    try:
        for node in (36, 41):
            try:
                pod, host = base.NODES[node]
                mapping = base.invoke(['kubectl get pod -n default ' + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=30)
                (run / ('node-' + str(node) + '-mapping-wire.json')).write_text(json.dumps(mapping, indent=2) + '\n')
                if mapping['rc'] != 0 or mapping['stdout'] != '172.16.12.' + str(node):
                    receipt['nodes'][str(node)] = {'error': 'pod_mapping_mismatch', 'mapping_rc': mapping['rc'], 'mapping_stdout': mapping['stdout'][:400], 'mapping_stderr': mapping['stderr'][-2000:]}
                    continue
                code = 'import socket\nassert socket.gethostname()==' + repr(host) + '\n' + codes[str(node)]
                ast.parse(code)
                command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod, '-c', 'hostctl', '--', 'nsenter', '-t', '1', '-m', '--', 'python3', '-'])
                wire = base.invoke([command], code, timeout=240)
                (run / ('node-' + str(node) + '-raw-wire.json')).write_text(json.dumps(wire, indent=2) + '\n')
                receipt['node_invocations'] += 1
                if wire['rc'] != 0:
                    receipt['nodes'][str(node)] = {'rc': wire['rc'], 'error': 'remote_program_nonzero', 'stderr': wire['stderr'][-4000:]}
                    continue
                try:
                    decoded = codec.decode(wire)
                    data = json.loads(decoded['stdout'])
                except Exception as error:
                    receipt['nodes'][str(node)] = {'rc': wire['rc'], 'error': repr(error)}
                    continue
                (run / ('node-' + str(node) + '.json')).write_text(json.dumps(decoded, indent=2) + '\n')
                (run / ('node-' + str(node) + '-context.json')).write_text(json.dumps(data, indent=2) + '\n')
                source_dir = run / 'source' / ('node-' + str(node))
                source_dir.mkdir(parents=True, exist_ok=True)
                source_rows = []
                for item in data.get('source_resolution', {}).get('sources', []):
                    name = item.get('name')
                    if not isinstance(name, str):
                        continue
                    source_rows.append({
                        'name': name,
                        'role': item.get('role'),
                        'exists': item.get('exists'),
                        'selected_type': item.get('selected_type'),
                        'bytes': item.get('bytes'),
                        'sha256': item.get('sha256'),
                        'error': item.get('error'),
                        'local_file': None,
                    })
                    text = item.get('text')
                    if isinstance(text, str):
                        target = source_dir / (name + '.py')
                        target.write_text(text)
                        source_rows[-1]['local_file'] = str(target.relative_to(run))
                (run / 'source' / ('node-' + str(node) + '-manifest.json')).write_text(json.dumps(source_rows, indent=2) + '\n')
                results[str(node)] = data
                receipt['nodes'][str(node)] = {
                    'rc': 0,
                    'collection_pass': data.get('collection_pass'),
                    'source_checks': data.get('checks'),
                    'worker_summary': data.get('worker_summary'),
                    'parent_model': data.get('parent_model'),
                    'source': {
                        'fixed_resolved': data.get('source_resolution', {}).get('fixed_resolved'),
                        'helper_resolved': data.get('source_resolution', {}).get('helper_resolved'),
                        'total_sources': data.get('source_resolution', {}).get('total_sources'),
                        'sources': source_rows,
                        'error': data.get('source_resolution', {}).get('error'),
                    },
                    'interpretation': data.get('interpretation'),
                }
            except Exception as error:
                receipt['nodes'][str(node)] = {'error': repr(error)}
        receipt['pass'] = len(results) == 2 and all(
            receipt['nodes'][str(node)].get('rc') == 0 and receipt['nodes'][str(node)].get('collection_pass') is True
            for node in (36, 41)
        )
        receipt['meta'] = meta
    finally:
        (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    plans_codes, plans_meta = plans()
    print(json.dumps({'meta': plans_meta, 'host_codes_sha256': {node: hashlib.sha256(code.encode()).hexdigest() for node, code in plans_codes.items()}}, indent=2))
