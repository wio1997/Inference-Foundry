"""Read-only L15-D existing native timeline and fixed-source inspector.

Importing this module performs no remote action. plans() reads frozen local
evidence, renders the two node programs and performs no SSH. execute() is the
sole remote entry point, is fixed to RUN_ID, requires an approved actual review
whose module/host-code hashes match exactly, and creates its run directory with
exist_ok=False. Neither program issues HTTP, NPU, SDK, model or profile work.

This is an attribution instrument only: it reads the already captured native
trace_view.json and the actual installed source modules. It does not optimize,
deploy, change flags or revise any frozen count.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = '20261019T-L15-sglang-existing-D-timeline-source-v1'
CAPTURE_ID = '20261019T-L13-sglang-C32-D-two-step-native-v1'
DIAG_RID = '20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1'
EVIDENCE = 'results/C32-L13-D-two-step-evidence-v1.json'
EVIDENCE_SHA = '5fccbbafa1072417625192d6f7f904a6815685f507866787e09526ae9bd768bf'
DECISION = 'results/C32-L13-D-diagnostic-acceptance-decision-v1.json'
DECISION_SHA = 'b4f297c2b8bfa5ecdd0691c0fa3a2100c5091aa683627c64389093753514ce6a'
AUDITOR = 'scripts/audit_C32_L13_D_diagnostic_local_v2.py'
AUDITOR_SHA = 'e9754f931295600e24cee438cb811dd04dae82b3e7678691156233f1b8c414ed'
L14_DECISION = 'results/C32-L14-existing-trace-attribution-decision-v1.json'
L14_DECISION_SHA = '9d3a1175d1af55f6c7554cb71385387bbd2e034c278a08e1069b5f3b518fa809'
L14_RECEIPT = 'runs/20261019T-L14-sglang-existing-D-trace-context-v1/receipt.json'
L14_RECEIPT_SHA = '72dadcd2a6d79292e4cdc62cfe6d58929bedac56a4088ee72f9b549e6c776788'
L14_MODULE = 'scripts/glm53_C32_L14_D_trace_context_v2.py'
L14_MODULE_SHA = 'a357d80e6aebbeefd0d72965946e2391cf354b267fe36feba946b2bf25567865'
L14_HOST_CODES = {
    '36': '0a62039595a0f427786fa02a4267bdb8e1d94181dd4b3997b6c181c92e759a90',
    '41': 'ae00150929c683a260ed9794310a6f87479af87cf0eadfe893e4f88b3f4c34fa',
}
L14_SOURCE_DIR = 'runs/20261019T-L14-sglang-existing-D-trace-context-v1/source'
CAPACITY = 'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json'
CAPACITY_SHA = '19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50'
DIAG_CONFIG = 'config/C32-L13-D-native-decode-v1/D-two-step-diagnostic-v1.json'
DIAG_CONFIG_SHA = '035d37267b9895790d0a733a4a219d15d15262382a211064305303b4aa2b9ee2'
NODE_CAPTURES = {
    36: 'runs/20261019T-L13-sglang-C32-D-two-step-native-v1/node-36.json',
    41: 'runs/20261019T-L13-sglang-C32-D-two-step-native-v1/node-41.json',
}
NODE_CAPTURE_SHA = {
    36: '4cee2c44897634fb53b985d37d8f13d358df61445fe356954bb11303e5375b75',
    41: '43ca3b1ef2f920c97751b8377ad97d8b8fcdf542131f5092b390a8ae32df0c32',
}
INVENTORY = 'scripts/inventory_readonly.py'
INVENTORY_SHA = 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
CODEC = 'scripts/glm53_native_output_transport_v1.py'
CODEC_SHA = 'ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5'
IDENTITY = 'scripts/glm53_C32_L12C_mixed_model_identity_v1.py'
IDENTITY_SHA = 'ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98'
MODEL_RUNNER_LOCAL = (
    'runs/20261018T-C32-alignment-installed-source-v1/source/sglang/'
    'sgl-workspace__sglang__python__sglang__srt__model_executor__model_runner.py'
)
MODEL_RUNNER_SHA = 'df648164f669f9bbf823fa63e4ba7d7f3e27205eed301235364809e90baf8ce2'
REVIEW = 'results/C32-L15-D-timeline-source-actual-review-v1.json'
REPRESENTATIVE = {36: 0, 41: 16}
RANK0_TRACE_BYTES = 21607164
RANK0_TRACE_SHA = '90f81d5a5674bbf69c69cfaadbd4c8b62f5d3b3ae8633d95b2ca36b7bbff779e'

MAX_TRACE = 128 * 1024 ** 2
MAX_TRACE_EVENTS = 2000000
MAX_SOURCE = 1024 ** 2
MAX_TOTAL_SOURCE = 6 * 1024 ** 2
MAX_NODE_JSON = 8 * 1024 ** 2
MAX_SELECTED = 256
MAX_ARGS_KEYS = 128
ARG_VALUE_LIMIT = 512
MAX_EXTRA = 4
RESOLVER_ROOT = '/sgl-workspace/sglang/python'
FIXED_MODULES = [
    'sglang.srt.speculative.eagle_info',
    'sglang.srt.speculative.draft_utils',
    'sglang.srt.model_executor.runner',
    'sglang.srt.model_executor.model_runner_components.attention_backend_setup',
]
EXTRA_PREFIXES = [
    'sglang.srt.layers.attention',
    'sglang.srt.hardware_backend.npu',
    'sglang.srt.model_executor.runner',
]
ALIAS_TOKENS = ['EagerRunner', 'Ascend', 'NPU']
KEY_TOKENS = [
    'EagerRunner',
    'Ascend',
    'prepare_for_draft_extend',
    'init_forward_metadata',
    'mark_ready',
    'draft_extend',
]

SOURCE_READER_TEMPLATE = r'''import ast,hashlib,json,pathlib
ROOT=pathlib.Path(__RESOLVER_ROOT__)
FIXED=__FIXED_MODULES__
PREFIXES=__EXTRA_PREFIXES__
ALIAS_TOKENS=__ALIAS_TOKENS__
KEY_TOKENS=__KEY_TOKENS__
MAX_SOURCE=__MAX_SOURCE__
MAX_TOTAL=__MAX_TOTAL_SOURCE__
MAX_EXTRA=__MAX_EXTRA__
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
def _walk_defs(body,prefix=()):
    for node in body:
        if isinstance(node,ast.ClassDef):
            yield node,prefix+(node.name,)
            yield from _walk_defs(node.body,prefix+(node.name,))
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            yield node,prefix+(node.name,)
            yield from _walk_defs(node.body,prefix+(node.name,))
def _index(text,tree):
    classes=[];keys=[];token_lines=[]
    for node,qual in _walk_defs(tree.body):
        if isinstance(node,ast.ClassDef) and len(classes)<64:
            classes.append({'name':node.name,'qualname':'.'.join(qual),'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',None)})
        name='.'.join(qual)
        if any(token in name for token in KEY_TOKENS) and len(keys)<8:
            lines=text.splitlines();segment='\n'.join(lines[node.lineno-1:(getattr(node,'end_lineno',None) or node.lineno)])
            keys.append({'kind':type(node).__name__,'qualname':name,'lineno':node.lineno,'end_lineno':getattr(node,'end_lineno',None),'source':segment[:8192],'truncated':len(segment)>8192,'original_chars':len(segment)})
    for number,line in enumerate(text.splitlines(),1):
        if any(token in line for token in KEY_TOKENS) and len(token_lines)<96:
            token_lines.append({'lineno':number,'text':line[:512],'truncated':len(line)>512})
    return {'classes':classes,'key_definitions':keys,'key_token_lines':token_lines}
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
        row['package']=name if chosen['kind']=='package_init' else name.rsplit('.',1)[0]
        row['index']=_index(text,tree)
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
matched=[];imports_observed=[];skipped=[]
for name in FIXED:
    tree=tree_by_name.get(name)
    if tree is None:continue
    package=package_by_name[name]
    parent_sha=next((x.get('sha256') for x in sources if x['name']==name),None)
    for node in ast.walk(tree):
        if not isinstance(node,ast.ImportFrom):continue
        level=node.level or 0
        for alias in node.names:
            alias_text=(alias.name or '')+':'+(alias.asname or '')
            if level==0:target=node.module
            elif level==1:target=(package+'.'+node.module) if node.module else (package+'.'+alias.name)
            else:target=node.module
            info={'parent_source':name,'parent_sha256':parent_sha,'parent_package':package,'lineno':node.lineno,'column':node.col_offset,'level':level,'module':node.module,'aliases':[x.name for x in node.names],'alias_basis':alias_text,'target':target}
            if len(imports_observed)<256:imports_observed.append(info)
            if level>1:
                if len(skipped)<128:skipped.append(info|{'reason':'relative_level_gt_1_not_followed'})
                continue
            if not target:continue
            prefix_match=any(target==p or target.startswith(p+'.') for p in PREFIXES)
            token_match=any(t in alias_text for t in ALIAS_TOKENS)
            if prefix_match and token_match:
                if len(matched)<64:
                    if 'EagerRunner' in alias_text:priority=0
                    elif target.startswith('sglang.srt.hardware_backend.npu'):priority=1
                    else:priority=2
                    matched.append(info|{'prefix_match':True,'alias_token_match':True,'import_priority':priority,'reason':'eligible_extra_import'})
            elif len(skipped)<128:
                skipped.append(info|{'prefix_match':prefix_match,'alias_token_match':token_match,'reason':'prefix_mismatch' if not prefix_match else 'alias_token_mismatch'})
seen=set(FIXED);chosen_extra=[]
for info in sorted(matched,key=lambda x:(x['import_priority'],FIXED.index(x['parent_source']),x['lineno'],x['column'],x['target'])):
    target=info['target']
    if target in seen:continue
    seen.add(target)
    if len(chosen_extra)>=MAX_EXTRA:
        if len(skipped)<128:skipped.append(info|{'reason':'extra_module_cap_reached'})
        continue
    row,package,tree=_read(target,'extra',parent={'source':info['parent_source'],'sha256':info['parent_sha256'],'package':info['parent_package']},basis={'ast_import_from':{'level':info['level'],'module':info['module'],'aliases':info['aliases'],'lineno':info['lineno'],'column':info['column']},'reason':'explicit_ImportFrom_alias_token_under_allowed_prefix'})
    row['import_reference']=info
    chosen_extra.append(row)
    if row.get('error'):errors.append({'name':target,'error':row['error']})
sources.extend(chosen_extra)
budget=MAX_TOTAL;included=0
for row in sources:
    text=row.get('text')
    if not isinstance(text,str):continue
    size=len(text.encode('utf-8'))
    if included+size>budget:
        row['text']=None;row['text_omitted']='total_source_text_budget_exceeded';row['text_bytes']=size
    else:
        included+=size;row['text_bytes']=size
out={'project_id':'glm53','resolver_root':str(ROOT),'fixed_modules':FIXED,'extra_prefixes':PREFIXES,'alias_tokens':ALIAS_TOKENS,'key_tokens':KEY_TOKENS,'sources':sources,'imports_observed':imports_observed,'skipped_imports':skipped,'errors':errors,'fixed_resolved':sum(1 for x in sources if x['role']=='fixed' and x.get('exists') is True),'extra_resolved':sum(1 for x in sources if x['role']=='extra' and x.get('exists') is True),'total_sources':len(sources),'total_source_bytes':sum(x.get('bytes',0) for x in sources if isinstance(x.get('bytes'),int)),'included_text_bytes':included,'limits':{'max_source_bytes':MAX_SOURCE,'max_total_source_text_bytes':MAX_TOTAL,'max_extra_modules':MAX_EXTRA,'follow_depth':1,'no_recursive_traversal':True,'framework_imported':False,'torch_imported':False,'SDK_imported':False,'path_boundary':str(ROOT),'symlink_allowed':False}}
print(json.dumps(out))
'''


def _build_source_reader():
    reader = SOURCE_READER_TEMPLATE
    for placeholder, value in (
        ('__RESOLVER_ROOT__', repr(RESOLVER_ROOT)),
        ('__FIXED_MODULES__', repr(FIXED_MODULES)),
        ('__EXTRA_PREFIXES__', repr(EXTRA_PREFIXES)),
        ('__ALIAS_TOKENS__', repr(ALIAS_TOKENS)),
        ('__KEY_TOKENS__', repr(KEY_TOKENS)),
        ('__MAX_SOURCE__', repr(MAX_SOURCE)),
        ('__MAX_TOTAL_SOURCE__', repr(MAX_TOTAL_SOURCE)),
        ('__MAX_EXTRA__', repr(MAX_EXTRA)),
    ):
        assert reader.count(placeholder) == 1, placeholder
        reader = reader.replace(placeholder, value)
    assert not re.search(r'__[A-Z_]+__', reader), 'unrendered source-reader placeholder'
    assert not re.search(r'\b(?:null|true|false)\b', reader), 'JSON literal in Python source readers'
    ast.parse(reader)
    return reader


SOURCE_READER = _build_source_reader()

TEMPLATE = r'''
import base64,collections,gzip,hashlib,json,pathlib,re,subprocess,time,traceback
out={'project_id':'glm53','run_id':L15_RUN_ID,'mode':'READ_ONLY_EXISTING_NATIVE_TIMELINE_AND_FIXED_SOURCE','capture_id':L15_CAPTURE_ID,'diagnostic_run_id':L15_DIAG_RID,'node':L15_NODE,'representative_TP_rank':L15_RANK,'evidence_sha256':L15_EVIDENCE_SHA,'read_only':True,'model_requests':0,'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'new_profiles':0,'remote_writes':0,'raw_trace_transferred_to_Mac':False,'raw_trace_remains_remote':True}
started=time.monotonic()
MAIN_FIELDS=('name','cat','ph','pid','tid','ts','dur','id','device')
def _number(value):return isinstance(value,(int,float)) and not isinstance(value,bool)
def _hashable(value):return value if isinstance(value,(str,int,float,bool)) or value is None else '<'+type(value).__name__+'>'
def _guarded(path,base):
    assert path.is_absolute() and not path.is_symlink()
    resolved=path.resolve(strict=True)
    assert resolved.is_relative_to(base.resolve(strict=True))
    cursor=path
    while cursor!=base:
        assert not cursor.is_symlink(),('symlink path component',str(cursor))
        cursor=cursor.parent
    return resolved
def _family(name):
    if name in ('draft','draft_extend'):return 'step_span'
    if name.startswith('step[VERIFY ') or name.startswith('step[DRAFT '):return 'step_span'
    if name in ('acl_memcpy_host_to_device','aten::copy_'):return 'host_copy'
    if name in ('aclrtMemcpy','aclrtMemcpyAsync'):return 'memcpy_api'
    if 'Synchronize' in name:return 'sync_api'
    if 'Nonzero' in name:return 'nonzero'
    if name=='aclmdlRIExecuteAsync':return 'graph_api'
    if 'MoeDistributeDispatch' in name or 'MoeDistributeCombine' in name:return 'mc2'
    return None
def _bucket(name,family):
    if family=='host_copy':return 'copy:'+name
    if family=='memcpy_api':return 'api:'+name
    if family=='mc2':return 'mc2:'+('dispatch' if 'MoeDistributeDispatch' in name else 'combine')
    return family
def _project(event,keep_device=False):
    row={key:event.get(key) for key in MAIN_FIELDS if key in event}
    args=event.get('args');keys=[];scalars={};omitted={}
    if isinstance(args,dict):
        for key in list(args)[:L15_MAX_ARGS_KEYS]:
            keys.append(key)
            lowered=str(key).lower()
            if 'tensor' in lowered:
                omitted[str(key)]='tensor_like_value_not_transferred';continue
            value=args[key]
            if isinstance(value,(str,int,float,bool)) or value is None:
                if keep_device and lowered in ('device','device_id'):
                    text=value if isinstance(value,str) else repr(value)
                    scalars[str(key)]={'value':text[:L15_ARG_VALUE_LIMIT],'truncated':len(text)>L15_ARG_VALUE_LIMIT,'original_chars':len(text)}
                elif re.search(r'(^id$|_id$|id$|stream|correlation)',str(key),re.I):
                    text=value if isinstance(value,str) else repr(value)
                    scalars[str(key)]={'value':text[:L15_ARG_VALUE_LIMIT],'truncated':len(text)>L15_ARG_VALUE_LIMIT,'original_chars':len(text)}
            else:omitted[str(key)]=type(value).__name__
        if len(args)>L15_MAX_ARGS_KEYS:omitted['__args_keys_omitted__']=len(args)-L15_MAX_ARGS_KEYS
    row['args_keys']=keys;row['args_selected_scalars']=scalars;row['args_omitted']=omitted
    return row
try:
    profile_base=pathlib.Path('/data/wio/inference-foundry/glm53')
    profile=shared_root/'profile'/L15_DIAG_RID
    _guarded(profile,profile_base)
    folder=profile/L15_FOLDER
    _guarded(folder,profile)
    output=folder/'ASCEND_PROFILER_OUTPUT'
    _guarded(output,folder)
    trace_path=_guarded(output/'trace_view.json',profile)
    assert trace_path.is_file()
    before=trace_path.stat();assert before.st_size==L15_TRACE_EXPECTED['bytes']<=L15_MAX_TRACE
    assert before.st_mtime_ns==L15_TRACE_EXPECTED['mtime_ns'],'trace mtime differs from frozen node index'
    raw=trace_path.read_bytes();assert len(raw)==L15_TRACE_EXPECTED['bytes']
    digest=hashlib.sha256(raw).hexdigest();assert digest==L15_TRACE_EXPECTED['sha256']
    data=json.loads(raw)
    if isinstance(data,list):
        root_kind='list';top_keys=None;display=None;other_top=None;events=data
    elif isinstance(data,dict):
        root_kind='dict';top_keys=list(data)
        assert 'traceEvents' in data and isinstance(data['traceEvents'],list),'traceEvents missing or not a list'
        events=data['traceEvents'];display=data.get('displayTimeUnit');other_top={}
        for key,value in data.items():
            if key=='traceEvents':continue
            if isinstance(value,str):
                other_top[key]={'type':'str','value':value[:512],'truncated':len(value)>512,'original_chars':len(value)}
            elif isinstance(value,(int,float,bool)) or value is None:other_top[key]={'type':type(value).__name__,'value':value}
            else:other_top[key]={'type':type(value).__name__,'length':(len(value) if hasattr(value,'__len__') else None)}
    else:raise RuntimeError('unsupported_top_level_shape')
    event_count=len(events);assert event_count<=L15_MAX_EVENTS
    ph_counts=collections.Counter();cat_counts=collections.Counter();phcat_counts=collections.Counter()
    meta=[];meta_total=0;all_counts=collections.Counter();all_distinct_overflow=0
    watch_counts=collections.Counter();watch_overflow=0
    step_spans=[];step_total=0;pools=collections.defaultdict(list)
    malformed=[];missing={key:0 for key in ('name','cat','pid','tid','ts','dur')};x_total=0
    for index,event in enumerate(events):
        if not isinstance(event,dict):
            if len(malformed)<8:malformed.append({'index':index,'type':type(event).__name__})
            continue
        phv=event.get('ph');ph_key=phv if isinstance(phv,str) else ('null' if phv is None else '<'+type(phv).__name__+'>')
        catv=event.get('cat');cat_key=catv if isinstance(catv,str) else ('null' if catv is None else '<'+type(catv).__name__+'>')
        ph_counts[ph_key]+=1;cat_counts[cat_key]+=1;phcat_counts[(ph_key,cat_key)]+=1
        if phv=='M':
            meta_total+=1
            if event.get('name') in ('process_name','thread_name') and len(meta)<256:
                args=event.get('args') if isinstance(event.get('args'),dict) else {}
                meta.append({'name':event.get('name'),'pid':event.get('pid'),'tid':event.get('tid'),'args_name':args.get('name')})
            continue
        if phv!='X':continue
        x_total+=1;name=event.get('name')
        if not isinstance(name,str):
            missing['name']+=1;continue
        for key in ('cat','pid','tid','ts','dur'):
            if event.get(key) is None:missing[key]+=1
        pid=event.get('pid');tid=event.get('tid');cat=catv if isinstance(catv,str) else None
        all_key=(name,_hashable(pid),_hashable(tid),cat)
        if len(all_counts)<4096 or all_key in all_counts:all_counts[all_key]+=1
        else:all_distinct_overflow+=1
        family=_family(name)
        if family is None:continue
        if len(watch_counts)<2048 or all_key in watch_counts:watch_counts[all_key]+=1
        else:watch_overflow+=1
        if family=='step_span':
            step_total+=1
            if len(step_spans)<4096:step_spans.append(event)
        pools[_bucket(name,family)].append((event,index))
    selected=[]
    for event in step_spans[:64]:selected.append(('step_span',event))
    caps=[('copy:acl_memcpy_host_to_device',12),('copy:aten::copy_',12),('api:aclrtMemcpy',12),('api:aclrtMemcpyAsync',12),('sync_api',12),('nonzero',8),('graph_api',8),('mc2:dispatch',24),('mc2:combine',24)]
    for bucket,cap in caps:
        items=sorted(pools.get(bucket,[]),key=lambda item:(-(item[0].get('dur') if _number(item[0].get('dur')) else -1),item[1]))
        for event,_ in items[:cap]:selected.append((bucket,event))
    assert len(selected)<=L15_MAX_SELECTED,'selected events exceed cap'
    def _enclosing(event):
        ts=event.get('ts');dur=event.get('dur')
        if not (_number(ts) and _number(dur)):return None,None,'missing_numeric_ts_or_dur'
        end=ts+dur;found=[]
        for span in step_spans:
            if span.get('pid')!=event.get('pid') or span.get('tid')!=event.get('tid'):continue
            st=span.get('ts');sd=span.get('dur')
            if not (_number(st) and _number(sd)):continue
            if st<=ts and st+sd>=end:found.append(span)
        if not found:return None,None,'no_same_pid_tid_step_or_draft_span_contains_event_interval'
        return [_project(item) for item in found[:16]],len(found)>16,None
    rows=[]
    for role,event in selected:
        item={'selection_role':role,'event':_project(event,keep_device=role.startswith('mc2:'))}
        if role.startswith('copy:'):
            enclosing,truncated,reason=_enclosing(event)
            item['enclosing_step_spans']=enclosing;item['enclosing_truncated']=truncated;item['enclosing_null_reason']=reason
            item['enclosing_scope']='Same trace file, same pid and same tid only; interval containment in raw ts/dur units; no cross-pid/tid/flow join.'
        rows.append(item)
    watch_rows=[{'name':k[0],'pid':k[1],'tid':k[2],'cat':k[3],'count':v} for k,v in sorted(watch_counts.items(),key=lambda kv:(-kv[1],str(kv[0][0]),str(kv[0][1]),str(kv[0][2]),str(kv[0][3])))[:2048]]
    all_rows=[{'name':k[0],'pid':k[1],'tid':k[2],'cat':k[3],'count':v} for k,v in sorted(all_counts.items(),key=lambda kv:(-kv[1],str(kv[0][0]),str(kv[0][1]),str(kv[0][2]),str(kv[0][3])))[:4096]]
    phcat_rows=[{'ph':k[0],'cat':k[1],'count':v} for k,v in sorted(phcat_counts.items(),key=lambda kv:(str(kv[0][0]),str(kv[0][1])))]
    after=trace_path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'trace changed during read'
    out['timeline']={'relative':L15_FOLDER+'/ASCEND_PROFILER_OUTPUT/trace_view.json','file':{'expected':L15_TRACE_EXPECTED,'bytes':len(raw),'sha256':digest,'mtime_ns_before':before.st_mtime_ns,'mtime_ns_after_read':after.st_mtime_ns},'root_shape':root_kind,'top_level_keys':top_keys,'displayTimeUnit':display,'top_level_other':other_top,'event_count':event_count,'x_event_count':x_total,'ph_counts':dict(ph_counts),'cat_counts':dict(cat_counts),'ph_cat_counts':phcat_rows,'process_thread_name_events_total':meta_total,'process_thread_names':meta,'all_x_name_counts_by_pid_tid_cat':all_rows,'all_x_distinct_key_overflow':all_distinct_overflow,'watch_name_counts_by_pid_tid_cat':watch_rows,'watch_distinct_key_overflow':watch_overflow,'selected_events':rows,'missing_field_counts':missing,'malformed_examples':malformed,'counts':{'step_spans_observed':step_total,'step_spans_used_for_enclosing':len(step_spans),'selected_total':len(rows)},'clock_note':'ts and dur are preserved as raw values; no unit conversion, clock-base guess or cross-file join is performed.','selection_note':'Step spans are the first 64 in file order. Host copy, API, sync, Nonzero, Graph and MC2 families are the longest durations per named bucket. Selected rows are a bounded context sample, not total coverage.'}
    reader=subprocess.run(['docker','exec',L15_CID,'python3','-S','-c',L15_SOURCE_READER],capture_output=True,text=True,timeout=120)
    if reader.returncode!=0:
        out['source_resolution']={'error':{'rc':reader.returncode,'stderr':reader.stderr[-4000:]},'sources':[]}
    else:out['source_resolution']=json.loads(reader.stdout)
    source=out['source_resolution'];sources=source.get('sources',[])
    out['source_read']={'method':'docker exec <CID> python3 -S -c <reader>','imports':['ast','hashlib','json','pathlib'],'framework_imported':False,'torch_imported':False,'SDK_imported':False,'resolver_root':L15_RESOLVER_ROOT,'fixed_modules':L15_FIXED_MODULES,'extra_prefixes':L15_EXTRA_PREFIXES,'alias_tokens':L15_ALIAS_TOKENS,'max_extra_modules':L15_MAX_EXTRA,'max_source_bytes':L15_MAX_SOURCE,'max_total_source_text_bytes':L15_MAX_TOTAL_SOURCE,'source_reader_sha256':hashlib.sha256(L15_SOURCE_READER.encode()).hexdigest()}
    out['provenance']={'capture_id':L15_CAPTURE_ID,'diagnostic_run_id':L15_DIAG_RID,'evidence_sha256':L15_EVIDENCE_SHA,'trace_index_origin':'runs/20261019T-L13-sglang-C32-D-two-step-native-v1 native_profile.files','raw_trace_remains_remote':True,'model_runner_source_local_only':True}
    out['checks']={'trace_bytes_sha_mtime_match':len(raw)==L15_TRACE_EXPECTED['bytes'] and digest==L15_TRACE_EXPECTED['sha256'] and before.st_mtime_ns==L15_TRACE_EXPECTED['mtime_ns'],'trace_mtime_stable':before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size,'trace_shape_supported':root_kind in ('list','dict'),'event_count_within_limit':event_count<=L15_MAX_EVENTS,'selected_count_within_limit':len(rows)<=L15_MAX_SELECTED,'source_reader_rc_zero':reader.returncode==0,'fixed_sources_resolved':source.get('fixed_resolved')==len(L15_FIXED_MODULES),'total_sources_within_cap':source.get('total_sources',len(sources))<=len(L15_FIXED_MODULES)+L15_MAX_EXTRA,'sources_ast_parsed':all(x.get('ast_parse_ok') is True for x in sources),'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'model_requests':0,'new_profiles':0,'remote_writes':0,'overlap_addition':False,'cross_rank_addition':False,'critical_path_claim':False,'GPU_utilization_claim':False,'header_missing_treated_as_resample_reason':False}
    out['collection_pass']=all([out['checks']['trace_bytes_sha_mtime_match'],out['checks']['trace_mtime_stable'],out['checks']['trace_shape_supported'],out['checks']['event_count_within_limit'],out['checks']['selected_count_within_limit'],out['checks']['source_reader_rc_zero'],out['checks']['fixed_sources_resolved'],out['checks']['total_sources_within_cap'],out['checks']['sources_ast_parsed']])
except BaseException as error:
    out['error']=repr(error);out['traceback']=traceback.format_exc();out['collection_pass']=False
out['duration_seconds']=time.monotonic()-started
raw=json.dumps(out).encode();assert len(raw)<=L15_MAX_NODE_JSON,'node context exceeds 8 MiB'
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
    return code


def _node_payload(node_document):
    if isinstance(node_document.get('native_profile'), dict):
        return node_document
    stdout = node_document.get('stdout')
    assert isinstance(stdout, str), 'node capture has no decoded stdout payload'
    payload = json.loads(stdout)
    assert isinstance(payload, dict)
    return payload


def _trace_entry(node_payload, folder, node):
    profile = node_payload.get('native_profile')
    assert isinstance(profile, dict) and profile.get('exists') is True, f'node {node} native profile missing'
    assert profile.get('raw_traces_transferred_to_Mac') is False, f'node {node} reports a transferred raw trace'
    relative = folder + '/ASCEND_PROFILER_OUTPUT/trace_view.json'
    matches = [item for item in profile.get('files', []) if item.get('relative') == relative]
    assert len(matches) == 1, f'node {node} trace index entry count {len(matches)}'
    entry = matches[0]
    assert type(entry.get('bytes')) is int and 0 < entry['bytes'] <= MAX_TRACE
    assert re.fullmatch('[0-9a-f]{64}', entry.get('sha256') or '')
    assert type(entry.get('mtime_ns')) is int
    if node == 36:
        assert entry['bytes'] == RANK0_TRACE_BYTES and entry['sha256'] == RANK0_TRACE_SHA, 'rank 0 trace index differs from frozen fact'
    return {'relative': entry['relative'], 'bytes': entry['bytes'], 'sha256': entry['sha256'], 'mtime_ns': entry['mtime_ns']}


def _inputs():
    evidence = _read(EVIDENCE, EVIDENCE_SHA)
    decision = _read(DECISION, DECISION_SHA)
    assert decision['project_id'] == 'glm53' and decision['decision'] == 'ACCEPTED'
    assert decision['evidence_sha256'] == EVIDENCE_SHA and decision['auditor_sha256'] == AUDITOR_SHA
    assert _sha(AUDITOR) == AUDITOR_SHA
    assert evidence['project_id'] == 'glm53' and evidence['capture_id'] == CAPTURE_ID
    assert evidence['diagnostic_evidence_pass'] is True and evidence['diagnostic_only'] is True
    assert evidence['performance_certificate_allowed'] is False
    assert evidence['all32_D_native_profile_workers'] is True
    assert evidence['requested_profile_forward_steps'] == 2
    assert evidence['actual_P_MemCache_read_workers'] == 32 and evidence['actual_D_SDMA_receiving_devices'] == 32
    assert evidence['physical_wire_bytes'] is None

    l14_decision = _read(L14_DECISION, L14_DECISION_SHA)
    assert l14_decision['project_id'] == 'glm53' and l14_decision['decision'] == 'PIVOTED'
    assert l14_decision['collection_pass'] is False
    assert l14_decision['collection_receipt_sha256'] == L14_RECEIPT_SHA
    l14_receipt = _read(L14_RECEIPT, L14_RECEIPT_SHA)
    assert l14_receipt['project_id'] == 'glm53' and l14_receipt['run_id'] == l14_decision['run_id']
    assert l14_receipt['pass'] is False
    assert l14_receipt['model_requests'] == 0 and l14_receipt['HTTP_requests'] == 0
    assert l14_receipt['NPU_calls'] == 0 and l14_receipt['SDK_init_calls'] == 0
    assert l14_receipt['raw_traces_remain_remote'] is True
    assert l14_receipt['source_sha256'] == L14_MODULE_SHA and _sha(L14_MODULE) == L14_MODULE_SHA
    assert l14_receipt['host_codes_sha256'] == L14_HOST_CODES
    preserved_sources = {}
    for node in (36, 41):
        preserved = {}
        for item in l14_receipt['nodes'][str(node)]['source_files']:
            if item.get('exists') is not True:
                continue
            path = ROOT / L14_SOURCE_DIR / ('node-' + str(node)) / item['name']
            raw = path.read_bytes()
            assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], path
            preserved[item['name']] = {'bytes': item['bytes'], 'sha256': item['sha256']}
        assert len(preserved) == 5, f'node {node} L14 preserved source count {len(preserved)}'
        preserved_sources[str(node)] = preserved
    assert preserved_sources['36'] == preserved_sources['41'], 'L14 five-source copies disagree across nodes'

    capacity = _read(CAPACITY, CAPACITY_SHA)
    assert capacity['pass'] is True and capacity['selected_variant'] == 'candidate'
    diag = _read(DIAG_CONFIG, DIAG_CONFIG_SHA)
    assert diag['run_id'] == DIAG_RID and diag['diagnostic_only'] is True
    assert diag['performance_certificate_allowed'] is False
    assert capacity['meta_container_id'] == diag['meta_container_id']
    for node in (36, 41):
        provider = capacity['providers'][str(node)]
        configured = diag['runtime_containers'][str(node)]
        assert (provider['container_id'], provider['name'], provider['remote_root'], provider['run_id']) == (
            configured['container_id'], configured['name'], configured['remote_root'], configured['run_id']
        )
    assert _sha(MODEL_RUNNER_LOCAL) == MODEL_RUNNER_SHA

    node_caps = {}
    trace_entries = {}
    for node, relative in NODE_CAPTURES.items():
        raw = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == NODE_CAPTURE_SHA[node]
        assert evidence['source_evidence_sha256'][relative] == digest
        document = json.loads(raw)
        assert document['rc'] == 0
        frame = document['native_output_transport']
        assert frame['version'] == 1 and 0 < frame['raw_bytes'] <= 256 * 1024 ** 2
        assert frame['packed_bytes'] <= 8 * 1024 ** 2
        node_caps[str(node)] = {
            'path': relative,
            'sha256': digest,
            'bytes': len(raw),
            'native_output_raw_sha256': frame['raw_sha256'],
            'native_output_raw_bytes': frame['raw_bytes'],
            'native_output_packed_sha256': frame['packed_sha256'],
        }
        trace_entries[str(node)] = None

    workers = evidence['workers']
    assert len(workers) == 32 and {worker['TP_rank'] for worker in workers} == set(range(32))
    assert len({worker['folder'] for worker in workers}) == 32
    assert evidence['profile_completion_by_node']['36']['worker_folders'] == 16
    assert evidence['profile_completion_by_node']['41']['worker_folders'] == 16
    representatives = {}
    for node, rank in REPRESENTATIVE.items():
        matches = [worker for worker in workers if worker['node'] == node and worker['TP_rank'] == rank]
        assert len(matches) == 1, f'node {node} rank {rank} worker'
        worker = matches[0]
        assert worker['folder'] == Path(worker['folder']).name
        document = json.loads((ROOT / NODE_CAPTURES[node]).read_bytes())
        trace = _trace_entry(_node_payload(document), worker['folder'], node)
        trace_entries[str(node)] = trace
        csv_index = {}
        for name in ('operator_details.csv', 'kernel_details.csv', 'task_time.csv'):
            item = worker['source_files']['ASCEND_PROFILER_OUTPUT/' + name]
            csv_index[name] = {'bytes': item['bytes'], 'sha256': item['sha256']}
        representatives[str(node)] = {
            'node': node,
            'TP_rank': rank,
            'folder': worker['folder'],
            'namespace_pid': worker['namespace_pid'],
            'native_device_id': worker['native_device_id'],
            'csv_index_link_only': csv_index,
            'csv_files_reread_by_this_loop': False,
            'trace_file': trace,
            'remote_profile_root': capacity['providers'][str(node)]['remote_root'] + '/profile/' + DIAG_RID + '/' + worker['folder'],
            'remote_trace_path': capacity['providers'][str(node)]['remote_root'] + '/profile/' + DIAG_RID + '/' + worker['folder'] + '/ASCEND_PROFILER_OUTPUT/trace_view.json',
            'selection': 'first worker of the D node: TP rank 0 on 36 and TP rank 16 on 41; no other rank trace is read',
        }
    assert trace_entries['36'] is not None and trace_entries['41'] is not None
    assert trace_entries['36']['sha256'] == RANK0_TRACE_SHA and trace_entries['36']['bytes'] == RANK0_TRACE_BYTES
    base = _load('inventory', INVENTORY, INVENTORY_SHA)
    codec = _load('codec', CODEC, CODEC_SHA)
    identity = _load('identity', IDENTITY, IDENTITY_SHA)
    capacity_subset = {
        'providers': {str(node): capacity['providers'][str(node)] for node in (36, 41)},
        'selected_variant': capacity['selected_variant'],
        'meta_container_id': capacity['meta_container_id'],
    }
    return {
        'evidence': evidence,
        'decision': decision,
        'l14_decision': l14_decision,
        'l14_receipt': l14_receipt,
        'l14_preserved_sources': preserved_sources,
        'capacity': capacity,
        'capacity_subset': capacity_subset,
        'diag': diag,
        'node_caps': node_caps,
        'representatives': representatives,
        'trace_entries': trace_entries,
        'base': base,
        'codec': codec,
        'identity': identity,
        'core_source_sha256': _sha(MODEL_RUNNER_LOCAL),
    }


def plans():
    inputs = _inputs()
    codes = {}
    for node in (36, 41):
        representative = inputs['representatives'][str(node)]
        values = {
            'L15_RUN_ID': RUN_ID,
            'L15_CAPTURE_ID': CAPTURE_ID,
            'L15_DIAG_RID': DIAG_RID,
            'L15_EVIDENCE_SHA': EVIDENCE_SHA,
            'L15_NODE': node,
            'L15_RANK': REPRESENTATIVE[node],
            'L15_CID': inputs['capacity_subset']['providers'][str(node)]['container_id'],
            'L15_FOLDER': representative['folder'],
            'L15_TRACE_EXPECTED': {'bytes': representative['trace_file']['bytes'], 'sha256': representative['trace_file']['sha256'], 'mtime_ns': representative['trace_file']['mtime_ns']},
            'L15_MAX_TRACE': MAX_TRACE,
            'L15_MAX_EVENTS': MAX_TRACE_EVENTS,
            'L15_MAX_SOURCE': MAX_SOURCE,
            'L15_MAX_TOTAL_SOURCE': MAX_TOTAL_SOURCE,
            'L15_MAX_NODE_JSON': MAX_NODE_JSON,
            'L15_MAX_SELECTED': MAX_SELECTED,
            'L15_MAX_ARGS_KEYS': MAX_ARGS_KEYS,
            'L15_ARG_VALUE_LIMIT': ARG_VALUE_LIMIT,
            'L15_SOURCE_READER': SOURCE_READER,
            'L15_RESOLVER_ROOT': RESOLVER_ROOT,
            'L15_FIXED_MODULES': FIXED_MODULES,
            'L15_EXTRA_PREFIXES': EXTRA_PREFIXES,
            'L15_ALIAS_TOKENS': ALIAS_TOKENS,
            'L15_MAX_EXTRA': MAX_EXTRA,
        }
        code = inputs['identity'].render(node, inputs['capacity_subset']) + '\n' + _render(TEMPLATE, values)
        ast.parse(code)
        codes[str(node)] = code
    meta = {
        'project_id': 'glm53',
        'run_id': RUN_ID,
        'mode': 'READ_ONLY_EXISTING_NATIVE_TIMELINE_AND_FIXED_SOURCE',
        'draft_status': 'DRAFT_PENDING_PRIMARY_REVIEW',
        'capture_id': CAPTURE_ID,
        'diagnostic_run_id': DIAG_RID,
        'purpose': 'Resolve the actual imported speculative/runner/attention-setup modules and read bounded event context from the already captured native trace; no new profile and no runtime change.',
        'evidence': {'path': EVIDENCE, 'sha256': EVIDENCE_SHA},
        'decision': {'path': DECISION, 'sha256': DECISION_SHA},
        'auditor': {'path': AUDITOR, 'sha256': AUDITOR_SHA},
        'l14_pivot_decision': {'path': L14_DECISION, 'sha256': L14_DECISION_SHA},
        'l14_collection_receipt': {'path': L14_RECEIPT, 'sha256': L14_RECEIPT_SHA, 'collection_pass': False},
        'capacity': {'path': CAPACITY, 'sha256': CAPACITY_SHA},
        'diagnostic_config': {'path': DIAG_CONFIG, 'sha256': DIAG_CONFIG_SHA},
        'node_captures': inputs['node_caps'],
        'representatives': [inputs['representatives']['36'], inputs['representatives']['41']],
        'trace_files': {
            '36': inputs['trace_entries']['36'],
            '41': inputs['trace_entries']['41'],
            'selection': 'indexed trace_view.json of the same representative worker folders; raw trace stays remote and is not copied to Mac',
            'rank0_expected_sha256_reused': RANK0_TRACE_SHA,
            'rank16_expected_from_index_not_guessed': True,
        },
        'source_resolution': {
            'resolver_root': RESOLVER_ROOT,
            'fixed_modules': FIXED_MODULES,
            'candidate_rule': 'package/__init__.py first, then module.py; both candidates recorded; no broad find',
            'extra_import_rule': 'explicit AST ImportFrom under sglang.srt.layers.attention / sglang.srt.hardware_backend.npu / sglang.srt.model_executor.runner whose imported alias contains EagerRunner, Ascend or NPU',
            'relative_import_rule': 'level 1 resolved from the selected source package; level > 1 not followed',
            'max_extra_modules': MAX_EXTRA,
            'max_total_sources': len(FIXED_MODULES) + MAX_EXTRA,
            'max_source_bytes': MAX_SOURCE,
            'max_total_source_text_bytes': MAX_TOTAL_SOURCE,
            'no_framework_or_torch_import': True,
            'model_runner_local_reuse': {'path': MODEL_RUNNER_LOCAL, 'sha256': MODEL_RUNNER_SHA, 'remote_fetch': False},
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
            'P_or_Meta_containers_contacted': False,
            'other_ranks_traces_read': False,
            'raw_traces_remain_remote': True,
            'full_raw_trace_copied_to_Mac': False,
            'source_text_local_only': True,
        },
        'limits': {
            'trace_file_bytes': MAX_TRACE,
            'trace_events': MAX_TRACE_EVENTS,
            'selected_events_per_node': MAX_SELECTED,
            'node_json_bytes': MAX_NODE_JSON,
            'process_thread_metadata_rows': 256,
            'all_x_count_rows': 4096,
            'watch_count_rows': 2048,
            'args_value_chars': ARG_VALUE_LIMIT,
            'cross_rank_addition': False,
            'overlap_addition': False,
            'critical_path_claim': False,
            'GPU_utilization_claim': False,
            'cross_file_join': None,
        },
        'source_pins': {
            'inventory': INVENTORY_SHA,
            'codec': CODEC_SHA,
            'identity': IDENTITY_SHA,
            'model_runner_local': MODEL_RUNNER_SHA,
        },
        'transport': {
            'inventory_source_sha256': INVENTORY_SHA,
            'codec_source_sha256': CODEC_SHA,
            'decoder': 'frozen codec.decode',
            'encoder': 'read-only frame-compatible variant; archive=null; no remote write',
            'raw_wire_written_before_decode': True,
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
        'supporting_question': 'Whether prepare_for_draft_extend pre-pad attention metadata initialization (already collected eagle_worker_common.py lines 106..210) and the actual EagerRunner post-pad initialization duplicate work, and whether the first side effect is necessary. Source/timeline attribution only; no optimization is authorized.',
        'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'host_codes_sha256': {node: hashlib.sha256(code.encode()).hexdigest() for node, code in codes.items()},
        'review_path': REVIEW,
        'review_created': False,
        'executed': False,
        'invoke_called': False,
        'no_publish': True,
    }
    return codes, meta


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
    assert review['evidence_sha256'] == EVIDENCE_SHA and review['module_sha256'] == module_sha
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
        'mode': 'READ_ONLY_EXISTING_NATIVE_TIMELINE_AND_FIXED_SOURCE',
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
        'raw_traces_remain_remote': True,
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
                timeline = data.get('timeline', {})
                source = data.get('source_resolution', {})
                receipt['nodes'][str(node)] = {
                    'rc': 0,
                    'collection_pass': data.get('collection_pass'),
                    'timeline': {
                        'relative': timeline.get('relative'),
                        'file': timeline.get('file'),
                        'root_shape': timeline.get('root_shape'),
                        'top_level_keys': timeline.get('top_level_keys'),
                        'event_count': timeline.get('event_count'),
                        'selected_total': timeline.get('counts', {}).get('selected_total'),
                    },
                    'source': {
                        'fixed_resolved': source.get('fixed_resolved'),
                        'extra_resolved': source.get('extra_resolved'),
                        'total_sources': source.get('total_sources'),
                        'sources': source_rows,
                        'error': source.get('error'),
                    },
                    'checks': data.get('checks'),
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
