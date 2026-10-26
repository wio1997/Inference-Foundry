"""Read-only L14 context collector for the accepted L13 D decode trace.

Importing this module performs no remote action. plans() only reads frozen local
evidence and renders the two node programs. execute() is the sole remote entry
point, is fixed to RUN_ID, and requires an approved actual review whose
host_codes_sha256 matches the rendered programs exactly. No HTTP, NPU, SDK or
model request is issued by either program.
"""
import ast,hashlib,importlib.util,json,os,re,shlex
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='20261019T-L14-sglang-existing-D-trace-context-v1'
CAPTURE_ID='20261019T-L13-sglang-C32-D-two-step-native-v1'
DIAG_RID='20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1'
EVIDENCE='results/C32-L13-D-two-step-evidence-v1.json'
EVIDENCE_SHA='5fccbbafa1072417625192d6f7f904a6815685f507866787e09526ae9bd768bf'
DECISION='results/C32-L13-D-diagnostic-acceptance-decision-v1.json'
DECISION_SHA='b4f297c2b8bfa5ecdd0691c0fa3a2100c5091aa683627c64389093753514ce6a'
AUDITOR='scripts/audit_C32_L13_D_diagnostic_local_v2.py'
AUDITOR_SHA='e9754f931295600e24cee438cb811dd04dae82b3e7678691156233f1b8c414ed'
CAPACITY='results/sglang-C32-L12C-selected-P-pool-capacity-v1.json'
CAPACITY_SHA='19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50'
DIAG_CONFIG='config/C32-L13-D-native-decode-v1/D-two-step-diagnostic-v1.json'
DIAG_CONFIG_SHA='035d37267b9895790d0a733a4a219d15d15262382a211064305303b4aa2b9ee2'
NODE_CAPTURES={36:'runs/20261019T-L13-sglang-C32-D-two-step-native-v1/node-36.json',41:'runs/20261019T-L13-sglang-C32-D-two-step-native-v1/node-41.json'}
NODE_CAPTURE_SHA={36:'4cee2c44897634fb53b985d37d8f13d358df61445fe356954bb11303e5375b75',41:'43ca3b1ef2f920c97751b8377ad97d8b8fcdf542131f5092b390a8ae32df0c32'}
INVENTORY='scripts/inventory_readonly.py'
INVENTORY_SHA='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
CODEC='scripts/glm53_native_output_transport_v1.py'
CODEC_SHA='ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5'
IDENTITY='scripts/glm53_C32_L12C_mixed_model_identity_v1.py'
IDENTITY_SHA='ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98'
CORE='scripts/glm53_profile_csv_analysis_v1.py'
CORE_SHA='c8e661f21c6a7413cd5e8ac34e490d7e457151c40811d566be4652ce2278a3a3'
REVIEW='results/C32-L14-D-trace-context-actual-review-v1.json'
CSV_NAMES=['operator_details.csv','kernel_details.csv','task_time.csv']
REPRESENTATIVE={36:0,41:16}
MAX_CSV=256*1024**2
MAX_SOURCE=1024**2
MAX_NODE_JSON=4*1024**2
CELL_LIMIT=8192
EXPECTED_EAGLE_SHA='67a76083ecc365604b7fa9e9d8dd64aa0d4cf3f023c9050e98dc48829985716d'
SOURCE_FILES=[
 ['eagle_worker_v2.py','/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py',EXPECTED_EAGLE_SHA],
 ['eagle_worker_common.py','/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_common.py',None],
 ['spec_utils.py','/sgl-workspace/sglang/python/sglang/srt/speculative/spec_utils.py',None],
 ['eagle_info_v2.py','/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info_v2.py',None],
 ['eagle_utils.py','/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_utils.py',None],
 ['forward_batch_info.py','/sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py',None],
]
SOURCE_READER=r'''import pathlib,hashlib,json
base=pathlib.Path('/sgl-workspace/sglang/python')
items=__ITEMS__
out=[]
for name,path,expected in items:
    row={'name':name,'path':path,'expected_sha256':expected}
    try:
        p=pathlib.Path(path)
        assert p.is_absolute() and not p.is_symlink()
        cursor=p
        while cursor!=base:
            assert not cursor.is_symlink(),('symlink path component',str(cursor))
            cursor=cursor.parent
        resolved=p.resolve(strict=True)
        assert resolved.is_relative_to(base.resolve(strict=True))
        assert resolved.is_file()
        raw=resolved.read_bytes()
        assert len(raw)<=1048576
        row.update(exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),resolved=str(resolved))
        if expected is not None:
            row['expected_sha256_match']=row['sha256']==expected
            if not row['expected_sha256_match']:row['error']='sha256_mismatch'
        else:row['expected_sha256_match']=None
        try:row['text']=raw.decode('utf-8')
        except UnicodeDecodeError as error:row['text']=None;row['decode_error']=repr(error)
    except Exception as error:
        row.update(exists=False,error=repr(error))
    out.append(row)
print(json.dumps(out))
'''
SOURCE_READER=SOURCE_READER.replace('__ITEMS__',json.dumps(SOURCE_FILES,separators=(',',':')))

TEMPLATE=r'''
import base64,collections,csv,gzip,hashlib,json,pathlib,re,subprocess,time,traceback
exec(compile(L14_CORE,'frozen-CSV-analysis','exec'),globals())
out={'project_id':'glm53','run_id':L14_RUN_ID,'mode':'READ_ONLY_EXISTING_TRACE_CONTEXT','capture_id':L14_CAPTURE_ID,'diagnostic_run_id':L14_DIAG_RID,'node':L14_NODE,'representative_TP_rank':L14_RANK,'evidence_sha256':L14_EVIDENCE_SHA,'read_only':True,'model_requests':0,'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'remote_writes':0}
started=time.monotonic()
def _guarded(path,base):
    assert path.is_absolute() and not path.is_symlink()
    resolved=path.resolve(strict=True)
    assert resolved.is_relative_to(base.resolve(strict=True))
    cursor=path
    while cursor!=base:
        assert not cursor.is_symlink(),('symlink path component',str(cursor))
        cursor=cursor.parent
    return resolved
def _cell(value):
    if value is None:return None,False,0
    text=str(value)
    if len(text)>L14_CELL_LIMIT:return text[:L14_CELL_LIMIT],True,len(text)
    return text,False,len(text)
def _row(fieldnames,row,index):
    fields={};truncated=[];lengths={}
    for key in fieldnames:
        value,cut,length=_cell(row.get(key));fields[key]=value
        if cut:truncated.append(key);lengths[key]=length
    return {'data_row_index':index,'fields':fields,'truncated_columns':truncated,'original_lengths':lengths}
def _checked(path,expected):
    before=path.stat();assert before.st_size==expected['bytes']<=L14_MAX_CSV
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):digest.update(chunk)
    after_hash=path.stat();assert (before.st_size,before.st_mtime_ns)==(after_hash.st_size,after_hash.st_mtime_ns),'CSV changed during hash'
    assert digest.hexdigest()==expected['sha256']
    return {'bytes':before.st_size,'sha256':digest.hexdigest(),'mtime_ns_before':before.st_mtime_ns,'mtime_ns_after_hash':after_hash.st_mtime_ns}
def _operator(path,expected):
    meta=_checked(path,expected)
    with path.open(newline='') as file:
        reader=csv.DictReader(file);fieldnames=list(reader.fieldnames or [])
        assert 'Name' in fieldnames and 'Host Total Duration(us)' in fieldnames
        total=0;matched=0;unscored=0;scored=[]
        for index,row in enumerate(reader):
            total+=1;name=row.get('Name') or ''
            if not (name in ('draft','draft_extend') or name.startswith('step[VERIFY ') or name.startswith('step[DRAFT ') or name in ('acl_memcpy_host_to_device','aten::copy_')):continue
            matched+=1;value=number(row.get('Host Total Duration(us)'))
            if value is None:unscored+=1;continue
            scored.append((value,index,row))
    selected=sorted(scored,key=lambda item:(item[0],item[1]),reverse=True)[:64]
    after=path.stat();assert (meta['bytes'],meta['mtime_ns_before'])==(after.st_size,after.st_mtime_ns),'CSV changed during read';meta['mtime_ns_after_read']=after.st_mtime_ns
    return {'relative':'ASCEND_PROFILER_OUTPUT/'+path.name,'fieldnames':fieldnames,'data_rows':total,'matched_rows':matched,'unscored_matching_rows':unscored,'selected_rows_count':len(selected),'selection':'Name exact draft/draft_extend, startswith step[VERIFY /step[DRAFT , acl_memcpy_host_to_device or aten::copy_; top 64 by Host Total Duration(us)','selected_rows':[_row(fieldnames,row,index) for _,index,row in selected],'timestamp_like_columns':[x for x in fieldnames if re.search('time|timestamp|start|stop',x,re.I)],'timestamp_scope':'Header text and values are preserved verbatim; no unit or clock base is inferred.','file':meta}
def _kernel(path,expected):
    meta=_checked(path,expected)
    with path.open(newline='') as file:
        reader=csv.DictReader(file);fieldnames=list(reader.fieldnames or [])
        assert 'Name' in fieldnames and 'Duration(us)' in fieldnames
        total=0;classes={'mc2_moe_dispatch':[],'mc2_moe_combine':[],'native_communication':[]};counts={k:0 for k in classes}
        for index,row in enumerate(reader):
            total+=1;name=row.get('Name') or ''
            if 'MoeDistributeDispatch' in name:kind='mc2_moe_dispatch'
            elif 'MoeDistributeCombine' in name:kind='mc2_moe_combine'
            elif COMM.search(name):kind='native_communication'
            else:continue
            counts[kind]+=1;value=number(row.get('Duration(us)'))
            if value is None:continue
            classes[kind].append((value,index,row))
    after=path.stat();assert (meta['bytes'],meta['mtime_ns_before'])==(after.st_size,after.st_mtime_ns),'CSV changed during read';meta['mtime_ns_after_read']=after.st_mtime_ns
    selected={k:sorted(v,key=lambda item:(item[0],item[1]),reverse=True)[:20] for k,v in classes.items()}
    return {'relative':'ASCEND_PROFILER_OUTPUT/'+path.name,'fieldnames':fieldnames,'data_rows':total,'class_counts':counts,'selected_rows_count':{k:len(v) for k,v in selected.items()},'selection':'Name contains MoeDistributeDispatch or MoeDistributeCombine, or frozen COMM regex on Name; top 20 longest by Duration(us) per class; no Graph name guessing','selected_rows':{k:[_row(fieldnames,row,index) for _,index,row in v] for k,v in selected.items()},'file':meta}
def _task(path,expected):
    meta=_checked(path,expected)
    kind_counts=collections.Counter();name_counts=collections.Counter();name_duration=collections.Counter();types=collections.Counter();counts=collections.Counter();all_i=[];comm_i=[];other_i=[];invalid=[];total=0
    with path.open(newline='') as file:
        reader=csv.DictReader(file);fieldnames=list(reader.fieldnames or [])
        required=['kernel_name','kernel_type','task_time(us)','task_start(us)','task_stop(us)'];assert all(x in fieldnames for x in required)
        for row in reader:
            total+=1;assert total<=2000000
            name=row['kernel_name'];kind=row['kernel_type'];kind_counts[kind]+=1
            start=number(row['task_start(us)']);duration=number(row['task_time(us)']);end=number(row['task_stop(us)'])
            if start is None or duration is None or end is None or end<start or duration<0:
                counts['invalid_rows']+=1
                if len(invalid)<8:invalid.append({k:row.get(k) for k in required})
                continue
            counts['valid_rows']+=1;types[kind]+=duration;name_duration[name]+=duration;name_counts[name]+=1
            pair=(start,end);all_i.append(pair);(comm_i if COMM.search(name+' '+kind) else other_i).append(pair)
    after=path.stat();assert (meta['bytes'],meta['mtime_ns_before'])==(after.st_size,after.st_mtime_ns),'CSV changed during read';meta['mtime_ns_after_read']=after.st_mtime_ns
    if all_i:
        window=max(x[1] for x in all_i)-min(x[0] for x in all_i);recorded=union(all_i);ucomm=union(comm_i);uother=union(other_i)
    else:
        window=0.0;recorded=0.0;ucomm=0.0;uother=0.0
    group={'window_us':window,'recorded_interval_union_us':recorded,'communication_name_interval_union_us':ucomm,'other_task_interval_union_us':uother,'communication_other_overlap_us':max(0.0,ucomm+uother-recorded),'uncovered_within_recorded_window_us':max(0.0,window-recorded),'coverage_fraction':(recorded/window if window else None),'native_type_duration_sums_us':dict(types),'counts':dict(counts),'invalid_examples':invalid,'scope':'Union of recorded task intervals; waiting tasks and overlapping streams prevent treating this as physical NPU utilization. No cross-rank addition.'}
    top=[{'name':name,'count':name_counts[name],'duration_sum_us':value} for name,value in name_duration.most_common(20)]
    timestamp_columns=[x for x in fieldnames if re.search('time|timestamp|start|stop',x,re.I)]
    return {'relative':'ASCEND_PROFILER_OUTPUT/'+path.name,'fieldnames':fieldnames,'data_rows':total,'native_kind_counts':dict(kind_counts),'top_names_by_duration_sum':top,'top_names_scope':'Top 20 valid task rows by duration sum; count is valid rows only.','recorded_interval_group':group,'timestamp_like_columns':timestamp_columns,'timestamp_note':('Timestamp-like header values are preserved as-is; no unit, clock base or cross-file join is inferred.' if timestamp_columns else 'No timestamp-like header is present; timestamp join is null, not fabricated.'),'cross_file_join':None,'file':meta}
try:
    profile_base=pathlib.Path('/data/wio/inference-foundry/glm53')
    profile=shared_root/'profile'/L14_DIAG_RID
    _guarded(profile,profile_base)
    folder=profile/L14_FOLDER
    _guarded(folder,profile)
    output=folder/'ASCEND_PROFILER_OUTPUT'
    _guarded(output,folder)
    paths={name:_guarded(output/name,profile) for name in L14_CSV_NAMES}
    out['profile']={'host_root':str(profile),'folder':str(folder),'representative_TP_rank':L14_RANK,'representative_source_files':L14_EXPECTED_FILES,'only_representative_rank_read':True,'other_ranks_traces_read':False,'other_indexed_files_read':[],'path_boundary':str(profile_base),'symlink_allowed':False}
    samples={}
    for name in L14_CSV_NAMES:
        path=paths[name];expected=L14_EXPECTED_FILES[name]
        if name=='operator_details.csv':samples[name]=_operator(path,expected)
        elif name=='kernel_details.csv':samples[name]=_kernel(path,expected)
        else:samples[name]=_task(path,expected)
    out['csv_samples']=samples
    reader=subprocess.run(['docker','exec',L14_CID,'python3','-S','-c',L14_SOURCE_READER],capture_output=True,text=True,timeout=60)
    if reader.returncode!=0:
        out['source_files']=[];out['source_read_error']={'rc':reader.returncode,'stderr':reader.stderr[-4000:]}
    else:out['source_files']=json.loads(reader.stdout)
    out['source_read']={'method':'docker exec <CID> python3 -S -c <reader>','imports':['pathlib','hashlib','json'],'framework_imported':False,'torch_imported':False,'SDK_imported':False,'per_file_limit_bytes':L14_MAX_SOURCE,'files':[{'name':x[0],'path':x[1],'expected_sha256':x[2]} for x in L14_SOURCE_FILES]}
    sources=out['source_files']
    out['checks']={'profile_csv_bytes_sha_match':all(v['file']['sha256']==L14_EXPECTED_FILES[k]['sha256'] and v['file']['bytes']==L14_EXPECTED_FILES[k]['bytes'] for k,v in samples.items()),'profile_csv_mtime_stable':all(v['file']['mtime_ns_before']==v['file']['mtime_ns_after_hash']==v['file']['mtime_ns_after_read'] for v in samples.values()),'leaf_paths_no_symlink':True,'representative_only':True,'source_files_all_present':len(sources)==len(L14_SOURCE_FILES) and all(x.get('exists') is True for x in sources),'eagle_worker_v2_expected_sha_match':any(x.get('name')=='eagle_worker_v2.py' and x.get('sha256')==L14_EXPECTED_EAGLE_SHA for x in sources),'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'model_requests':0,'remote_writes':0}
    out['collection_pass']=all([out['checks']['profile_csv_bytes_sha_match'],out['checks']['profile_csv_mtime_stable'],out['checks']['source_files_all_present'],out['checks']['eagle_worker_v2_expected_sha_match']])
except BaseException as error:
    out['error']=repr(error);out['traceback']=traceback.format_exc();out['collection_pass']=False
out['duration_seconds']=time.monotonic()-started
raw=json.dumps(out).encode();assert len(raw)<=L14_MAX_NODE_JSON,'node context exceeds 4 MiB'
packed=gzip.compress(raw,mtime=0);assert len(packed)<=8*1024**2
print('GLM53_NATIVE_GZIP='+json.dumps({'version':1,'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'packed_bytes':len(packed),'packed_sha256':hashlib.sha256(packed).hexdigest(),'archive':None,'gzip_base64':base64.b64encode(packed).decode()}),flush=True)
'''

def _read(relative,digest):
    raw=(ROOT/relative).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest,relative
    return json.loads(raw)

def _sha(relative):
    return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()

def _load(name,relative,digest):
    path=ROOT/relative;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,relative
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def _render(template,values):
    pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b'
    code=re.sub(pattern,lambda match:repr(values[match[0]]),template);ast.parse(code)
    assert not ({k for k in values if re.search(r'\b'+re.escape(k)+r'\b',code)}),'unrendered placeholder'
    return code

def _inputs():
    evidence=_read(EVIDENCE,EVIDENCE_SHA)
    decision=_read(DECISION,DECISION_SHA)
    assert decision['project_id']=='glm53' and decision['decision']=='ACCEPTED'
    assert decision['evidence_sha256']==EVIDENCE_SHA and decision['auditor_sha256']==AUDITOR_SHA
    assert _sha(AUDITOR)==AUDITOR_SHA
    assert evidence['project_id']=='glm53' and evidence['capture_id']==CAPTURE_ID
    assert evidence['diagnostic_evidence_pass'] is True and evidence['diagnostic_only'] is True
    assert evidence['performance_certificate_allowed'] is False
    assert evidence['all32_D_native_profile_workers'] is True
    assert evidence['requested_profile_forward_steps']==2
    assert evidence['actual_P_MemCache_read_workers']==32 and evidence['actual_D_SDMA_receiving_devices']==32
    assert evidence['physical_wire_bytes'] is None
    capacity=_read(CAPACITY,CAPACITY_SHA);assert capacity['pass'] is True and capacity['selected_variant']=='candidate'
    diag=_read(DIAG_CONFIG,DIAG_CONFIG_SHA)
    assert diag['run_id']==DIAG_RID and diag['diagnostic_only'] is True and diag['performance_certificate_allowed'] is False
    assert capacity['meta_container_id']==diag['meta_container_id']
    for n in (36,41):
        provider=capacity['providers'][str(n)];configured=diag['runtime_containers'][str(n)]
        assert (provider['container_id'],provider['name'],provider['remote_root'],provider['run_id'])==(configured['container_id'],configured['name'],configured['remote_root'],configured['run_id'])
    node_caps={}
    for n,relative in NODE_CAPTURES.items():
        raw=(ROOT/relative).read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==NODE_CAPTURE_SHA[n]
        assert evidence['source_evidence_sha256'][relative]==digest
        node=json.loads(raw);assert node['rc']==0
        frame=node['native_output_transport'];assert frame['version']==1 and 0<frame['raw_bytes']<=256*1024**2 and frame['packed_bytes']<=8*1024**2
        node_caps[str(n)]={'path':relative,'sha256':digest,'bytes':len(raw),'native_output_raw_sha256':frame['raw_sha256'],'native_output_raw_bytes':frame['raw_bytes'],'native_output_packed_sha256':frame['packed_sha256']}
    workers=evidence['workers'];assert len(workers)==32
    assert {w['TP_rank'] for w in workers}==set(range(32))
    assert len({w['folder'] for w in workers})==32
    assert evidence['profile_completion_by_node']['36']['worker_folders']==16
    assert evidence['profile_completion_by_node']['41']['worker_folders']==16
    assert evidence['profile_completion_by_node']['36']['current_RID_markers']==32
    assert evidence['profile_completion_by_node']['41']['current_RID_markers']==32
    representatives={}
    for n,rank in REPRESENTATIVE.items():
        matches=[w for w in workers if w['node']==n and w['TP_rank']==rank];assert len(matches)==1
        worker=matches[0];assert worker['folder']==Path(worker['folder']).name
        files={}
        for name in CSV_NAMES:
            item=worker['source_files']['ASCEND_PROFILER_OUTPUT/'+name]
            assert type(item['bytes']) is int and 0<item['bytes']<=MAX_CSV
            assert re.fullmatch('[0-9a-f]{64}',item['sha256'])
            files[name]={'bytes':item['bytes'],'sha256':item['sha256']}
        representatives[str(n)]={'node':n,'TP_rank':rank,'folder':worker['folder'],'namespace_pid':worker['namespace_pid'],'native_device_id':worker['native_device_id'],'source_files':files,'remote_profile_root':capacity['providers'][str(n)]['remote_root']+'/profile/'+DIAG_RID+'/'+worker['folder'],'selection':'first worker of the D node: TP rank 0 on 36 and TP rank 16 on 41; no other rank trace is read'}
    base=_load('inventory',INVENTORY,INVENTORY_SHA)
    codec=_load('codec',CODEC,CODEC_SHA)
    identity=_load('identity',IDENTITY,IDENTITY_SHA)
    core=(ROOT/CORE).read_text();assert hashlib.sha256(core.encode()).hexdigest()==CORE_SHA
    capacity_subset={'providers':{str(n):capacity['providers'][str(n)] for n in (36,41)},'selected_variant':capacity['selected_variant'],'meta_container_id':capacity['meta_container_id']}
    return {'evidence':evidence,'decision':decision,'capacity':capacity,'diag':diag,'node_caps':node_caps,'representatives':representatives,'base':base,'codec':codec,'identity':identity,'core':core,'capacity_subset':capacity_subset}

def plans():
    inputs=_inputs();codes={}
    for n in (36,41):
        values={'L14_RUN_ID':RUN_ID,'L14_CAPTURE_ID':CAPTURE_ID,'L14_DIAG_RID':DIAG_RID,'L14_EVIDENCE_SHA':EVIDENCE_SHA,'L14_NODE':n,'L14_RANK':REPRESENTATIVE[n],'L14_CID':inputs['capacity_subset']['providers'][str(n)]['container_id'],'L14_FOLDER':inputs['representatives'][str(n)]['folder'],'L14_EXPECTED_FILES':inputs['representatives'][str(n)]['source_files'],'L14_CSV_NAMES':CSV_NAMES,'L14_SOURCE_FILES':SOURCE_FILES,'L14_SOURCE_READER':SOURCE_READER,'L14_EXPECTED_EAGLE_SHA':EXPECTED_EAGLE_SHA,'L14_MAX_CSV':MAX_CSV,'L14_MAX_SOURCE':MAX_SOURCE,'L14_MAX_NODE_JSON':MAX_NODE_JSON,'L14_CELL_LIMIT':CELL_LIMIT,'L14_CORE':inputs['core']}
        code=inputs['identity'].render(n,inputs['capacity_subset'])+'\n'+_render(TEMPLATE,values);ast.parse(code);codes[str(n)]=code
    meta={'project_id':'glm53','run_id':RUN_ID,'mode':'READ_ONLY_EXISTING_TRACE_CONTEXT','draft_status':'DRAFT_PENDING_PRIMARY_REVIEW','capture_id':CAPTURE_ID,'diagnostic_run_id':DIAG_RID,'evidence':{'path':EVIDENCE,'sha256':EVIDENCE_SHA},'decision':{'path':DECISION,'sha256':DECISION_SHA},'auditor':{'path':AUDITOR,'sha256':AUDITOR_SHA},'capacity':{'path':CAPACITY,'sha256':CAPACITY_SHA},'diagnostic_config':{'path':DIAG_CONFIG,'sha256':DIAG_CONFIG_SHA},'node_captures':inputs['node_caps'],'representatives':[inputs['representatives']['36'],inputs['representatives']['41']],'expected_source_files':[{'name':x[0],'path':x[1],'expected_sha256':x[2]} for x in SOURCE_FILES],'boundaries':{'nodes':[36,41],'sequential':True,'max_concurrent_SSH':1,'node_invocations':2,'extra_SSH':0,'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'model_requests':0,'new_profiles':0,'other_ranks_traces_read':False,'P_or_Meta_containers_contacted':False,'remote_writes':0,'raw_traces_remain_remote':True},'limits':{'csv_file_bytes':MAX_CSV,'source_file_bytes':MAX_SOURCE,'node_json_bytes':MAX_NODE_JSON,'operator_rows':64,'kernel_rows_per_class':20,'cell_chars':CELL_LIMIT,'cross_rank_addition':False,'overlap_addition':False,'cross_file_join':None},'source_pins':{'inventory':INVENTORY_SHA,'codec':CODEC_SHA,'identity':IDENTITY_SHA,'csv_analysis':CORE_SHA,'diagnostic_config':DIAG_CONFIG_SHA,'expected_eagle_worker_v2':EXPECTED_EAGLE_SHA},'transport':{'inventory_source_sha256':INVENTORY_SHA,'codec_source_sha256':CODEC_SHA,'decoder':'frozen codec.decode','encoder':'read-only frame-compatible variant; archive=null; no remote write'},'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'host_codes_sha256':{n:hashlib.sha256(code.encode()).hexdigest() for n,code in codes.items()},'review_path':REVIEW,'review_created':False,'executed':False,'invoke_called':False,'no_publish':True}
    return codes,meta

def execute(run_id,review_path=REVIEW):
    assert run_id==RUN_ID and review_path==REVIEW
    review_abs=ROOT/review_path;assert review_abs.is_file() and not review_abs.is_symlink()
    assert review_abs.resolve().is_relative_to(ROOT.resolve())
    review_raw=review_abs.read_bytes();review=json.loads(review_raw)
    codes,meta=plans();expected={n:hashlib.sha256(code.encode()).hexdigest() for n,code in codes.items()}
    module_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['project_id']=='glm53' and review['run_id']==RUN_ID
    assert review['decision']=='APPROVED_READ_ONLY' and review['read_only'] is True
    assert review['evidence_sha256']==EVIDENCE_SHA and review['module_sha256']==module_sha
    assert review['host_codes_sha256']==expected and review.get('model_requests_authorized')==0 and review.get('new_profile_authorized') is False
    base=_load('inventory',INVENTORY,INVENTORY_SHA);codec=_load('codec',CODEC,CODEC_SHA)
    run=ROOT/'runs'/RUN_ID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()))
    receipt={'project_id':'glm53','run_id':RUN_ID,'mode':'READ_ONLY_EXISTING_TRACE_CONTEXT','review':{'path':str(review_path),'sha256':hashlib.sha256(review_raw).hexdigest()},'source_sha256':module_sha,'host_codes_sha256':expected,'nodes':{},'model_requests':0,'HTTP_requests':0,'NPU_calls':0,'SDK_init_calls':0,'max_concurrent_SSH':1,'node_invocations':0,'extra_SSH':0,'raw_traces_remain_remote':True,'source_text_written_local_only':True,'no_publish':True,'performance_certificate_allowed':False}
    results={}
    try:
        for n in (36,41):
            try:
                pod,host=base.NODES[n]
                mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
                (run/('node-'+str(n)+'-mapping-wire.json')).write_text(json.dumps(mapping,indent=2)+'\n')
                if mapping['rc']!=0 or mapping['stdout']!='172.16.12.'+str(n):
                    receipt['nodes'][str(n)]={'error':'pod_mapping_mismatch','mapping_rc':mapping['rc'],'mapping_stdout':mapping['stdout'][:400],'mapping_stderr':mapping['stderr'][-2000:]}
                    continue
                code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+codes[str(n)];ast.parse(code)
                command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
                wire=base.invoke([command],code,timeout=180);(run/('node-'+str(n)+'-raw-wire.json')).write_text(json.dumps(wire,indent=2)+'\n');receipt['node_invocations']+=1
                if wire['rc']!=0:
                    receipt['nodes'][str(n)]={'rc':wire['rc'],'error':'remote_program_nonzero','stderr':wire['stderr'][-4000:]}
                    continue
                try:
                    decoded=codec.decode(wire);data=json.loads(decoded['stdout'])
                except Exception as error:
                    receipt['nodes'][str(n)]={'rc':wire['rc'],'error':repr(error)};continue
                (run/('node-'+str(n)+'.json')).write_text(json.dumps(decoded,indent=2)+'\n')
                (run/('node-'+str(n)+'-context.json')).write_text(json.dumps(data,indent=2)+'\n')
                source_dir=run/'source'/('node-'+str(n));source_dir.mkdir(parents=True,exist_ok=True)
                for item in data.get('source_files',[]):
                    name=item.get('name');assert isinstance(name,str) and Path(name).name==name
                    if item.get('exists') is True and isinstance(item.get('text'),str):(source_dir/name).write_text(item['text'])
                results[str(n)]=data
                receipt['nodes'][str(n)]={'rc':0,'collection_pass':data.get('collection_pass'),'profile':data.get('profile'),'checks':data.get('checks'),'source_files':[{'name':x.get('name'),'exists':x.get('exists'),'bytes':x.get('bytes'),'sha256':x.get('sha256'),'error':x.get('error')} for x in data.get('source_files',[])]}
            except Exception as error:
                receipt['nodes'][str(n)]={'error':repr(error)}
        receipt['pass']=len(results)==2 and all(receipt['nodes'][str(n)].get('rc')==0 and receipt['nodes'][str(n)].get('collection_pass') is True for n in (36,41))
        receipt['meta']=meta
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    plans_codes,plans_meta=plans()
    print(json.dumps({'meta':plans_meta,'host_codes_sha256':{n:hashlib.sha256(code.encode()).hexdigest() for n,code in plans_codes.items()}},indent=2))
