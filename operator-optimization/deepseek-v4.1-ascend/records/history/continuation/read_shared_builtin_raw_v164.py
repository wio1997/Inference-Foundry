"""Stdlib-only terminal-cohort raw field/marker/guard census.

Preserves process/profile identity; never joins historical tasks by timestamps,
shape, pointers or names. Does not read Tensor contents or initialize CANN.
"""
import argparse,collections,hashlib,json,struct
from pathlib import Path

def pin(p):
    p=Path(p);h=hashlib.sha256();n=0
    with p.open('rb') as f:
        while True:
            b=f.read(4*1024**2)
            if not b:break
            n+=len(b);h.update(b)
    return {'path':str(p),'bytes':n,'sha256':h.hexdigest()}
def tlvs(raw):
    at=0
    while at<len(raw):
        if len(raw)-at<6:raise ValueError('Truncated TLV header')
        kind,n=struct.unpack_from('<HI',raw,at);at+=6
        if at+n>len(raw):raise ValueError('Truncated TLV body')
        yield kind,raw[at:at+n];at+=n
def outer_file(path):
    size=path.stat().st_size;at=0
    with path.open('rb') as f:
        while at<size:
            b=f.read(6);at+=len(b)
            if len(b)!=6:raise ValueError('Incomplete outer TLV header')
            kind,n=struct.unpack('<HI',b)
            if n>8*1024**2 or n>size-at:raise ValueError('Oversize/incomplete record;scope unknown')
            body=f.read(n);at+=len(body)
            if len(body)!=n:raise ValueError('Incomplete outer body')
            yield kind,body
def tensor_meta(value):
    result=[]
    for item in value.split(')'):
        if not item:continue
        parts=item.split(';')
        if len(parts)!=8:raise ValueError('Unrecognized8-field Tensor schema')
        number=lambda x:int(x,16) if x.startswith('0x') else int(x)
        vector=lambda x:[int(v) for v in x.split(',')] if x else []
        result.append({'TensorImpl':number(parts[0]),'storage_base':number(parts[1]) if parts[1] else None,
            'dtype':parts[2],'element_bytes':int(parts[3]),'shape':vector(parts[4]),'stride':vector(parts[5]),
            'device_type':int(parts[6]),'device_index':int(parts[7]),
            'storage_offset_extent_epoch_not_emitted_by_this_record':True})
    return result
def profile(path,label):
    result={'profile_dir':str(path),'kind':label,'events':0,'field_tags':collections.Counter(),
        'operator_names':collections.Counter(),'regimes':collections.Counter(),'marker_names':[],'interesting':[],'parse_errors':[],
        'NPU_task_or_template_bridge_proved':False}
    framework=path/'FRAMEWORK';op=framework/'torch.op_range'
    info=path/'profiler_info.json'
    if info.exists():result['profiler_configuration']=json.loads(info.read_text()).get('config')
    if not op.exists():result['parse_errors'].append('Missing torch.op_range');return result
    result['raw_op_range']=pin(op);fixed=struct.Struct('<3q4QB?')
    try:
        for tag,raw in outer_file(op):
            if tag!=1 or len(raw)<fixed.size:raise ValueError('Unknown Torch outer tag/body')
            header=fixed.unpack_from(raw);pairs=list(tlvs(raw[fixed.size:]));fields={k:v for k,v in pairs}
            if len(fields)!=len(pairs):raise ValueError('Duplicate Torch field tags')
            result['events']+=1;result['field_tags'].update(fields.keys());name=fields.get(2,b'UNKNOWN').decode()
            result['operator_names'][name]+=1
            dtypes=fields.get(3,b'').decode();shapes=fields.get(4,b'').decode()
            result['regimes'][json.dumps([name,dtypes,shapes],separators=(',',':'))]+=1
            if name.startswith('wio_shared/'):
                if len(result['marker_names'])<8192:result['marker_names'].append(name)
                else:result['marker_names_truncated']=True
            candidate=name in ['aten::zero_','aten::zeros_like','aten::constant_pad_nd'] or 'bool' in dtypes.lower()
            tensors=tensor_meta(fields[5].decode()) if candidate and 5 in fields else []
            wanted=name in ['aten::zero_','aten::zeros_like','aten::constant_pad_nd'] or any(t['dtype']=='bool' and t['shape'] in [[8],[16],[20],[24]] for t in tensors)
            if wanted:
                if len(result['interesting'])<512:
                    result['interesting'].append({'event_ordinal':result['events'],'name':name,
                        'raw_syscnt_begin':header[0],'raw_syscnt_end':header[1],'pid':header[3],'tid':header[4],
                        'input_tensors':tensors,'dtype_labels':dtypes,'shape_text':shapes,
                        'scalar_type_labels_not_values':fields.get(7,b'').decode(),'stack_excerpt':fields.get(8,b'').decode()[:6000],
                        'original_model_operands_promoted':0})
                else:result['interesting_rows_truncated']=True
    except Exception as e:result['parse_errors'].append(type(e).__name__+':'+str(e))
    memory=framework/'torch.memory_usage'
    if memory.exists():
        result['raw_memory_usage']=pin(memory);counter=collections.Counter()
        try:
            for tag,b in outer_file(memory):
                if len(b)!=77:raise ValueError('Unrecognized allocator record layout')
                x=struct.unpack('<7q2b3B2Q',b)
                counter[str((x[7],x[8],x[10],x[11],x[12]))]+=1
            result['allocator_category_counts']=dict(counter)
        except Exception as e:result['parse_errors'].append(type(e).__name__+':'+str(e))
    result['field_tags']={str(k):v for k,v in result['field_tags'].items()}
    result['operator_names']=dict(result['operator_names'])
    result['regimes']=dict(result['regimes'])
    result['raw_syscnt_not_cross_joined_to_realtime_by_nearest_timestamp']=True
    return result
def supplements(root):
    results=[];all_markers=set();guard_counts=collections.Counter()
    for p in sorted(root.glob('*.jsonl')):
        info={'file':pin(p),'kinds':collections.Counter(),'headers':[],'footer':None,'guard_variants':{},
            'scope_marker_bindings':[],'graph_bindings':[],'scheduler_totals':collections.Counter(),'errors':[]}
        try:
            with p.open() as f:
                for line in f:
                    if len(line)>4*1024**2:raise ValueError('Oversize supplementary record')
                    r=json.loads(line);kind=r.get('kind','UNKNOWN');info['kinds'][kind]+=1
                    if kind=='header':info['headers'].append(r)
                    elif kind=='footer':info['footer']=r
                    elif kind=='runner_guard_snapshot':
                        key=json.dumps({k:r.get(k) for k in ['use_compress','use_sparse','attn_backend_MRO','enabled_dispatch_mode']},sort_keys=True)
                        guard_counts[key]+=1;info['guard_variants'][key]=info['guard_variants'].get(key,0)+1
                    elif kind=='scope_begin':
                        marker=r.get('identity',{}).get('profile_marker')
                        if marker:
                            all_markers.add(marker)
                            if len(info['scope_marker_bindings'])<8192:info['scope_marker_bindings'].append({'profile_marker':marker,'scope_id':r['scope_id'],'name':r['name'],'identity':r['identity']})
                            else:info['scope_bindings_truncated']=True
                    elif kind=='wrapper_graph_binding':
                        if len(info['graph_bindings'])<512:info['graph_bindings'].append(r)
                        else:info['graph_bindings_truncated']=True
                    elif kind=='scheduler_host_snapshot':info['scheduler_totals'][str(r.get('total_num_scheduled_tokens'))]+=1
                    elif kind=='supplementary_metadata_error':info['errors'].append(r)
            info['kinds']=dict(info['kinds']);info['scheduler_totals']=dict(info['scheduler_totals'])
            info['complete_footer']=bool(info['footer'])
        except Exception as e:info['errors'].append({'parse_error':type(e).__name__+':'+str(e)})
        results.append(info)
    return results,all_markers,[{'guard':json.loads(k),'observations':v} for k,v in guard_counts.items()]
def analyze(root):
    status_path=root/('CORRECTION-STATUS.json' if (root/'CORRECTION-STATUS.json').exists() else 'STATUS.json')
    status=json.loads(status_path.read_text())
    if not status.get('finished_unix'):
        # The raw writer cohort can be closed while a different original service
        # is restoring. Verify Docker's state,not merely a status-file promise.
        import subprocess
        prep=json.loads((root/'control/PREPARED.json').read_text())
        d=json.loads(subprocess.check_output(['docker','inspect',prep['diagnostic_container_id']],text=True))[0]
        if d['State']['Running'] or not status.get('profile_case_valid'):
            raise RuntimeError('Diagnostic writers not terminal;poll the same handle')
        status['raw_writer_container_closed_restore_still_pending']=True
    data=root/'runtime/diagnostics';profiles=[]
    for label in ['startup','formal']:
        for path in sorted((data/label).glob('*_ascend_pt')):profiles.append(profile(path,label))
    extra,markers,guards=supplements(data/'supplement')
    for p in profiles:
        names=p['marker_names'];p['exact_marker_names_found_in_supplement']=sum(n in markers for n in names)
        p['marker_equality_binds_CPU_scope_only_not_native_tasks']=True
    cases={}
    for label in ['baseline','profile']:
        path=root/'outputs'/label/'window-status.json'
        cases[label]=json.loads(path.read_text()) if path.exists() else None
    result={'status':'TERMINAL_COHORT_RAW_FIELD_SCOPE_CENSUS','controller':status,'case_receipts':cases,
        'profiles':profiles,'supplementary_files':extra,'actual_runner_guard_variants':guards,
        'counts':{'startup_profiles':sum(p['kind']=='startup' for p in profiles),
          'formal_profiles':sum(p['kind']=='formal' for p in profiles),
          'profiles_with_input_detail':sum(int(p['field_tags'].get('5',0))>0 for p in profiles),
          'profiles_with_parse_errors':sum(bool(p['parse_errors']) for p in profiles),
          'supplementary_files':len(extra),'files_with_footer':sum(r.get('complete_footer',False) for r in extra)},
        'scope':'New171 only;raw fields/markers/guards,not original1352 native operands or all-reader/physical evidence.',
        'new_historical_native_promotions':0,'new_independence_proofs':0,'whole_modules_closed':0,'overall_complete':False}
    out=root/'RAW-FIELD-CENSUS.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    result=analyze(args.root)
    print(json.dumps({'status':result['status'],'counts':result['counts'],'actual_runner_guard_variants':result['actual_runner_guard_variants'],
        'restore':result['controller'].get('restore'),'case_valid':{k:bool(v and v.get('valid')) for k,v in result['case_receipts'].items()},
        'overall_complete':False},ensure_ascii=False))
