"""CPU-only, streaming intact native ET node prefix;never invent its footer."""
import collections
import hashlib
import json
import os
import time
from pathlib import Path

ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint176')
OUT=ROOT.parent/'joint176-derived'
OUT.mkdir(exist_ok=True)

def write(name,value):
    p=OUT/name;p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    assert p.stat().st_size<64*1024**2
def pin(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def nodes(path):
    # Exact native pretty-print delimiters,validated for every parsed node.
    # A valid completed node remains evidence even if the top-level file lacks
    # its final array/object/footer. No missing suffix is synthetically added.
    buf=[]
    with path.open() as f:
        for line in f:
            if line.rstrip()=='      {':
                if buf:raise RuntimeError('Unexpected nested native ET node delimiter')
                buf=[line]
            elif buf:
                buf.append(line)
                if line.rstrip() in ['      },','      }']:
                    text=''.join(buf).rstrip().removesuffix(',');r=json.loads(text)
                    assert isinstance(r.get('id'),int) and isinstance(r.get('name'),str)
                    yield r;buf=[]
        if buf:raise RuntimeError('Truncated individual node;stop and preserve prior parser result')
def tensor_values(v):
    if isinstance(v,list):
        if len(v)==6 and all(isinstance(v[i],int) for i in range(5)) and isinstance(v[-1],str):
            yield v
        else:
            for z in v:yield from tensor_values(z)

started=time.time();all_workers=[];selected=None
for q in sorted((ROOT/'workers').glob('*')):
    f=q/'bindings.jsonl'
    if not f.exists():continue
    kinds=collections.Counter();names=collections.Counter();identity=None;formal=[];omissions=collections.Counter();controls={};bank_records=0
    with f.open() as z:
        for l in z:
            r=json.loads(l);kinds[r['kind']]+=1
            if r['kind']=='worker_identity':identity=r
            if r['kind']=='formal_start':formal.append(r)
            if r['kind']=='binding':
                names[r['name'].split('/')[0]]+=1
                for s in r.get('omitted',[]):omissions[s]+=1
                bank_records+=sum(isinstance(v,dict) and 'tensor_impl' in v for v in r.get('controls',{}).values())
                for k,v in r.get('controls',{}).items():
                    if not isinstance(v,(dict,list)) and any(t in k.lower() for t in ['all_greedy','all_random','no_penalties','logprob','entropy','synthetic','kv_sharing_fast_prefill','mamba_cache_mode']):
                        controls.setdefault(k,[])
                        if v not in controls[k]:controls[k].append(v)
    if not identity:continue
    rec={'worker':q.name,'identity':identity,'sidecar':pin(f),'event_kinds':dict(kinds),'binding_name_groups':dict(names),'raw_tensor_metadata_records':bank_records,'formal_start_events':formal,'selected_scalar_values':controls,'omissions':dict(omissions),'full_case_completed':False}
    all_workers.append(rec)
    if identity['EP']['rank_in_group']==0:selected=q
assert len(all_workers)==16 and selected is not None
write('all16-sidecar-evidence-index.json',{'workers':all_workers,'scope':'Actual startup/capture/formal-start state;full frozen C8 failed before completion.','diagnostic_markers_not_baseline_consumers':True})

bindings={};module_records=[];scalar_bindings=[]
with (selected/'bindings.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        if r['kind']=='module_identity':module_records.append(r)
        if r['kind']=='binding':
            bindings[r['seq']]=r
            if not r.get('tensor_paths') or r.get('omitted'):scalar_bindings.append(r)
write('ep0-module-identities.json',module_records)
write('ep0-scalar-bindings-and-omissions.json',scalar_bindings)

trace=selected/'execution-trace.json';opcounts=collections.Counter();marker_counts=collections.Counter();bool_storages=set();matched=set();count=0;first=None;last=None
bind_path=OUT/'ep0-named-public-tensor-bindings.jsonl'
with bind_path.open('w') as bf:
    for n in nodes(trace):
        count+=1;first=n['id'] if first is None else first;last=n['id'];name=n['name'];opcounts[name]+=1
        if name.startswith('wio176/'):
            marker_counts[name.split('/')[1]]+=1
            vals=n['inputs']['values']
            if len(vals)>=2 and isinstance(vals[0],int) and isinstance(vals[1],str):
                seq=vals[0];r=bindings.get(seq)
                if r:
                    paths=json.loads(vals[1]);ts=vals[2:]
                    assert paths==r['tensor_paths'] and len(paths)==len(ts),(seq,name)
                    for path,t in zip(paths,ts):
                        assert list(tensor_values(t))==[t],(seq,path,t)
                        bf.write(json.dumps({'binding_seq':seq,'native_node':n['id'],'name':name,'path':path,'ET_tensor':t,'raw_metadata':r.get('controls',{}).get(path)},ensure_ascii=False)+'\n')
                    matched.add(seq)
        if not name.startswith('wio176/'):
            for side in ['inputs','outputs']:
                io=n[side]
                for v,shape,typ in zip(io.get('values',[]),io.get('shapes',[]),io.get('types',[])):
                    if shape==[24] and 'bool' in typ.lower():
                        bool_storages.update(t[1] for t in tensor_values(v))
assert bind_path.stat().st_size<64*1024**2

related_count=0;related_ops=collections.Counter();storage_events=collections.defaultdict(list);bool_path=OUT/'ep0-BOOL24-storage-native-neighborhood.jsonl'
assert bool_path.exists() and bool_path.stat().st_size<160*1024**2
with bool_path.open() as f:
    for line in f:
        n=json.loads(line);related_count+=1;related_ops[n['name']]+=1
        hit={t[1] for side in ['inputs','outputs'] for t in tensor_values(n[side].get('values',[])) if t[1] in bool_storages}
        for st in hit:
            if len(storage_events[st])<12:storage_events[st].append({'node':n['id'],'name':n['name'],'ctrl_deps':n['ctrl_deps']})

with trace.open('rb') as f:f.seek(max(0,trace.stat().st_size-300));tail=f.read().decode(errors='replace')
summary={'status':'INTACT_NODE_PREFIX_AND_SIDECAR_BINDINGS_RECOVERED_CPU_ONLY',
    'selected_worker':str(selected),'trace':pin(trace),'sidecar':pin(selected/'bindings.jsonl'),
    'trace_document_has_closing_footer':tail.rstrip().endswith('}'),
    'trace_document_status':'UNCLOSED_NATIVE_ET_DOCUMENT;COMPLETE_NODE_PREFIX_ONLY',
    'parsed_intact_nodes':count,'first_emitted_node':first,'last_emitted_node':last,
    'original_operator_counts':{k:v for k,v in opcounts.items() if not k.startswith('wio176/')},
    'diagnostic_marker_groups':dict(marker_counts),'matched_binding_markers':len(matched),
    'sidecar_bindings':len(bindings),'unmatched_sidecar_binding_seqs':sorted(set(bindings)-matched),
    'BOOL24_related_storages':len(bool_storages),'BOOL24_related_native_nodes':related_count,
    'BOOL24_related_native_operator_counts':dict(related_ops),'BOOL24_storage_node_samples':dict(storage_events),
    'output_files':[pin(p) for p in [bind_path,bool_path]],
    'scope_limits':['No complete frozen C8,closed worker FINAL,performance timing or all-case runtime proof.',
        'Native ET emitted order and Tensor IDs are not device execution/retirement order.',
        'RF diagnostic bindings are observations,not required baseline data consumers.',
        'Different ET storage IDs alone do not prove independent allocation/address/lifetime.'],
    'CPU_seconds_elapsed':time.time()-started,'new_NPU_context_or_model_calls':0,'overall_complete':False}
# A trailing node ends with },too;it is not a valid top-level closing footer.
summary['trace_document_has_closing_footer']='finish_ts' in tail
write('SUMMARY.json',summary)
print(json.dumps({'status':summary['status'],'intact_nodes':count,'matched_markers':len(matched),'BOOL24_storages':len(bool_storages),'CPU_seconds':summary['CPU_seconds_elapsed']}))
