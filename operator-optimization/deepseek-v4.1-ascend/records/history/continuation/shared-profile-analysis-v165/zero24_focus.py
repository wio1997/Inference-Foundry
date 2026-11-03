"""Fill only the 24 raw examples omitted by the previous 512-row sample cap."""
import collections,importlib.util,json,struct
from pathlib import Path
C=Path('/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b')
spec=importlib.util.spec_from_file_location('raw',C/'read_raw_census.py');raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
census=json.loads((C/'RAW-FIELD-CENSUS.json').read_text());fixed=struct.Struct('<3q4QB?');records=[];rescanned=[]
for profile in census['profiles']:
    expected=sum(n for key,n in profile['regimes'].items() if (lambda x:x[0] in ['aten::zero_','aten::zeros_like'] and 'bool' in x[1].lower() and x[2] in ['24','[24]','[[24]]'])(json.loads(key)))
    # Use the parser's structured tensors to avoid relying on the shape text encoding.
    kept=[r for r in profile['interesting'] if r['name'] in ['aten::zero_','aten::zeros_like'] and any(t['dtype']=='bool' and t['shape']==[24] for t in r['input_tensors'])]
    # The actual field4 encoding is stable in this cohort; derive expected from its observed kept records.
    shape_texts={r['shape_text'] for r in kept}
    expected=sum(n for key,n in profile['regimes'].items() if (lambda x:x[0] in ['aten::zero_','aten::zeros_like'] and 'bool' in x[1].lower() and x[2] in shape_texts)(json.loads(key)))
    if expected>len(kept):
        kept=[]
        for ordinal,(_,body) in enumerate(raw.outer_file(Path(profile['profile_dir'])/'FRAMEWORK/torch.op_range'),1):
            fields=dict(raw.tlvs(body[fixed.size:]));name=fields.get(2,b'').decode()
            if name not in ['aten::zero_','aten::zeros_like'] or b'bool' not in fields.get(3,b'').lower():continue
            tensors=raw.tensor_meta(fields[5].decode()) if 5 in fields else []
            if not any(t['dtype']=='bool' and t['shape']==[24] for t in tensors):continue
            header=fixed.unpack_from(body)
            kept.append({'event_ordinal':ordinal,'name':name,'raw_syscnt_begin':header[0],'raw_syscnt_end':header[1],
                         'pid':header[3],'tid':header[4],'input_tensors':tensors,'shape_text':fields.get(4,b'').decode(),
                         'stack_excerpt':fields.get(8,b'').decode()[:6000]})
        rescanned.append(profile['profile_dir'])
    if len(kept)!=expected:raise ValueError('Zero24 index mismatch '+profile['profile_dir']+': '+str((len(kept),expected)))
    records.extend({'profile':profile['profile_dir'],'phase':profile['kind'],**r} for r in kept)
groups=collections.Counter()
for r in records:
    frame=r['stack_excerpt'].split(';')[0]
    for t in r['input_tensors']:
        if t['dtype']=='bool' and t['shape']==[24]:groups[(r['phase'],r['name'],t['device_type'],frame)]+=1
out={'records':records,'groups':[{'phase':k[0],'op':k[1],'device_type':k[2],'first_frame':k[3],'count':v} for k,v in groups.items()],
     'targeted_rescan_profiles':rescanned,'source_profile_count':len(census['profiles']),'records_count':len(records),
     'same_shape_does_not_identify_role_or_alias':True,'historical_native_promotions':0,'all_reader_proof':False}
(C/'ZERO24-COMPLETE-FOCUS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['records','targeted_rescan_profiles']},ensure_ascii=False))
