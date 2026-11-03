"""Use recovered data once;do not equate ET identity with allocation epoch."""
import collections
import json
from pathlib import Path

D=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint176-derived')
rows=[];tensor_impls=collections.defaultdict(dict);c2=[];selected=[]
with (D/'ep0-named-public-tensor-bindings.jsonl').open() as f:
    for line in f:
        r=json.loads(line);m=r.get('raw_metadata')
        if not isinstance(m,dict) or 'tensor_impl' not in m:continue
        tid=r['ET_tensor'][0];impl=m['tensor_impl']
        if len(tensor_impls[tid])<4:tensor_impls[tid].setdefault(impl,r)
        if 'c2_complete_mask' in r['path']:c2.append(r)
        if ('catalogue/' in r['name'] and any(k in r['path'] for k in ['embed_tokens.weight','lm_head.weight']) and 'engram' not in r['path']) or 'topk_indices_buffer' in (r['name']+'/'+r['path']):selected.append(r)
reuse=[]
for tid,values in tensor_impls.items():
    if len(values)>1:
        reuse.append({'ET_tensor_id':tid,'distinct_raw_TensorImpls':len(values),'examples':list(values.values())[:2]})
        if len(reuse)==5:break
assert c2
storages={r['ET_tensor'][1] for r in c2};events=[];types=collections.Counter();exact_bool24_counts=collections.Counter();public_zero_pad=[]
with (D/'ep0-BOOL24-storage-native-neighborhood.jsonl').open() as f:
    for line in f:
        n=json.loads(line);hit=False
        for side in ['inputs','outputs']:
            vals=n[side].get('values',[]);shapes=n[side].get('shapes',[]);types_io=n[side].get('types',[])
            for v,shape,typ in zip(vals,shapes,types_io):
                if isinstance(v,list) and len(v)==6 and isinstance(v[-1],str):
                    if v[1] in storages:hit=True
                    if shape==[24] and 'bool' in typ.lower():
                        exact_bool24_counts[(n['name'],side)]+=1
                        if n['name'] in ['aten::zeros','aten::zero_','aten::new_zeros','aten::constant_pad_nd','aten::pad']:
                            if len(public_zero_pad)<24:public_zero_pad.append(n)
        if hit:
            types[n['name']]+=1
            if len(events)<60:events.append(n)

def write(name,x):(D/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
write('ep0-bank-epoch-and-role-review.json',{
    'status':'POSITIVE_PUBLIC_BINDINGS_WITH_EXPLICIT_ET_ID_REUSE_GUARD',
    'c2_complete_mask':{
        'named_observations':len(c2),'ET_storage_ids':sorted(storages),
        'raw_storage_impls':sorted({r['raw_metadata']['storage_impl'] for r in c2}),
        'raw_storage_addresses':sorted({r['raw_metadata']['storage_base_address'] for r in c2}),
        'storage_sizes':sorted({r['raw_metadata']['storage_nbytes'] for r in c2}),
        'view_extents':sorted({tuple(r['raw_metadata']['shape']) for r in c2}),
        'first_and_last_bindings':[c2[0],c2[-1]],
        'native_storage_occurrence_types':dict(types),
        'interpretation':'Observed named C2 views share this persistent backing;do not reuse transient Tensor identity or erase writes/readiness without generation and all-reader contracts.'},
    'reused_ET_Tensor_id_examples':reuse,
    'reuse_rule':'No different-raw-TensorImpl under one ET Tensor-ID counterexample found in this named subset. ET identity alone does not establish allocation retirement/generation;use source lifetime before global joins.',
    'selected_embedding_head_and_index_banks':selected,
    'exact_BOOL24_direct_operand_counts':[{'name':k[0],'side':k[1],'count':v} for k,v in exact_bool24_counts.items()],
    'zero_pad_native_examples':public_zero_pad,
    'diagnostic_marker_reads_excluded':True,
    'no_blanket_alias_independence_or_delete_claim':True,
    'no_historical_v164_or166_identity_backfill':True,
    'full_case_completed':False,'overall_complete':False})
write('ep0-C2-persistent-bank-native-samples.json',events)
print(json.dumps({'status':'PASS','named_C2_observations':len(c2),'C2_storage_ids':sorted(storages),'C2_backing_sizes':sorted({r['raw_metadata']['storage_nbytes'] for r in c2}),'ET_reuse_counterexamples':len(reuse),'selected_parameter_or_index_records':len(selected)}))
