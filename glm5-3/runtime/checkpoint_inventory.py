"""Read safetensors headers without reading tensor payloads.

Byte accounting is checkpoint storage, not loaded-device memory or a fit claim.
The header digest identifies metadata only, never the actual weight contents.
"""
import argparse,collections,hashlib,json,re,struct
from pathlib import Path

def inventory(directory):
    rows=[];totals=collections.Counter();tensor_names=set()
    for path in sorted(Path(directory).glob('*.safetensors')):
        with path.open('rb') as stream:
            prefix=stream.read(8)
            if len(prefix)!=8:raise ValueError('truncated header length: '+str(path))
            length=struct.unpack('<Q',prefix)[0]
            if length>64*1024*1024:raise ValueError('unreasonably large header: '+str(path))
            raw=stream.read(length)
        if len(raw)!=length:raise ValueError('truncated header: '+str(path))
        header=json.loads(raw);size=path.stat().st_size;payload=size-8-length;file_total=0
        for name,tensor in header.items():
            if name=='__metadata__':continue
            if name in tensor_names:raise ValueError('duplicate tensor: '+name)
            tensor_names.add(name);start,end=tensor['data_offsets']
            if not 0<=start<=end<=payload:raise ValueError('invalid offset: '+name)
            count=end-start;file_total+=count
            layer=re.search(r'model\.layers\.(\d+)\.',name)
            mtp=bool(layer and int(layer.group(1))>=78)
            if '.mlp.experts.' in name:category='mtp_routed_experts' if mtp else 'routed_experts'
            elif mtp:category='mtp_other'
            elif '.self_attn.' in name:category='attention_and_indexer'
            elif '.mlp.shared_experts.' in name:category='shared_experts'
            elif '.mlp.' in name:category='dense_mlp_and_router'
            elif 'embed_tokens' in name or name.startswith('lm_head.'):category='embedding_and_output_head'
            else:category='norms_and_other'
            totals[category]+=count
        rows.append({'file':str(path),'file_bytes':size,'header_bytes':length+8,'header_sha256':hashlib.sha256(prefix+raw).hexdigest(),'tensor_bytes':file_total,'payload_bytes':payload,'unassigned_bytes':payload-file_total})
    if not rows:raise ValueError('no safetensors files')
    return {'model_dir':str(Path(directory).resolve()),'kind':'checkpoint_header_inventory','shards':len(rows),'tensors':len(tensor_names),'file_bytes':sum(r['file_bytes'] for r in rows),'tensor_bytes':sum(totals.values()),'category_bytes':dict(totals),'files':rows,'limits':['does not read or hash tensor payloads','storage dtypes/padding/loading transforms are not runtime device allocations','GLM-specific MTP classification uses layer index >=78','no TP/EP/DP replication or activation/KV/graph fit inferred']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('output');a=p.parse_args()
    result=inventory(a.directory);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['shards','tensors','tensor_bytes','category_bytes']}))
