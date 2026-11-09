"""Build a separate, verified metadata view over immutable checkpoint shards.

Recovers tensors omitted from a local index when complete shards are already
present. Does not modify the source or read tensor payloads. Rejects conflicting
names, out-of-range offsets and incomplete files.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

_SIZES={'F32':4,'BF16':2,'F16':2,'F8_E4M3':1,'F8_E8M0':1,'I8':1,'U8':1,'I16':2,'I32':4,'I64':8,'BOOL':1}

def build_view(source:Path,destination:Path):
    source=source.resolve();destination=destination.absolute()
    if destination.exists():raise FileExistsError(f'view already exists: {destination}')
    original_bytes=(source/'model.safetensors.index.json').read_bytes()
    original=json.loads(original_bytes);mapping={};entries={};total=0
    shards=sorted(source.glob('model-*-of-*.safetensors'))
    if not shards:raise ValueError('no checkpoint shards')
    for shard in shards:
        with shard.open('rb') as f:
            prefix=f.read(8)
            if len(prefix)!=8:raise ValueError(f'incomplete header: {shard.name}')
            size=struct.unpack('<Q',prefix)[0]
            if not 0<size<=min(100_000_000,shard.stat().st_size-8):raise ValueError(f'unsafe header size: {shard.name}')
            header=json.loads(f.read(size))
        for name,desc in header.items():
            if name=='__metadata__':continue
            begin,end=desc['data_offsets'];dtype=desc['dtype']
            if dtype not in _SIZES:raise ValueError(f'unsupported storage dtype {dtype}')
            count=math.prod(desc['shape'])*_SIZES[dtype]
            if not 0<=begin<=end<=shard.stat().st_size-8-size or end-begin!=count:raise ValueError(f'invalid byte range: {shard.name}:{name}')
            if name in mapping:raise ValueError(f'duplicate tensor: {name}')
            mapping[name]=shard.name;entries[name]=desc;total+=count
    for name,shard in original['weight_map'].items():
        if mapping.get(name)!=shard:raise ValueError(f'original index/shard mismatch: {name}')
    canonical=[name.removeprefix('model.') for name in mapping]
    if len(set(canonical))!=len(canonical):raise ValueError('ambiguous model.-prefixed names')
    config=json.loads((source/'config.json').read_text())
    expected_engram=config.get('text_config',{}).get('engram_layer_ids',[])
    for layer in expected_engram:
        for suffix in ['embed.weight','embed.scale','q_weight','k_weight','wkv.weight','wkv.scale']:
            if f'layers.{layer}.engram.{suffix}' not in mapping:raise ValueError(f'incomplete Engram layer {layer}: {suffix}')
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.name!='model.safetensors.index.json':(destination/item.name).symlink_to(item,target_is_directory=item.is_dir())
    index={**original,'metadata':{**original.get('metadata',{}),'total_size':total},'weight_map':mapping}
    (destination/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
    assert (source/'model.safetensors.index.json').read_bytes()==original_bytes
    report={'source_index_sha256':hashlib.sha256(original_bytes).hexdigest(),'view_index_sha256':hashlib.sha256((destination/'model.safetensors.index.json').read_bytes()).hexdigest(),'source_tensor_count':len(original['weight_map']),'view_tensor_count':len(mapping),'added_tensors':sorted(set(mapping)-set(original['weight_map'])),'source_index_unchanged':True,'indexed_payload_bytes':total,'engram_layers':expected_engram,'boundary':'Headers/ranges/config checked; weight payload checksums and numerical model correctness not established'}
    (destination/'checkpoint-view-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build_view(args.source,args.destination),indent=2))
