"""Header-only residency estimates; never materialize a model or modify weights."""
import argparse
import collections
import json
import struct
from pathlib import Path


def plan(root):
    root=Path(root);index=json.loads((root/'model.safetensors.index.json').read_text())
    categories=collections.defaultdict(lambda:collections.Counter());tensor_count=0
    for shard in sorted(set(index['weight_map'].values())):
        path=root/shard
        with path.open('rb') as f:
            prefix=f.read(8)
            if len(prefix)!=8:raise ValueError('incomplete header')
            size=struct.unpack('<Q',prefix)[0]
            if not 0<size<=min(100_000_000,path.stat().st_size-8):raise ValueError('invalid header size')
            header=json.loads(f.read(size))
        for name,entry in header.items():
            if name=='__metadata__' or index['weight_map'].get(name)!=shard:continue
            begin,end=entry['data_offsets'];stored=end-begin
            if not 0<=begin<=end<=path.stat().st_size-8-size:raise ValueError('invalid tensor range')
            dtype=entry['dtype'];factor=1
            if dtype=='I8' and '.experts.' in name and name.endswith('.weight'):factor=4
            elif dtype=='F8_E4M3' and name.endswith('.weight'):factor=2
            if name.startswith('mtp.'):category='mtp'
            elif '.engram.embed.' in name:category='engram_tables'
            elif '.experts.' in name:category='routed_experts'
            elif name.startswith('layers.'):category='decoder_other'
            elif name in ('embed.weight','head.weight','norm.weight'):category='embedding_head'
            else:category='other_including_vision'
            group=categories[category];group['tensor_count']+=1;group['stored_bytes']+=stored;group['bf16_reference_bytes']+=stored*factor;tensor_count+=1
    if tensor_count!=len(index['weight_map']):raise ValueError('index coverage mismatch')
    return {'tensor_count':tensor_count,'shard_count':len(set(index['weight_map'].values())),'categories':dict(categories),'total_stored_bytes':sum(g['stored_bytes'] for g in categories.values()),'total_bf16_reference_bytes':sum(g['bf16_reference_bytes'] for g in categories.values()),'boundary':'Header-only lower-bound residency estimate: FP4 expert payload x4, FP8 weight payload x2, other tensors including scales retained. Excludes padding, replicas, activations, KV, compiler, workspace and allocator overhead. Streaming/offload can change residency; no accepted deployment layout.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=plan(args.model_dir);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
