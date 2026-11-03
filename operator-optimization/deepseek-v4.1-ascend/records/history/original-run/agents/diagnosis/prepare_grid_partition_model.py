#!/usr/bin/env python3
"""Recompute frozen helper's logical row partition for selected shapes, conditional on 48 cores."""
import collections,hashlib,json,math,pathlib
R=pathlib.Path(__file__).resolve().parents[2]
src=R/'agents/fusion/frozen-source/vllm_ascend/ops/triton/prepare_indexer_indices.py'
data=R/'agents/diagnosis/indexer-attention-event-identity.json'
pmu=R/'agents/diagnosis/prepare-pmu-32-selected.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(data.read_text());b=json.loads(pmu.read_text());assert len(a['pairs'])==len(b['samples'])==32
joined={(x['ep'],x['scope_id']):x for x in b['samples']};assert len(joined)==32
shapes=collections.Counter()
for x in a['pairs']:
 k=(x['ep'],x['scope_id']);assert k in joined and joined[k]['compute_info']['blockNum']==48
 t=[t for t in x['middle_tasks'] if t['info']['name']=='_prepare_indexer_indices_kernel'];assert len(t)==1
 raw=t[0]['info']['inputShapes'].strip('"').split(';')[0].split(',');assert len(raw)==2
 shapes[tuple(map(int,raw))]+=1
results=[]
for (rows,topk),count in sorted(shapes.items()):
 cores=48;cols=1<<(topk-1).bit_length();max_block_rows=128*1024//(cols*4*8);aligned_rows=8//math.gcd(topk,8)
 block_rows=min(max_block_rows,1<<((rows+cores-1)//cores-1).bit_length());blocks=(rows+block_rows-1)//block_rows;aligned_blocks=(aligned_rows+block_rows-1)//block_rows
 grid=min((blocks+aligned_blocks-1)//aligned_blocks,cores);per_core=((blocks+grid-1)//grid+aligned_blocks-1)//aligned_blocks*aligned_blocks
 counts=[max(0,min((i+1)*per_core,blocks)-i*per_core) for i in range(grid)]
 assert sum(counts)==blocks and grid==48
 floor,extra=divmod(blocks,grid);balanced=[floor+int(i<extra) for i in range(grid)]
 assert sum(balanced)==blocks and max(balanced)==max(counts)
 results.append({'rows':rows,'topk':topk,'samples':count,'hypothesized_vectorcores':cores,'block_rows':block_rows,'logical_blocks':blocks,'grid':grid,'blocks_per_core':per_core,'current_blocks_per_program':dict(sorted(collections.Counter(counts).items())),'example_balanced_blocks_per_program':dict(sorted(collections.Counter(balanced).items())),'conditional_nonempty_programs':sum(v>0 for v in counts)})
obj={'source_sha256':sha(src),'selected_input_sha256':sha(data),'pmu_summary_sha256':sha(pmu),'assumptions':['Frozen Python helper actually executed for the selected tasks, get_vectorcore_num returned 48, and per-program loop follows the saved source; none is established by a reported blockNum alone.','The alternative block distribution is an arithmetic comparison, not a supported or implemented runtime schedule.'],'results':results,'limits':['Logical program assignments are not physical core utilization, timing, or proof of speedup.','Evenly distributing blocks has the same largest whole-block count; individual tail sizes, overhead, caches, waves and other work affect performance.','No proof of performance bottleneck or actual unprofiled execution.']}
out=R/'agents/diagnosis/prepare-grid-partition-model.json';out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');print(json.dumps(obj,indent=2));print('output_sha256',sha(out))
