#!/usr/bin/env python3
"""Map ScatterNdUpdateSk placement by target-layer GMM markers, offline."""
import glob, json, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
ratios=json.loads(Path('/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json').read_text())['compress_ratios'][:43]
scatter='aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore_ScatterNdUpdateSk'
gmm='aclnnGroupedMatmulSwigluQuantWeightNzV2_GroupedMatmulSwigluQuantV2_GroupedMatmulSwigluQuantV2'
rows=[]
for rank in range(8):
    paths=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    assert len(paths)==1
    events=json.loads(Path(paths[0]).read_text())
    scopes=sorted((e for e in events if e.get('cat')=='cpu_op' and e.get('name')=='extreme::target'),key=lambda e:float(e['ts']))
    assert len(scopes)==2
    for cycle,scope in zip((64,65),scopes):
        if (rank,cycle) in excluded:continue
        a=float(scope['ts']);b=a+float(scope['dur'])
        g=sorted((float(e['ts']) for e in events if e.get('name')==gmm and a<=float(e['ts'])<b))
        s=sorted((e for e in events if e.get('name')==scatter and a<=float(e['ts'])<b),key=lambda e:float(e['ts']))
        assert len(g)==43 and len(s)==126,(rank,cycle,len(g),len(s))
        cuts=[a]+g+[b]
        bins=[[e for e in s if cuts[i]<=float(e['ts'])<cuts[i+1]] for i in range(44)]
        rows.append({'rank':rank,'cycle':cycle,'pre_gmm_scatter_counts':[len(v) for v in bins],
                     'pre_gmm_scatter_ms':[sum(float(e['dur']) for e in v)/1000 for v in bins]})
assert len(rows)==15
patterns={tuple(r['pre_gmm_scatter_counts']) for r in rows}
assert len(patterns)==1,patterns
pattern=list(patterns.pop())
out={'run':'run135','source':'Run107 15 valid target scopes; ordered scatter kernels binned before each of 43 GMM1 layer markers',
     'ratios':ratios,'scatter_counts_before_layer_gmm':pattern,'scatter_per_target':126,
     'median_scatter_kernel_sum_ms':statistics.median(sum(r['pre_gmm_scatter_ms']) for r in rows),
     'source_call_sites':{'swa_cache':'attention/context_parallel/dsa_cp.py writes SWA KV via DeviceOperator.dsa_kv_compress_scatter in every layer',
                          'compressor_cache':'attention/context_parallel/dsa_cp.py writes compressed KV when compress_ratio>1',
                          'indexer_cache':'attention/context_parallel/dsa_cp.py compress_ratio==4 path writes indexer K and scale via two DeviceOperator scatters'},
     'interpretation':'Observed exact pattern 1,1, then 4 at c4 and 2 at c128 matches one SWA write per layer, one compressed KV write for all ratio>1 layers, and two indexer writes at c4. This count/source correspondence does not establish cache tensor addresses, but offers no duplicate-write candidate.',
     'caveat':'GMM marker bins give ordering only; asynchronous streams and graph scheduling may move a scatter across GMM boundary. Count and source suggest distinct cache tensors, but per-kernel address/slot parity is not in trace_view. Disabling any write needs same-state 8-rank KV/DSpark differential gate.',
     'windows':rows}
p=root/'evidence/20260925_loop041_hc_copy/run135/cache_scatter_map.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'pattern':pattern,'median_ms':out['median_scatter_kernel_sum_ms']},indent=2))
