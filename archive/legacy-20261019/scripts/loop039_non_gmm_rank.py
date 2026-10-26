#!/usr/bin/env python3
"""Rank stable non-GMM target kernel families from Run107 diagnostic windows."""
import json
import statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
trace=json.loads((ROOT/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in trace['excluded_windows']}
windows=[c for r in trace['ranks'] for c in r['cycles'] if (r['rank'],c['cycle']) not in excluded]
assert len(windows)==15
by_name=defaultdict(list)
for w in windows:
 for name,ms in w['top_kernel_sums_ms']:
  by_name[name].append(ms)
families=[]
for name,x in by_name.items():
 if len(x)!=15 or name.startswith('hcom_') or 'GroupedMatmul' in name:
  continue
 families.append({'kernel':name,'median_ms':statistics.median(x),
                  'min_ms':min(x),'max_ms':max(x),
                  'range_over_median':(max(x)-min(x))/statistics.median(x)})
families.sort(key=lambda x:-x['median_ms'])
source_map={
 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3':'multiple quantized Linear projections; exact per-layer calls unresolved',
 'Compressor':'DeepSeek V4 compressor path; vllm_ascend/models/deepseek_v4.py Compressor',
 'HcPre':'DeepSeek V4 hyper-connection pre; deepseek_v4.py hc_pre calls npu_hc_pre_v2 twice per layer',
 'aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore_ScatterNdUpdateSk':'cache writes; device/device_op.py npu_scatter_nd_update_sk calls',
 'VllmQuantLightningIndexer':'DSA indexer; device/device_op.py quant lightning indexer',
 'SparseAttnSharedkv':'DSA attention; device/device_op.py sparse attention',
}
for item in families:
 item['source_mapping']=source_map.get(item['kernel'],'not yet resolved')
out={
 'run':'run123','source':'Run107 15 valid synchronized target windows',
 'method':'Top 15 kernel-duration families per window; require appearance in all 15; report per-family sums, not union or removable benefit.',
 'non_gmm_compute_union_lower_bound_median_ms':29.97925,
 'families':families,
 'decision':'Prioritize a concrete target-stage intervention only after separating critical-path overlap and validating semantics. No single non-GMM family exceeds 5ms; Compressor+HcPre total is ~6.3ms kernel sum but are distinct mandatory model operations, not jointly removable by default.',
 'limits':['Top-15 truncates smaller families.','Profiled synchronized cycles, not formal E2E.','Source mapping is at Python call-family level, not a unique kernel-to-call proof.']}
p=ROOT/'evidence/20260925_loop039_gmm/run123'
p.mkdir(parents=True,exist_ok=True)
(p/'non_gmm_rank.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(families[:9],indent=2))
