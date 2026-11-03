#!/usr/bin/env python3
"""Recompute exact-operator, exact-physical-signature summaries from saved capture.

Uses the existing integrity-checking group extractor. All activity is local CPU.
"""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
extractor = root/'agents/diagnosis/analyze_recapture_all_kernel_groups.py'
raw = subprocess.check_output(['python3', str(extractor), '--all-groups'], cwd=root)
x = json.loads(raw)
assert x['source_sha256']=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert x['aic_eligible_group_count']==1596 and x['aiv_eligible_group_count']==3966
names = {'aic':['DispatchFFNCombine','aclnnMatmul_MatMulCommon_MatMulV2'],
         'aiv':['aclnnInplaceCopy_TensorMoveAiCore_TensorMove']}
def fingerprint(g):
    return json.dumps({k:g[k] for k in ('csv_type','core','physical_shape_dtype_format','per_rank_task_count')},sort_keys=True,separators=(',',':'))
def point(g):
    return {'layer':g['layer'], 'weighted_displayed_ratio':g['weighted_displayed_ratio']}
result=[]
for engine, wanted in names.items():
    entries=x['all_'+engine+'_eligible_groups']
    for digest in sorted({g['digest'] for g in entries}):
        for name in wanted:
            groups=[g for g in entries if g['digest']==digest and g['operator_name']==name]
            if not groups:continue
            counts=Counter(map(fingerprint,groups))
            # Deterministic tie-breaker, no assumption about what each signature represents.
            chosen=min(counts,key=lambda s:(-counts[s],s))
            same=[g for g in groups if fingerprint(g)==chosen]
            ordered=sorted(same,key=lambda g:(Decimal(g['weighted_displayed_ratio']),g['layer']))
            min_any=min(groups,key=lambda g:(Decimal(g['weighted_displayed_ratio']),g['layer'],fingerprint(g)))
            result.append({'engine':engine,'digest':digest,'operator_name':name,
                'eligible_groups_all_physical_signatures':len(groups),'distinct_physical_signatures':len(counts),
                'dominant_signature':json.loads(chosen),'dominant_signature_groups':len(same),
                'dominant_signature_min':point(ordered[0]),'dominant_signature_max':point(ordered[-1]),
                'minimum_across_signatures':point(min_any),'minimum_signature':json.loads(fingerprint(min_any))})
out={'source_sha256':x['source_sha256'],'extractor_sha256':sha256(extractor.read_bytes()).hexdigest(),
     'scope':'Per digest and raw operator name, eligible 8-EP physically matched groups only; layer groups use recorded physical shape, dtype, format, core and task count. Effective work and actual logical semantics are not bound.',
     'rows':result}
print(json.dumps(out,ensure_ascii=False,indent=2))
