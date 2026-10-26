#!/usr/bin/env python3
"""Audit Run132 W8A8 apply call probe and classify target versus draft."""
import glob,json,collections
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry/evidence/20260925_loop040_quant/run132')
files=sorted(glob.glob(str(ROOT/'calls/*.jsonl')))
assert len(files)==8
rows=[]
for f in files:
 a=[json.loads(line) for line in open(f)]
 by_prefix=collections.Counter(x['prefix'] for x in a)
 target={k:v for k,v in by_prefix.items() if k.startswith('model.layers.')}
 mtp={k:v for k,v in by_prefix.items() if k.startswith('mtp.')}
 assert len(target)==43 and set(target.values())=={2}
 assert len(mtp)==3 and set(mtp.values())=={271}
 assert len(a)==899
 assert {tuple(x['x_shape']) for x in a}=={(96,4096)}
 assert {tuple(x['weight_shape']) for x in a}=={(4096,512)}
 rows.append({'file':f,'pid':a[0]['pid'],'records':len(a),
              'target_layer_prefixes':len(target),'target_calls_per_prefix':2,
              'mtp_prefixes':len(mtp),'mtp_calls_per_prefix':271,
              'x_shape':[96,4096],'weight_shape':[4096,512],
              'unique_x_ptr_count':len({x['x_ptr'] for x in a}),
              'unique_q_ptr_count':len({x['q_ptr'] for x in a})})
bench=json.loads((ROOT/'bench.json').read_text())['summary']
assert bench['success']==12 and bench['fail']==0 and bench['max_tokens']==1024
restore=json.loads((ROOT/'restore.json').read_text())
assert restore['restored_sha256']=='ac9bd6ec05356b639665ec05ae21b19e1d0f0dc6723f7568b2f0ab70821285de'
out={'run':'run132','status':'valid_partial_call_capture','contract':bench,
     'rank_count':8,'per_rank':rows,
     'interpretation':'Probe of AscendW8A8DynamicLinearMethod.apply captures only self_attn.wkv: 43 target layers twice during graph setup, and three MTP layer prefixes 271 times during run. It does not cover other quant-matmul direct calls. Cross-layer pointer reuse is graph-memory reuse, not semantic same-input proof.',
     'limitations':['Diagnostic bench, not formal E2E.','The probe filters first dimension 96, so other linear shapes are not represented.','MTP repeated Python calls do not by themselves quantify removable proposer time.']}
(ROOT/'call_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'rank_count':8,'records_per_rank':899,'target_prefixes':43,'mtp_prefixes':3,'bench_success':12},indent=2))
