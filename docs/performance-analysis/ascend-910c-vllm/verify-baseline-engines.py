#!/usr/bin/env python3
"""Check reported engine histogram arithmetic, not underlying request binding."""
import json
from decimal import Decimal
from pathlib import Path

data=json.loads((Path(__file__).parent/'baseline-engine-histograms.json').read_text())
for phase,row in data['rounds'].items():
    assert row['counts_per_engine']==[4,4]
    ttft=list(map(Decimal,row['means']['time_to_first_token_seconds']))
    prefill=list(map(Decimal,row['means']['request_prefill_time_seconds']))
    decode=list(map(Decimal,row['means']['request_decode_time_seconds']))
    assert row['ttft_bucket_deltas']=={'0.5':[0,2],'0.75':[1,2],'1.0':[3,4],'2.5':[4,4]}
    print(phase,'TTFT means ms:',*[round(x*1000,3) for x in ttft],'; delta ms:',round((ttft[0]-ttft[1])*1000,3))
    print('  prefill means ms:',*[round(x*1000,3) for x in prefill],'; decode means s:',*[round(x,3) for x in decode])
print('Aggregated histogram observations only: not matched per-request or causal evidence.')
