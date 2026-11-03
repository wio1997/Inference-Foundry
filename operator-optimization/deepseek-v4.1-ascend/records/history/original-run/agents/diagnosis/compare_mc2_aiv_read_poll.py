#!/usr/bin/env python3
"""Offline arithmetic on the 16 already-matched task rows; no NPU, no remote access."""
import json, hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[2]
src=p/'diagnostic-20261029/mc2-extra-pipe-fields-16-tasks.json'
x=json.loads(src.read_text())
r={(q['scope']['ep'],q['scope']['layer']):q['row'] for q in x['rows']}
assert len(x['rows'])==16 and len(r)==16
fields=['Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Accelerator Core','Block Num','Mix Block Num']
print('input_sha256',hashlib.sha256(src.read_bytes()).hexdigest())
print('EP duration_L14/L39 aiv_mte2_field_div1000_L14 aiv_mte2_field_div1000_L39 mte2_L14/L39 mte3_L14/L39')
for ep in range(8,16):
 a,b=r[ep,14],r[ep,39]
 assert all(a[k]==b[k] for k in fields)
 def v(row,k):
  z=float(row[k]); assert z>0; return z
 print(ep,round(v(a,'Duration(us)')/v(b,'Duration(us)'),3),round(v(a,'aiv_mte2_time(us)')/1000,3),round(v(b,'aiv_mte2_time(us)')/1000,3),round(v(a,'aiv_mte2_time(us)')/v(b,'aiv_mte2_time(us)'),3),round(v(a,'aiv_mte3_time(us)')/v(b,'aiv_mte3_time(us)'),3))
