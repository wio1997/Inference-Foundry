#!/usr/bin/env python3
"""Recalculate AIV proxy composition from the stage266 frozen aggregate."""
from pathlib import Path
from decimal import Decimal
import hashlib,json
p=Path(__file__).resolve().parent/'layer0-aic-aiv-intersection.json'
b=p.read_bytes();expected='0393893f08cae3421eb385727e8c658a769a1da9b5311dfec6de6784cc709d94'
if hashlib.sha256(b).hexdigest()!=expected:raise RuntimeError('input changed')
x=json.loads(b);rows=x['group_by_type']; mc=[v for v in rows if v['type']=='DispatchFFNCombine']
if len(mc)!=1 or mc[0]['rows']!=48:raise RuntimeError('unexpected MC2 sample')
d=sum(Decimal(v['aiv_cycle_sum']) for v in rows); n=sum(Decimal(v['aiv_vec_weighted_numerator']) for v in rows)
a=Decimal(mc[0]['aiv_cycle_sum']);q=Decimal(mc[0]['aiv_vec_weighted_numerator'])
if d!=Decimal(4508918621) or n!=Decimal('20082171.015') or d<=a or a<=0:raise RuntimeError('unexpected counts')
print(json.dumps({'input_sha256':expected,'total_cycles':str(d),'total_vector_numerator':str(n),'mc2_cycles':str(a),'mc2_vector_numerator':str(q),'other_cycles':str(d-a),'other_vector_numerator':str(n-q),'mc2_cycle_share':str(a/d),'mc2_vector_proxy':str(q/a),'other_vector_proxy':str((n-q)/(d-a)),'all_vector_proxy':str(n/d),'limits':'Cycle-weighted exporter arithmetic only, not time, real NPU activity, waiting, causality, or achievable gain.'},indent=2))
