#!/usr/bin/env python3
"""Exact conditional INT8 throughput arithmetic; does not validate premises."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

here=Path(__file__).resolve().parent
run=here.parent.parent
files={
 'calculator':run/'sources/primary/msprof-analyze/chip_peak_flops.py',
 'profile':run/'evidence/profile-device-capacity-info.json',
 'actual_sample':run/'evidence/conditional-performance-comparisons.json',
 'matrix_case':run/'bounds/Wq_b_rank0_M3.input.json',
 'capacity_review':run/'evidence/conditional-nominal-capacity.json',
}
code=files['calculator'].read_text()
assert re.search(r'DataType\.INT8:\s*16\s*\*\s*32\s*\*\s*16\s*\*\s*2',code)
info=json.loads(files['profile'].read_text())['device_info']
devices=[entry['device_info'][0] for entry in info]
assert len(devices)==16 and len({d['id'] for d in devices})==16
assert all(d['ai_core_num']==24 and d['aic_frequency']=='1800' for d in devices)
per_cycle_ops=16*32*16*2
ops=24*1800*1_000_000*per_cycle_ops
mac=ops//2
F=3*1280*32768
t=Fraction(F,mac)
rows=Fraction(3,1)/t
report={
 'scope':'Hypothetical standard dense full-output M3 x K1280 x N32768 Wq_b, one modeled device, all counted work uses modeled INT8 Cube pool.',
 'estimator_int8_ops_per_core_cycle':per_cycle_ops,
 'profile_ai_core_num':24,'profile_aic_frequency_field_mhz':1800,
 'conditional_estimator_ops_per_s':ops,
 'conditional_estimator_mac_per_s':mac,
 'conditional_estimator_16_device_mac_per_s':16*mac,
 'independence_not_necessary_for_summing_valid_per_device_upper_bounds':True,
 'example_MAC_count':F,
 'example_time_lower_fraction_seconds':str(t),
 'example_time_lower_seconds_approx':float(t),
 'example_output_row_per_second_upper_fraction':str(rows),
 'example_output_row_per_second_upper_approx':float(rows),
 'physical_ceiling_proven':False,
 'observed_sample_membership_in_domain_proven':False,
 'model_output_token_tps_conclusion':'UNKNOWN',
 'assumptions':['Applicable maximum INT8 OPS/cycle on 9382','Hard AIC frequency <=1800 MHz','All counted MAC use modeled Cube pool','Standard dense full-output MxKxN arithmetic is mandatory','No equivalent bypass or work sharing'],
 'sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},
 'no_new_device_work':True,
}
(here/'int8_ceiling_certificate.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
