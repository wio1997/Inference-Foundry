#!/usr/bin/env python3
"""Extract bounded evidence from saved CPU-side files, never access hardware."""
import hashlib
import json
from pathlib import Path
import re

here=Path(__file__).resolve().parent
run=here.parent.parent
project=run.parents[4]
files={
 'profile':run/'evidence/profile-device-capacity-info.json',
 'sdk':run/'evidence/actual-sku-and-cp.json',
 'estimator':run/'sources/primary/msprof-analyze/chip_peak_flops.py',
 'board_query':project/'analysis/operator-ceiling-20261028/model-source-inventory.json',
 'precision_report':project/'evidence/precision-paths-20261028/REPORT.md',
}
binary=project/'evidence/precision-paths-20261028/binaries'
variants={
 'int8':('QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.o.asm','MMAD.s8'),
 'int4':('QuantBatchMatmulV3_ND_ND_int4_int4_bf16_high_performance.o.asm','MMAD.s4'),
 'bf16':('MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.o.asm','MMAD.bf162f32'),
 'fp16':('MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance.o.asm','MMAD.f162f32'),
 'fp32':('MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance.o.asm','MMAD.f322f32'),
}
profiles=json.loads(files['profile'].read_text())['device_info']
devs=[v['device_info'][0] for v in profiles]
assert len(devs)==16 and len({d['id'] for d in devs})==16
assert all(d['ai_core_num']==24 and d['aiv_num']==48 and d['aic_frequency']=='1800' for d in devs)
queries=json.loads(files['board_query'].read_text())['hardware_queries']
assert re.search(r'NPU Name\s*:\s*9382',queries['board0']['stdout'])
assert re.search(r'HBM Capacity\(MB\)\s*:\s*65536',queries['memory0']['stdout'])
sdk=json.loads(files['sdk'].read_text())['Ascend910_9382.ini']['text']
for x in ('SoC_version=Ascend910_9382','cube_freq=1800','ai_core_cnt=24','vector_core_cnt=48'):
 assert x in sdk
est=files['estimator'].read_text()
assert 'DataType.FLOAT16: 16 * 16 * 16 * 2' in est
assert 'DataType.INT8: 16 * 32 * 16 * 2' in est
ops={'FLOAT16':24*1800*1_000_000*16*16*16*2,'INT8':24*1800*1_000_000*16*32*16*2}
instructions={}
for name,(filename,instr) in variants.items():
 path=binary/filename
 matching=[line.strip() for line in path.read_text(errors='replace').splitlines() if instr in line]
 assert matching,filename
 instructions[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'first_match':matching[0], 'evidence':'installed binary static disassembly only, not runtime path or throughput proof'}
out={
 'actual_board_direct_query_count':1,
 'actual_board0_name_observed':'9382',
 'profile_logical_device_count':len(devs),
 'each_profile_metadata':{'aic_count':24,'aiv_count':48,'aic_frequency_field_mhz':1800},
 'sku_runtime_to_ini_exact_mapping_proven':False,
 'conditional_estimator_ops_per_sec':ops,
 'estimator_conditional_only':True,
 'hardware_unconditional_max_proven':False,
 'other_dtype_hard_throughput_proven':False,
 'instructions':instructions,
 'input_hashes':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in files.items()},
 'no_hardware_access':True,
}
(here/'hardware_sources_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='instructions'},ensure_ascii=False,indent=2))
