#!/usr/bin/env python3
"""Check documented formula numerically against six archived EP8 rows; no device calls."""
from pathlib import Path
from decimal import Decimal
import hashlib,json
root=Path(__file__).resolve().parents[2]
source=root/'agents/diagnosis/engram-matmul-metrics.json'
capacity=root/'evidence/profile-device-capacity-info.json'
j=json.loads(source.read_text()); cap=json.loads(capacity.read_text())
infos=[z for x in cap['device_info'] for z in x['device_info'] if z.get('id')==8]
assert len(infos)==1
freq=Decimal(infos[0]['aic_frequency']); cores=Decimal(infos[0]['ai_core_num'])
assert freq==1800 and cores==24 and len(j['samples'])==6
D=Decimal
rows=[]
for s in j['samples']:
 c=s['csv_fields']; assert c['Input Shapes']=='"3,6144;25600,6144"' and c['Input Data Types']=='DT_BF16;DT_BF16'
 dur=D(c['Duration(us)']);cyc=D(c['aic_total_cycles']);aic=D(c['aicore_time(us)']);mac=D(c['aic_mac_time(us)']);rep=D(c['cube_utilization(%)']);ratio=D(c['aic_mac_ratio']);blocks=D(c['Block Num'])
 calc=D(100)*cyc/(freq*cores*dur); mac_per=mac/aic; aic_ideal=cyc/(freq*blocks)
 rows.append({'run':s['target']['run'],'after_layer':s['target']['after_layer'],'global_task_id':s['task']['globalTaskId'],
   'duration_us':str(dur),'total_aic_cycles':str(cyc),'block_num':str(blocks),
   'reported_cube_pct':str(rep),'calculated_cube_pct':str(calc),'cube_difference_pct_points':str(calc-rep),
   'reported_aicore_us':str(aic),'calculated_aicore_us':str(aic_ideal),
   'reported_mac_us':str(mac),'reported_mac_ratio':str(ratio),'mac_time_div_aicore_time':str(mac_per)})
assert all(abs(D(r['cube_difference_pct_points'])) < D('0.00051') for r in rows)
out={'input_sha256':{'engram_matmul_metrics':hashlib.sha256(source.read_bytes()).hexdigest(),'device_capacity_info':hashlib.sha256(capacity.read_bytes()).hexdigest()},
 'document_url':'https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/devaids/Profiling/atlasprofiling_16_0067.html',
 'document_version':'CANNCommunityEdition 910beta2, op_summary PipeUtilization',
 'scope':'EP8 six recorded BF16 MatMulV2 rows; arithmetic consistency with separately versioned documentation, not proof of exporter implementation',
 'freq_mhz_from_device8_metadata':str(freq),'ai_core_num_from_device8_metadata':str(cores),
 'candidate_formula':'cube_pct=100*aic_total_cycles/(frequency_MHz*ai_core_num*task_duration_us)',
 'notes':['1 MHz = 1 cycle/us','aicore time is documented as theoretical equal-block scheduling, not physical wall time',
 'ratios and units differ; matching formula does not establish physical FLOPS utilization, speedup, or exact runtime producer version',
 'pipeline counters can overlap; do not add or subtract','frequency metadata is not proved to be hardware maximum'],
 'samples':rows,'no_npu_calls':True}
p=root/'agents/diagnosis/engram-cube-mac-formula-check.json'
if p.exists(): raise SystemExit('STOP_OUTPUT_ALREADY_EXISTS')
p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'OK','output':str(p),'rows':len(rows),'max_cube_error_pct_points':str(max(abs(D(x['cube_difference_pct_points'])) for x in rows))}))
