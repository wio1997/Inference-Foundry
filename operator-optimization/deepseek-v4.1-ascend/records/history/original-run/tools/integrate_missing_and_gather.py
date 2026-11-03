# -*- coding: utf-8 -*-
from pathlib import Path
import json,datetime,hashlib
T=Path(__file__).resolve().parents[1]
def load(n):return json.loads((T/n).read_text())
def save(n,x):(T/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
a=load('evidence/eager-no-attention-patterns.json');z=load('evidence/eager-empty-query-example.json');g=load('evidence/engram-gather-candidate-example.json');e=load('evidence/captured-layer0-engine-diagnosis.json');f=load('evidence/frequency-field-review.json')
assert a['windows']==720 and sorted(x['windows'] for x in a['groups'])==[320,400]
assert z['task']['task_id']==44214 and z['status']=='EXACT_ZERO_FILL_CALL_CHAIN_NOT_QUERY_LENGTH_PROOF'
assert g['status']=='LOOKUP_GATHER_CANDIDATE_NOT_PROVEN_DATA_DEPENDENCY' and g['communication_op'][3]==4915200 and g['device_gap_ns']==960
assert abs(e['mc2_aic_cycle_share_of_attributed_layer']-0.8173185160471488)<1e-12 and f['status']=='FREQUENCY_FIELD_NOT_PROVEN_LIVE_OR_HARD_MAX'
now=datetime.datetime.now(datetime.timezone.utc).isoformat();r=load('records.json');ca=next(x for x in r['module_causes'] if x['id']=='C_ARRIVAL_IMBALANCE')
ca['supporting_evidence'].append('evidence/captured-layer0-engine-diagnosis.json: within 48 captured layer0 calls, MC2 contributes 81.73% of attributed AIC-cycle denominator and has a 2.49% MAC proxy; this describes the metric composition, not a measured causal wait.')
ca['missing_evidence'].append('MC2 internal time decomposition, device-wide busy time and exposed critical path are not recorded; per-pipe counters overlap and cannot isolate communication waiting.')
save('records.json',r)
s=load('STATE.json');s.update(last_update=now,checkpoint_status='LAYER0_MULTIPIPE_AND_TRACE_GAPS_REVIEWED',next_read_only_action='Investigate remaining exact metadata/collective ownership or complete the explicit evidence-gap report; 720 attention branches and physical peak remain unknown.');save('STATE.json',s)
out={'timestamp':now,'status':'INTEGRATED_PARTIAL_PROFILE_EVIDENCE','evidence_files':{n:hashlib.sha256((T/n).read_bytes()).hexdigest() for n in ['evidence/eager-no-attention-patterns.json','evidence/eager-empty-query-example.json','evidence/engram-gather-candidate-example.json','evidence/captured-layer0-engine-diagnosis.json','evidence/frequency-field-review.json']},'limits':['No measured query length for the 720 missing-attention windows','One exact zero-fill API chain does not identify the branch','One gather/lookup pair is a candidate without tensor identity or group mapping','Layer0 multi-pipeline ratios are proxies with overlapping timing and separate denominators','Profiler frequency field is not proven as live or hard maximum'],'no_device_work':True}
save('evidence/missing-gather-integration.json',out)
with (T/'REPORT.md').open('a') as f:f.write('''\n\n## 未见 attention 的窗口和 Engram 通信候选\n\n720 个未见 attention 的 eager target 窗口中，按同一 model/stream 的第一 attention 子窗口扫描，400 个出现 `ZerosLike` 且形状末尾为 `[64,512]`，320 个没有该形状；两组均未看到指定输出形状的 QuantBatchMatmulV3。一个补零样本可用连接 ID 精确追到 `vllm::dsa_v41_forward` 的 CPU 调用，但调用的输入形状未记录。现象与源码中的空 query 或补齐分支相容，不能据此确定任一窗口的 query 长度或将 720 个窗口归为同一原因。见 `evidence/eager-no-attention-patterns.json` 和 `evidence/eager-empty-query-example.json`。\n\n一个 Engram host-UVA lookup 样本与随后同线程 all-gather 的启动链均可各自精确追踪；通信元素数与 lookup 输出形状一致，设备时间相隔 0.960 微秒。它是值得继续核对的候选，但没有 tensor ID 和通信组与 TP 组的映射，不能证明二者直接依赖，也不能把时间差当作等待或可掩盖时长。见 `evidence/engram-gather-candidate-example.json`。\n''')
print(json.dumps({'status':out['status'],'missing_window_patterns':[400,320],'engram_gather':'CANDIDATE_ONLY','hard_cap':'UNKNOWN'}))
