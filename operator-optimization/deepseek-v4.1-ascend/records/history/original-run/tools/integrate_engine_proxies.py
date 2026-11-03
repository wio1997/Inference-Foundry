# -*- coding: utf-8 -*-
from pathlib import Path
from decimal import Decimal
import csv,datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def load(p):return json.loads((T/p).read_text())
def save(p,o):(T/p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
s=load('captured-layer-engine-proxies/summary.json');assert s['rank_count']==16 and len(s['layers'])==40
l=s['layers'][0];assert l['layer']==0 and l['calls']==48
for n in ['aic_mac_ratio','aic_mte2_ratio','aic_scalar_ratio','aiv_vec_ratio','aiv_mte2_ratio','aiv_mte3_ratio','aiv_scalar_ratio']:assert l['metrics'][n]['missing_with_positive_cycles']==0
rows=list(csv.DictReader((T/'captured-layer-attribution/operator_metrics.csv').open()))
mc=next(r for r in rows if r['sample_scope']=='INTERIOR_RUNS_1_2_3' and r['layer']=='0' and r['operator_type']=='DispatchFFNCombine');assert int(mc['calls'])==48
all_den=Decimal(l['metrics']['aic_mac_ratio']['denominator']);all_num=Decimal(l['metrics']['aic_mac_ratio']['numerator']);mc_den=Decimal(mc['aic_reported_cycles']);mc_ratio=Decimal(mc['reported_cycle_weighted_mac_ratio']);mc_num=mc_den*mc_ratio
assert mc_den<all_den
non_ratio=(all_num-mc_num)/(all_den-mc_den);share=mc_den/all_den
now=datetime.datetime.now(datetime.timezone.utc).isoformat();out={'timestamp':now,'status':'CAPTURED_TARGET_LAYER0_MULTI_PIPE_PROXIES','samples':48,'ranks':16,'layer0':l,'mc2_aic_cycle_share_of_attributed_layer':float(share),'mc2_aic_mac_ratio':float(mc_ratio),'non_mc2_aic_mac_ratio':float(non_ratio),'mc2_duration_sum_over_layer_windows':float(Decimal(mc['task_duration_sum_us'])/(Decimal(str(l['mean_layer_window_us']))*48)),'arithmetic':'subtract MC2 weighted numerator and cycles only for AIC MAC; source CSV ratios are rounded','comparators':[{'layer':x['layer'],'metrics':{k:x['metrics'][k]['cycle_weighted_ratio'] for k in x['metrics']},'mean_layer_window_us':x['mean_layer_window_us'],'mean_mc2_duration_us':x['mean_mc2_duration_us']} for x in s['layers'] if x['layer'] in [1,16,20]],'limits':['AIC and AIV pipe ratios have different denominators and may overlap; none is full NPU utilization, a wait percentage, or MFU.','Task interval union coverage is not proof of physical busy or idle; kernel intervals may include internal waits and unrelated streams.','MC2 cycle share and relative window-time share are within this sampled layer group, not end-to-end critical-path contribution.','Existing cross-DP arrival asymmetry supports a synchronization/arrival hypothesis, but task internals do not establish how much is communication wait versus other causes.'],'source_summary_sha256':hashlib.sha256((T/'captured-layer-engine-proxies/summary.json').read_bytes()).hexdigest()}
save('evidence/captured-layer0-engine-diagnosis.json',out)
a=load('module-analysis.json');a['supplementary_layer0_engine_diagnosis']='evidence/captured-layer0-engine-diagnosis.json';save('module-analysis.json',a)
p=load('profile-summary.json');p['supplementary_layer0_engine_diagnosis']='evidence/captured-layer0-engine-diagnosis.json';assert len(json.dumps(p).encode())<32768;save('profile-summary.json',p)
st=load('STATE.json');st.update(last_update=now,checkpoint_status='LAYER0_MULTIPIPE_DIAGNOSED',next_read_only_action='Investigate no-attention eager target metadata, Engram collective ownership and physical hardware ceiling.');save('STATE.json',st)
with (T/'REPORT.md').open('a') as f:f.write(f'''\n\n## 第 0 层：Cube 和 Vector 流水的进一步对照\n\n已归属 target 的 48 个内部采样窗口中，第 0 层 AIC MAC 计数代理为 {l['metrics']['aic_mac_ratio']['cycle_weighted_ratio']*100:.2f}%，AIV Vector 代理为 {l['metrics']['aiv_vec_ratio']['cycle_weighted_ratio']*100:.2f}%，AIV Scalar 代理为 {l['metrics']['aiv_scalar_ratio']['cycle_weighted_ratio']*100:.2f}%。不同流水的分母、覆盖和可重叠关系不同，不能相加或当成整卡利用率。窗口内已导出任务的时间并集覆盖约 {l['known_task_time_union_over_window']*100:.2f}%，但 kernel 区间内可能等待，因此不表示 NPU 一直有效工作。\n\n同一组样本中，融合的 DispatchFFNCombine 占第 0 层已归属 AIC 周期分母约 {float(share)*100:.2f}%，其 MAC 代理约 {float(mc_ratio)*100:.2f}%；其他已归属 kernel 的 MAC 代理约 {float(non_ratio)*100:.2f}%。这说明该代理的低值受到 MC2 样本显著影响，与先前观察到的跨 DP 到达不齐现象一致；但仍无法仅据此分离通信等待、工作量不均或其他内部因素，也不是端到端耗时占比。原始分子、分母和其它层对照见 `evidence/captured-layer0-engine-diagnosis.json`。\n''')
print(json.dumps({'status':out['status'],'mc2_cycle_share':float(share),'mc2_mac_proxy':float(mc_ratio),'non_mc2_mac_proxy':float(non_ratio)}))
