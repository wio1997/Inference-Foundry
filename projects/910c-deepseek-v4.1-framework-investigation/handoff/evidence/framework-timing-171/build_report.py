"""Generate a human-readable entry plus machine-readable, denominator-explicit ledgers."""
from pathlib import Path
from collections import defaultdict
import ast,csv,json,statistics as st
P=Path(__file__).resolve().parent
load=lambda name:json.loads((P/name).read_text())
fine=load('function-analysis.json');raw=load('detailed-performance.json')['ranks'];host=fine['host_logs'];ref=load('balanced-reference-1.json');prof=load('balanced-steady.json');off=load('balanced-reference-0.json')
oldast=ast.parse((P.parent/'performance-breakdown-171/summarize.py').read_text())
info=ast.literal_eval(next(n.value for n in oldast.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INFO' for t in n.targets)))

def export(name,rows):
    if not rows:return
    fields=list(rows[0]);f=(P/name).open('w',encoding='utf-8-sig',newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.close()

requests=[]
for r in ref['requests']:
    requests.append({'dp':r['dp_rank'],'pair':r['pair_id'],'prompt_tokens':r['usage']['prompt_tokens'],'output_tokens':r['usage']['completion_tokens'],
      'total_ms':(r['end_mono_ns']-r['start_mono_ns'])/1e6,'ttft_ms':(r['first_token_mono_ns']-r['start_mono_ns'])/1e6,
      'decode_delivery_ms':(r['end_mono_ns']-r['first_token_mono_ns'])/1e6,'output_sha256':r['output_sha256']})
export('requests.csv',requests)

core=[];workers=[]
for p in host['processes']:
    rows=[r for r in p['rows'] if r['label']=='balanced-reference-1']
    roots=[r for r in rows if r['name'].endswith(':EngineCoreProc._process_engine_step')]
    if roots:
        root=roots[0];wait=sum(r['self_wall_ms'] for r in rows if '/future.result@' in r['name'] or '/exec_model_fut.result@' in r['name'])
        scheduler=sum(r['self_wall_ms'] for r in rows if ':Scheduler.' in r['name'])
        submit=sum(r['self_wall_ms'] for r in rows if '/self.model_executor.' in r['name'])
        other=root['duration_ms']-wait-scheduler-submit
        assert other>=0
        core.append({'pid':p['pid'],'recorded_step_ms':root['duration_ms'],'calls':root['calls'],
                     'result_wait_ms':wait,'schedule_update_ms':scheduler,'submit_ms':submit,'other_ms':other,
                     'wait_thread_cpu_ms':sum(r['thread_cpu_ms'] for r in rows if '/future.result@' in r['name']),
                     'schedule_call_median_ms':next(r['call_median_ms'] for r in rows if r['name'].endswith(':Scheduler.schedule')),
                     'functions':rows})
    if any(r['name'].endswith(':NPUModelRunner.execute_model') for r in rows):workers.append({'pid':p['pid'],'functions':rows})
core.sort(key=lambda r:r['pid']);assert len(core)==2 and len(workers)==16
export('engine-core.csv',[{k:v for k,v in r.items() if k!='functions'} for r in core])

worker_names=sorted({r['name'] for p in workers for r in p['functions']});worker_rows=[]
for name in worker_names:
    xs=[next((r for r in p['functions'] if r['name']==name),{}) for p in workers]
    out={'name':name}
    for k in ['calls','duration_ms','self_wall_ms','thread_cpu_ms','self_thread_cpu_ms']:
        out[k]=sum(r.get(k,0) for r in xs)/16
    out['call_median_ms']=st.median(r['call_median_ms'] for r in xs if r)
    worker_rows.append(out)
worker_rows.sort(key=lambda r:-r['duration_ms']);export('worker-full-request.csv',worker_rows)

operator_groups={}
for group,ranks in [('all',raw),('dp0',[r for r in raw if r['dp']==0]),('dp1',[r for r in raw if r['dp']==1])]:
    names=sorted({r['name'] for rank in ranks for r in rank['operators']});ops=[];window=st.mean(r['window_ms'] for r in ranks)
    for name in names:
        xs=[next((r for r in rank['operators'] if r['name']==name),{}) for rank in ranks];values={k:sum(r.get(k,0) for r in xs)/len(ranks) for k in ['calls','union_ms','wall_sum_ms','target_graph_union_ms']}
        values['window_pct']=values['union_ms']/window*100
        values['call_median_us']=st.median(r['call_median_us'] for r in xs if r)
        label,module,meaning,next_step=info.get(name,(name,'调用点待绑定','保留全部shape与设备事件。','先核调用点与交付链，再估净收益。'))
        ops.append({'name':name,'label':label,'module':module,'meaning':meaning,'next':next_step,'kind':next(r['kind'] for r in xs if r),**values,
          'shape_cases':[{'ep':rank['ep'],**case} for rank,x in zip(ranks,xs) for case in x.get('shape_cases',[])]})
    operator_groups[group]={'window_ms':window,'operators':sorted(ops,key=lambda r:-r['union_ms'])}
export('operators.csv',[{k:v for k,v in r.items() if k!='shape_cases'} for r in operator_groups['all']['operators']])

for group in fine['groups']:
    for r in fine['groups'][group]['functions']:
        source=host['catalog'].get(r['name'],{})
        r['source']=source.get('path');r['line']=source.get('line');r['end_line']=source.get('end_line')
export('functions.csv',fine['groups']['all']['functions'])
for group in fine['groups'].values():group['activity_boundary_residual_ms']=group['window_ms']-sum(group['activity_ms'].values())
view={'schema':1,'request_seconds':ref['batch_seconds'],'tokens_s':ref['output_tokens_per_second'],'requests':requests,
      'capture_seconds':prof['batch_seconds'],'probe_off_seconds':off['batch_seconds'],'historical_uninstrumented_seconds':16.090767783,
      'baseline_note':'本轮coarse诊断与既有真正无探针参考分列；off仍有控制文件轮询，20.163s不作无探针基线。单轮接近不证明零开销。',
      'source_details':load('source-details.json'),'groups':fine['groups'],'core':core,'workers':worker_rows,'operators':operator_groups,'catalog':host['catalog'],
      'checks':{'reference':ref['balance_checks'],'profile':prof['balance_checks']},'rank_count':16,
      'coverage':{'prepared_functions':97,'loaded_function_definitions':len(host['catalog']),'executed_trace_function_names':len(fine['groups']['all']['functions']),
                  'logged_processes':len(host['processes']),'engine_core_processes':2,'worker_processes':16,
                  'unmeasured':['API预处理的本次逐函数拆分','跨进程request/batch代次自动因果关联','DFC内部设备阶段','内核编译代码内部等待原因','off-CPU细因如GIL/内核调度的独立证据']}}
export('clock-alignment.csv',[{'ep':r['ep'],'matched':r['body_marker_matched'],'unmatched':r['body_marker_unmatched'],**r['clock_alignment']} for r in fine['ranks']])
export('probe-costs.csv',[{'ep':r['ep'],'window_ms':r['window_ms'],'prepare_free_ms':r['prepare_free_ms'],'added_marker_prepare_free_ms':r['added_marker_prepare_free_ms'],'function_assigned_prepare_free_ms':sum(v for k,v in r['prepare_free_function_partition_ms'].items() if k not in ['added_marker_boundary','unmarked_prepare_self']),'unmarked_prepare_free_ms':r['prepare_free_function_partition_ms']['unmarked_prepare_self']} for r in fine['ranks']])
(P/'view-data.json').write_text(json.dumps(view,ensure_ascii=False,indent=2))

A=fine['groups']['all'];fn=[r for r in A['functions'] if r['prepare_self_free_ms']>0];top=fn[:10]
lines=['# 171：从请求到函数、算子的性能诊断','',f"本轮轻量记录：{ref['batch_seconds']:.6f}s，{ref['output_tokens_per_second']:.2f}输出token/s；8条请求、DP各4条、32K/1K、prefix口径与输出哈希核验通过。详细profile为{prof['batch_seconds']:.6f}s，不作生产速度。",'',
'完整性入口：[完整timeline逻辑](../../tools/framework-timing/TIMELINE.md) · [本轮覆盖账与明确缺口](COVERAGE.md)。','',
'## 结论：应先优化哪里','',f"详细稳态同窗：每卡平均{A['window_ms']:.3f}ms，计算{A['activity_ms']['Computing']:.3f}ms，设备空档{A['activity_ms']['Free']:.3f}ms，暴露通信{A['activity_ms']['Communication(Not Overlapped)']:.3f}ms。输入准备对应空档{A['prepare_free_ms']:.3f}ms。",'',
'|具体函数，自身区间计一次|与输入准备空档重叠ms/卡|占输入准备空档|入口|','|---|---:|---:|---|']
for r in top:lines.append(f"|{r['name'].split(':')[-1]}|{r['prepare_self_free_ms']:.3f}|{r['prepare_free_pct']:.2f}%|{r['source']}:{r['line']}|")
lines+=['',f"旧样本155.35ms无法在原日志恢复；本轮重新测得无cpu_op覆盖部分{A['prepare_without_cpu_op_ms']:.3f}ms，并按新增函数边界归属。两次采样绝对量分列，不能声称逐字节补回原155ms。函数自身时间仍可含Python、未标记的库调用和等待，不能等同纯Python执行。",'',
f"新增FT标记边界与输入准备空档的交集为{A['added_marker_prepare_free_ms']:.3f}ms/卡，已从具体函数自身贡献中单列扣除。该扣除不能回放不存在的无探针时序，仍不当生产收益。",'',
f"设备三类活动与窗口的分类边界残差为{A['activity_boundary_residual_ms']:.3f}ms/卡，单列不命名成额外调度工作。",'',
'## 完整请求的调度器与worker','',
'EngineCore结果等待与worker/设备执行并行。这一等待是等下游完成，不是独立新增的同等调度开销；线程CPU接近墙钟时也可能是在轮询。全部数字见engine-core.csv，子表可加总到各自记录到的step范围。']
for i,c in enumerate(core):lines.append(f"- EngineCore进程{c['pid']}：step {c['recorded_step_ms']:.3f}ms = 等结果{c['result_wait_ms']:.3f} + 调度/更新{c['schedule_update_ms']:.3f} + 提交{c['submit_ms']:.3f} + 其余{c['other_ms']:.3f}。调度单次中位{c['schedule_call_median_ms']:.4f}ms。")
lines+=['','worker全请求表是16卡平均Host墙钟，可含异步设备提交与等待。包含关系不相加；不能将_model_forward的Host提交时间当整图设备执行时间。','',
'## 设备热点：同一次新采样','', '|算子|任务区间ms/卡|占新稳态窗口|','|---|---:|---:|']
for o in operator_groups['all']['operators'][:14]:lines.append(f"|{o['label']} / {o['name']}|{o['union_ms']:.3f}|{o['window_pct']:.2f}%|")
lines+=['','各算子有并行重叠，类型时长不能直接相加；Computing/Free/暴露通信按设备活动并集，通信任务区间另列。全量shape/dtype/format和图内分属见view-data.json。','',
'## 调优价值与下一步','',
'1. 输入/metadata：以真实CP构造和共同frame大段为首要Host候选；从最大重复计算与批量交付判断数学共享、融合与调度，净省扣掉替代工作、栅栏、存储与争用。停止单个微小Cast/where的独立研究。',
'2. 目标DFC：设备最大单项，拆权重读取、GMM、通信和同步；设备占比是成本上限而非可实现收益。',
'3. Attention/mHC/矩阵乘：按完整shape与真实读者绑定大段，考虑数学重写、布局贯穿及megakernel；通用Type不是单一开发任务。',
'所有新优化仍待确认。未做性能改写，不宣称吞吐已改善。两倍吞吐须在同请求交付边界将总时间减半，不能靠加和不同采样或局部比例证明。','',
'## 测量边界与复用','',
view['baseline_note'],
'细函数与profiler时钟按长函数包围关系校准，16卡约36–42ppm漂移；最大所需包围残差2.012µs。FT范围成功匹配数量与未匹配数量按逐卡CSV保留。标记边界扣除是有时钟误差的诊断估算，不能重放无探针时序。见clock-alignment.csv与probe-costs.csv。',
'对照既有无探针批次，8组prompt SHA与完整输出SHA一致；这是本负载结果核验，不是所有分支的数值证明。见historical-output-check.json。',
'详细标记与profiler均会扰动Host。本轮报告是定位依据；任何收益需要正确性及真正无探针端到端复测。',
'API前处理的本次逐函数、跨进程自动batch关联、DFC内部、off-CPU具体原因均未完整测出，保留为明确缺口。不是整个框架已经无未知。',
'原始spans与16份trace保存在launch.json指向的171自有目录；源码SHA、全部标记、控制器/恢复回执、函数和算子CSV均保留。',
'复用入口：[通用方法与适配契约](../../tools/framework-timing/README.md)。SGLang/DeepInfer需绑定实际版本入口后验证，不照搬本框架数值。','']
(P/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({'window_ms':A['window_ms'],'top_functions':[(r['name'].split(':')[-1],r['prepare_self_free_ms']) for r in top[:8]],'operators':len(operator_groups['all']['operators'])}))
