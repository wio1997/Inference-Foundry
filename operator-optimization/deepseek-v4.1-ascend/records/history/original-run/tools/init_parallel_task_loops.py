"""Create three isolated Task/Loop records for the resumed diagnosis."""
from pathlib import Path
import json,subprocess,sys
T=Path(__file__).resolve().parents[1];S=Path('/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py');root=T/'task-loop-runs';src=T/'sources/vllm-workspace/vllm-ascend'
items=[
 {'id':'fusion-coverage','agent':'fusion_work','title':'融合机会与数据依赖覆盖','operator':'DeepSeek V4.1 fusion boundaries','shape':[3,5120],'dtype':'INT8/BF16 mixed','layout':'source-specific','hypothesis':'已冻结源码中可辨认尚未建图的依赖和消费者，能够拓展或否定特定融合机会','success':'记录至少一个具体源码边界及其全部已知消费者，或提供可审查的缺口证据','falsification':'源码或现有证据不足以判定边界，记 UNKNOWN','action':'读取现有融合覆盖和冻结源码，检查未覆盖跨模块或多消费者边界'},
 {'id':'layer0-diagnosis','agent':'diagnose_work','title':'第零层 MC2 低活动原因','operator':'DispatchFFNCombine / layer0','shape':[3,5120],'dtype':'INT8/BF16 mixed','layout':'model-specific','hypothesis':'已有 trace 可进一步区分目标层 MC2 的到达时间差与其他资源瓶颈','success':'形成可复现时间或流水数据，能够支持或排除至少一条竞争原因','falsification':'缺乏可关联时间和资源字段，标记因果不可识别','action':'复核 16 rank trace 的现成汇总和低活动机制的竞争解释'},
 {'id':'theoretical-ceiling','agent':'bounds_work','title':'算子与推理吞吐严格上界','operator':'Wq_b / inference graph','shape':[3,1280,32768],'dtype':'INT8 inputs, BF16 output','layout':'ND/FRACTAL_NZ','hypothesis':'在明确的允许实现域中，可以用必要工作与硬件容量证明更强的时间下界或界定未知','success':'给出含证明、单位和域边界的约束，或明确证实无法收紧的缺口','falsification':'所需必要工作或硬容量上限无可验证依据，则保持符号/UNKNOWN','action':'检查已知理论界的必要工作、资源上限、数据依赖和跨资源重叠假设'}]
root.mkdir(exist_ok=True)
for z in items:
 d=root/z['id'];agent=T/'agents'/z['agent'].replace('_work','').replace('diagnose','diagnosis');agent.mkdir(parents=True,exist_ok=True)
 def run(*args):
  r=subprocess.run([sys.executable,str(S),*map(str,args)],capture_output=True,text=True,check=True);return r.stdout.strip()
 if not d.exists():
  run('init-task','--tasks-root',root,'--task-id',z['id'],'--title',z['title'],'--operator',z['operator'],'--framework','vLLM-Ascend 910C','--source-root',src,'--goal',{'fusion-coverage':'在已声明的冻结实现范围内持续扫描融合与拆分重组，完整记录数据依赖、覆盖、候选及无法证明的缺口。','layer0-diagnosis':'按一致指标定位 NPU 低活动模块，尽可能区分调度、访存、计算、通信和数据依赖，保留无法识别的原因。','theoretical-ceiling':'建立可复现、条件明确的算子和整体吞吐理论上界，并与同口径实测比较；缺证据标记 UNKNOWN。'}[z['id']],'--allowed-path',agent,'--non-goal','不改目标模型、框架或算子；不调参，不启动新 NPU 负载，不访问其他环境')
 attrs={'hardware':'Ascend 910C logical 16','case':'32K input / 1K output / concurrency 8 / approx 90% prefix hit','shape_interpretation':'shape lists are representative dimensions, not full interface specification','semantics_precision':'preserve frozen implementation; missing metadata UNKNOWN','baseline_evidence':'../../evidence/baseline-result.json'}
 if not json.loads((d/'cases.json').read_text()).get('cases'):
  run('add-case','--task-dir',d,'--case-id','frozen-169','--shape',json.dumps(z['shape']),'--dtype',z['dtype'],'--layout',z['layout'],'--attributes-json',json.dumps(attrs,ensure_ascii=False),'--group','existing-profile','--protected','yes','--reason','冻结基线和已有 trace；只做证据驱动诊断')
 if not (d/'loops/loop-001').exists():
  run('new-loop','--task-dir',d,'--loop-id','loop-001','--title',z['title'],'--mode','design','--hypothesis',z['hypothesis'],'--success',z['success'],'--falsification',z['falsification'],'--allowed-path',agent,'--case','frozen-169','--next-action',z['action'])
 if not (d/'loops/loop-001/runs/frozen-169/run-001').exists():
  run('new-run','--loop-dir',d/'loops/loop-001','--run-id','run-001','--case-key','frozen-169','--kind','review','--command','Subagent '+z['agent']+': read existing frozen source/trace and save exact commands and evidence under '+str(agent),'--code-ref','vllm-ascend 8fde6d4e335fe300e2b949aab0e31659cce0811d')
 print(json.dumps({'task':str(d),'loop':str(d/'loops/loop-001'),'agent':z['agent'],'artifact_dir':str(agent)},ensure_ascii=False))
