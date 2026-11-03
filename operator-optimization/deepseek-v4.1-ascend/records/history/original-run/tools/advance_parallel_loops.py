# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess,sys
T=Path(__file__).resolve().parents[1];S='/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'
def run(*a):
 r=subprocess.run([sys.executable,S,*map(str,a)],capture_output=True,text=True);print(r.stdout.strip()[-400:]);
 if r.returncode:raise RuntimeError(r.stderr)
def complete(task,loop,art,summary,result,gs,causal,reason,next_action,others=()):
 d=T/'task-loop-runs'/task;L=d/'loops'/loop;args=['finish-run','--run-dir',L/'runs/frozen-169/run-001','--status','pass','--correctness','not-applicable','--summary',summary,'--artifact',art]
 for f in others:args+=['--artifact',f]
 run(*args);run('verdict','--loop-dir',L,'--result',result,'--goal-satisfied',gs,'--comparability','not-applicable','--causal-conclusion',causal,'--reason',reason,'--next-action',next_action,'--evidence-path',art)
def newloop(task,title,hyp,success,fail,action,agent,ref):
 d=T/'task-loop-runs'/task;L=d/'loops/loop-002';run('new-loop','--task-dir',d,'--loop-id','loop-002','--title',title,'--mode','design','--hypothesis',hyp,'--success',success,'--falsification',fail,'--allowed-path',T/'agents'/agent,'--case','frozen-169','--next-action',action);run('new-run','--loop-dir',L,'--run-id','run-001','--case-key','frozen-169','--kind','review','--command',ref,'--code-ref','vllm-ascend 8fde6d4e335fe300e2b949aab0e31659cce0811d')
complete('fusion-coverage','loop-001',T/'agents/fusion/FINDINGS.md','发现末层 MTP 缓冲和最终输出的条件多消费者；四种 SP/MTP 分支结构枚举已独立复算。','accepted','yes','not-applicable','源码与 G1/G5 对照支持原图遗漏条件分支；实际是否启用、全部消费者和融合收益仍未证。','核查冻结配置的实际 SP、PP 和 MTP 分支。',[T/'agents/fusion/final-boundary-structural-scan.json'])
newloop('fusion-coverage','确定末层条件分支的基线适用性','冻结配置与源码足以判定末层是否分配 MTP 缓冲和是否启用 SP','确认实际条件或明确剩余无法验证项','缺 PP/SP 配置证据时保持 UNKNOWN','读冻结 speculative 和 parallel 源码及基线进程配置','fusion','Subagent fusion_work: inspect frozen configuration source and baseline flags')
complete('theoretical-ceiling','loop-001',T/'agents/bounds/cold_gemm_bound.md','给出精确 INT32 冷态独立 GEMM 的全 1 输入必读 41946880 字节及符号时间下界；不可直接应用于热态真实推理。','accepted','yes','not-applicable','论证和整数算术已核对；这是条件辅助模型，物理带宽与真实基线适用性仍未知。','围绕真实固定权重热态语义审查可证明的约束。',[T/'agents/bounds/cold_gemm_bound.json'])
newloop('theoretical-ceiling','真实热态 Wq_b 与推理下界','固定权重热态语义可支持额外必要计算或 I/O 界','建立有证明的非平凡界，或识别不可约缺失前提','无法证明必要工作或硬上限时保留 UNKNOWN','检查真实语义、运行数据初态和硬件容量界','bounds','Subagent bounds_work: review necessary work for actual warm fixed-weight inference')
complete('layer0-diagnosis','loop-002',T/'agents/diagnosis/loop-002-ep8-single-sample.md','一个 EP8 样本中未找到可直接关联内部阶段的 MSTX 时间标识；PMU 名称各有 7 条，是否为单次调用未知。','inconclusive','unknown','not-identifiable','单样本与局部导出不足以拆分 dispatch/GMM/combine 或等待；不能推广到全部 rank。','核对 graph replay 的 PMU 键与通信任务关联，不重复已排除的查询。',[T/'agents/diagnosis/mc2-raw-sample-ep8.json'])
print('three loops reviewed; next loops retained active where appropriate')
