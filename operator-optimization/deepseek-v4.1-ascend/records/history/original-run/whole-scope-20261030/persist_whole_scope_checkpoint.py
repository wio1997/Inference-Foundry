#!/usr/bin/env python3
"""Publish a truthful local recovery checkpoint; never complete the goal/task."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

W = Path(__file__).resolve().parent
R = W.parent
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
read = lambda path: json.loads(path.read_text())
def save(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
verification = read(W/'contract-checkpoint-verification.json')
assert verification['status']=='PASS_LOCAL_CONTRACT_CHECKPOINT_ONLY'
module = read(W/'module-coverage-ledger.json')
assert not module['overall_complete'] and all(not x['complete'] for x in module['surfaces'])
backup = W/'root-records-before-whole-scope-checkpoint'
backup.mkdir(exist_ok=False)
for rel in ['STATE.json','CONTINUATION-STATUS.md','RESEARCH-INDEX.md','RESEARCH-CATALOG.json','publication-draft/RESEARCH-REPORT.md']:
    p=R/rel; dest=backup/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
for rel in ['TAIL-HEAD-SAMPLER-REVIEW.md','COVERAGE-LEDGER.md','SCOPE-AND-GATES.md']:
    (backup/rel).write_bytes((W/rel).read_bytes())

next_action = ('继续M00/M01/M05的实际初始化与范围闭包：按真实import补generate/base、serve/engine及HF generation fallback，'
    '核实aux层ID映射、wrapper图像驻留与logits全消费者；继而补M04/M21/M24的版本化状态和同次因果边。'
    '并行概念槽不代表并行Agent；不启动目标优化实现或无目的重复采集。')
gaps = [
    '实际resolved模型/头/采样配置与历史同调用绑定、传递依赖和primitive语义闭包',
    '全部相关分支及有依据的不适用排除、全部消费者/alias/状态/跨流跨卡跨层跨迭代边',
    '完整请求关键路径、worker准备与device真实因果关系、通信组/消息/事件身份',
    '同调用MoE专家有效行与容量、内部阶段；Indexer真实长度/mode；完整Engram host/UVA代价',
    'HBM/L2实际字节与适用硬件容量上限；可证资源账、候选重叠/冲突和范围内排除依据',
]
state = {'objective':read(W/'GOAL-ACTIVE.json')['goal']['objective'], 'goal_status':'active',
    'overall_complete':False,'source_scope_closed':False,'updated_at_utc':now,
    'case':'169 Ascend910C/A3, DeepSeek V4.1 Flash W8A8, TP8xDP2, C8, 32768in/1024out, ~90%prefix',
    'scope_and_gates':'SCOPE-AND-GATES.md','source_ledger':'coverage-ledger-v2/summary.json',
    'module_ledger':'module-coverage-ledger.json','progress':'PROGRESS.md',
    'source_files_indexed':verification['source_files'],'module_surfaces_provisional':26,'whole_modules_closed':0,
    'selected_local_branch_contracts':23,'these_counts_are_not_completion_percentage':True,
    'source_only_facts':['frozen two-repository commits and hash-verified selected sources','DSpark imports legacy qwen3_dspark directly; earlier alias claim corrected','existing greedy reject kernel and existing top-token APIs are not new optimizations'],
    'measured_facts':['same official later capture:16rank CSV hashes matched; eight vocab shape/dtype families arithmetic rechecked','matching shapes/counts do not prove tensor or invocation identity'],
    'conditional_derivations':['STANDARD base vocab/REPLICATED Markov under frozen normal initialization','target/draft greedy and no requested logprobs; entropy/block/dynamic-confidence disabled under saved configuration','A3 standard adaptor excludes A5 packed ABI under the inspected dispatch'],
    'candidate_local_reviews':['WS-VOCAB-01','WS-REJECT-01','WS-BONUS-01','F27/F28/F29/F30/F31 refinements; overlapping directions, no additive gain claim'],
    'conclusion_changing_gaps':gaps,'next_action':next_action,
    'target_operator_changes':False,'new_inference_or_npu_work_this_goal':False,
    'blockers':[],'blocked_consecutive_turns':0,
    'progress_audit':'Previous interrupted/rejected report-only completion is historical. Current goal has substantive source, local semantic and same-capture measurement progress; not blocked.',
    'historical_complete_receipt':'../continuation-20261030/GOAL-COMPLETION-20261030.json; preserved, not current acceptance',
    'local_checkpoint_verification':'contract-checkpoint-verification.json; validation of references/identities only'}
save(W/'STATE.json',state)

progress = '''# 整体优化机会分析进度

**整体未完成，目标active；尚无完整模块通过八类覆盖门槛。** 不给缺少分母依据的完成百分比。旧报告、历史complete回执及本次检查点PASS都不代表整体验收。

本轮完成的实质工作：

- 固定两库commit并保留566份去重源码，建立版本化站点索引；较早558文件索引原样保留。最新索引含7479类/函数、20291控制站点、39948调用、7772导入，全部只是定位底账。
- 建立26个暂定pipeline审阅面、260个变换规则槽和每模块8类门槛；范围仍扩展，不以四节点枚举限制完成范围。
- 对输出头/拒绝采样/DSpark的23个选定分支补充局部效果合同；区分源码事实、配置推导、实际观测和未知，没有关闭整模块。
- 核对同次16rank原CSV的hash和8类词表形状/dtype族，修复FLOAT/BF16 ArgMax混选；未以kernel时长之和宣称关键路径收益。
- 纠正DSpark导入路径误读；补真实Markov/Confidence头及采样参数源码。按固定启动/请求合同细化greedy、entropy/block、动态confidence、词表分片与A3 cache适用范围，历史resolved对象仍待证。
- 细化WS-VOCAB-01、WS-REJECT-01及按需求拆分bonus的WS-BONUS-01；补F27–F31跨位置/层/卡/迭代重组约束。候选存在重叠，未实现、未实测模型收益。

当前决定性缺口：

'''+ '\n'.join('- '+gap for gap in gaps) + '''

下一步：'''+next_action+'''

读取顺序：[STATE](STATE.json) → [范围门槛](SCOPE-AND-GATES.md) → [模块台账](COVERAGE-LEDGER.md) → [输出头审阅](TAIL-HEAD-SAMPLER-REVIEW.md) → [DSpark合同](DSPARK-CONTRACT-REVIEW.md)。原始数据、失败版本、错误推断修订前记录和历史目标回执全部保留。

当前研究仅只读取证和离线分析；未修改目标代码、重启服务或发送新的推理/NPU负载。没有外部阻塞，可继续推进。
'''
(W/'PROGRESS.md').write_text(progress)

tail = W/'TAIL-HEAD-SAMPLER-REVIEW.md'
text = tail.read_text().replace(
    '- DSpark引用的 `vllm.models.qwen3_dspark` 在所查新路径不存在，但tracked legacy `model_executor/models/qwen3_dspark.py` 存在。导入重定向机制还未闭合，不把“找到legacy文件”当运行对象证据。',
    '- **已纠正取证误读：**V4.1 DSpark直接导入legacy `vllm.model_executor.models.qwen3_dspark`，无需假设新旧路径重定向。真实头源码已补采；运行对象和其它传递依赖仍待闭合，见[DSPark合同](DSPARK-CONTRACT-REVIEW.md)。')
assert 'DSpark引用的 `vllm.models.qwen3_dspark`' not in text
text = text.replace('## 6. 余下的关键证据','## 6. 余下的关键证据\n\n后续已依据固定serve/client/source补配置适用推导，见[合同台账](contract-applicability.json)。推导与直接历史对象观测继续分开；本节中运行绑定未知仍保留其具体含义。')
tail.write_text(text)

lines=['# 整体覆盖台账入口','','**整体未完成；范围闭包仍在扩展。** 源码/站点数是定位分母，未审阅范围不能当不适用。26个暂定模块均未整体关闭。','',
    '| 模块 | 已关联源码数 | 分支站点关联数（可重复） | 结论性缺口 |','|---|---:|---:|---|']
for m in module['surfaces']:
    lines.append(f"| {m['id']} {m['name']} | {len(m['indexed_source_paths'])} | {len(m['indexed_branch_ids'])} | {'；'.join(m['conclusion_changing_gaps'])} |")
lines.extend(['','[机器可读台账](module-coverage-ledger.json)保存260个规则槽和每模块8类门槛，目前均OPEN或PARTIAL。',
    '23个选定分支的局部合同见[tail delta](tail-branch-review-delta.json)与[DSpark delta](dspark-branch-review-delta.json)，不能当作23个完整模块。',
    '[最新源码索引](coverage-ledger-v2/summary.json)含566份文件；[原558文件索引](coverage-ledger/summary.json)保留。',
    '[配置适用推导](contract-applicability.json)与直接运行观测分开；[下一步和决定性缺口](PROGRESS.md)。'])
(W/'COVERAGE-LEDGER.md').write_text('\n'.join(lines)+'\n')
scope=W/'SCOPE-AND-GATES.md'
s=scope.read_text()
begin=s.index('[模块台账](COVERAGE-LEDGER.md)')
end=s.index('\n\n首版站点ID',begin)
s=s[:begin]+('[模块台账](COVERAGE-LEDGER.md)建立26个暂定pipeline审阅面与260个规则槽，范围仍扩展。最新三批源包含566个去重文件、517个Python文件，源级底账有7479个类/函数、20291个控制站点、39948个调用和7772个导入。**均非完成语义审阅的数量，不提供整体完成百分比。** 23个选定分支有局部合同，0个完整模块关闭。')+s[end:]
s+='\n最新条件合同、误读修订与配置推导见 [DSPARK-CONTRACT-REVIEW.md](DSPARK-CONTRACT-REVIEW.md)。AST中的decorator/default/base执行阶段、lambda归属、动态调用与C++内部尚未成为完整语义模型；源码索引的存在不能关闭这些缺口。\n'
scope.write_text(s)

root_state=read(R/'STATE.json')
root_state.setdefault('historical_goal_before_whole_scope',root_state.get('current_goal_20261030'))
root_state.update(current_goal_20261030=state['objective'],state='ACTIVE_WHOLE_SCOPE_INCOMPLETE',
    current_phase='WHOLE_SCOPE_SEMANTIC_AND_EVIDENCE_CLOSURE',next_read_only_action=next_action,
    last_update=now,overall_work_paused=False,overall_complete=False,
    checkpoint_status='LOCAL_CONTRACT_CHECKPOINT_ONLY_WHOLE_TASK_OPEN',goal_tool_state_last_checked='active',
    goal_tool_state_note='Latest explicit /goal is active; old report-only completion receipt is historical and rejected as acceptance.',
    whole_scope_current={'state':'whole-scope-20261030/STATE.json','progress':'whole-scope-20261030/PROGRESS.md',
        'module_ledger':'whole-scope-20261030/module-coverage-ledger.json','closed_whole_modules':0,'overall_complete':False})
save(R/'STATE.json',root_state)

status=R/'CONTINUATION-STATUS.md'
old=status.read_text();marker='以下为新目标启动前的原始交接要求'
assert marker in old
status.write_text('''# 2026-10-30 整体优化机会目标：active，尚未完成

当前有效目标是用户最新 `/goal`：完整分析指定模型/版本/硬件/负载的全部模块、条件分支及跨算子、层、卡、迭代融合拆分重组；未分析范围不得因报告成稿而标完成。

最新入口：[进度](whole-scope-20261030/PROGRESS.md)、[STATE](whole-scope-20261030/STATE.json)、[覆盖台账](whole-scope-20261030/COVERAGE-LEDGER.md)。已有566份源码索引和23个局部分支合同，26个暂定模块中尚无整模块关闭。这些数量不代表完成百分比。

[新goal的真实active回执](whole-scope-20261030/GOAL-ACTIVE.json)与[完整性更正](continuation-20261030/COMPLETENESS-CORRECTION-20261030.md)保留。历史 `continuation-20261030/GOAL-COMPLETION-20261030.json` 记录的是此前错误验收时的实际工具返回，不能作为当前完成依据。阶段报告仍是既有证据入口，新的语义与适用合同见整体覆盖目录。

'''+old[old.index(marker):])

index=R/'RESEARCH-INDEX.md';old=index.read_text();marker='这是本次研究的总入口。'
assert marker in old
index.write_text('''# 169 DeepSeek V4.1 Flash W8A8：整体研究目录

> **当前整体目标active，尚未完成。** 最新入口为[完整范围进度](whole-scope-20261030/PROGRESS.md)、[覆盖台账](whole-scope-20261030/COVERAGE-LEDGER.md)、[恢复STATE](whole-scope-20261030/STATE.json)。0个完整模块通过覆盖门槛；旧报告成稿与历史complete回执不作验收依据。

> 已补输出头/拒绝采样/DSpark局部合同、固定配置适用推导和同次16rank词表测量族，详见[tail审阅](whole-scope-20261030/TAIL-HEAD-SAMPLER-REVIEW.md)及[DSpark审阅](whole-scope-20261030/DSPARK-CONTRACT-REVIEW.md)。完整调用/状态/primitive闭包、关键证据与所有变换仍待推进。

'''+old[old.index(marker):])
report=R/'publication-draft/RESEARCH-REPORT.md'
report.write_text('> **最新整体目标仍未完成（2026-10-30）。** 本文为阶段报告；继续研究的有效状态、完整覆盖门槛与新增合同见[整体进度](../whole-scope-20261030/PROGRESS.md)。不得以本文成稿或历史complete回执标整体完成。\n\n'+report.read_text())

catalog=read(R/'RESEARCH-CATALOG.json')
catalog['read_first']=['whole-scope-20261030/STATE.json','whole-scope-20261030/PROGRESS.md','whole-scope-20261030/SCOPE-AND-GATES.md','whole-scope-20261030/COVERAGE-LEDGER.md','whole-scope-20261030/DSPARK-CONTRACT-REVIEW.md','CONTINUATION-STATUS.md','publication-draft/RESEARCH-REPORT.md']
catalog['latest_user_status']='ACTIVE/INCOMPLETE: latest whole-scope /goal; no whole module closed; historical complete receipt is not acceptance.'
catalog['continuation_updated_at_utc']=now
catalog['whole_scope_current']=root_state['whole_scope_current']
changed=['STATE.json','CONTINUATION-STATUS.md','RESEARCH-INDEX.md','publication-draft/RESEARCH-REPORT.md']
changed += ['whole-scope-20261030/'+x for x in ['STATE.json','PROGRESS.md','SCOPE-AND-GATES.md','COVERAGE-LEDGER.md','TAIL-HEAD-SAMPLER-REVIEW.md','DSPARK-CONTRACT-REVIEW.md','contract-applicability.json','module-coverage-ledger.json','contract-checkpoint-verification.json','coverage-ledger-v2/summary.json']]
existing={x['relative_path']:x for x in catalog['indexed_files']}
for relative in changed:
    p=R/relative;data=p.read_bytes()
    item=existing.get(relative,{})
    item.update(relative_path=relative,local_path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
        current_local_record=True,server_scope='Updated local checkpoint; this revision has not been synchronized to remote/public copies.')
    item.pop('server_mirror_path',None);item.pop('stage_mirror_path',None)
    existing[relative]=item
catalog['indexed_files']=list(existing.values())
save(R/'RESEARCH-CATALOG.json',catalog)

# Record the completed local evidence Run, without closing its Loop/Task or goal.
run=W/'tasks/whole-opportunity-coverage/loops/loop-001/runs/c8-input32k-output1k-prefix90/run-20261030T121953Z'
tool='/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'
command=['python3',tool,'finish-run','--run-dir',str(run),'--status','pass','--correctness','not-applicable',
    '--summary','Hash/ID/arithmetic and selected local-contract checkpoint passed;566 files,23 selected branch contracts,26 provisional surfaces,0 whole modules closed. Research review only: no target implementation, numerical correctness test, guaranteed gain or global completion.']
for rel in ['checkpoint-verification.json','contract-checkpoint-verification.json','PROGRESS.md','TAIL-HEAD-SAMPLER-REVIEW.md','DSPARK-CONTRACT-REVIEW.md','contract-applicability.json']:
    command += ['--artifact',str(W/rel)]
result=subprocess.run(command,capture_output=True,text=True,check=False)
receipt={'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
save(W/'taskctl-checkpoint-execution.json',receipt)
assert result.returncode==0,receipt

# Keep the active research Task directed at actual open work, retaining its schema maturity label.
task=W/'tasks/whole-opportunity-coverage';task_index=read(task/'index.json')
task_index['next_action']=next_action;task_index['updated_at']=now
task_index['research_coverage_status']='PARTIAL_LOCAL_CONTRACTS; NOT_WHOLE_SCOPE_CLOSED'
save(task/'index.json',task_index)
result=subprocess.run(['python3',tool,'compact','--task-dir',str(task)],capture_output=True,text=True,check=False)
save(W/'taskctl-compact-execution.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
assert result.returncode==0
print(json.dumps({'status':'LOCAL_ACTIVE_CHECKPOINT_PERSISTED','source_files':566,'selected_local_contracts':23,
    'whole_modules_closed':0,'goal_and_task_still_active':True,'target_or_service_changes':False},ensure_ascii=False))
