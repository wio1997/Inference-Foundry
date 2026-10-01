# 优化点、实验与Git提交规范

GitHub保存权威代码/配置、事实摘要、裁决和可追溯证据。运行机器保存大raw artifact，Git记录稳定位置、大小、SHA256及关键归约；不提交权重、大trace或凭据。文件数量不作为进度。

## 1. 一点一身份，一次执行一个Run

- 优化点：`GLM-OPT-NNNN`，按问题/机制划分，允许成套重构与多个相关实验。摘要放`records/points/<ID>/point.md`，索引`records/points.jsonl`。
- Run：`GLM-RUN-NNNN`，记录真实执行的代码、合同、命令、观察器、结果与证据。默认在优化点下`runs/<Run>/`，不同配置/采集不得伪装同一次执行。
- 沿用根`scripts/taskctl.py`的Task/Loop/Run、compare和resume机制；若采用它，其目录为唯一权威路径，点索引只引用，避免复制第二份状态。先恢复真实源码目录，勿用假路径初始化；其算子字段/固定Loop习惯不限制GLM研究。

初始点允许PLANNED且没有Run；未执行不填性能。索引仅保留ID、标题、类别、状态、权威路径和关键Run/Current引用。

## 2. 优化点最小内容

问题与合同；必要依赖/资源及假设；预期改变的有效输出链；相关历史与条件差异；本次新增信息；修改范围/配置；验收与裁决。已有证据足够则标`REUSED`并引用，省去重复实验。原型前无需写长证明，可随证据修正。

## 3. Run最小证据

`manifest.json`记录：

- ID/优化点、时间、kind（formal_e2e/diagnostic/profile/microbenchmark/simulation等）、实际执行状态。
- 框架/source repo与commit；有未提交改动则记录base+patch artifact/hash及关键文件hash。代码、镜像、模型/量化、设备/rank/部署和effective config身份。
- 实际命令/cwd/内层exit/超时；负载、到达、采样/MTP/Graph/cache条件及相关有效工作量。比对Run和可比性说明。
- 风险相称的功能/状态验证、完整请求/有效输出、吞吐/延迟/错误与观察器开关。缺失填null/unknown，不臆造。
- artifact路径/大小/hash、关键日志归约、必要的清理/恢复与活动资源状态。

短`summary.md`只写结果、因果证据、限制、裁决及下一问题；无需重复manifest。已失效、超时或缺产物保留`INVALID`，不归为性能成功/失败。指标可增减，足以复现和裁决即可。

## 4. 提交与裁决

提交按可审阅变化组织，不每个编辑都commit，也不强制一轮一个机制：

```text
glm(opt): GLM-OPT-0007 describe final code/config change
glm(exp): GLM-OPT-0007 GLM-RUN-0012 record evidence and verdict
glm(knowledge): record applicability and revalidation conditions
```

代码提交与实验记录相互链接；Run固定被测code ref，记录提交固定该Run证据。推送前核对实际diff，不夹带他人改动；研究分支按任务选择，Git更新不force覆盖并发工作。

裁决分`KEEP / REJECT / INCONCLUSIVE / INVALID / REUSED / NOT_APPLICABLE`。KEEP提升Current需满足声明合同的真实重复E2E及功能状态验证；诊断结论可以成立，不能冒充产品KEEP。部署/工作量改变单列收益类别。REJECT写原因、未知与值得重验的条件，不永久否定整个方向。

checkpoint更新活动点索引、HANDOFF Current引用和必要复用知识；不每Run重写全部文档。旧Run保持原始身份，新归约写新文件并引用旧hash，不改写历史数据。Current只有一个权威证据引用。
