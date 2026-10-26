# 之前的vLLM GLM-5.3 W8A8测试脚本、方案和C32数据

这是2026-10-08的旧GLM-5.3 W8A8、六机4P2D历史资料，不能当成Flash 1P1D结果或直接运行Flash部署。输入原始80000，服务封装后80012 token；输出600，temperature0、ignore_eos。不是81920输入的早期2P2D问题复现。

| 配置/统计 | 请求数 | 输入token/s | 输出token/s | TTFT P50/P90/P99 秒 | TPOT P50/P90 ms |
|---|---:|---:|---:|---|---|
| 原始C32，全程 |384|138072.58|1035.39|3.552/6.356/9.028|23.366/25.628|
| 原始C32，AISBench原生稳态 |352|148360.37|1112.54|3.470/5.776/7.903|23.2/25.3|
| A参数C32，全程 |1024|145216.03|1088.96|3.611/5.786/9.230|22.515/24.830|
| A参数C32，AISBench原生稳态 |992|149382.76|1120.20|3.559/5.647/8.708|22.482/24.695|

上述TTFT P50均小于最新5秒要求，TPOT P50均超过18ms，整体SLA未通过。A组和原始组N不同，不能将差值当成严格匹配配置提升。原计划使用TTFT4秒门槛，原文保留，未改写成5秒。

本地前缀命中92.7861%；P加external复用后92.9461%，输入token/s包括缓存命中，P/D不重复计输入。原始C32实际HTTP并发均值30.97/峰值32；A组31.613/32。原始稳态窗口189.837秒，A组531.332秒；窗口条件通过不等于SLA通过。

## 测试方案

1. 固定角色与代码：P DP4×TP16/EP64，D DP16×TP2/EP32；先验证真实PD、权重、600输出计数与连接器路径。
2. 生成80000输入/600输出、共享前缀74400的负载；正式服务侧计数80012。前缀预热单独记录，核对每个P/D rank覆盖，正式轮计数从预热后取差。
3. 先做固定0.93RPS C16/C32，各48请求，检验到达限制；容量轮request_rate=0（该版本表示不限外部RPS），C16/C32/C64/C96，各384请求，闭环持续补足并发。
4. A组C32是后续独立N1024验证：P token budget12288，D MTP5、fused MC2=1，D FlashComm1/shared-DP/overlap关闭；具体服务身份见c32_A_validation/serving_evidence.json。
5. 输出全程和AISBench原生稳态两套输入/输出token/s、TTFT/TPOT分位数、错误、实际并发；同时保存逐请求时间、角色metrics、NPU和CPU采样。稳态不可覆盖全请求SLA。
6. 缓存每轮核对；历史C16/C64发现一条跨轮完整输入重复，聚合命中未抬高，但原生成器不能证明跨轮唯一。后续必须用独立run命名空间并做完整输入哈希/LCP检查，不原样复用历史seed+row盐逻辑。

## 脚本入口与边界

- scripts/original/prepare.py：历史数据与AISBench配置生成（有上述跨轮盐缺陷，保留用于溯源）。
- scripts/original/run_prepared.sh：前缀预热、正式AISBench、metrics采集与离线完整性检查。需要已准备manifest/数据/配置与客户端依赖，不是独立一键部署器。
- scripts/original/capture_metrics.py、analyze_run.py、analyze_metrics.py：采集、全请求统计与缓存审计。
- scripts/diagnostic/analyze_stable.py：调用当时AISBench原生稳态选择器；必须使用同一版本。
- c32_A_validation/configs/：当时实际C32/N1024客户端配置；仍包含历史服务器路径/模型名。
- original_4p2d/：冻结测试方案、部署说明、输入合同、结果、指标口径和来源哈希。

历史执行形式（只展示，不要对当前Flash直接运行）：
```bash
RUN_BENCH=1 RUN_NAME=sat_c32_pilot PHASE=warmup bash scripts/original/run_prepared.sh
RUN_BENCH=1 RUN_NAME=sat_c32_pilot PHASE=full WARMUP_APPROVED=1 bash scripts/original/run_prepared.sh
```
需设置ROOT、API地址、模型与tokenizer，并提前准备输入；原脚本默认路径是历史aisbench目录。新实验要求所有运行产物、缓存、日志落在/data/tiankuan/wio，不能直接沿用默认路径。模型生成器GSM8K依赖与AISBench软件未打包，原始大型数据/设备trace不在这个精简包；其来源/原始包哈希在PROVENANCE中。

当前Flash继续使用独立的/data/tiankuan/wio/glm53-flash-1p1d-20261016/。先做真实PD短请求和81920输入/600输出功能验证，随后正式性能比较须统一Flash tokenizer后的80000原始输入、600输出、命中策略与并发，保留新旧模型/拓扑差异。
