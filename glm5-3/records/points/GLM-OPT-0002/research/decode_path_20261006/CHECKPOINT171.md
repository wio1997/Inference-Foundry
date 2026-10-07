# checkpoint171 — H9 closed, H10 source cause and CPU parity

本阶段没有新增代码级性能 KEEP。H6/H5的已验证研究收益继续保留，正式Current/API/SLA另验。用户最终workload80000输入/600输出/93%prefix，单列93%声明KV命中条件；166 AISBench与两机配置、GitHub main c0159929/research e8e30671 已同步，未启动正式压测。公开授权已由checkpoint160用户“允许”恢复，不再额外阻塞。

[Run266](../../runs/GLM-RUN-0266/summary.md) H9 correctness/全16/golden/MTP/同worker通过，但两次完整D saving0.000883129/0.008071115s均小于drift0.017917528s，短请求也不重复改善，独立归约INCONCLUSIVE_NOT_REPEATED/H9off。原controller20请求完成后的跨host active_epoch读取错误保留FAILED；独立冻结terminal reconciliation完成且只用剩余1个预定warmup，总21，无重载/重跑，确认H6/H5on、H9off、health/idle/all16/同worker。D266 root419075/start309643580/boot65c53cfa，P249 root1916718/start303049910/boot6d9cf06f；下次操作须fresh核验，不能继承旧D265根。

唯一后续候选[H10源码裁决](early_padded_mtp/SOURCE_DECISION.md)：ModelRunner同步CPU记账在普通PP1 padded MTP提交前等待target D2H。实际贪心采样、备份token和DCP消费者证明受限K1路径不依赖追加后的CPU输出历史；最小单文件patch沿用原early-PP框架。336实际AST状态用例/4096guard通过，48次只改变提交次序，NPU未初始化。设备正确性、暴露预算、收益unknown，不把CPU doubles当性能。没有新性能patch加入stack，也没有恢复全部有效patch。

Run267准备唯一固定FULL/H6H5/纯KVAPI对相同stack+H10，旧H9setter移除、target host wait始终保留；复用恒定mode0薄Graph见证，新增H10一字节selector仅warmup模式转换写全rank witness， measured同模式无文件写。20固定correctness/短A/B/A/B/23token自然EOS PD加≤1终态warm，最多一次owned D重载与一次失败baseline recovery，不扫描或追加profile。源代码/spec/CPU/live preflight冻结通过后才执行；当前准备记录不宣称已运行。

Run267于06:14:38Z已由唯一controller1079159/start309928676启动；spec202a2a2548a67b3531bec7156e5e334c1dba671b1ddc1eda8e0f1b3433269bcc/51pins。CPU336/4096/16384/1152及80installed-source/dead-controller/locks/health/idle/all16 preflight通过。启动不等于正确性或收益；实际D与进度用Run267 state/active_epoch/guards/retained_stack，不继承D266旧PID。该段覆盖上方准备中事实，冻结Reset/spec不修改。
