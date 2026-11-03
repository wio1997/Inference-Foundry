本轮继续按 profile 定点推进 M21。复用 v37 的 186 条完整调用链，24 次普通 slot mapping、64 次 adaptive slot mapping、72 次 prepare_inputs_padded 以及状态修正/输入 packing/索引 gather 的确切现场决定源码范围。没有扩大为仓库或 SDK 扫描。

slot mapping 已融合多组：profile 的两个 UINT64[13] 是指针数组，实际写的是各组的 INT32 slot bank。当前源码已有普通/自适应 tile 策略、每组尾部 PAD 程序和 circular 负位置处理；这些不算新增优化。新的条件方向包括公共位置/page-offset 算术复用、与位置 producer 重组、lookup/小请求布局调整，以及在完整尾部读者与代际证明后缩小 PAD 发布。各组 block table、stride、block size、circular 标志和写入所有权必须保留。现有 lookup 中 INT32 block number 经 FP32 gather 再转回 INT32，直接整数读取需保留原舍入或证明值域，不能暗改数值语义。选中 fused 路径要求 dcp1 且无 compressed position/index 列表；其他 fallback 不借此关闭。

拒绝采样 padded kernel 原本已经同时输出 rejected 和 selector：D 是无前导零的 inclusive draft 前缀，d=D[i]-D[i-1]；r=d+1-count（d>0），否则 r=0；selector=QSL[i+1]-1-r。可在同一 request、D/QSL/count 代际与 ready 边界下并入已有 count producer；保留 INT32 输出、无额外 clamp、全部 padded 输入和 metadata。不能从 selector 推出非因果 decoder 的 query/KV 行可删。现有 count 的 sentinel/vocab 与 count−1 语义仍沿用 v36。

跨迭代状态修正的关键是先 gather 旧状态再提交。对 prev_positions=[1,0]、old_computed=[100,200]、counts=[1,2]，正确结果是 [202,101]；若逐行原位写，第二行可能读到已覆盖的 202，得到 [202,203]。这是融合方案的反例，不是现有实现错误。应保留快照、分阶段或明确单一同步域。新请求/无 draft 的 CPU fallback、old accepted、INT64→INT32/copy_ 转换和 CP rebuilt positions 分支保留。公共位置/seq_len/slot pipeline 可作为条件重组，但不能凭一个 launch 消除跨 program 依赖。

输入 ID general packing 的四组索引可按完整请求映射和 schedule/draft 几何复用，token 内容始终是 live 值。保持 PP receive wait、新请求/prefill 上传、optional embeds/is_token_ids 和现有 contiguous 快路径。该方向细化既有 v13 候选，不重复计收益。两个相邻 draft-ID gather 可在合法 baseline 索引/错误域与稳定输入下组合索引，避免中间 ID gather；不把该范围全部 360 TASK 宣称为可删。

本轮将 22 条链、2952 TASK 的局部合同从未分析/待查转为条件分析，登记 8 项细化和 32 项约束。仍有 39 条未归属链、4280 TASK：30 条 C2 metadata 链/3256 TASK，8 条 block/slot 初始化 fill 链/936 TASK，以及 worker1462 fill88；此外仍有部分 CP/proposer 字段、QLI 空 query 的 eager/captured 读者与 M22-M25。不能因此标记整体完成。条件方案未实现/未测速本身不等于未分析范围。

远端 CPU 点读 2 次通过；首次 AST 函数名 guard 因 profile 展示名少一个下划线失败，真实错误保留，显式 v2 修正后通过。加主审共 4 runs：3 pass / 1 error。全局保持 35 模型候选、7 框架候选、0 整模块关闭；26 理论比较 INCOMPLETE、3136 candidate 资源行、849 缺 PMU、16 phase-unbound 图不变。无模型/torch 导入、新 NPU 负载、服务更改或新健康检查。Goal active。
