# Astra High：Loop066 第一gate代码预检

2026-09-26，只读以下当前文件：
`scripts/loop066_owner_fixture.py`、`loop066_owner_fixture_patch.py`、`run_loop066_owner_fixture.sh`。未运行服务/NPU或修改源码。

## 必须先修

### P0：借用Graph capture可能先命中fixture

patch把hook直接放在borrowed `dsa_cp._forward`。fixture只检查layer2、x=[96,4096]、非draft，没有真实Runtime执行标记或capture排除。`EXTREME_RUNTIME_TARGET_GRAPH=0`只管专项Runtime使用Target图，不能证明borrowed runner的启动capture关闭；其FULL decode capture仍可走到同shape。

因此启动期可能执行qsl.cpu()/torch.npu.synchronize()/JSON等非capture安全动作，或先置`_extreme_owner_fixture_done=True`消耗掉真正fixture，造成启动失败或dummy fixture假证据。

最小修正：只有从真实Runtime eager Target调用显式携带的opt-in上下文标记才能进入；同时排除捕获/编译/dummy warmup。skip时不得置done；真实fixture完整写盘后再置done。不要靠shape、非prefill或CPU start值猜真实上下文。runner需确认采样发生在runtime handoff后的实际cycle并记录cohort/cycle/req identity或等价关联。

### P1：slot交集比较存在假通过

`_comparison`只比较common集合，missing和duplicate只是记录。若B仅有一个有效slot且该行相等，`matched_values_exact=True`仍成立；重复slot会覆盖前面的row，错误B→错误owner地址也可能在full集合中找到共同项。

必须：
- 从真实qsl/start推导每request有效compressed行数；本冻结有效c4/8输入为A共24、B共4，padding独立忽略；
- 用req0/req1在A的**compressed request prefix**取出预期owner slot序列，检查B与这4行严格对应（不能只要求B是A的某个子集）；
- 对重复slot、缺失slot、预期多余slot采取显式INVALID处理，除非实现了有证明的多行alias比较；不能字典覆盖后继续宣称通过；
- 检查所有有效slot行号落在对应Compressor输出第一维内，输出dtype/尾维匹配；缺行不能悄悄被交集隐藏；
- 输出parity状态必须联合覆盖门槛，不能只打印matched_values_exact作为A/B gate。

### P1：state比较使用混合storage的浮点解释，易假失败且scope不清

`pages=unique(state_bt_b)`选取owner整个blocktable，不是本次native真实state读写页；`torch.equal(state_a[pages],state_a2[pages])`会把共享key区/无效区作为FP32比较。相同NaN bit也返回false，不能据此判A/A不稳定或B不等价。A/B在整个owner表范围的差异也可能来自未闭合物理alias，当前布尔无法归因。

最小修正：
1. immutable prestate与各arm初值恢复用全raw byte一致性校验，A/A/A3报告raw byte差异范围而非混合FP32 maxabs；若合法state浮点自身有变化，须在实际typed state有效范围量化noise，而不是强制所有raw poststate位相等。
2. 记录owner当前输入[start,start+8)以及native历史state读包络，按4160B stride、512×FP32行和物理页构建明确byte区间；A/B state gate比较这个有定义的state域，其他diff单列，不先忽略所有nonowner逻辑页。
3. 页号/extent做0及上下界校验；page0 native读写语义单列。当前pages>0只够排除常见padding，不是read closure。
4. 若首轮不准备闭合实际state读域，报告`state_gate=INCONCLUSIVE`及byte diff，不能仅用现有owner_state_exact布尔做通过/拒绝。

### P1：runner没有fixture完成/覆盖验收

bench12成功即容器命令结束；代码skip或未进入Runtime时也可成功。必须在cleanup之前读验all8 fixture产物：真实Runtime采样关联、三个typed ABI、四arm初值同一快照、A/A与A_after_B覆盖完整、B四个预期slot、无缺rank/重复文件。数值不稳定可以明确INCONCLUSIVE，但不能等同“第一gate通过”。不要用HTTP 12×1024代替fixture检查。

## 其余实现判断（未发现阻塞错误）

- `_private_alias_views`以untyped storage完整byte clone，再按各view原dtype/storage_offset/size/stride set_，方向正确；scale offset2048是FP16元素offset，传set_原offset正确。四个private backing互相独立，tuple强引用保留到synchronize，未见直接写原cache。
- 四个arm都在任何Compressor调用前从原storage克隆，之后A→A2→B→A3分别使用独立初值，构成同一prestate的四次调用，不需要先在原cache上restore。仍宜显式记录四backing与原storage区间无交及initial raw equality；A3输出需确保并非native输出workspace重用，记录storage或clone有效结果后比较。
- qsl全0:8:96后按rank计算两个完整owner请求、x16切片、qsl_B=[0,8,16]、start原token单位、两个独立group的blocktable选择都符合当前冻结语义。建议补验真实CP本地query区间与该rank-owner表一致；不能将此公式推广其他TP/CP布局。
- B直接调用native compressor_metadata，避开forward-context的12请求metadata memo；ncmp=6是容量而非有效4行，负slot padding应保留显式判断。A使用原metadata可供后续原路径复用，B没有覆盖原memo。
- 第一gate未调用rotate/quant/scatter/QLI，scope声明正确。通过后只能证明局部Compressor owner compact假设，不能证明未来消费者安全或正式性能。
- patch的SHA/唯一anchor/restore guard合理；runner的active+RUN_TS检查沿用已修复隔离。per-run marker不是互斥锁、cleanup的全局stop和restore失败被吞掉仍是既有操作风险，运行时保持单服务并核验恢复日志。

**优先级：先修真实Runtime/capture gate及比较覆盖，再跑。** 否则容易取得不能回答owner16假设的dummy或假通过证据。
