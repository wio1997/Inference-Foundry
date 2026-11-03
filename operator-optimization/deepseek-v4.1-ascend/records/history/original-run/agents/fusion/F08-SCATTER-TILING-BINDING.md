# F08：已保存任务与当前安装二进制、tiling 的绑定边界

## 问题与结论

排查 EP3 同一 stream 35 的短样本 L9（task 13226，globalTaskId 2078）和长样本 L30（task 13468，globalTaskId 2942）：**已导出 TASK、COMPUTE_TASK_INFO、trace 不能确定它们实际使用的二进制文件、tiling key 或参数。** 当前 stage 容器的已安装算子包中，有一个二进制描述的输入输出 dtype 与两条任务导出 dtype 匹配；这只是当前文件的候选，不能证明历史进程选中它，也不能证明两次选用相同 tiling 分支。

## 证据及具体范围

1. 对 EP3 的 `ascend_pytorch_profiler_3.db` 用 `mode=ro` 查询，`TASK` 与 `COMPUTE_TASK_INFO` 各以 globalTaskId 唯一匹配；后者除 globalTaskId 外的 17 个导出字段及解码后的字符串完全一致。`name` 解析为逻辑算子名 `aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore_ScatterNdUpdateSk`，`attrInfo` 解析为空字符串，`opState` 为 `dynamic`，blockNum 都为 48。上述表没有列名为 tiling key、tiling data、二进制文件名或哈希。**空 attrInfo 不证明调用时 strides 属性为空**，也不证明数据布局相同。
2. `trace_view.json` 对同 track、同 task ID 的目标 X 事件各出现一次；事件只显示逻辑名称、Model Id、Task Type、Physic Stream Id、Task Id、Batch Id、Subtask Id 和 connection_id，没有导出的二进制名、key 或参数。CSV 此外也没有这些字段；这只描述当前导出结果，不证明原始采集一定没有这些信息。
3. 当前 stage 安装的 `ascend910_93/scatter_nd_update_sk` 目录有 16 个 JSON 描述与 `.o`，其中唯一同时声明 BF16/INT32/BF16 输入及 BF16 输出的是 `ScatterNdUpdateSk_3b70bd1dfff14ea76aa6465d526da019`。其 `.o` 的实际 SHA-256 为 `e92bc71e8f8c76a1d906ca27a1f8766e2e3a3eae257037aa2efa6e4726951323`，与同目录 JSON 声明一致。JSON 列出以 `_10`、`_11`、`_20`、`_21`、`_30` 结尾的五个 kernel 名称；这不能从逻辑名称推断运行时选中了哪一个。声明的动态属性默认值也不等同于历史 task 的实际参数。
4. 对照当前安装的 `torch_npu/profiler/analysis/prof_view/_kernel_view_parser.py`，`KernelViewParser.generate_view()` 从 CANN `OP_SUMMARY` 读取记录，再选择和映射显示列生成 `kernel_details.csv`。因此该 CSV 是加工导出，不能把它与相同采集的数据库/trace 当成互相独立的测量；也不能只凭 CSV 缺字段断定原始采集绝无该信息。未找到足以解释 CANN 原始设备 slice 格式及完整导出规则的同版本源码。

当前 device 目录存在 `ffts_profile`、`stars_soc` 等原始 slice；本次仅枚举文件，没有解码，也没有声称其中一定包含或不包含 key。时间对照、PMU 和源码分支的其他限制见 `F08-SCATTER-VARIATION-AUDIT.md` 与 `F08-SCATTER-INSTALLED-KERNEL-SEMANTICS.md`。无论当前候选的 key 列表与源码中的分支命名是否看似对应，历史运行绑定仍是独立证据要求。

## 复现与下一步

只读脚本 `f08_scatter_tiling_binding_audit.py` 在既有 stage 容器中运行，不调用 NPU API；脱敏输出为 `f08-scatter-tiling-binding-audit.json`。脚本从指定 DB 精确查询两条任务、流式筛选目标 trace 事件、核对当前已安装描述符和文件哈希。脚本结果没有读取 prompt、输出或权重；它不解码 raw slice，也不将当前文件自动追溯成历史文件。

后续优先查同版原始 profile 解码器或历史运行时日志能否按 device/stream/task/时间绑定实际 binary、key 和结构化 tiling 数据，并同时记录导出器版本与时钟口径；如果不可获得，应维持未知。在当前不允许新增 NPU 负载的范围内不能补做线上捕获，也不因找不到 key 就宣称已经确定低利用率根因。
