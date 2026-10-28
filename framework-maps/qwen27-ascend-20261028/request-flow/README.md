# 请求流程分解页

先下载 [中文详解HTML](report.html)，打开即可离线阅读。`index.html` 是同一份新版报告；GitHub 文件页不执行 HTML。也可读 [Markdown分解对比](../REQUEST_FLOW_COMPARISON.md)。

新版按“结论 → 流程 → 组件机制 → 请求时间线 → 调度与图 → 算子通信 → 输出链路 → 优化与自研参考”组织。原字段被翻译为带单位的中文说明，逐轮记录改用卡片，缺失值明确说明未采集。正文不再输出JSON对象。

- `data.json`：正式请求、源码机制、profile 窗口、诊断步与输出链路。页面内嵌交互所需的数据子集，不依赖网络加载。
- `gap-analysis.json`：28场景的吞吐比值复算及源数据SHA。阅读报告将修复前native mixed路径的大幅退化，与正式矩阵剩余0.9%～10.9%的吞吐差距分开说明；后者尚未完成最终根因归因。
- `formal-requests.csv`：1,344 条请求的客户端起点、首非空文本、结束端点和可复算指标。
- `source-manifest.json`：16 个源对象的指纹；`public/` 前缀表示公开归档中的文件，其余表示原始诊断来源，不是可下载原生 trace 的承诺。
- `verification.json`：数据与 JavaScript 语法校验结果；`view-verification.json`：最小 DOM 替身中的逻辑测试结果。
- `human-browser-qa.json`：实际本地Chromium离线检查1440/768/390宽度、展开记录、672组请求及500个窗口的结果。仅检查页面与展示，不判定性能收益。
- `build_report.py` 和 `human_report_template.html`：以同目录数据和归档快照重新生成中文报告，不需要NPU或网络。

复核命令（需要 Python3 和 Node.js）：

```sh
python3 -P build_report.py
python3 -P verify.py
node test_view.cjs
```

数据区间保留原始时间域和证据范围。`decode[].intervals_ms` 内的值统一为毫秒，内部字段名为兼容源表保留 `_us` 后缀；原始微秒值可由此乘以 1000 得到。其余 `profiles` 和 `phases[].intervals_us` 保持微秒。表格显示时已按对应范围换算。

尚未取得全部正式请求的内部互斥阶段时间；不能用这些诊断区间拼接出一条虚构的完整请求瀑布，也不能把未覆盖区间直接当成可优化收益。
