# 请求流程分解页

先读 [组件与耗时对比](../REQUEST_FLOW_COMPARISON.md)，或下载本目录后打开 [交互页](index.html)。GitHub 文件页不执行 HTML。

- `data.json`：正式请求、源码机制、profile 窗口、诊断步与输出链路。页面内嵌同一数据，不依赖网络加载。
- `formal-requests.csv`：1,344 条请求的客户端起点、首非空文本、结束端点和可复算指标。
- `source-manifest.json`：16 个源对象的指纹；`public/` 前缀表示公开归档中的文件，其余表示原始诊断来源，不是可下载原生 trace 的承诺。
- `verification.json`：数据与 JavaScript 语法校验结果；`view-verification.json`：最小 DOM 替身中的逻辑测试结果，均不是浏览器验收或性能收益验收。

复核命令（需要 Python3 和 Node.js）：

```sh
python3 -P verify.py
node test_view.cjs
```

数据区间保留原始时间域和证据范围。`decode[].intervals_ms` 内的值统一为毫秒，内部字段名为兼容源表保留 `_us` 后缀；原始微秒值可由此乘以 1000 得到。其余 `profiles` 和 `phases[].intervals_us` 保持微秒。表格显示时已按对应范围换算。

尚未取得全部正式请求的内部互斥阶段时间；不能用这些诊断区间拼接出一条虚构的完整请求瀑布，也不能把未覆盖区间直接当成可优化收益。
