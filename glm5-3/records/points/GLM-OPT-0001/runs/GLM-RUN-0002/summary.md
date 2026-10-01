# GLM-RUN-0002 — 并发1/2/1短E2E诊断

同81932实际输入/2048输出，4/4返回usage、length与DONE，三个阶段真实退出均0。并发1两次TPOT22.825/20.451ms；并发2两条21.869/26.812ms。整arm有效输出TPS38.851/63.100/43.478（包含TTFT；不能当稳态decode容量）。

采样D MTP接受率64.3%/66.8%/75.9%，接受长度2.929/3.004/3.276；2秒采样未观察到waiting队列，不能排除未采到的短等待。P prefix hits增量79872/151552/79872，响应cached_tokens却为0。P累计prefill增量1.258/5.891/1.167s，TTFT5.990/10.022–12.576/5.242s；两个累计或不同请求窗口不可直接相减归因。

裁决INCONCLUSIVE：降并发1未稳定达TPOT18ms；不能证明并发2长输出达标、缓存失效、容量已到顶。c1a/c1b同prompt输出hash不同，MTP/生成轨迹也不同，不把bracket差异算代码收益。下一步完整c2长输出及请求阶段时间线。数字详见reduction.json；原始计数、SSE、metrics留166。

2026-10-01测量修订：早期parser未包含native delta.reasoning，原first_content是后续可见正文端点，不能称完整TTFT。离线读取原SSE、未新发请求；正确首输出与TPOT见first_output_reduction.json：

- c1a/0: first native output 2.324280s，TPOT 24.615357ms；旧content端点 5.989543s。
- c1b/0: first native output 2.221923s，TPOT 21.924436ms；旧content端点 5.242178s。
- c2/0: first native output 7.372125s，TPOT 24.411150ms；旧content端点 12.575994s。
- c2/1: first native output 5.772811s，TPOT 28.888002ms；旧content端点 10.021914s。
