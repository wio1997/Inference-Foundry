# 短诊断采集出处与质量边界补证（待整合、未编号）

此文作为gen25未编号归档草稿的追加材料，不覆盖旧草稿，不占R，不改变原性能数值。正式记录仍须按发布时最新私有仓规则处理。

建议在README“质量/适用边界”与RESULT的Result/Produces字段追加：**已保存的vLLM gen81与SG gen67插桩512/C1/out32诊断，在相同prompt IDs、payload、seed和temperature=0下重建文本不一致；只覆盖这两个诊断请求，无法据此定位根因或判定原性能矩阵的无插桩输出等价。协议/长度有效不等于质量通过，整体质量/greedy未闭合，未采用优化。** 不把它写成所有场景质量失败或某补丁正确性失败。

入Git候选仅新增[两行采集/导出时段表](diagnostic-acquisition.csv)与[八行rank部署表](diagnostic-rank-deployments.csv)；逐字节复制进新候选record/assets并记录本代文件SHA，旧19CSV保持不变。表内SHA精确对应原文件，诊断时间不能放入正式out1024性能计分。不要把导出时间当采集时间。

EVIDENCE引用本代外部小证据包及其中`diagnostic-provenance.json`、`output-text-comparison.json`、`runtime-config-lines.json`、`validation.json`、完整下载收据。保留两份原SSE/result、source manifest、8rank profiler_info和SDK原raw-copy manifest的路径/SHA；transport/key从未打包。原始profile/完整trace/服务日志不进Git，外部证据包按原规则保管。

当前地图仍是gen26预览，不能声称已显示本代质量实证。实际204完成后按gen26 runbook合并；再把本代scope明确的质量说明和小表放入最后的新树，对实际最终树重新核对文字、链接/下载SHA、native按需读取和浏览器。独立补证不构成actual204验收，也不能替代root最终接受。
