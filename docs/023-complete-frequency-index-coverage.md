# 频率索引调用覆盖修正

G425完整D图已经经过此前压缩KV写入处的频率索引，随后在索引器 `scoring.decode_scores` 的 `inputs.freqs_cis[pos]` 再次因complex64的aclnnIndex失败。8份安装凭证说明此前选中的6处已安装；故障来自原源码子集遗漏了索引器调用，不能据此认定已验收的逐位helper失效。D控制器已退出并清空8卡。

本次对固定镜像完整V4 attention目录、V4模型文件及后端共26份源码做AST扫描，发现7个函数中的8处频率下标。G437补齐 `scoring.prefill_requests`、`scoring.decode_scores` 两处，保留其余6处及已在8张910B1验证的同一helper字节；新增scoring源文件SHA门槛。扫描范围和每处函数、行号、表达式均留存。

G423组件检查的224用例、192图、576次变化回放仍只证明helper和指定原生RoPE/FP4链的范围。G437新调用安装、完整target/draft图、PD状态传输、语义和四档性能尚未验收；本记录不增加完整服务或性能通过声明。固定镜像、权重、EP、DSPARK及完整D图要求不变。

- [G425完整错误与调用栈](../evidence/full_model_G425/2204/server.log)
- [D退出和8卡清空](../evidence/full_model_G425/2204/state.json)
- [故障原因、原始归档SHA与验收范围](../evidence/full_model_G425/2204/coverage_assessment.json)
- [26份源码的8处频率索引](../code/full_model_candidates/sg_pd_g437/full_frequency_index_audit.json)
- [8处安装和原生源码SHA检查](../code/full_model_candidates/sg_pd_g437/freqs_install.py)
- [沿用的逐位索引helper](../code/full_model_candidates/sg_pd_g437/freqs_compat.py)
- [原组件验收](../evidence/frequency_gather_G423/acceptance.json)
