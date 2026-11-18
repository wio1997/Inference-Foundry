# G316 D 编译缓存验收更正

128K 首轮 cold32/hot32 的原始64条 usage、五项 AISBench 指标、首次提交身份和独立 P 461文件证明仍有效。完整“P/D 均无编译干扰”协议验收已撤回，当前 D 状态为未验证。

实际客户端源码的 compile_cache() 固定读取旧 G181 路径，而当前服务运行在 G316。现场核对 Docker 挂载、环境和六组实际路径后，旧目录496项、当前目录411项，六组路径均不互为别名。旧目录前后相同不能证明当前目录没有编译。当前文件最新 mtime 早于后三轮预热仅作旁证，不能补造历史前后快照。

保留历史验收文件与所有原始测量，用本更正覆盖其中 D 编译通过的结论。禁止重新发送正式请求来替代首次结果。表中的吞吐差值只能作为带此限制的首次观测，不能据此认定因果性能收益。

- [更正记录](../evidence/matched_SG_G316/128K/compile_root_correction/G487_G316_D_compile_acceptance_correction.json)
- [实际目录、挂载与缓存快照](../evidence/matched_SG_G316/128K/compile_root_correction/G483_actual_D_cache_2202_parsed.json)
- [现场检查源码](../code/SG_G316_compile_root_correction/inspect_current_D_cache_g483.sh)
- [保留的首次128K数据](024-matched-SG-first-128K-cold-hot.md)

256K 已在第三轮预热期间、完整第四和第五轮开始前取得当前 D 的独立基线；仍需末态比较和修正后的正式阶段检查才能验收。本页没有接受任何256K结果。
