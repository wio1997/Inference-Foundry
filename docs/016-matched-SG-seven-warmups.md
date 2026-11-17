# 同机 SGLang 的七轮独立预热与首轮正式门槛

G181 同一硬件、同一服务容器完成七遍完整32768→1024、C16、每遍16请求，共112个实际usage核验通过的请求。前五遍原始结果全部保留；第4遍 Decode 新增 alloc_extend_kernel 的二进制和编译产物，因此五遍结束时不接受“充分预热”，未发送正式请求。

追加第6、7遍使用冻结数据集中两个独立前缀的数据集，不重发原始五遍。第6、7遍均无新的D编译产物，最后两遍duration变化1.5193%、TTFT变化0.2373%、TPOT变化3.3613%；三项均低于5%。P从首遍后到七遍完成后的331个缓存文件大小和mtime均未改变。缓存观察是编译文件证据，不声称证明所有异步硬件活动绝对不存在。

- [原始第4遍新编译的逐文件证据](../evidence/matched_SG_G181/seven_warmups/G219_SG_warm_gate_2202_parsed.json)
- [追加两遍的时延与D编译门槛](../evidence/matched_SG_G181/seven_warmups/SG_G181_C16_32K_g220_extra/warmup_stability_32768.json)
- [七遍后P独立缓存核查](../evidence/matched_SG_G181/seven_warmups/G227_Pcache_DSstart_2201_parsed.json)
- [全部Run索引：含每遍原始五项指标、无损请求详情与SQLite](../records/runs.json)

追加预热保持原始整体截止时间，不延长原campaign生命周期。同一服务epoch的首轮正式cold32/hot32单独提交并核对，只能在首轮原始证据齐备且P/D编译及cold命中审计通过后形成正式结论。本页仅记录预热与门槛，不将预热速度转换为框架最终收益。
