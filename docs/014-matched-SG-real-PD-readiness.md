# 同机 SGLang 的实际加载、完整图和 PD 功能门槛

G181 在相同两台 GLM 主机上完成 native target/draft 各八 rank 的严格加载分派账本，共32份；缺失和未分派参数均为0。每角色八个rank的非空内存注册均返回0。D端八rank完成 target verify NPU完整图，batch尺寸为2/4/6/8/10/12/14/16，原生 draft decode和extend完整图也完成；真实生成日志记录npu graph=True。

G182先完成数学、127/129/131、8191/8193及OpenAI短输入，共七项功能检查。随后32K请求返回503，原生router明确报告server_selection_failed：P被健康检查超时标为不健康，因此没有选出PD pair。原失败、健康检查日志和响应保留。两端自动恢复健康后，G204等待连续两次健康且空闲的worker观察，用新前缀仅补做剩余诊断，未重跑原七项，也未发送正式数据。

补充请求实际为32768输入、32输出，无retraction，MTP verify8轮、平均接受长度4。八项选定的原生PD功能与长度门槛通过，但不代表整模型质量套件、全部状态逐字节传输证明或C16性能通过。G204是诊断恢复，不能把G182原尝试改写为首轮全成功。

- [P端实际容器、配置、设备与图日志](../evidence/matched_SG_G181/p/readiness.json)
- [D端实际容器、配置、设备与图日志](../evidence/matched_SG_G181/d/readiness.json)
- [原503诊断](../evidence/matched_SG_G181/router_first_503_diagnosis.json)
- [保留七项原结果及新32K响应的合并验收](../evidence/matched_SG_G181/p/functional_PD_g204/result.json)
- [新32K原始响应](../evidence/matched_SG_G181/p/functional_PD_g204/native_32768_32_response.json)
- [有界恢复代码](../code/matched_SG_migration/functional_completion_g204.py)
- [复用固定token数据与AISBench配置](../code/matched_SG_migration/prepare_matched_SG_client_g199.py)
- [只允许首遍完整C16预热的启动门槛](../code/matched_SG_migration/prepare_first_SG_warm_g210.py)

后续先跑独立输入的完整32K/C16/1024预热，第一遍后暂停采集P编译缓存，所有预热与正式分账。当前门槛尚未授权该编排器发送正式冷热请求：P/D编译稳定与时延稳定通过后，才进入首轮cold32/hot32。其余三档、同机匹配profile、差异根因和优化地图仍未完成，没有新的框架性能收益结论。
