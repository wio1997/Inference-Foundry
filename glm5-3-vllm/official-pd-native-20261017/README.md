# 官方原生 PD 部署（2026-10-17）

最终验收：官方P/D/proxy功能、实际远程KV READ及本次32rank SDMA/HCCS路径窗口核查通过；原始P302,908,782B、D416,348,365B均已回传Mac并核SHA。最后健康检查正常，当前服务保留至下述有限期限。详见`evidence/20261017-official-pd/deployment-validation-v1.json`。

使用现场固定镜像 v0.30.0rc1-a3-arm64、Ascend提交5f828ad638ee4f9b12db87f9d5481f29076d37f9、既有GLM5.3 W8A8C8/Indexer-int8正式权重。原样MooncakeConnectorV2与原样官方load_balance_proxy_server_example.py，未叠加自写连接器、同步、缓存对齐或调度补丁。

| 组件 | 地址 | 配置 |
| --- | --- | --- |
| P，32 | 172.16.12.32:29481 | TP16、DP1、PP1、eager |
| D，33 | 172.16.12.33:29482 | TP16、DP1、PP1、FULL_DECODE_ONLY |
| 官方代理，32 | http://172.16.12.32:29880 | CPU-only、一个P与一个D |

served model为glm5。两端8K上下文、4seq、1024批token，容量/拓扑适配，不称官方四节点200K方案原样复现。TP通信保持AI_CPU，HCCL_INTRA_ROCE_ENABLE=0、HCCL_INTRA_PCIE_ENABLE=1、HCCS、ASCEND_LOCAL_COMM_RES1.2及/etc/hccn.conf只读挂载；没有EP/MC2/FlashComm/DSA-CP/MTP。c8_enable_reshape_optim=false是已确认的旧HDK兼容项。

P/D与代理均已Ready。256/16、1024/16经原版官方代理均HTTP200且基础2+3烟测通过；D每条请求16rank均记录成功远端KV READ，逻辑接收合计分别223,346,688与893,386,752B。D已真实capture/replay，无原生错误。收据见evidence/20261017-official-pd。配置选择SDMA不单独证明所有实际数据链路；保留运行路径验收边界。函数响应、基础任务正确性、逐token一致性和完整精度验收分别记录，不把5与5.0的格式差异作为整条部署路线的阻断。

服务是有限窗口：P/D各3600秒，约12:37–12:38Z（北京时间20:37–20:38）停止；4500秒NPU holder约12:52Z到期。只使用32/33自己的16die资源，续期前已核两机8卡空闲和无其他有效claim，没有停止其他作业。

两份启动计划为node32-prefill-v1.json与node33-decode-v1.json，使用已有tools/start_capacity.py启动。reference保存匹配提交的官方原文和SHA；source中的404读取失败明确保留，实际V2注册到kv_p2p/mooncake/connector.py。旧Mooncake小测完成payload但析构异常，不能据此预先否决/宣布本服务成功；本次不重复该小测，以实际服务证据为准。

```sh
curl http://172.16.12.32:29880/healthcheck
curl http://172.16.12.32:29481/v1/models
curl http://172.16.12.33:29482/v1/models
curl http://172.16.12.32:29880/v1/completions -H 'Content-Type: application/json' -d '{"model":"glm5","prompt":"What is 2 + 3? Answer only the number:","max_tokens":16,"temperature":0}'
```

匹配V2源码未更新Mooncake传输统计对象，Prometheus相应counter保持0；原客户端计数门槛失败保留，独立CPU校验按原生成功READ日志、请求ID、16rank/地址/字节与镜像源SHA确认实际传输，未补写官方指标源码。256直接D与PD文本一致，1024有5.0/5格式差异，完整逐token/任务精度未验收。真实请求profile和CPU离线导出已完成；P/D共32rank有效trace均有SDMA任务，D各rank有hccl_batchGet，32通信矩阵仅LOCAL/SIO/HCCS，无RDMA命名任务。该结果限定于实际profile窗口；不把任务计数当作每byte归因或普遍证明，不计性能收益。
