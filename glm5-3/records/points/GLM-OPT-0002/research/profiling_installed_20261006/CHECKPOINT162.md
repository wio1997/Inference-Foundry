# Checkpoint162 — actual native PD profiling and SLA diagnostic

产品目标与SLA保持MISSION。Current=None，active PERF_KEEP为空；无代码优化收益声明。Run249在实际安装源码核验后，仅增加profiler config并按fresh idle/boot/start/ancestry/NPU ownership退休Run246，保留原始模型/算子与布局。Run249为当前驻留P/D；完成控制器已退出，profile已stop，未运行额外observer或正式测试。

三个PD诊断均2334input/8generated tokenIDs，P内部token不计有效输出，native external hits各2334，MTP输出1/2/2/2/1。暖态profOFF TTFT1.309065938s；TPOT265.290943ms/token。非正式分位数，语义最终答案尚未生成；无法据小样本宣布SLA达标，Decode明显仍远于18/40ms目标。profON测得1.451518115s/336.029329ms，正式测量应stop重型profile。

两端32rank CPU/NPU原始采集已落盘，daemon解析限制由实际torch_npu _npu_profiler.py22–29确认。官方离线接口analyse(max_process_number=4)完成所有32rank非空Ascend Hardware tracks；全部原始框架/设备binary sha未变。最初离线审计对新增.complete错误比较导致JobINVALID，修正后C复用P16、仅解析D16；B本地准备语法错误/缺工作单的工具preflight失败也保留，均无额外模型请求或重启。原始Job/Result/CLI与capture不覆盖，运行成功不等于性能KEEP。

Sol亲读实际profiler/router/config/worker/torch_npu parser源码及P/D rank0原始事件索引、全rank归约。D0 hardware extent2.859885s，COMMUNICATION union1.854119s，EVENT_WAIT union2.820010s，大量同步与collective重叠。不能把2.82s直接删掉；profile中没有完整request/step/KV-ready标记，不足以判明设备安全完成或控制成本。

方向裁决：H3从conditional置PARKED，未证明/未否定其并发场景。当前唯一源码问题是D逐步collective/peer进度与host提交的依赖和成本；尚无性能patch或新NPU Run队列。先关联现有trace中的communication/CANN connection与实际runner/collector消费者，最大代码可删除Gap仍unknown，不做配置扫描或猜测同步删除。完整API/unified8000、SLA分位数、matched真实完整E2E仍是产品门槛。

证据：Run249 execution_summary.json、jobs/GLM53-PD-PROFILE-OFFLINE-20261006-C/{summary,device_166,device_167}.json；大trace/server binary refs保留于raw_capture_index_refs.json。

本阶段没有新增代码级性能 KEEP。
