# 接续断点

介绍：把空档里的工作从父阶段/函数名继续拆到实际数据、用途和消费者。入口仍为同一份全局 HTML，默认“空档操作与数据”。

到了哪里：28 项操作、8 个交付阶段、9 份冻结源、28 个准备空档函数自身账；可查 KV 地址表/写入槽、QSL、CPU 与设备长度、CP 裁剪、压缩/ring/RoPE、SMLA/QLI packet、回调与跨轮事件，真实 KV/index/compressor 数据另列。逐项源码片段及 SHA 保存在 view-data.json。

结果：CP build/base build/local-token helper 合计311.050016ms/卡/11周期，占准备空档65.10%；函数自身387.452688ms+可辨识标记边界90.346922ms=477.799609ms。诊断窗1025.502734ms，Free491.193438ms。所有字段独立时间未知，不从函数时间按比例拆分，不把 SMLA 总时间叫纯 KV 读取。

下一步：在此字典上追加字段级 producer ready / H2D done / last reader / consumer 其余依赖就绪证据，才能算实际可提前量。优先复用 loop046 已有读者证据：后25次命中长度投影死值，首 producer 保留；每资源 QSL owner、CPU/setup 重复和原生集成仍待确认。不要重采同样的父 scope，不要默认把全部 metadata 都提前。

本轮新模型/profile/服务动作0；未重核 live 环境。全框架范围尚未证明穷尽、逐字段时长与完整因果 DAG 未测齐。其它对话全模型分支搜索断点继续有效。
