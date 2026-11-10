# 接续断点

介绍：统计业务token、算法位置与padding后的行域，新增全局图“有效Token与Padding”页。

到了哪里：新配平参考批次8请求的usage/counter严格一致；原C8 227活跃波、454个DP组、3632个rank窗全部行域对账；9个冻结源码片段。请求输出8192，草稿8805/接受6450，接受率73.253833%；prefix命中236032，不把逻辑prompt量当新计算量。

结果：原C8每层草稿body真实query8840/物理行域10688=82.709581%；额外1848行=17.290419%，由对齐1808、DP统一35、synthetic5组成。此为行域，不是全框架FLOPs、实际空算时间或预计收益。固定target FULL图若进24行桶，B1/B2/B3/B4的条件padding为75/50/25/0%；命中人口未测。

下一步：复用已有值，按阶段记录scheduled、DP统一、图档位、stage-specific有效量、tensor extent、有效slot、MoE收发及rejected-context占位。实际字段已有CPU值直接写账，设备计数批末异步取，避免每轮同步引入新空档。当前全模型逐阶段实测比例未知；不拿历史17.29%或拒绝率代填。先追有效域是否已经被kernel mask裁掉，再价可省工作。

本轮全部本机旧数据复算，新增NPU/模型请求/profile/服务操作0，未核live环境。原融合源码搜索的70/86断点继续有效，不改其进度。统计与图UI校验通过，见validation.json及相邻balanced-profile-171/view-validation.json。
