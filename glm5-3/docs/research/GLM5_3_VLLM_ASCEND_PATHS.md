# GLM-5.3 vLLM-Ascend源码与PD路径

当前模型路径`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务名`glm-53`；用户2026-10-06上传中。现有安装版本锚点见[ENVIRONMENT_RECOVERY](../../ENVIRONMENT_RECOVERY.yaml#software)，不据旧模型可运行或公开main代码认定5.3已受支持。

按完成上传后的真实加载路径逐项确认：

1. 从新config/index确定architectures、model_type、量化、tokenizer和MTP；定位实际安装registry/model class/loader，不重用旧模型版本名代替核验。
2. 从实际argv/PYTHONPATH确认V1/V2 runner、attention backend、cache spec/layout、MTP/Graph消费者；保持既有正确算子，不通过新kernel或改计算语义绕过不支持。
3. 标准PD沿现有proxy→Mooncake producer/consumer→scheduler→实际device执行→MTP提交→用户输出定位request/engine/rank/block身份及完成/取消回收。
4. P/D各自组织DP/TP/EP通信域，参考DP1TP16PP1须在新模型验证fit与正确性。旧布局参数、兼容补丁和controller队列不直接重放。
5. 缺必要软件支持明确列阻塞；不切旧模型、不把完整请求副本叫5.3 PD，也不预选AsyncLLM hook/API重构。基线、patch与E2E按[PLAN](../../PLAN.md)推进。

具体实际源码入口仍经[CURRENT_PRODUCT_MAP](../../CURRENT_PRODUCT_MAP.md)→manifest/argv发现；其中ACTIVE当前是旧模型驻留链，尚无5.3加载证明。历史[源码研究](GLM5_2_VLLM_ASCEND_PATHS.md)及SOURCE_AUDIT.json保留原版本/commit，只作匹配机制线索。
