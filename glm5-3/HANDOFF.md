# GLM — HANDOFF

2026-10-01T20:47Z checkpoint，持续自主执行。研究、代码、执行、原始日志全部在166/167，Mac只作SSH入口。真实clone /data/tiankuan/wio/Inference-Foundry，分支glm5-3-autonomous-20261001。恢复根/局部AGENTS和本文件，再按问题读[活动点](records/points/GLM-OPT-0002/point.md)、[源证据](records/points/GLM-OPT-0002/source_notes.md)、MISSION/PLAN/REUSE/RECORDING。不要恢复旧测试队列。

- Current：None。尚无满足合同的重复KEEP；达标稳定容量与上下界unknown。正式Run21四条81932→61440严格原生提交245760输出，100.124194TPS；TTFT P50 6.003388667s和未舍入TPOT P50 18.006747585ms均FAIL。单窗口/四样本不证明稳定容量/P99。
- 任务：完整GLM专用推理框架，动态请求/可变负载/PD与并行可行域；不改算子计算，不做模型/量化精度比较。GPT研究/代码/裁决，真实Zcode --prompt负责执行/监控/日志归约。唯一controller用.controller.lock/.formal-test.lock及PID/boot/start/heartbeat排程，只操作本任务资源。
- 现场：/data/tiankuan/wio/glm52-pd/deploy，glm52-single。1669081和1679900是同一个DP2/TP16/DCP16/EP32 group，两物理成员必须同时存活；不是独立副本。Stock PDproxy8000自Run24停用，诊断gateway8002仅有限窗口运行。
- 最新执行：Run32 controller PID1001047，boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025/start263318003；spec SHA41f4d722cda8bde90dc9600bcc88b639fc5d17633fba2d6fd0b67f3dbc58a913，24 pins，启动Job BUDGET16K-LAUNCH32-20261001T2042Z真实bridge VALID。state.json/controller-owner.json及startup_model_identities.json为动态权威；本checkpoint在deploy阶段，尚未声明新配置fit或收益。
- Run32改变：预填充budget4096→16384，其余DP2/TP16/DCP16/EP32、K5、FULL_DECODE_ONLY、gmu.87、seed1024、checkpoint、解析器、hierarchy MC2/fused0/DSA-CPfalse相同。metadata V2只把logger改为vllm namespace（导入对应V2），两控制谓词与V1完全一致；复用原12 CPU合同，controller先验证当前16k native EngineArgs CPU配置，再按原联合组PID/boot/start/argv/healthy-idle精确共同停止/重建。缺INFO日志写unknown，不另设完美遥测前置门槛。有限包目标31completed+原生400/cancel共33attempts；所有原始协议/Usage/finish/DONE/errorunknown/lease/idle需真实验收。模型计算未改，缓存因必要重建重新观测。

## 最近终态与下一个问题

[Run29](records/points/GLM-OPT-0002/runs/GLM-RUN-0029/summary.md)：保留Run28未执行请求的精确健康组，33attempts/31completed/23161提交输出，原生400/cancel零信用。真实Zcode REDUCE-DP29-20261001T2004Z-v2 VALID，82748B SHA dd43d2e97781e8716cf9d70f50ffdfa875e85e9e54361da95897f333641f14a3。动态9088/121.8479870040901s=74.584736TPS，低于Run26的88.900785；缓存计数相同，MTP轨迹不同，无代码收益/KEEP。冷81932→2048 TTFT47.8756/48.2228s。四匹配prefix-tail命中71680/uncached10252，TTFT6.66–8.89s，nativeprefill6.18–7.12s，queue<=38.4us/preemption0：研究更大prefill预算的直接依据。EP32每worker权重25.53GiB、activation.35–.36、non-torch2.57–2.58、Graph.91–.92、KV24.85–24.87，nativecapacity2338614tokens；startup内存不是runtimepeak/容量界。

[Run30](records/points/GLM-OPT-0002/runs/GLM-RUN-0030/summary.md) driver INVALID：首条native Responses JSON实际HTTP200/completed/READY/Usage13+2=15，GPT误用不存在的obj.error属性后失败。已执行唯一请求2输出作为valid partial evidence保留，不重放；真实Zcode REDUCE-RESP30-20261001T2024Z VALID，3441B SHA cf61e3f9c3998f9d3338e968ccbee2466dcc0a5c8c7f1a947b1caac8a8b0bbe2。

[Run31](records/points/GLM-OPT-0002/runs/GLM-RUN-0031/summary.md) 完成20:27:06Z，controller450032/boot6d9/start263201604已死。仅余8attempts，3新completed/6输出+5预期400/404拒绝，第一Run30响应只离线复用/新信用0。真实Zcode REDUCE-RESP31-20261001T2032Z VALID，7977B SHA 2bdb1f56d13986a67006959344a116641ad9f335f584109ed41d1038f7915cba。原生Responses SSE每条9个named typed事件、sequence0..8、response.completed终态+Usage，不发Chat [DONE]。5代理lease全部释放，客户端wire与gateway原始wire逐字节/hash一致，两端idle。实际两端VLLM_ENABLE_RESPONSES_API_STORE未设/native默认0；store关闭，background400、未知/previousID404属于配置行为。当前只证明无状态短请求与拒绝合同，不声称背景/持久状态功能完备。若开启store，native各APIserver本地dict需要owner亲和；当前router首输出观察Responses仍保守unknown，提交以独立native协议审计。

## 控制原型与复用边界

metadata opt-in Worker仅替换实例的两个native源guard控制谓词，保留SP/oproj/embedding/draft/Graph、CPU collective和MC2全局max/masks/buffer。native_sync SHA3ae98d968ade0f78463b04143be79792d41c484fb938f25cbb6bf3ee8b93ee8e；determine SHA812c989f97f1a271731f4b7d1c41938014d567f541aff92aac588dbc1373af50。V2 helper SHA33bee48a7bac3b48cbfd4f86c4f626721dfb1e8cf6333ff013030da5870c30a4；worker SHAc329333f50dc57f63b6b580d008c3d0516490a2e66f50085a59676ebba567660。实际vendor源码未改。
CPU12源/dispatcher案例与native fullEngineArgs通过只证明配置和控制合同；Run29证明有限E2E但没有直接live padding shape/收益证据。Run28成功加载却因外部INFO过滤造成安装日志assert0/16失败，0推理，INVALID driver不是fit/语义REJECT；原code/spec冻结。CPUlogger-v3证实外部INFO过滤。

旧Run8/9/11的16k OOM发生在47.28GiB权重/约1GiBfree/PD merged draft，不能据此否定现在EP32约25.53GiB权重的新内存条件。Run24继承PD-only recompute flag使local kv_roleNone配置失败，0weights/0requests；Run25合法省略此flag后fit。Run25部分driver INVALID，但cold/tools valid复用；Run26/27在同组证明功能/动态/prefix，均无KEEP。Run23旧独立两端dynamic101.365TPS与新跨机EP32不可孤立因果比较。

工具插件glm_tool_contract默认缺tools/choice none保留原生文本，提供tools沿用native结构约束；解决Run19四条usage却只有三条严格有效的错误。Run20/21/25/29实际工具分支通过。prefill_aware为默认关闭的路由选项，Run15/16重复证明局部短请求TTFT约6.7→.7s，但总TPS无收益。完整历史以Run manifest/summary及point/source_notes定位，不恢复任何旧队列。

## 接入与操作

166：ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166。
167 Mac socket当前失效；从166 ssh -o BatchMode=yes root@172.16.10.167。
现场读SSH_AND_OPERATIONS.md、PD_START_AND_TEST_GUIDE.md及deploy/HANDOFF.md。native请求显式禁HTTP proxy（curl --noproxy '*'、ProxyHandler({})/trust_env=False）；Zcode保留既有互联网proxy。
CLI0.16.5的默认deepseek-flash SSE曾十分钟timeout；帮助列--settings但实际拒绝。当前真实/usr/local/bin/zcode在任务私有cwd /data/tiankuan/wio/glm52-pd/deploy/private/zcode-relay-work使用已有tiankuan-dsv/deepseek-v4-flash-0731 provider，Job→CLI→Result→bridge已多次VALID。全局配置未改，凭据private600不发布。所有失败Jobs原样保留，不冒充执行成功。
