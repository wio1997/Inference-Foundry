# 当前产品代码导航

**当前目标GLM-5.3 W8A8，用户已确认两机上传完成**：目录`/data/tiankuan/wio/GLM-5.3-w8a8`、服务名`glm-53`。Run245已退休旧完整副本/public/observer，保留历史状态；两边新权重/MTP实际加载后因KV预算校验失败并清理。Run246修正2GiB缓存后实际ready，Run247/248已通过一条2334prompt/32生成token的原生P166→KV→D167验证；最新证据沿[HANDOFF](HANDOFF.md)与Run248 execution_summary发现，驻留身份仍须fresh重验。`glm52-single`、`glm52-pd/deploy`仍是实际资源名称。8000公共入口尚未替换验收，完整API/性能未成立，Current=None。

**Current=None**，依据[HANDOFF](HANDOFF.md)、[当前性能stack](CURRENT_PERFORMANCE_STACK.md)。本轮原生PD启动入口是Run246冻结的`pd_functional_run.py`→`runtime_bundle/native_acl_lifecycle.py`→实际vLLM/Ascend scheduler/runner/Mooncake；没有旧PP/worker插件或gateway fallback。单条nativePD功能参考已通过，结束后两机idle/KVusage0，无性能patch或完整产品裁决。按[目标纠正裁决](records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)与新Reset定向研究后续完整入口。

以下表保留checkpoint159/Run244时点的旧完整副本代码导航，所列ACTIVE仅表示当时采用，当前服务已由Run245停止；不按这些历史入口启动或操作。Run242诊断后当时恢复过原始插件与V14。下次先用HANDOFF→最新Run→现场argv/config重验；不按最大版本号猜现役代码。

路径缩写（服务器绝对根见[RECOVERY_INDEX](RECOVERY_INDEX.md)）：`U=G/records/points/GLM-OPT-0002/runs/GLM-RUN-0241`；`R=G/records/points/GLM-OPT-0002/runs/GLM-RUN-0242`；`B=U/runtime_bundle`；`E166=D/plugins/local_pp239`；`E167=D/plugins/local_pp211`；`V=/vllm-workspace/vllm/vllm`；`A=/vllm-workspace/vllm-ascend/vllm_ascend`（V/A在容器内）。证据简称：**M**=[Run244 manifest](records/points/GLM-OPT-0002/runs/GLM-RUN-0244/manifest.json)，沿其public_config / native_fresh_audit / artifacts读取；**S**=`R/restored/service_config.json`；**N**=`R/restored/native_engines_resident.json`（plans/roots/members）；**I**=M.native_fresh_audit。S/N等服务器文件由M追溯，不是第二份事实表。

| Component | checkpoint159 historical code path | Status at checkpoint159 | Historical owner/consumer | Evidence |
| --------- | ----------------- | ------ | -------------- | -------- |
| public service entry | `B/native_engines_service_entry_v14.py`；Git阅读[冻结V14](records/points/GLM-OPT-0002/runs/GLM-RUN-0240/runtime_bundle/native_engines_service_entry_v14.py) | ACTIVE | public；`D/plugins/local_engines137/native_acl_lifecycle.py`包裹生命周期 | M→public_config；`R/restored/public_host_owner.json`的argv + process |
| final product candidate entry | 标准PD入口尚未确定；历史参考为[Run101](records/points/GLM-OPT-0002/runs/GLM-RUN-0101/summary.md)实际V9 + PDv2 + 原生Mooncake链；当前V14仍是现役完整副本入口 | UNKNOWN | Sol研究完整P→KV→D合同；旧PDv2/v3的fallback及DP1限制不能自动成为产品设计 | PD_SCOPE_CORRECTION、实际connector源码与Run101 raw；待完整功能、代码级matched A/B及完整E2E产品裁决 |
| gateway/router | `B/response_affinity_gateway_v12.py`→`B/response_affinity_gateway_v11.py`；Git读[V12](runtime/response_affinity_gateway_v12.py)与[冻结V11](records/points/GLM-OPT-0002/runs/GLM-RUN-0240/runtime_bundle/response_affinity_gateway_v11.py) | ACTIVE | V14 public，含Responses/affinity | V14的runpy路径；B中import链、S |
| placement | `B/capability_placement.py`→shape_split_placement.py→work_seconds_placement.py；Git读[冻结入口](records/points/GLM-OPT-0002/runs/GLM-RUN-0240/runtime_bundle/capability_placement.py) | ACTIVE | 完整请求副本选择；兼容性过滤 | N.placement / compatibility；冻结入口import链 |
| scheduler | `E166/issue_budget_scheduler_v5.py`、`E167/issue_budget_scheduler_v5.py`→v3 mixin→native scheduler；Git读[V5](runtime/issue_budget_scheduler_v5.py) | ACTIVE | native engines；admission/issue budget | N.plans/roots的scheduler-cls + I；plugin policy与源码hash |
| controller | [controller.py](runtime/controller.py)→[phase_runner.py](runtime/phase_runner.py)；实际执行用Run冻结sources | UNKNOWN | 唯一实验排程者；服务驻留不代表controller活跃 | M.controller_state（242 completed、243/244 failed并退出）；RECOVERY_INDEX中owner/lock→现场process，禁止重放 |
| P-side runtime | `A/worker/`中实际runner的native prefill；没有独立现役P服务 | NATIVE_UNCHANGED | E166/E167完整请求engine；members命名不能证明PD | N.plans/roots及native argv；无独立P→KV→D链 |
| D-side runtime | `E166/atomic_mq_worker.py`、`E167/atomic_mq_worker.py`→`A/worker/worker.py`及各自runner；Git读[worker入口](runtime/atomic_mq_worker.py) | ACTIVE | 166 V2 / 167 V1完整请求engine的控制扩展 | N启动计划 + I；按实际PYTHONPATH选plugin，算子沿用native |
| PD transport | [native_pd_transport_v3.py](runtime/native_pd_transport_v3.py) / [native_pd_geometry.py](runtime/native_pd_geometry.py)（现役不接入） | HISTORICAL | 旧标准PD/bridge原型；现役走完整请求HTTP | N与native argv；旧transport Run/REUSE证据，非当前KV链 |
| KV identity/state | `V/v1/core/` + 实际`A/worker/`runner的native KV/cache state | NATIVE_UNCHANGED | 各native engine独立拥有 | N、I及当前安装源码；public response owner路径从S.environment发现 |
| KV bridge identity | [registered_kv_identity.py](runtime/registered_kv_identity.py)（现役不接入） | HISTORICAL | 旧PD状态/identity原型 | transport import与对应历史manifest；不把文件存在当部署 |
| MTP control | `A/spec_decode/`及当前runner draft/accept/commit路径 | NATIVE_UNCHANGED | 各native engine，按runner/speculative config选实现 | N.plans、I、实际安装源码；native输出提交不由public接管 |
| MTP candidate | [grammar_draft_control.py](records/points/GLM-OPT-0002/research/native_grammar_immediate_20261005_v2/grammar_draft_control.py) | CANDIDATE | 未接入现役；不自动继承为下一研究主线 | 同目录[candidate.json](records/points/GLM-OPT-0002/research/native_grammar_immediate_20261005_v2/candidate.json) / source_analysis.json |
| Graph control | 166：`E166/draft_dense_capture.py`及native runner；167：native runner（无该plugin文件） | ACTIVE | 166控制扩展；167 native Graph | N.plans、I的capture proof与plugin hash；开关由实际effective config发现 |
| sampling | `V/v1/sample/`及实际native runner采样/输出提交 | NATIVE_UNCHANGED | native engines | 当前安装路径 + N runner；不按public转发行为推断采样改造 |
| tokenization/request preparation | `B/native_chat_token_memo.py`→native renderer/tokenizer；Git读[token memo](runtime/native_chat_token_memo.py) | ACTIVE | V12 transport包装 + 各engine原生准备 | B中NativeChatTokenMemoTransport import，S.environment，模型/tokenizer argv |
| benchmark adapter | [adapters/aisbench](adapters/aisbench/README.md)；容器`/opt/aisbench-benchmark` | CANDIDATE | 后续matched A/B客户端；不是驻留服务组件 | RECORDING + 选定Run spec/stage argv；安装路径按RECOVERY_INDEX发现 |
| observer | [native_identity_observer.py](runtime/native_identity_observer.py)、`B/sse_observer.py` | ACTIVE | 现役身份观测 / public SSE观测 | `R/restored/identity_observer/`及B import链；只读恢复不启动其quarantine逻辑 |
| profiler/trace | [profiling入口](docs/research/VLLM_ASCEND_PROFILING.md)；实际profile/trace配置指向的工具 | UNKNOWN | 按需诊断；不宣称正在采集 | M.artifacts→profile gate/trace；S.environment→trace/raw路径 |

版本判定：先查Current/路径缩写、public入口与目标组件行；研究public时从V14冻结入口读起，其他问题直接读对应组件，再沿**实际B的import/runpy与native argv/PYTHONPATH**定向深入。Run241的B继承Run240；Git只保存Run240部分冻结源码，其余实际文件由M.artifacts核验，不能假设root `runtime/`同名文件完全相同。没有`response_affinity_gateway_v14.py`这个现役入口；V12/V11及scheduler v3仍是现役依赖，不能因编号旧而退休。

未在上述启动/import链中的v1/v2/v10等版本是HISTORICAL研究源码；只有新退出记录明确标记不再进入默认路径的才称RETIRED。旧入口被V14替代的原因和边界以相应Run证据为准，不批量推断REJECT。保留全部源码和证据；不按目录最大版本号选代码、不扫描所有Run重建依赖、不由本表激活新架构或候选。
