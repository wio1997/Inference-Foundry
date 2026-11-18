# 完整目标和草稿图已捕获，前缀缓存布局仍待接通

G474在相同固定SGLang镜像、同一8卡910B1解码主机、原始权重和TP8/EP8/DSPARK5下，八个rank分别完成目标bs4/8/12/16与草稿bs8/16完整图捕获，草稿sampling折入图内。16份target/draft装载ledger均通过。此前CUDA RoPE路径与强制INT8覆盖草稿BF16问题在本次图捕获中未再出现。这是捕获阶段证据，尚不等于完整服务或语义通过。

随后原生decode radix布局检查拒绝非unified的DeepSeekV4池，D按自身控制器清理并释放八卡。完整KV分配617088 tokens，不代表已执行263296长请求。P已开放API，但原生PD、质量、缓存复用与AISBench尚未验证。G479功能检查等待指定P/D就绪900秒到期，没有提交生成请求；该超时是观察边界，不是质量失败。

固定源码明确说明unified布局不支持C1/C2低压缩比，不能直接切换标志来通过检查。原生注册器还仅依据请求表存在sidecar字段添加C128组件，而本模型没有C128池；组件后续读取C128 allocator.page_size，需核对和限定真实组件能力。保留前缀缓存目标，先验证FULL页派生的C1/C2所有权、SWA尾部生命周期和请求私有C2状态，再接受任何适配。

- [逐rank图事件与限定验收](../evidence/full_model_G474/2204/graph_radix_assessment.json)
- [完整日志](../evidence/full_model_G474/2204/server.log)
- [D清理释放](../evidence/full_model_G474/2204/state.json)
- [功能等待终态](../evidence/functional_G479/functional/state.json)
- [固定镜像布局检查源码](../code/native_radix_G491/fixed_source/sglang/srt/mem_cache/kv_cache_builder.py)
- [固定镜像注册器](../code/native_radix_G491/fixed_source/sglang/srt/mem_cache/registry.py)
- [真实C128组件契约](../code/native_radix_G491/fixed_source/sglang/srt/hardware_backend/npu/dsv4/c128_sidecar_component.py)
