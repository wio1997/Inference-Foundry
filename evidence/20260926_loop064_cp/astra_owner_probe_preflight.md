# Astra High：Run286 ownership probe 启动前复核

2026-09-26，仅只读源码审查。已确认state范围修正后，无阻止启动P0 census的已见Graph/单位错误；尚不构成删写安全证明。

## 已修正问题

初稿state按最后token页±1取页。历史Run75 layer2 state block_size=2，8token跨4–5页，会漏前页。现已改为 [floor(start/block_size), ceil((start+qlen)/block_size))，并将历史state读域标unknown。
压缩页标为conservative_prefix_page_envelope_including_partial，正确，不能称精确native read集。

## 单位与Graph

FixedTargetMetadataUpdater.update将state.target_seq_lens复制到每个req.seq_lens，start_pos=seq−8；bootstrap/vllm_target_metadata_handoff.py绑定这些张量。因此seq/start是原token单位，compressed包络除ratio*block_size一次正确。
当前包络用global seq覆盖完整请求前缀；local_seq带CP causal offset，应另存以区分当前read与未来可见前沿。
local_counts按实测qsl与CP区间交集，owner_full_update_rows按完整owner请求qsl差求和，正确区分12本地query与条件下16更新行。
observe在derived metadata后、Target前，只读且不改metadata；FULL Graph保留。CPU拷贝/文件写会同步并扰动时序，此Run不能作性能裁决。

## 证据边界及最小补项

- limit=0不截断ownership，因为该调用在limit检查之前；也意味着常规snapshot/actual compressor slot验证不运行，不要声称actual scatter已测。
- cohort_seq是worker局部构造计数，不是request身份或天然Runtime cohort id。需与all8 Runtime报告关联；缺request_id须标局限。
- serve.sh max-num-seqs16仍可能handoff缺口；48+12客户端不自动等于5个Runtime cohort，按实际记录计数。
- min(table.width,end)可隐藏越界；建议输出unclipped_end/clipped及负/零页计数。
- 各source现使用主attn seq/start/qsl推导，固定绑定预期一致，但宜一次assert或记录source向量指纹。
- inventory只覆盖layer2有关views；混合layer alias按storage/byte区间解释，不同cache同page号不是alias。
- DSpark全部storage、state历史读域、实际scatter、prefix hash/refcount/COW/connector及下一cohort有效前沿仍未闭合。P0可以先跑，但unknown不能变成安全。
- emitted_token_count是Host mirror可落后一cycle，不作当轮精确acceptance；parked的local_qsl仍可非零。

下一步先审owner稳定性/完整请求行数/明显physical冲突，再对具体未知加probe，不先删96→16。
