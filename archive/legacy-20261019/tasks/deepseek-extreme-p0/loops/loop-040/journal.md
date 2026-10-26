# 执行日志

- `2026-09-25T00:48:01Z` Loop 已冻结。下一步：Run131: parse Run107 bounded trace to map 96-token target quant-matmul device kernels to CPU operator calls and shapes; use source audit, no service restart.

- `2026-09-25T00:54:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run131`（profile）。

- `2026-09-25T00:54:57Z` Run `run131` 记录为 `pass`；正确性为 `not-applicable`。Run107 15 valid synchronized windows: exactly 236 quant-matmul and 43 GMM1 kernels/target. Layer-position bins match config ratios: first layers4/5, c4 layers6, c128 layers5, one tail kernel. Quant kernel sum median4.8223ms; c4 layer median120.28us vs c128 103.83us. Ordered association, no exact module/pointer proof.

- `2026-09-25T00:56:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run132`（profile）。

- `2026-09-25T01:17:19Z` Run `run132` 记录为 `pass`；正确性为 `not-applicable`。Legal 12/12 1024-token requests; 8-rank W8A8 call logs. Per rank 43 target self_attn.wkv prefixes seen twice at graph setup and 3 MTP prefixes seen 271 times. All x[96,4096], w[4096,512]. Probe covers only wkv, so no shared-input conclusion. Borrowed source restored, service stopped.

- `2026-09-25T01:19:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run133`（design-check）。

- `2026-09-25T01:20:35Z` Run `run133` 记录为 `pass`；正确性为 `not-applicable`。Source-backed DSA CP audit under frozen flags: split wq_b and c4 indexer wq_b reuse one qr dynamic quant; wq_a local and wkv gathered token domains differ; weights_proj unquantized. Obvious repeated-input quant hypothesis unsupported. Full per-call runtime tags still absent.

- `2026-09-25T01:20:35Z` 主控结论为 `PIVOTED`。Run131 mapped 236 quant kernels/target and alternating c4/c128 pattern. Run132 8-rank legal call probe covered wkv but no new shared-input pair. Run133 source audit shows the apparent split/indexer repeats already share quantization; no semantics-valid projection fusion candidate currently warrants another 8-rank service run. Shift to HC/clone/cache source and trace attribution; quant family remains later candidate if a concrete legal replacement appears.
