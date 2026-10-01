# Native D prefill budget candidate

Run22 raises only D max-num-batched-tokens to8192, retaining native K5, eager draft, target FULL_DECODE_ONLY, TP/EP/DCP16, gmu.87 and request-contract parser. P21 stays4096. Run21 full counters show10252 uncached tokens/request and native prefill5.6–5.7s with almost no queue. Chunk count, shape costs and KV/headroom tradeoff require real E2E. Native operator implementation unchanged. Not Current or capacity proof.
