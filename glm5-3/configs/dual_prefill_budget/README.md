# Native D prefill budget candidate

Run22 raises only D max-num-batched-tokens to8192, retaining native K5, eager draft, target FULL_DECODE_ONLY, TP/EP/DCP16, gmu.87 and request-contract parser. P21 stays4096. Run21 full counters show10252 uncached tokens/request and native prefill5.6–5.7s with almost no queue. Chunk count, shape costs and KV/headroom tradeoff require real E2E. Native operator implementation unchanged. Not Current or capacity proof.

Run22 cold valid but fixture INVALID; Run23 corrected six matched prefix/tails +PD+dynamic valid,13 new/17538 outputs. P/D tailTTFT~6.00/5.89s still above4s;D8192 KV258264 vs296902,13.01% less. Host/cache conditions retained, no causal/repeated gain or Current. See Run23 reduction; both engines remain resident while researching native DP/EP alternatives.
